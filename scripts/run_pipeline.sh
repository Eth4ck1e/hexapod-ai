#!/usr/bin/env bash
# scripts/run_pipeline.sh — scaffold priors → BC pretrain → AMP-PPO, one robot.
#
# The v27 PhantomX recipe (scripts/launch.py), parameterised by robot so the
# mini hexapod gets the identical treatment. Stages whose output already
# exists are skipped, so a crashed run resumes where it stopped.
#
#   HEXAPOD_ROBOT=mini_hexapod TAG=v1 bash scripts/run_pipeline.sh
#
# Env knobs (all optional):
#   HEXAPOD_ROBOT   phantomx | mini_hexapod            (default phantomx)
#   TAG             version tag for artifact names     (default v1)
#   PRIOR_STEPS     prior rollout steps per env        (default 250 → ~1M transitions @ 4096 envs)
#   AMP_SEGMENTS    outer PPO+disc iterations          (default 10)
#   AMP_STEPS       PPO env-steps per segment          (default 50000000)
#   STYLE_WEIGHT    λ_style                            (default 1.0)
#   PY              python interpreter                 (default ~/.venv-mjx/bin/python)
#   STAGES          subset to run, e.g. "priors bc"    (default "priors bc amp")
#   DR_SCALE        dynamics domain randomization 0..1 (default 0 = nominal physics)
#   PRIORS          override priors npz (default checkpoints/amp_priors_<robot>_<TAG>.npz)
#   BC_DIR          override BC checkpoint dir (default checkpoints/bc_pretrained_jax_<robot>_<TAG>)
#   AMP_RESTORE     policy params.pkl to start AMP from (default $BC_DIR/params.pkl;
#                   point at a previous run's iterN/final/params.pkl to continue it)
#   AMP_RESTORE_DISC  matching discriminator.pkl (default: fresh discriminator)
#   NICE            nice level for the python processes (default 10 — the
#                   workstation runs other CPU work; training is GPU-bound)
#   MEM_BUDGET_MB   kill the stage if its host RSS exceeds this (default 10000;
#                   0 = no cap). Protects the user's other jobs from an OOM.
#
# Artifacts (names follow the robot so lineages never collide):
#   checkpoints/amp_priors_<robot>_<TAG>.npz
#   checkpoints/bc_pretrained_jax_<robot>_<TAG>/params.pkl
#   checkpoints/amp_to_<robot>_<TAG>/iter*/final/{params,discriminator}.pkl
#   logs/stdout/<stage>_<robot>_<TAG>.log
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export HEXAPOD_ROBOT="${HEXAPOD_ROBOT:-phantomx}"
export PYTHONPATH=.
TAG="${TAG:-v1}"
PRIOR_STEPS="${PRIOR_STEPS:-250}"
AMP_SEGMENTS="${AMP_SEGMENTS:-10}"
AMP_STEPS="${AMP_STEPS:-50000000}"
STYLE_WEIGHT="${STYLE_WEIGHT:-1.0}"
PY="${PY:-$HOME/.venv-mjx/bin/python}"
STAGES="${STAGES:-priors bc amp}"
DR_SCALE="${DR_SCALE:-0}"
NICE="${NICE:-10}"
MEM_BUDGET_MB="${MEM_BUDGET_MB:-10000}"

ROBOT="$HEXAPOD_ROBOT"
PRIORS="${PRIORS:-checkpoints/amp_priors_${ROBOT}_${TAG}.npz}"
BC_DIR="${BC_DIR:-checkpoints/bc_pretrained_jax_${ROBOT}_${TAG}}"
AMP_RESTORE="${AMP_RESTORE:-$BC_DIR/params.pkl}"
AMP_RESTORE_DISC="${AMP_RESTORE_DISC:-}"
RUN="amp_to_${ROBOT}_${TAG}"
mkdir -p logs/stdout checkpoints

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
want() { [[ " $STAGES " == *" $1 "* ]]; }

# run_stage <logfile> <cmd...> — runs the command under `nice`, tees its
# output, and watches the process tree's resident memory: if it crosses
# MEM_BUDGET_MB the stage is killed (exit 137) instead of starving the rest
# of the machine. Peak RSS is logged either way.
run_stage() {
  local logfile="$1"; shift
  nice -n "$NICE" "$@" > >(tee "$logfile") 2>&1 &
  local pid=$! peak=0 rss
  while kill -0 "$pid" 2>/dev/null; do
    rss=$(ps -o rss= --pid "$pid" --ppid "$pid" 2>/dev/null | awk '{s+=$1} END {print int(s/1024)}')
    rss="${rss:-0}"
    (( rss > peak )) && peak=$rss
    if (( MEM_BUDGET_MB > 0 && rss > MEM_BUDGET_MB )); then
      log "MEMORY BUDGET EXCEEDED: rss=${rss} MB > ${MEM_BUDGET_MB} MB — killing pid $pid"
      pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null
      sleep 5; pkill -KILL -P "$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      log "stage killed (peak rss ${peak} MB)"
      return 137
    fi
    sleep 5
  done
  local rc=0; wait "$pid" || rc=$?
  log "stage exit=$rc  peak host rss=${peak} MB"
  return "$rc"
}

log "pipeline start  robot=$ROBOT tag=$TAG stages='$STAGES' dr_scale=$DR_SCALE nice=$NICE mem_budget_mb=$MEM_BUDGET_MB python=$PY"
log "priors=$PRIORS  bc_dir=$BC_DIR  amp_restore=$AMP_RESTORE  amp_restore_disc=${AMP_RESTORE_DISC:-<fresh>}"
"$PY" -c "import jax; print('  jax devices:', jax.devices())"

if want priors; then
  if [[ -f "$PRIORS" ]]; then
    log "priors exist → skip ($PRIORS)"
  else
    log "STAGE 1/3  scaffold priors → $PRIORS"
    run_stage "logs/stdout/priors_${ROBOT}_${TAG}.log" \
        "$PY" -u amp/prior_data.py --cmd-mask paper_stance --n-steps "$PRIOR_STEPS" \
        --out "$PRIORS"
  fi
fi

if want bc; then
  if [[ -f "$BC_DIR/params.pkl" ]]; then
    log "BC checkpoint exists → skip ($BC_DIR)"
  else
    log "STAGE 2/3  BC pretrain → $BC_DIR"
    run_stage "logs/stdout/bc_${ROBOT}_${TAG}.log" \
        "$PY" -u scripts/pretrain_bc_jax.py --cmd-mask paper_stance --action-space foot \
        --out-dir "$BC_DIR"
  fi
fi

if want amp; then
  if [[ -f "checkpoints/$RUN/iter${AMP_SEGMENTS}/final/params.pkl" ]]; then
    log "AMP run complete → skip (checkpoints/$RUN)"
  else
    log "STAGE 3/3  AMP-PPO → checkpoints/$RUN  (${AMP_SEGMENTS} × ${AMP_STEPS} steps, dr_scale=$DR_SCALE)"
    disc_args=()
    [[ -n "$AMP_RESTORE_DISC" ]] && disc_args=(--restore-discriminator "$AMP_RESTORE_DISC")
    run_stage "logs/stdout/amp_${ROBOT}_${TAG}.log" \
        "$PY" -u scripts/train_jax_amp.py \
        --restore "$AMP_RESTORE" \
        --priors "$PRIORS" \
        --cmd-mask paper_stance --action-space foot \
        --style-weight "$STYLE_WEIGHT" \
        --segments "$AMP_SEGMENTS" --steps-per-segment "$AMP_STEPS" \
        --partition-disc --dr-scale "$DR_SCALE" "${disc_args[@]}" \
        --run "$RUN"
  fi
fi

log "pipeline done  robot=$ROBOT tag=$TAG"
echo PIPELINE_DONE
