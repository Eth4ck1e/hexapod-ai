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
#   BC_ARGS         extra pretrain_bc_jax.py flags, e.g. "--n-epochs 18"
#   CMD_MASK        cmd slots sampled in every stage (default paper_stance; paper_tilt = + pitch/roll)
#   DR_SCALE        dynamics domain randomization 0..1 (default 0 = nominal physics)
#   LOG_STD         exploration log-std baked by BC (default -4.0; joint-target policies: -6)
#   LEARNING_RATE   PPO learning rate (default 3e-4; joint-target policies: 1e-4)
#   ENTROPY_COST    PPO entropy bonus (default 1e-3)
#   CLIP_EPS        PPO ratio clip (default 0.3; joint-target policies: 0.2)
#   VALUE_EPOCHS    BC critic-pretraining epochs (default 3; 0 = random critic as before)
#   DET_EVAL=1      score [EVAL] with the deterministic policy (joint-target policies)
#   NUM_ENVS        PPO parallel envs (default chain_train.NUM_ENVS=4096; 2048 roughly halves the
#                   host-RSS compile spike — joint-target graphs hit 11–20 GB at 4096)
#   ACTION_SPACE    foot | joint (default: the chain_train session's)
#   PRIORS          override priors npz (default checkpoints/amp_priors_<robot>_<TAG>.npz)
#   BC_DIR          override BC checkpoint dir (default checkpoints/bc_pretrained_jax_<robot>_<TAG>)
#   AMP_RESTORE     policy params.pkl to start AMP from (default $BC_DIR/params.pkl;
#                   point at a previous run's iterN/final/params.pkl to continue it)
#   AMP_RESTORE_DISC  matching discriminator.pkl (default: fresh discriminator)
#   NICE            nice level for the python processes (default 10 — the
#                   workstation runs other CPU work; training is GPU-bound)
#   MEM_BUDGET_MB   kill the stage if its host RSS exceeds this (default 10000;
#                   0 = no cap). Protects the user's other jobs from an OOM.
#   SEGMENT_ISOLATION  1 (default): run every AMP segment in its own python
#                   process (restore from the previous iter) so host RSS resets
#                   each segment (~6 GB peak instead of +0.6 GB/segment growth,
#                   which killed v1dr at segment 10). 0 = one long process.
#
# Artifacts (names follow the robot so lineages never collide):
#   checkpoints/amp_priors_<robot>_<TAG>.npz
#   checkpoints/bc_pretrained_jax_<robot>_<TAG>/params.pkl
#   checkpoints/amp_to_<robot>_<TAG>/iter*/final/{params,discriminator}.pkl
#   logs/stdout/<stage>_<robot>_<TAG>.log
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export HEXAPOD_ROBOT="${HEXAPOD_ROBOT:-phantomx}"
# Persistent XLA compilation cache: every AMP segment is its own process compiling
# the same graphs, and the compile is where host RSS spikes (11–20 GB observed on
# joint-target graphs). With the cache only the first segment pays that peak.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax_xla}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="${JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES:--1}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
export PYTHONPATH=.
TAG="${TAG:-v1}"
PRIOR_STEPS="${PRIOR_STEPS:-250}"
AMP_SEGMENTS="${AMP_SEGMENTS:-10}"
AMP_STEPS="${AMP_STEPS:-50000000}"
STYLE_WEIGHT="${STYLE_WEIGHT:-1.0}"
PY="${PY:-$HOME/.venv-mjx/bin/python}"
STAGES="${STAGES:-priors bc amp}"
CMD_MASK="${CMD_MASK:-paper_stance}"   # paper_tilt adds pitch/roll (mini web controller tilt stick)
DR_SCALE="${DR_SCALE:-0}"
NICE="${NICE:-10}"
MEM_BUDGET_MB="${MEM_BUDGET_MB:-10000}"
SEGMENT_ISOLATION="${SEGMENT_ISOLATION:-1}"

ROBOT="$HEXAPOD_ROBOT"
# Action space follows the chain_train session (foot residual for phantomx /
# mini_hexapod, joint targets for mini_hexapod_pure) unless overridden.
ACTION_SPACE="${ACTION_SPACE:-$("$PY" -c 'import sys; sys.path.insert(0, "scripts"); from chain_train import ACTION_SPACE; print(ACTION_SPACE)' 2>/dev/null | tail -1)}"
ACTION_SPACE="${ACTION_SPACE:-foot}"
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
  : > "$logfile"
  # tee -a: the watchdog appends [mem] lines to the same file; without O_APPEND
  # tee keeps its own offset and overwrites them.
  nice -n "$NICE" "$@" > >(tee -a "$logfile") 2>&1 &
  local pid=$! peak=0 rss tick=0
  while kill -0 "$pid" 2>/dev/null; do
    rss=$(ps -o rss= --pid "$pid" --ppid "$pid" 2>/dev/null | awk '{s+=$1} END {print int(s/1024)}')
    rss="${rss:-0}"
    (( rss > peak )) && peak=$rss
    # Memory profile every ~60 s so a kill (or a close call) is explainable.
    (( tick++ % 12 == 0 )) && printf '[%s] [mem] rss=%s MB peak=%s MB budget=%s MB\n' "$(date '+%T')" "$rss" "$peak" "$MEM_BUDGET_MB" >> "$logfile"
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

log "pipeline start  robot=$ROBOT tag=$TAG stages='$STAGES' cmd_mask=$CMD_MASK action_space=$ACTION_SPACE log_std=${LOG_STD:-default} lr=${LEARNING_RATE:-default} entropy=${ENTROPY_COST:-default} dr_scale=$DR_SCALE nice=$NICE mem_budget_mb=$MEM_BUDGET_MB python=$PY"
log "priors=$PRIORS  bc_dir=$BC_DIR  amp_restore=$AMP_RESTORE  amp_restore_disc=${AMP_RESTORE_DISC:-<fresh>}"
"$PY" -c "import jax; print('  jax devices:', jax.devices())"

if want priors; then
  if [[ -f "$PRIORS" ]]; then
    log "priors exist → skip ($PRIORS)"
  else
    log "STAGE 1/3  scaffold priors → $PRIORS"
    run_stage "logs/stdout/priors_${ROBOT}_${TAG}.log" \
        "$PY" -u amp/prior_data.py --cmd-mask "$CMD_MASK" --n-steps "$PRIOR_STEPS" \
        --out "$PRIORS"
  fi
fi

if want bc; then
  if [[ -f "$BC_DIR/params.pkl" ]]; then
    log "BC checkpoint exists → skip ($BC_DIR)"
  else
    log "STAGE 2/3  BC pretrain → $BC_DIR"
    run_stage "logs/stdout/bc_${ROBOT}_${TAG}.log" \
        "$PY" -u scripts/pretrain_bc_jax.py --cmd-mask "$CMD_MASK" --action-space "$ACTION_SPACE" \
        ${LOG_STD:+--log-std "$LOG_STD"} ${VALUE_EPOCHS:+--value-epochs "$VALUE_EPOCHS"} ${BC_ARGS:-} --out-dir "$BC_DIR"
  fi
fi

if want amp; then
  if [[ -f "checkpoints/$RUN/iter${AMP_SEGMENTS}/final/params.pkl" ]]; then
    log "AMP run complete → skip (checkpoints/$RUN)"
  else
    log "STAGE 3/3  AMP-PPO → checkpoints/$RUN  (${AMP_SEGMENTS} × ${AMP_STEPS} steps, dr_scale=$DR_SCALE, isolation=$SEGMENT_ISOLATION)"
    amp_common=(--priors "$PRIORS" --cmd-mask "$CMD_MASK" --action-space "$ACTION_SPACE"
                --style-weight "$STYLE_WEIGHT" --steps-per-segment "$AMP_STEPS"
                ${LEARNING_RATE:+--learning-rate "$LEARNING_RATE"} ${ENTROPY_COST:+--entropy-cost "$ENTROPY_COST"}
                ${CLIP_EPS:+--clipping-epsilon "$CLIP_EPS"} ${NUM_ENVS:+--num-envs "$NUM_ENVS"}
                ${DET_EVAL:+--deterministic-eval}
                --partition-disc --dr-scale "$DR_SCALE" --run "$RUN")
    if [[ "$SEGMENT_ISOLATION" == "1" ]]; then
      # One process per segment. Resume-aware: segments whose params.pkl exist
      # are skipped and the next one restores from the latest complete iter.
      restore="$AMP_RESTORE"; restore_disc="$AMP_RESTORE_DISC"
      for (( seg=1; seg<=AMP_SEGMENTS; seg++ )); do
        it="checkpoints/$RUN/iter${seg}/final"
        if [[ -f "$it/params.pkl" ]]; then
          log "segment $seg/$AMP_SEGMENTS exists → skip"
        else
          log "segment $seg/$AMP_SEGMENTS  restore=$restore  disc=${restore_disc:-<fresh>}"
          disc_args=()
          [[ -n "$restore_disc" ]] && disc_args=(--restore-discriminator "$restore_disc")
          run_stage "logs/stdout/amp_${ROBOT}_${TAG}_seg${seg}.log" \
              "$PY" -u scripts/train_jax_amp.py --restore "$restore" --segments 1 \
              --iter-offset $(( seg - 1 )) "${disc_args[@]}" "${amp_common[@]}"
        fi
        restore="$it/params.pkl"
        [[ -f "$it/discriminator.pkl" ]] && restore_disc="$it/discriminator.pkl"
      done
    else
      disc_args=()
      [[ -n "$AMP_RESTORE_DISC" ]] && disc_args=(--restore-discriminator "$AMP_RESTORE_DISC")
      run_stage "logs/stdout/amp_${ROBOT}_${TAG}.log" \
          "$PY" -u scripts/train_jax_amp.py --restore "$AMP_RESTORE" \
          --segments "$AMP_SEGMENTS" "${disc_args[@]}" "${amp_common[@]}"
    fi
  fi
fi

log "pipeline done  robot=$ROBOT tag=$TAG"
echo PIPELINE_DONE
