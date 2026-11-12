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

ROBOT="$HEXAPOD_ROBOT"
PRIORS="checkpoints/amp_priors_${ROBOT}_${TAG}.npz"
BC_DIR="checkpoints/bc_pretrained_jax_${ROBOT}_${TAG}"
RUN="amp_to_${ROBOT}_${TAG}"
mkdir -p logs/stdout checkpoints

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
want() { [[ " $STAGES " == *" $1 "* ]]; }

log "pipeline start  robot=$ROBOT tag=$TAG stages='$STAGES' python=$PY"
"$PY" -c "import jax; print('  jax devices:', jax.devices())"

if want priors; then
  if [[ -f "$PRIORS" ]]; then
    log "priors exist → skip ($PRIORS)"
  else
    log "STAGE 1/3  scaffold priors → $PRIORS"
    "$PY" -u amp/prior_data.py --cmd-mask paper_stance --n-steps "$PRIOR_STEPS" \
        --out "$PRIORS" 2>&1 | tee "logs/stdout/priors_${ROBOT}_${TAG}.log"
  fi
fi

if want bc; then
  if [[ -f "$BC_DIR/params.pkl" ]]; then
    log "BC checkpoint exists → skip ($BC_DIR)"
  else
    log "STAGE 2/3  BC pretrain → $BC_DIR"
    "$PY" -u scripts/pretrain_bc_jax.py --cmd-mask paper_stance --action-space foot \
        --out-dir "$BC_DIR" 2>&1 | tee "logs/stdout/bc_${ROBOT}_${TAG}.log"
  fi
fi

if want amp; then
  if [[ -f "checkpoints/$RUN/iter${AMP_SEGMENTS}/final/params.pkl" ]]; then
    log "AMP run complete → skip (checkpoints/$RUN)"
  else
    log "STAGE 3/3  AMP-PPO → checkpoints/$RUN  (${AMP_SEGMENTS} × ${AMP_STEPS} steps)"
    "$PY" -u scripts/train_jax_amp.py \
        --restore "$BC_DIR/params.pkl" \
        --priors "$PRIORS" \
        --cmd-mask paper_stance --action-space foot \
        --style-weight "$STYLE_WEIGHT" \
        --segments "$AMP_SEGMENTS" --steps-per-segment "$AMP_STEPS" \
        --partition-disc \
        --run "$RUN" 2>&1 | tee "logs/stdout/amp_${ROBOT}_${TAG}.log"
  fi
fi

log "pipeline done  robot=$ROBOT tag=$TAG"
echo PIPELINE_DONE
