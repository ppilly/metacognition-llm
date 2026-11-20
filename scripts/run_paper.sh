#!/usr/bin/env bash
# Full replication of the LLM-based results (Tables 2-5, Fig. 9).
#   CONFIG=configs/default.yaml scripts/run_paper.sh            # Mistral-7B-Instruct-v0.2: Tables 2-4, Fig. 9
#   CONFIG=configs/openorca.yaml AGENTS="react reflexion muse" scripts/run_paper.sh   # Table 5
#   CONFIG=configs/openelm.yaml  AGENTS="react reflexion muse" scripts/run_paper.sh   # Table 5
# Every stage is a separate process (each loads its own model). collect and eval resume where they stopped;
# the training stages start over, so restart after training with STAGES= (see README, Restarting after a hard pause or a crash).
# Extra arguments are passed through as config overrides, e.g. `scripts/run_paper.sh env.num_tasks=10`.
set -euo pipefail

CONFIG=${CONFIG:-configs/default.yaml}
AGENTS=${AGENTS:-"react reflexion reflexion_sl muse_lowest muse_random muse"}
: "${ALFWORLD_DATA:?set ALFWORLD_DATA to the ALFWorld data directory}"
PY=${PY:-python}
STAGES=${STAGES-"collect train-sft train-dpo train-sa"}

needs_pretraining=false
for a in $AGENTS; do
  case $a in reflexion_sl|muse*) needs_pretraining=true ;; esac
done

if $needs_pretraining; then
  # Pre-deployment training (Alg. 3): Reflexion on the 140 seen tasks, then SFT / DPO / M_sa
  for stage in $STAGES; do
    $PY -m muse.cli "$stage" -c "$CONFIG" "$@"
  done
fi

# Deployment on the 134 unseen tasks (Alg. 4); each run has No Adaptation, adaptation and test episodes
for agent in $AGENTS; do
  $PY -m muse.cli eval -c "$CONFIG" --agent "$agent" "$@"
done
