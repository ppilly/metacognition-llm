#!/usr/bin/env bash
# Local pipeline on the 4 GB GPU for run RUN (default: local): config configs/$RUN.yaml, the fixed task
# subsets configs/tasks/${RUN}_train.json and configs/tasks/${RUN}_eval.json, every stage appending to
# runs/${RUN}_<stage>.log, then a report in runs/${RUN}_report. MUSE and its random-selection control (the
# comparison that isolates M_sa) run before the lowest-selection check. OVERRIDES="key=value ..." applies to every stage, e.g. MUSE with
# the paper's M_sa on the same models (README, "Running"):
#   OVERRIDES="self_assessment.input=paper self_assessment.chunk_stride=null self_assessment.adapt_rollouts=none
#   run.eval_variant=paper_sa" STAGES=train-sa AGENTS=muse scripts/run_local.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs

PY=${PY:-python}   # the muse conda env; or set PY to its interpreter
export ALFWORLD_DATA=${ALFWORLD_DATA:-$PWD/data/alfworld}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
RUN=${RUN:-local}
read -ra OV <<< "${OVERRIDES:-}"
CFG=configs/$RUN.yaml
RUN_DIR=$($PY -c "from muse.config import load_config; print(load_config('$CFG').run.out_dir)")
TRAIN=("env.task_ids=$(cat configs/tasks/${RUN}_train.json)" env.num_tasks=null)
EVAL=("env.task_ids=$(cat configs/tasks/${RUN}_eval.json)" env.num_tasks=null)
AGENTS=${AGENTS-"react reflexion reflexion_sl muse muse_random muse_lowest"}   # AGENTS= runs no evals
STAGES=${STAGES-"collect train-sft train-dpo train-sa"}   # STAGES= skips straight to the evals (they resume per task)

for stage in $STAGES; do
  echo "$(date '+%F %T') $stage"
  $PY -m muse.cli "$stage" -c $CFG "${TRAIN[@]}" "${OV[@]}" >> "runs/${RUN}_$stage.log" 2>&1
done
for agent in $AGENTS; do
  echo "$(date '+%F %T') eval $agent"
  $PY -m muse.cli eval -c $CFG --agent "$agent" "${EVAL[@]}" "${OV[@]}" >> "runs/${RUN}_eval_$agent.log" 2>&1
done
if [ -n "$AGENTS" ]; then
  $PY -m muse.cli report --runs "$RUN_DIR" --out "runs/${RUN}_report" >> "runs/${RUN}_report.log" 2>&1
fi
echo "$(date '+%F %T') done"
