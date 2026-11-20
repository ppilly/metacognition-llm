# Metacognition for LLM Agents

[![tests](https://github.com/ppilly/metacognition-llm/actions/workflows/test.yml/badge.svg)](https://github.com/ppilly/metacognition-llm/actions/workflows/test.yml)

An independent implementation of the LLM-based agents from [*Metacognition for Unknown Situations
and Environments (MUSE)*](https://doi.org/10.1016/j.neunet.2025.108131) (Valiente & Pilly, *Neural
Networks*, 2026), written by the paper's corresponding author.

MUSE is a self-improving AI agent that performs test-time adaptation to out-of-distribution tasks
from limited data by pairing metacognition learned during post-training with continual learning at
deployment.

This repository reproduces the comparative trends on ALFWorld reported in the paper (MUSE vs.
baselines), using Qwen2.5-1.5B-Instruct on a single 4 GB GPU. The paper used 3B and 7B models, so
absolute success rates here are not directly comparable. This README notes where the code follows
the paper and where it makes its own choices, some informed by additional experiments.

This implementation was written from the published paper, without the original code or data, and is
not affiliated with or endorsed by HRL Laboratories.

The repository covers the paper's LLM-based results (Sec. 4):

| Paper result | What produces it |
|---|---|
| Table 2, Fig. 9: self-assessment (AUROC2, metacognitive accuracy) | `eval --agent muse` |
| Tables 3 & 4: success rate / time-to-completion, No Adaptation vs. Adaptation | `eval` for all six agents |
| Table 5: Mistral-7B-OpenOrca and OpenELM-3B-Instruct | `configs/openorca.yaml`, `configs/openelm.yaml` |

The agents are ReAct, Reflexion, Reflexion with supervised learning (SFT + DPO), MUSE, and the two
MUSE controls that select by the lowest or by a random self-assessment.

[docs/beyond_the_paper.md](docs/beyond_the_paper.md) summarizes the implementation choices beyond
the paper, what it changes to run on a 4 GB GPU, and the extensions it adds, with the reasons.

## Results

One run of `configs/local.yaml`: Qwen2.5-1.5B-Instruct in 4-bit on a 4 GB GPU, 10 unseen tasks, each
with 5 adaptation episodes and then 5 independent test episodes (the No Adaptation episodes were
skipped).

| Agent | Success rate (95% CI) | Mean steps (failure = 100) |
|---|---|---|
| ReAct | 22% (7–44) | 82 |
| Reflexion | 14% (2–33) | 88 |
| Reflexion + SL | 24% (8–47) | 81 |
| MUSE | 44% (21–70) | 64 |

MUSE ranks first on both success rate and steps, as in the paper. The per-agent confidence intervals
overlap, so the paired comparison is the better test. Paired by game (9 games), MUSE succeeds 20 to
30 points more often than each baseline (permutation p = 0.039–0.055) and needs 16 to 24 fewer steps
(p = 0.031–0.039). After adaptation, its self-assessment separates successful from failed chunks
with AUROC 0.84. One ordering differs from the paper: Reflexion falls below ReAct. This is a single
run on 10 tasks, so these results support the paper's trends but do not establish effect sizes.

## Setup

```bash
git clone https://github.com/ppilly/metacognition-llm.git && cd metacognition-llm
conda create -n muse python=3.11 -y && conda activate muse   # vLLM >= 0.32 needs Python >= 3.11
conda install -y -c conda-forge make cmake cxx-compiler    # needed to build ALFWorld's PDDL backend
pip install -r requirements.txt                            # versions the local runs were tested with
pip install vllm                                           # AWS/large-GPU runs only
export ALFWORLD_DATA=$PWD/data/alfworld && alfworld-download
```

`valid_seen` has 140 games (pre-deployment training) and `valid_unseen` has 134 (deployment). These
match the paper's counts exactly.

**ALFWorld version note.** ALFWorld 0.4.x changed two things from the version ReAct, Reflexion and
the paper used:
- It renamed `put X in/on Y` to `move X to Y`.
- It prints `You arrive at drawer 1. ` before observations.

`muse/env.py` translates both back, so the agents see exactly the original interface ("You put the
bowl 1 in/on the cabinet 1.", as in the Supplementary Material). Without this, every placement
action silently fails and no episode can succeed.

## Quick start

Check the install with the tests (no GPU, about 10 s), then run ReAct on one unseen task with one
adaptation and one test episode. ReAct needs no training, so this checks ALFWorld, the model and the
agent loop in about 4 minutes on a 4 GB GPU, after a one-time download of Qwen2.5-1.5B-Instruct.

```bash
pytest -q tests
python -m muse.cli eval -c configs/local.yaml --agent react run.name=quickstart env.num_tasks=1 \
  protocol.skip_noadapt=true protocol.adapt_episodes=1 protocol.test_episodes=1
python -m muse.cli report --runs runs/quickstart --out runs/quickstart/report    # -> report.md
```

## Running

```bash
export ALFWORLD_DATA=$PWD/data/alfworld

# Local (4 GB GPU): Qwen2.5-1.5B-Instruct in 4-bit, HF backend, on the fixed task lists
# configs/tasks/local_train.json (10 training) and configs/tasks/local_eval.json (10 unseen)
mkdir -p runs && scripts/run_local.sh >> runs/local_pipeline.log 2>&1 &

# Then MUSE with the paper's M_sa on the same models: retrain M_sa only, evaluate into eval/muse+paper_sa
OVERRIDES="self_assessment.input=paper self_assessment.chunk_stride=null self_assessment.adapt_rollouts=none \
  run.eval_variant=paper_sa" STAGES=train-sa AGENTS=muse scripts/run_local.sh >> runs/local_pipeline.log 2>&1 &

# Paper (AWS, >=1x A100/H100 or L40S): Mistral-7B-Instruct-v0.2 on vLLM
scripts/run_paper.sh                                                      # Tables 2-4, Fig. 9
CONFIG=configs/openorca.yaml AGENTS="react reflexion muse" scripts/run_paper.sh   # Table 5
CONFIG=configs/openelm.yaml  AGENTS="react reflexion muse" scripts/run_paper.sh   # Table 5

python -m muse.cli report --runs runs/mistral7b runs/openorca7b runs/openelm3b   # -> runs/report/
```

Each stage can also be run on its own:
`python -m muse.cli {collect,train-sft,train-dpo,train-wm,train-sa,eval}` (`train-wm` only with
`world_model.enabled`, after `train-sft` and before `train-sa`). Any config key can be overridden by
appending `key=value`, e.g., `env.num_tasks=10 run.workers=8`. Results are written one task per
JSONL line, so an interrupted stage resumes where it left off.

`report --tasks common` scores the main run only on the tasks every agent finished, which keeps a
stopped eval comparable. `--tasks solved` keeps only those that at least one agent solved at least
once, dropping tasks where every agent failed every episode. Neither is comparable with the paper's
numbers, which cover all tasks.

Every stage launch appends to `runs/<run>/provenance.jsonl` the git commit (and any uncommitted
diff), the full config, package versions, the GPU, the model revisions and hashes of the trained
models, so you can check that all evals used the same models.

### Pausing

From the repository root, `pkill -STOP -f muse.cli` freezes the running stage and `pkill -CONT -f
muse.cli` resumes it mid-step. This frees CPU and GPU compute but not memory (a local eval holds
about 3.9 GB of GPU memory). To free everything, stop it:

```bash
pkill -f 'run_(local|paper).sh'; pkill -f muse.cli    # stop the script first, or it starts the next stage
```

This loses the episode in progress (up to about 15 min for local MUSE), or the whole stage during
training. Restart as below.

### Restarting after a stop or a crash

Nothing restarts automatically. Check that nothing is still running (`pgrep -af muse.cli`) and
which stage was reached (`runs/${RUN}_pipeline.log`; each stage also appends to
`runs/${RUN}_<stage>.log`). Then rerun the same command, adding `STAGES=` if training had started:
it skips collection and training and goes straight to the evals.

```bash
STAGES= scripts/run_local.sh >> runs/local_pipeline.log 2>&1 &
STAGES= CONFIG=configs/default.yaml scripts/run_paper.sh               # paper runs, same rule
```

Training stages always start over, so don't rerun them once evals have begun: they would replace
the models the finished evals used. `collect` and `eval` skip finished tasks and resume an
unfinished one after its last checkpointed episode (`episodes.partial/`); the log says
`N tasks (K already done)`. `env.keep_task_order=true` with a reordered task list changes only which
tasks finish first.

Models load from the local Hugging Face cache after the first download. Set `HF_HUB_OFFLINE=1` to
skip the check against the Hub, or set `model.name` and `self_assessment.encoder` to local
directories.

### Progress dashboard

```bash
python -m muse.cli status --watch 15     # terminal table + runs/dashboard.html (reloads every 15 s)
```

Every stage writes `runs/<run>/status/<stage>.json`, containing the state, progress, live metrics
and process ID. The dashboard shows, for each model run and each agent:
- the agent's pipeline: pre-deployment stages, which only Reflexion + SL and the MUSE variants
  need, followed by deployment;
- the current stage, its progress and an ETA;
- running success rates for No Adaptation and Adaptation;
- AUROC2 for MUSE.

A stage whose process has died without finishing is shown as *stopped*. A stage that raised an
error is shown as *failed*, with the error message. On AWS, run the command there and view the
terminal output, or copy or port-forward `runs/dashboard.html`.

## Layout

```
muse/env.py              ALFWorld games (same filtering as AlfredTWEnv), one subprocess per worker
muse/prompts.py          ReAct 2-shot actor prompt, Reflexion memory + reflection queries (fail & success)
muse/llm.py              Batched HF / vLLM backends with named LoRA adapters (actor, reflector)
muse/agents.py           ReActAgent, ReflexionAgent, MuseAgent (rollouts -> M_sa scores -> selection)
muse/self_assessment.py  M_sa: frozen all-MiniLM-L6-v2 + MLP g_eta, chunking, replay-based adaptation
muse/runner.py           Episode loop, per-task protocol (Alg. 3/4), concurrent tasks, crash-safe resumable output
muse/datasets.py         SFT / DPO / M_sa training sets from the collected Reflexion episodes
muse/train_lora.py       LoRA SFT (actor) and DPO (reflector) training loops
muse/metrics.py          Success rate, time-to-completion (+100 failure penalty), metacognitive accuracy, AUROC2
muse/report.py           Tables 2-5 and Fig. 9 (the paper's numbers alongside only for its model)
muse/provenance.py       Per-launch record of code version, config, packages and model revisions
muse/data/               ReAct `alfworld_3prompts.json`, Reflexion few-shot reflections (both MIT,
                         unmodified; notices in `muse/data/THIRD_PARTY_LICENSES`)
```

## How the paper maps to code

**Pre-deployment (Alg. 3).** `collect` runs the base-weight Reflexion agent for
`collect.num_episodes` sequential episodes on every seen task (all of them; no stop on success).

Raising `collect.num_episodes` and running `collect` again extends every task where it stopped,
reflections included (eval results are never extended). Train only after collection is complete.

The resulting episodes train three models. Each holds out whole training tasks (`val_fraction`)
and writes `curves.json` and `curves.png` next to the model, so you can check that it learns:
- **M_a (actor):** LoRA SFT on the successful episodes. The loss covers only the tokens the actor
  generates (thoughts and actions). Logits are computed only at the positions that predict those
  tokens (`_sft_loss`), the same loss as over full-sequence logits, which take several GB with Qwen's
  152k-token vocabulary. DPO likewise scores only the reflection's tokens (`_seq_logp`).
- **M_rx (reflector):** DPO-like training on reflection preferences, built within each task.
  - A positive reflection is one where episode e failed and e+1 succeeded after it.
  - A negative reflection is one where e succeeded and e+1 failed despite it.
  - Fallback: a task with no positives uses success→success reflections, and one with no
    negatives uses fail→fail ones.
  - Every positive is paired with every negative of the same task. Each reflection is scored
    under its own prompt.
- **M_sa (self-assessment):** the MLP is trained with BCE on every non-overlapping 4-step chunk,
  labelled with the outcome of its episode. No chunk carries the terminal signal (`STATUS: OK` or
  `STATUS: FAIL`): the last chunk would otherwise give its label away. The same holds for the
  adaptation updates during deployment.
  Its input, class balance, training and adaptation settings are in the M_sa rows below.

The World Model M_w is not trained separately. It is the actor LLM (M_a), prompted with the same
transcript, but with each of the two ReAct examples ending in `STATUS: OK`. This lets it end an
imagined trajectory with a predicted outcome.

**MUSE action selection.** Every action, the first included, is chosen by best-of-N:
- The World Model continues the transcript 5 times at temperature 0.5. Each rollout looks 4
  steps ahead (`muse.chunk_size`). Generation stops at a predicted `STATUS` line or the
  `muse.rollout_max_tokens` budget, and anything past the 4th step is dropped.
- M_sa scores each 4-step rollout, given I and P_e. With the paper's input, the predicted `STATUS`
  is included if it falls within those 4 steps; the actor input scores only the steps.
- The agent executes the first action of the highest-scoring rollout.

For the paper's M_sa, each rollout and chunk is written as `action.observation.` per step, joined
by `> ` (e.g., `go to cabinet 1.The cabinet 1 is closed..> open cabinet 1.…`), copying the
Supplementary Material's "Imagine Trajectory" examples.

**Deployment (Alg. 4).** Each unseen task runs three phases, with an M_sa update before the last:
1. 5 **No Adaptation** episodes. Each is independent: the model as it was at the end of
   pre-deployment training, with no reflections and the pre-deployment M_sa.
2. 5 **adaptation** episodes, with reflections accumulating.
3. After the adaptation episodes, only g_eta is updated, on this task's adaptation episodes mixed
   with chunks replayed from the pre-deployment buffer. Each novel task starts from its own copy of
   g_eta and the buffer, so no task's update reaches another. Held-out pre-deployment chunks are
   scored before and after each update, to detect forgetting. The mix, epochs and data are in the
   "M_sa adaptation" rows below.
4. 5 **test** episodes (the Adaptation condition), each seeing the adaptation episodes'
   reflections (see "Test episodes" below).

With the actor sampling at temperature 0.5, the 5 episodes of each test condition are independent
samples of the same agent, so their mean estimates its success rate on the task.

**Episodes.**
- Every actor output, including a `think:` step, counts as a step. The limit is 50.
- As in Reflexion's code, repeating the previous action ends the episode as a failure.

**Metrics.**
- Success rate is the share of episodes solved, averaged over tasks.
- Time-to-completion is the step count per episode, with failures counted as 100.
- Self-assessment metrics pool the chunk-wise M_sa predictions over the condition's executed
  episodes. AUROC2 is a plain AUROC of each chunk's prediction against its episode's outcome.
  These chunks carry no terminal signal, which would give the outcome away. Metacognitive
  accuracy uses a 0.5 threshold.
  - Each novel task has its own adapted copy of M_sa, so pooled AUROC2 also compares chunks scored
    by differently calibrated models. The report therefore also gives the mean of per-task AUROC2
    over tasks with both outcomes.
- Uncertainty: the report gives 95% bootstrap CIs and paired comparisons of MUSE with each agent
  (mean per-game difference, bootstrap CI, two-sided sign-flip permutation test, exact up to 16
  units). A game is a task id without its trial: trials of one game share scene, object, and
  receptacle, so they are averaged before resampling.

## Implementation decisions

These are config values in `configs/default.yaml`. Values marked `ASSUMPTION` are open choices that
could reasonably be set differently; the "Note" column gives the paper section a value follows, or
the reason for it. "Lessons" links to the findings behind a choice, in
[docs/beyond_the_paper.md](docs/beyond_the_paper.md#7-lessons-learned).

| Item | Choice | Note |
|---|---|---|
| Prompting | Completion-style ReAct prompt, no chat template (`model.chat_template`) | As in Reflexion |
| Randomness | One global seed, `run.seed`, seeds Python's `random`, NumPy, and PyTorch once per process. There is no per-task generator: LLM sampling and `muse_random`'s choice draw from the global generator | Chosen for simplicity. A run is reproducible only if it runs straight through, in the same task order, with one worker. Each restart reseeds from `run.seed`, so a resumed or extended task takes a new random path, and a task's draws depend on the tasks before it |
| P_e (initial plan) | First step, forced to be a plan: prefilled with `think: To solve the task,` for every agent (`agent.force_plan`) | Supplementary Material: "To solve the task, I need to …"; every ReAct example opens this way. MUSE's first-step rollouts are alternative plans scored by M_sa |
| Actor decoding | ReAct, Reflexion, Reflexion + SL (collection included): sampled at `agent.actor_temperature: 0.5`, retried at +0.2 temperature (up to 6 tries) until the output has 5 non-blank characters, at most 256 tokens. MUSE: rollouts at `muse.rollout_temperature` (0.5); when no rollout parses, it falls back to the Reflexion + SL actor with this decoding | ASSUMPTION. Sampling makes repeated test episodes independent samples, as MUSE's are ([lessons](docs/beyond_the_paper.md#7-lessons-learned)). As in Reflexion, a 4-character action such as `look` is never accepted from the model; after 6 tries the agent sends `look` anyway |
| Test episodes | The 5 No Adaptation and the 5 Adaptation test episodes are each independent: no reflections from earlier test episodes, so every Adaptation test episode sees exactly the adaptation episodes' reflections (`protocol.test_independent: true`) | ASSUMPTION. `false` runs the test episodes as a sequence, reflecting after each one so memory keeps growing (5 to 9 reflections) |
| Reflection memory | All prior reflections with their outcomes, for actor and reflector | `agent.max_memory: k` keeps the last k (Reflexion used 3). `agent.max_memory_tokens: n` then drops the oldest (FIFO) until the memory block fits n LLM tokens |
| Success-reflection prompt | `agent.success_reflection: plan`: asks for the specific decisions and actions that made the difference, distilled into a concise reusable plan naming the essential actions in order, with no examples (`SUCCESS_REFLECTION` in `muse/prompts.py`). Alternatives: `reflexion` (Reflexion's query rewritten for success, with its two failure examples); `plan_strategy` (the strategy behind the successful steps); `plan_all_actions` (every essential action through the last, worded as in the history); `plan_grounded` (extension: why the episode succeeded, the strategy and its reasoning, referring only to what happened, leaving out actions that failed or had no effect and anything absent from the history) | ASSUMPTION: prompt chosen here. Failure queries are Reflexion's in all. A trained reflector is evaluated with the query its collection used. None of the other four did clearly better than `plan` ([lessons](docs/beyond_the_paper.md#7-lessons-learned)) |
| Rollouts | 5, each 4 steps ahead (or to a predicted `STATUS`) | 320-token budget; 4 steps took at most 257 Qwen tokens in collected episodes. HF generation stops each rollout when its 5th step begins, or when a line passes 500 characters (`muse.rollout_max_line_chars`): a sign of an imagined observation looping, since real lines are at most 398 characters |
| Collection episodes per seen task | 30, all run (`collect.num_episodes`) | ASSUMPTION: value chosen here for Alg. 3's "max. training episodes". With 10, a third of the tasks had no success with Qwen2.5-1.5B ([lessons](docs/beyond_the_paper.md#7-lessons-learned)) |
| DPO pairs | All positive × negative combinations per task, with fallback | `dpo.max_pairs_per_task` subsamples |
| M_sa input | Z = E(task), S = E(`Plan: <P_e>\nTrajectory: <chunk>`), encoder limit 512 tokens | Truncation cuts the chunk's end, so it is tracked: `train-sa` logs the count; eval records `truncated` per scored rollout and chunk, and `summary.json` gives `truncated_chunks` / `truncated_rollouts` (%) |
| World Model adapter (extension) | `world_model.enabled` (default false): `train-wm` continues the actor adapter into `models/world_model`, adding an observation loss (`obs_weight` 0.3). `action_term: ce` retrains actions on the successes; `anchor` penalizes changes to the actor's action log-probabilities. Training keeps the held-out check with the lowest observation loss whose action loss stays within `action_tolerance` (5%). MUSE's rollouts, its actor fallback and `train-sa` then use this adapter | The actor's rollouts went to places the room does not have. Off by default: no gain in a pilot ([lessons](docs/beyond_the_paper.md#7-lessons-learned)) |
| Admissible-command filter (extension) | `agent.admissible_filter` (default false), for every agent, in collection and deployment: rejects an action that is neither a think step nor among the game's admissible commands. Actors sample again, up to `admissible_resamples` (4) times; MUSE drops the rollout before ranking (only its first step can be checked). If all are rejected, the first is kept | Not in the paper or Reflexion's code. `configs/local.yaml` uses it: the 1.5B model often proposes actions the game rejects ([lessons](docs/beyond_the_paper.md#7-lessons-learned)) |
| Reflection resampling (extension) | `agent.reflection_resamples` (default 0): when a reflection repeats an earlier one of the same task, exactly or more than 90% similar (difflib), sample it again at `reflection_resample_temperature` (1.0), up to that many times; the episode records `resamples`. Applies to every agent that reflects | Qwen2.5-1.5B copies its previous reflection ([lessons](docs/beyond_the_paper.md#7-lessons-learned)) |
| Memory layout (extension) | `agent.max_failure_memory` (default null: all) keeps only the last k failure reflections and every success reflection; `agent.successes_last` (default false) orders the memory as failures, then successes, each in episode order, so the success reflections sit nearest the end of the actor's and reflector's prompts. Applied before `max_memory_tokens` | The paper keeps all reflections (Sec. 4.1.2). With a long memory, the 1.5B reflector continued earlier reflections instead of diagnosing the episode ([lessons](docs/beyond_the_paper.md#7-lessons-learned)) |
| No Adaptation episodes | `protocol.skip_noadapt` (default false): true runs only the adaptation and test episodes of each task | Saves a third of the deployment time when the No Adaptation condition is not needed |
| Action voting (extension) | `agent.action_votes` (default 1: off): ReAct/Reflexion actors sample N actions per step in one call at `actor_temperature`, drop those the game would reject (`admissible_filter`), and take the most frequent; think steps count as one choice, ties go to the earliest sample | Tests whether MUSE's gain comes from sampling candidates or from M_sa ranking them. No gain in a pilot ([lessons](docs/beyond_the_paper.md#7-lessons-learned)) |
| Eval variants | `run.eval_variant` (default null): a name makes `eval` write to `eval/<agent>+<name>`; the report gives it its own rows in Tables 2-4 and its own paired comparisons, labelled `<agent> (<name>)` | Compares settings that leave the trained actor and reflector unchanged next to the same baselines, e.g., the paper's M_sa (`self_assessment.input=paper chunk_stride=null adapt_rollouts=none`, after `train-sa` with the same overrides; it saves to `models/sa`, the actor-feature M_sa to `models/sa_actor`) |
| M_sa input (extension) | `self_assessment.input: actor`: g_eta on the actor LLM's hidden states (layer: next row) over the chunk (mean over its tokens, then its last token), given the actor's full context: the World Model prompt, memory, and every earlier step. Rollouts are scored as continuations of the World Model prompt, without a predicted `STATUS` line; the prompt's KV cache is reused across them. Trained on the merged actor's features, saved to `models/sa_actor`; hf backend only | The paper's M_sa sees only I, P_e, and the chunk (Eq. 6), not what happened before it. Compare offline with `scripts/analysis/compare_sa.py` ([lessons](docs/beyond_the_paper.md#7-lessons-learned)) |
| M_sa actor layer | `self_assessment.layer`: null takes the final layer (after the model's last norm); k takes decoder layer k's output. One forward pass serves several layers (`scripts/analysis/compare_sa.py --layers`) | The final layer is shaped by next-token prediction. Layer 14 of Qwen2.5-1.5B's 28 gave the best offline within-task AUROC; an outcome question appended after the chunk did not help ([lessons](docs/beyond_the_paper.md#7-lessons-learned)) |
| M_sa held-out tasks | `self_assessment.val_tasks: mixed`: the held-out tasks (val_fraction of them, at least one) are drawn only from training tasks with both successful and failed episodes; `random` draws from all tasks | Random held-out tasks can have no successes, leaving held-out AUROC undefined |
| M_sa early stopping | `self_assessment.early_stop: true` (ASSUMPTION): pre-deployment training keeps the epoch with the best held-out within-task AUROC; adaptation runs all its epochs. `lr: 3e-4` (ASSUMPTION) | At lr 1e-3 with class weighting, the held-out loss swung widely from epoch to epoch ([lessons](docs/beyond_the_paper.md#7-lessons-learned)) |
| M_sa input scaling | `self_assessment.standardize: false`. true scales every feature to zero mean and unit variance over the pre-deployment training chunks, kept through adaptation | Offline it lowered within-task AUROC for every input ([lessons](docs/beyond_the_paper.md#7-lessons-learned)) |
| M_sa class balance | `self_assessment.class_balance: weight` (ASSUMPTION): BCE weights successful chunks by #failed / #successful in each epoch, replayed chunks included. `sample` redraws successful chunks to match the failed ones; `none` is plain BCE | Successful chunks are rare. Balancing raises the level of the scores, so metacognitive accuracy at 0.5 is not comparable with unbalanced runs; MUSE's choice uses only their order ([lessons](docs/beyond_the_paper.md#7-lessons-learned)) |
| M_sa training chunks | Non-overlapping, as in the paper (Sec. 4.1.3). `self_assessment.chunk_stride: 1` (extension) adds a chunk starting at every step, for pre-deployment training and adaptation | Rollouts start at any step, while non-overlapping chunks start at steps 1, 5, 9, … Overlapping chunks are about 4 times as many but strongly correlated. Compare offline with `scripts/analysis/compare_sa.py` |
| M_sa adaptation | `adapt_new_fraction: 0.2` (ASSUMPTION): 20% new / 80% replay from the pre-deployment buffer, with `class_balance` over the mix; 30 epochs at lr 1e-3 (`adapt_epochs`, `adapt_lr`; ASSUMPTION); g_eta and buffer reset for each novel task | Replay guards against forgetting; offline on training tasks, any update helped and the mixes did not differ clearly (`scripts/analysis/sweep_sa_adapt.py`, [lessons](docs/beyond_the_paper.md#7-lessons-learned)). Forgetting check in `summary.json` / report |
| M_sa adaptation data (extension) | `adapt_rollouts: none` (default): executed chunks only, as in the paper. `selected` (`configs/local.yaml`): the adaptation episodes' chosen rollouts, as M_sa scored them, are added as examples labelled with their episode's outcome. `all`: every candidate | M_sa scores imagined rollouts but is trained on executed chunks ([lessons](docs/beyond_the_paper.md#7-lessons-learned)) |
| LoRA / SFT / DPO / MLP hyper-parameters | r=16, α=32; SFT lr 1e-4 ×2 epochs; DPO β=0.1, lr 5e-5 ×2 epochs; effective batch 8 (one episode or pair at a time, gradients accumulated; `configs/local.yaml`: SFT 2, DPO 4, for enough optimizer steps on small data); MLP 768→256→64→1 (its training settings are in the M_sa rows) | ASSUMPTION; check the learning curves |
| OpenELM context | 2048 tokens, prompts left-truncated | The 2-shot prompt alone is ~1.7k tokens |

## Compute notes

- Local (RTX 500 Ada, 4 GB): Qwen2.5-1.5B-Instruct in 4-bit.
  - An unmerged LoRA adapter doubles decoding time. `model.merge_adapters: true` serves the actor
    and reflector as two merged 4-bit models (~3.1 GB peak), cached in `<adapter>/merged`, with the
    SFT loss unchanged.
  - The task lists are drawn over all six task types by `scripts/sample_tasks.py --seed 0`;
    [docs/beyond_the_paper.md](docs/beyond_the_paper.md) lists how `configs/local.yaml` differs from
    the paper setup.
  - In the last run, the longest World Model prompt was 4,340 tokens (5 reflections in memory), and
    a MUSE step (5 × 4-step rollouts) took 22–46 s (median 31 s). `model.max_batch_tokens: 10000`
    splits longer calls, keeping generation at about 3.3 GB. The 7,500-token memory cap allows
    prompts up to about 13k tokens.
  - SFT on a ~3.9k-token episode and DPO on a ~3.9k-token pair peak at about 3.9 GB, spilling into
    shared system memory under WSL. A DPO pair at that length takes 1–2 minutes.
- AWS: vLLM with prefix caching batches all concurrent tasks (`run.workers`).
  - MUSE is the most expensive agent: up to 134 tasks × 15 episodes × 50 steps × 5 rollouts of
    4 steps each.
  - LoRA SFT/DPO of a 7B model with 8k-token episodes needs ~40 GB in bf16. Set
    `model.load_in_4bit=true` to use QLoRA on 24 GB cards.
  - Training stages always use transformers + PEFT, whatever `model.backend` is set to. vLLM
    serves the trained adapters at deployment time.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{valiente2026muse,
  title   = {Metacognition for Unknown Situations and Environments ({MUSE})},
  author  = {Valiente, Rodolfo and Pilly, Praveen K.},
  journal = {Neural Networks},
  volume  = {194},
  pages   = {108131},
  year    = {2026},
  doi     = {10.1016/j.neunet.2025.108131}
}
```

The experiments also build on the ALFWorld benchmark ([Shridhar et al., ICLR
2021](https://arxiv.org/abs/2010.03768)) and the ReAct ([Yao et al., ICLR
2023](https://arxiv.org/abs/2210.03629)) and Reflexion ([Shinn et al., NeurIPS
2023](https://arxiv.org/abs/2303.11366)) agents; please cite them as well where relevant.

## AI assistance

This implementation was developed with Claude Code. I directed it to produce an initial draft from
the published paper, then steered it through many iterations: correcting deviations from the
protocol, filling in implementation details, and directing additional experiments. This continued
until the code matched the paper's experimental protocol and reproduced its comparative trends.
Claude Code also wrote the unit tests and initial drafts of the documentation.

I made the research and design decisions, reviewed and validated the code and documentation against
the method as described in our paper, and checked the results against its findings.

## License

MIT (see `LICENSE`). The two few-shot files in `muse/data/` are unmodified copies from ReAct and
Reflexion, and a few prompt lines and helpers derive from their code; all keep their MIT licenses
(`muse/data/THIRD_PARTY_LICENSES` lists them). ALFWorld, TextWorld, the ALFWorld data and the
models are not included; they are installed or downloaded separately under their own licenses.
