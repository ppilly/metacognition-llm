# Beyond the paper: what this replication adds, and why

The MUSE paper (Valiente & Pilly, *Neural Networks*, 2026) and its Supplementary Material describe
the methods and results of the ALFWorld experiments. This document records the implementation
choices this repository makes, and why. The README's table "Implementation decisions" lists every
config key. Section 7 gives the measurements behind the choices.

Three kinds of change are covered:

1. **Assumptions:** values the paper does not specify. Each is a key in `configs/default.yaml`,
   marked `ASSUMPTION`.
2. **Hardware adaptations:** what changes to run on a 4 GB laptop GPU (`configs/local.yaml`).
3. **Extensions:** mechanisms added beyond the paper. Each is a config option that is off in
   `configs/default.yaml`, so the default config stays the paper's method.

## 1. Assumptions

### How the agents act

| Decision | Choice | Why |
|---|---|---|
| Prompt format | Completion-style ReAct prompt, no chat template | As in Reflexion, whose agents MUSE builds on |
| First step | Forced to be a plan: every agent's first output is prefilled with `think: To solve the task,` | The Supplementary Material's examples open this way. P_e (the initial plan), which M_sa is conditioned on, then always exists |
| Actor decoding | Sampling at temperature 0.5 for every agent, collection included | Greedy decoding results in identical test episodes (Section 7). Sampling makes them independent, as MUSE's are |
| Episode end | 50 steps. Every output counts as a step, `think:` included. Repeating the previous action ends the episode as a failure | Reflexion's code |
| World Model | The actor LLM, prompted with the same transcript but with examples that end in `STATUS: OK` | The paper prompts the actor as the World Model (Sec. 4.1.3). The `STATUS` line lets an imagined trajectory end with a predicted outcome |

### Reflections and memory

| Decision | Choice | Why |
|---|---|---|
| Success-reflection prompt | `plan`: the decisions and actions that made the difference, as a concise reusable plan | As in the paper, the agent also reflects on successes; this prompt is this repository's. Reflexion's failure prompt, reworded for success, made the 1.5B model copy its failure examples. Several other wordings were tried; none did clearly better (README, "Success-reflection prompt") |
| Failure-reflection prompt | Reflexion's, unchanged | Kept identical so that Reflexion is the published baseline |
| Memory | All earlier reflections, with their outcomes | The paper keeps every reflection (Sec. 4.1.2). Locally, a token cap bounds the prompt length (Section 2) |

### Pre-deployment training (Alg. 3)

| Decision | Choice | Why |
|---|---|---|
| Collection episodes per training task | 30, all run | Value chosen here for Alg. 3's "max. training episodes". With 10, a third of the tasks had no successful episode with the 1.5B model, which left nothing to imitate |
| M_a (actor) | LoRA SFT on the successful episodes, with the loss only on the tokens the actor generates | LoRA, as in the paper; settings chosen here: r = 16, α = 32, lr 1e-4, 2 epochs |
| M_rx (reflector) | DPO on reflection pairs within each task: a positive reflection is followed by a success after a failure, a negative one by a failure after a success. Each positive is paired with each negative, with a fallback for tasks lacking one kind | Positive and negative reflections as defined in Sec. 4.1.3. Pairing within each task, and the fallback, are implementation choices. β = 0.1, lr 5e-5, 2 epochs |
| M_sa (self-assessment) training data | Every non-overlapping 4-step chunk of the collected episodes, labelled with its episode's outcome. No chunk includes the final `STATUS` line | The last chunk would otherwise reveal its own label |
| M_sa training | 60 epochs, lr 3e-4. The epoch with the best held-out within-task AUROC is kept. Successful chunks are weighted up (#failed / #successful) | Few chunks come from successes. At a higher learning rate the held-out loss swung from epoch to epoch |
| M_sa held-out tasks | Drawn only from training tasks with both outcomes | Otherwise held-out AUROC can be undefined |

### Deployment (Alg. 4)

| Decision | Choice | Why |
|---|---|---|
| Test episodes | The 5 No Adaptation and 5 Adaptation test episodes are independent: each Adaptation test episode sees only the adaptation episodes' reflections | With temperature 0.5, the 5 episodes then estimate one agent's success rate. Run as a sequence, the memory kept growing during testing |
| M_sa update after adaptation | Only g_eta is updated: 30 epochs at lr 1e-3. Each epoch is 20% new chunks and 80% replayed pre-deployment chunks. Each novel task starts from a fresh copy of g_eta and the buffer | Mix chosen here. Replay guards against forgetting the pre-deployment training |
| Metrics | Failures count as 100 steps in time-to-completion. Self-assessment AUROC is reported pooled and as a per-task mean. 95% bootstrap CIs and paired permutation tests are computed over games (a task id without its trial number) | Each task has its own adapted M_sa, so pooled AUROC mixes differently calibrated models. Trials of one game share their scene, so they are averaged before resampling |

## 2. Running on a 4 GB GPU

`configs/local.yaml` runs the same pipeline with Qwen2.5-1.5B-Instruct, on 10 training and 10 unseen
tasks (`configs/tasks/local_train.json`, `configs/tasks/local_eval.json`), drawn over all six task
types.

| Change | Paper setup (`default.yaml`) | Local | Why |
|---|---|---|---|
| Model | Mistral-7B-Instruct-v0.2, vLLM | Qwen2.5-1.5B-Instruct, 4-bit, transformers | Fits in 4 GB |
| LoRA adapters | Served as adapters | Merged into the 4-bit model | An unmerged adapter doubles decoding time |
| Context and batches | 32k tokens, 64 sequences, 32 workers | 16k tokens, 5 sequences (10k tokens), 1 worker | Memory |
| SFT and DPO | 8k tokens, effective batch 8 | 6k tokens, effective batch 2 (SFT) and 4 (DPO), at most 8 DPO pairs per task | Memory, and enough optimizer steps on little data; DPO under about 5 h |
| Memory cap | None | 7,500 tokens; the oldest reflections are dropped first | Bounds the prompt over 30 collection episodes |
| Reflector decoding | Greedy, as in Reflexion | Temperature 0.5 | Greedy reflections mostly copied the previous one |
| Task order | Sorted by id | The given order, hardest first | Only changes which tasks finish first |

A smaller model is a much weaker actor and reflector, so every agent's absolute success rate is
lower than the paper's. The comparison between agents is what this setup can test.

## 3. Extensions

Each extension is off in `configs/default.yaml`. The last column says whether `configs/local.yaml`
uses it.

| Extension | What it does | Why | Local |
|---|---|---|---|
| Admissible-command filter (`agent.admissible_filter`) | Rejects an action the game would not accept. The actor samples again, up to 5 samples in all; MUSE drops the rollout before M_sa ranks the rest | The 1.5B model often proposes actions the game rejects. The filter is used by every agent, so the comparison stays fair. Reflexion's own code does not filter | Yes |
| M_sa on the actor's hidden states (`self_assessment.input: actor`, `layer: 14`) | g_eta reads the actor's layer-14 hidden states over the chunk, given the full context, instead of sentence embeddings of the task, plan, and chunk | The paper's M_sa sees only the task, plan, and chunk, not what happened before it. Offline, layer 14 gave the best within-task AUROC, above the paper's input | Yes |
| Overlapping training chunks (`self_assessment.chunk_stride: 1`) | Adds a training chunk starting at every step | Rollouts start at any step, while non-overlapping chunks start only at steps 1, 5, 9 and so on | Yes |
| Rollouts in the M_sa update (`self_assessment.adapt_rollouts: selected`) | The adaptation episodes' chosen imagined rollouts join the executed chunks as new training examples | M_sa scores imagined rollouts but was trained only on executed chunks. It gave high scores to imagined rollouts in failed episodes | Yes |
| World Model adapter (`world_model.enabled`) | A separate adapter trained to predict observations, used for rollouts | Rollouts went to places the room does not have. It reduced observation loss but gave no gain on a pilot task at twice the decoding time | No |
| Reflection resampling (`agent.reflection_resamples`) | Samples a reflection again when it repeats an earlier one | The 1.5B reflector copies its previous reflection | No |
| Memory layout (`agent.max_failure_memory`, `agent.successes_last`) | Keeps only recent failure reflections, and puts success reflections last | Long memory made the reflector continue its earlier reflections instead of diagnosing the episode | No |
| Action voting (`agent.action_votes`) | Samples N actions per step and takes the most frequent | Tests whether MUSE's gain comes from sampling candidates or from M_sa ranking them. With 5 votes, ReAct did not improve: the samples mostly agreed | No |
| Eval variants (`run.eval_variant`) | Writes an eval to `eval/<agent>+<name>` and reports it as its own row | Compares settings that keep the trained actor and reflector, e.g., the paper's M_sa against the extended one | Optional |

To run MUSE locally with the paper's M_sa on the same trained models, see the README's Running
section. The overrides are `self_assessment.input=paper self_assessment.chunk_stride=null
self_assessment.adapt_rollouts=none run.eval_variant=paper_sa`.

## 4. Tried and not adopted

These ran as pilots or offline checks; Section 7 gives the measurements.

- **Greedy decoding** for the actor and reflector: no gain over temperature 0.5 on two tasks.
- **`plan_grounded` success reflections**, which refer only to what happened: no more accurate than
  `plan` offline, and no better on a pilot task.
- **An outcome question for M_sa** ("will this attempt complete the task?") after the chunk: its
  features scored below plain layer-14 features.
- **Standardizing M_sa's inputs**: lowered within-task AUROC for every input.

## 5. Current results

See [Results](../README.md#results) in the README.

## 6. Open problems

- **Search failures.** On the hardest tasks every agent fails by searching the wrong places or
  looping, even when every action is valid.
- **Reflections that record failed steps as facts.** A success reflection sometimes turns a rejected
  action into the plan for the next episode.
- **Copying.** The 1.5B reflector tends to repeat its earlier reflections whatever the prompt.
- **OpenELM's 2,048-token window** (Table 5) cannot hold the 2-shot prompt, a long history, and the
  reflections together.

## 7. Lessons learned

These come from developing the local setup: Qwen2.5-1.5B-Instruct in 4-bit, 10 to 12 training tasks,
and pilots on 1 to 10 unseen tasks with 5 episodes each, plus offline checks on the collected
episodes. The samples are small. Read the findings as reasons for a setting, not as established
effects, and expect a 7B model to behave differently on several of them.

### Decoding and reflection

- **Greedy decoding hides variance.** With the actor at temperature 0, all 5 No Adaptation episodes
  were identical on every task, for every baseline: 5 episodes measured one sample. Sampling at 0.5
  makes them independent samples of the agent.
- **A small reflector copies itself.** Greedy reflections were exact copies of the previous one 253
  times out of 290 (87%). Replaying 30 of those prompts, the copy rate (exact or near) at
  temperature 0 / 0.5 / 0.7 / 1.0 was 100 / 65 / 40 / 7% for the untrained reflector and 90 / 89 /
  68 / 27% for the DPO-trained one. At 1.0 the reflections vary but sometimes stray off-task.
- **Long memory makes it worse.** With 22 earlier reflections (about 2,000 of 4,900 prompt tokens)
  just before "New plan:", the untrained reflector continued their template ("drawers, then
  sidetables, then dressers...") instead of diagnosing the episode.
- **The success-reflection wording matters, but only so much.** Reflexion's failure prompt reworded
  for success made the model copy its two failure examples after every success. Over 300 collection
  episodes, a plan-style prompt gave 30 successes, a strategy-style one 32, and one asking for every
  action 14. A wording that asks to refer only to what happened did not reduce invented object names
  (27% of reflections, against 21% for the plan prompt): the reflector mostly copies the task's
  earlier reflections whatever the prompt.
- **Success reflections can record failures as facts.** A rejected action (e.g., taking a mug from a
  place it was not) sometimes became the plan the next episode followed.

### Pre-deployment training

- **10 collection episodes are too few for a small model.** With 10, a third of the training tasks
  had no successful episode, leaving SFT nothing to imitate for them. 30 fixed that.
- **Compute the SFT loss only where it is needed.** Full-sequence logits for a 6k-token episode with
  a 152k-token vocabulary take several GB and ran out of memory on 4 GB. Computing logits only at
  the positions that predict the actor's tokens gives the same loss.
- **Keep the outcome out of M_sa's input.** A chunk that ends with the episode's `STATUS` line gives
  its label away.

### Self-assessment (M_sa)

All comparisons are mean within-task AUROC with one training task left out at a time.

- **What happened before the chunk matters.** On 9 tasks, the actor's layer-14 hidden states (given
  the full context) scored 0.818, the paper's sentence-encoder input 0.729, and the chunk's position
  in the episode alone 0.742. Position alone is a strong baseline: report it next to any M_sa.
- **Middle layers beat the last one.** On 5 tasks: layer 7 0.742, layer 14 0.769, layer 21 0.710,
  final layer 0.701. The final layer is shaped for next-token prediction.
- **Asking the model doesn't help.** On 9 tasks, the actor's layer-14 features gave 0.818. Appending
  "will this attempt complete the task?" and reading the question's last token gave 0.759 (layer 14)
  and 0.769 (final layer); adding those features to the actor's gave 0.802. The model's own Yes/No
  log-probability gap gave 0.661 zero-shot.
- **Class weighting helps actor features only.** About 6% of training chunks come from successes.
  Weighting them up helped every actor layer (+0.005 to +0.04) and hurt the paper input (−0.03).
  Resampling them hurt everywhere, and standardizing the inputs hurt every input (layer 14 0.761 →
  0.683).
- **Training is noisy at small scale.** At lr 1e-3 with class weighting, the held-out loss swung
  between 0.21 and 0.68 from epoch to epoch while training AUROC reached 0.99. Use a lower rate with
  early stopping, and draw held-out tasks from those with both outcomes, or held-out AUROC can be
  undefined.
- **Adapting helps; the mix matters little.** Each of 9 training tasks in turn stood in for a novel
  task: M_sa was trained on the others, adapted on the task's first 5 episodes, and scored on the
  rest (`scripts/analysis/sweep_sa_adapt.py`). Any update raised within-task AUROC (0.820 without,
  0.834 to 0.850 with 10% to 50% new chunks per epoch), with no clear winner among the mixes.
  Forgetting, as the loss on held-out training tasks, was unchanged except at 10% new, where it
  fell. The 20% mix is an assumption.
- **Train M_sa on what it will score.** M_sa is trained on executed chunks but ranks imagined
  rollouts, and it gave high scores to imagined rollouts in failed episodes. Whether adding the
  chosen rollouts to the adaptation data helps has not been checked on training tasks: collection
  episodes have no rollouts.

### Acting

- **Invalid actions dominate early, search failures later.** On one task, 69% of the small model's
  actions were rejected by the game. Filtering by the admissible commands cut that to 12%, but the
  agent still failed: it searched the wrong places or looped through valid moves. On another task,
  the filter took MUSE's test success from 0/5 to 3/5.
- **Majority voting adds little.** Five sampled actions agreed on 70–86% of steps, so the vote
  mostly repeats one sample, and the failures (search loops, wrong plans) are shared. ReAct with 5
  votes did not improve on two tasks.
- **A trained World Model needs its policy anchored.** In MUSE every executed action is a rollout's
  first, so a World Model adapter is also the policy. Training it on actions again overfit the few
  successes (held-out action loss +31%). Anchoring the action log-probabilities to the actor's kept
  it within 3.5% while the observation loss fell 16%, but it gave no gain in a pilot.
- **Merge LoRA adapters on small GPUs.** An unmerged adapter doubled decoding time in 4-bit (162–182
  against 85 ms per step for 5 sequences); merged, it took 90 ms with the SFT loss unchanged
  (0.126).
