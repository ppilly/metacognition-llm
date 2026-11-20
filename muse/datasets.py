"""Build MUSE's pre-deployment training sets from Reflexion episodes on the training tasks (Alg. 3)."""
import logging
import random
from collections import defaultdict

from . import prompts
from .agents import Step
from .self_assessment import chunk_starts, chunk_texts

log = logging.getLogger(__name__)


def _steps(ep):
    return [Step(s["action"], s["observation"]) for s in ep["steps"]]


def sft_examples(episodes, include_fewshot=True) -> list[dict]:
    """Actor M_a: successful episodes only, loss on the actor's own outputs (actions and thoughts)."""
    out = []
    for ep in episodes:
        if not ep["success"]:
            continue
        text, spans = prompts.actor_text_with_action_spans(
            ep["task_type"], ep["init_obs"], ep["memory"], _steps(ep), fewshot=include_fewshot)
        out.append({"text": text, "spans": spans, "task_id": ep["task_id"], "episode": ep["episode"]})
    return out


def wm_examples(episodes, max_failed=None, seed=0, max_label_steps=None) -> list[dict]:
    """World Model adapter (extension): every episode's transcript under the World Model prompt, with
    observation spans from all episodes, action spans from successful ones only (as for the actor) and
    anchor spans, every episode's actions (world_model.action_term: anchor).
    max_failed keeps a seeded sample of the failed episodes (each a full 50-step transcript).
    max_label_steps splits an episode into items that label at most that many steps each, every item's
    text ending after its last step: the logits of a 50-step episode's labels overflow a 4 GB GPU."""
    failed = [ep for ep in episodes if not ep["success"]]
    if max_failed is not None and len(failed) > max_failed:
        keep = set(map(id, random.Random(seed).sample(failed, max_failed)))
        episodes = [ep for ep in episodes if ep["success"] or id(ep) in keep]
    out = []
    for ep in episodes:
        text, actions, observations = prompts.world_model_text_with_spans(
            ep["task_type"], ep["init_obs"], ep["memory"], _steps(ep))
        size = max_label_steps or len(actions) or 1
        for i in range(0, max(1, len(actions)), size):
            end = observations[min(i + size, len(actions)) - 1][1] if actions else len(text)
            out.append({"text": text[:end], "action_spans": actions[i:i + size] if ep["success"] else [],
                        "obs_spans": observations[i:i + size], "anchor_spans": actions[i:i + size],
                        "task_id": ep["task_id"], "episode": ep["episode"], "success": ep["success"]})
    return out


def _reflection_items(episodes):
    """Label each reflection r_e by the outcomes of episode e and e+1 of the same task."""
    by_task = defaultdict(dict)
    for ep in episodes:
        by_task[ep["task_id"]][ep["episode"]] = ep
    items = []
    for task_id, eps in by_task.items():
        for e, ep in eps.items():
            nxt = eps.get(e + 1)
            if ep["reflection"] is None or nxt is None:
                continue
            items.append({"task_id": task_id, "task_type": ep["task_type"], "before": ep["success"],
                          "after": nxt["success"], "prompt": ep["reflection"]["query"],
                          "text": ep["reflection"]["text"]})
    return items


def dpo_pairs(episodes, fallback=True, max_pairs_per_task=None, seed=0) -> list[dict]:
    """DPO-like reflection preferences for M_rx (Sec. 4.1.3), built within each task.

    Positive: episode e failed and e+1 succeeded after the reflection. Negative: e succeeded and
    e+1 failed despite it. With `fallback`, a task with no positives uses success->success
    reflections instead, and one with no negatives uses fail->fail ones. Every positive is paired
    with every negative of the same task (optionally subsampled to `max_pairs_per_task`). Each side
    keeps its own reflection prompt, since the two reflections answer different episodes.
    """
    rng = random.Random(seed)
    by_task = defaultdict(list)
    for item in _reflection_items(episodes):
        by_task[item["task_id"]].append(item)
    pairs, counts = [], defaultdict(int)
    for task_id, items in by_task.items():
        kinds = {(b, a): [i for i in items if (i["before"], i["after"]) == (b, a)]
                 for b in (False, True) for a in (False, True)}
        for k, v in kinds.items():
            counts[k] += len(v)
        pos, neg = kinds[(False, True)], kinds[(True, False)]
        if fallback:
            pos = pos or kinds[(True, True)]
            neg = neg or kinds[(False, False)]
        combos = [(p, n) for p in pos for n in neg]
        if max_pairs_per_task and len(combos) > max_pairs_per_task:
            combos = rng.sample(combos, max_pairs_per_task)
        pairs += [{"task_id": task_id, "prompt_chosen": p["prompt"], "chosen": p["text"],
                   "prompt_rejected": n["prompt"], "rejected": n["text"]} for p, n in combos]
    log.info("reflections: %d fail->success, %d success->fail, %d success->success, %d fail->fail; "
             "%d pairs from %d tasks", counts[(False, True)], counts[(True, False)], counts[(True, True)],
             counts[(False, False)], len(pairs), len({p["task_id"] for p in pairs}))
    return pairs


def split_by_task(task_ids: list[str], val_fraction: float, seed=0, labels=None) -> tuple[list[int], list[int]]:
    """Indices of a train/validation split that holds out whole tasks. With `labels`, the held-out tasks
    are drawn only from those with both labels, val_fraction of them (at least one), so that held-out
    AUROC is defined; tasks with one label all train."""
    if labels is not None:
        both = {t for t in set(task_ids) if len({l for l, u in zip(labels, task_ids) if u == t}) == 2}
        if both:
            sub = sorted(both)
            _, val = split_by_task(sub, val_fraction, seed)
            val_tasks = {sub[i] for i in val}
            return ([i for i, t in enumerate(task_ids) if t not in val_tasks],
                    [i for i, t in enumerate(task_ids) if t in val_tasks])
    tasks = sorted(set(task_ids))
    random.Random(seed).shuffle(tasks)
    n_val = int(round(val_fraction * len(tasks))) if val_fraction else 0
    if len(tasks) > 1:
        n_val = min(max(n_val, 1 if val_fraction else 0), len(tasks) - 1)
    val_tasks = set(tasks[:n_val])
    train = [i for i, t in enumerate(task_ids) if t not in val_tasks]
    val = [i for i, t in enumerate(task_ids) if t in val_tasks]
    return train, val


def sa_examples(episodes, chunk_size=4, stride=None) -> dict:
    """M_sa: every executed chunk of 4 action-observation pairs labelled with the episode outcome
    (non-overlapping unless `stride` is given; see chunk_starts). No chunk carries the terminal signal
    (STATUS: OK / FAIL), which would leak the label."""
    out = {"task_ids": [], "descs": [], "plans": [], "texts": [], "labels": [], "starts": []}
    for ep in episodes:
        steps = _steps(ep)
        for start, text in zip(chunk_starts(len(steps), chunk_size, stride), chunk_texts(steps, chunk_size, stride)):
            out["starts"].append(start)
            out["task_ids"].append(ep["task_id"])
            out["descs"].append(ep["task_desc"])
            out["plans"].append(ep["plan"])
            out["texts"].append(text)
            out["labels"].append(float(ep["success"]))
    return out
