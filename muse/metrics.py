"""Self-regulation and self-assessment metrics (Sec. 4.2.1)."""
from collections import defaultdict

import numpy as np


PHASE = {"noadapt": "noadapt", "adapt": "test"}   # condition -> episode phase (see muse/runner.py)


def condition_episodes(episodes, condition: str) -> list[dict]:
    """'noadapt': the independent episodes of the pre-deployment model. 'adapt': the test episodes
    after the adaptation episodes."""
    return [ep for ep in episodes if ep["phase"] == PHASE[condition]]


def _per_task(episodes, fn):
    by_task = defaultdict(list)
    for ep in episodes:
        by_task[ep["task_id"]].append(ep)
    return float(np.mean([fn(eps) for eps in by_task.values()])) if by_task else float("nan")


def per_task(episodes, metric: str, fail_penalty=100) -> dict[str, float]:
    """Per-task value of 'success_rate' (%) or 'time_to_completion' (failures = fail_penalty)."""
    by_task = defaultdict(list)
    for ep in episodes:
        by_task[ep["task_id"]].append(ep)
    if metric == "success_rate":
        return {t: 100 * float(np.mean([ep["success"] for ep in eps])) for t, eps in by_task.items()}
    return {t: float(np.mean([ep["num_steps"] if ep["success"] else fail_penalty for ep in eps]))
            for t, eps in by_task.items()}


def by_game(values: dict[str, float]) -> dict[str, float]:
    """Average the trials of each game (scene + object + receptacle, the part of the task id before '/'):
    trials of one game are correlated, so games are the independent units for CIs and tests."""
    groups = defaultdict(list)
    for task_id, v in values.items():
        groups[task_id.split("/")[0]].append(v)
    return {g: float(np.mean(v)) for g, v in groups.items()}


def bootstrap_ci(values, n_boot=10000, level=0.95, seed=0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling tasks."""
    v = np.asarray(list(values), dtype=float)
    if len(v) < 2:
        return float("nan"), float("nan")
    means = np.random.default_rng(seed).choice(v, (n_boot, len(v))).mean(1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def paired_comparison(a: dict, b: dict, n_perm=10000, seed=0) -> dict:
    """a - b over the tasks both agents ran: mean per-task difference, bootstrap 95% CI, and a two-sided
    sign-flip permutation p-value (exact up to 16 tasks)."""
    tasks = sorted(set(a) & set(b))
    d = np.array([a[t] - b[t] for t in tasks])
    out = {"n_tasks": len(tasks), "mean_diff": float(d.mean()) if len(d) else float("nan")}
    out["ci"] = list(bootstrap_ci(d, seed=seed))
    if len(d) == 0:
        out["p"] = float("nan")
        return out
    exact = len(d) <= 16
    if exact:
        signs = np.array(np.meshgrid(*[[1, -1]] * len(d))).reshape(len(d), -1).T
    else:
        signs = np.random.default_rng(seed).choice([1, -1], (n_perm, len(d)))
    hits = int(np.sum(np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12))
    out["p"] = hits / len(signs) if exact else (hits + 1) / (len(signs) + 1)
    return out


def success_rate(episodes) -> float:
    """% of test episodes solved within the time limit, averaged across tasks."""
    return 100 * _per_task(episodes, lambda eps: np.mean([ep["success"] for ep in eps]))


def time_to_completion(episodes, fail_penalty=100) -> float:
    """Average steps to success per episode; failures count as `fail_penalty` steps."""
    return _per_task(episodes, lambda eps: np.mean(
        [ep["num_steps"] if ep["success"] else fail_penalty for ep in eps]))


def sa_predictions(episodes):
    preds, labels = [], []
    for ep in episodes:
        for c in ep.get("sa_chunks") or []:
            preds.append(c["pred"])
            labels.append(int(ep["success"]))
    return np.array(preds), np.array(labels)


def sa_truncation(episodes) -> dict:
    """% of M_sa inputs (plan + chunk) cut by the encoder's length limit: executed chunks, and the
    imagined rollouts scored for action selection."""
    chunks = [c["truncated"] for ep in episodes for c in ep.get("sa_chunks") or [] if "truncated" in c]
    rollouts = [c["truncated"] for ep in episodes for s in ep["steps"]
                for c in (s.get("info") or {}).get("candidates", []) if "truncated" in c]
    return {k: 100 * float(np.mean(v)) for k, v in
            (("truncated_chunks", chunks), ("truncated_rollouts", rollouts)) if v}


def auroc2_per_task(episodes) -> dict:
    """Mean of per-task AUROC2 over the tasks whose episodes include both outcomes. Unlike the pooled
    AUROC2, it is unaffected by per-task copies of M_sa being calibrated differently."""
    from sklearn.metrics import roc_auc_score

    by_task = defaultdict(list)
    for ep in episodes:
        by_task[ep["task_id"]].append(ep)
    vals = []
    for eps in by_task.values():
        preds, labels = sa_predictions(eps)
        if len(set(labels)) == 2:
            vals.append(roc_auc_score(labels, preds))
    return {"auroc2_per_task": float(np.mean(vals)) if vals else float("nan"), "auroc2_n_tasks": len(vals)}


def self_assessment(episodes, threshold=0.5) -> dict:
    """Metacognitive accuracy and AUROC2 over chunk-wise predictions pooled across tasks."""
    from sklearn.metrics import roc_auc_score, roc_curve

    preds, labels = sa_predictions(episodes)
    if len(preds) == 0:
        return {"n_chunks": 0}
    out = {"n_chunks": int(len(preds)), "positive_rate": float(labels.mean()),
           "meta_accuracy": float(100 * np.mean((preds >= threshold) == labels)),
           **sa_truncation(episodes), **auroc2_per_task(episodes)}
    if len(set(labels)) == 2:
        out["auroc2"] = float(roc_auc_score(labels, preds))
        fpr, tpr, _ = roc_curve(labels, preds)
        out["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    else:
        out["auroc2"] = float("nan")
    return out


def sa_adaptation(episodes) -> dict:
    """Mean effect of the per-task M_sa update: BCE on the task's own adaptation chunks, and BCE/AUROC
    on held-out pre-deployment chunks (a rise in that BCE means forgetting)."""
    recs = [ep["sa_adaptation"] for ep in episodes if ep.get("sa_adaptation")]
    if not recs:
        return {}

    def mean(key, metric):
        vals = [r[key][metric] for r in recs if r.get(key) and r[key][metric] == r[key][metric]]
        return float(np.mean(vals)) if vals else float("nan")

    return {"num_tasks": len(recs), "new_bce_before": mean("new_before", "bce"),
            "new_bce_after": mean("new_after", "bce"),
            "heldout_bce_before": mean("val_before", "bce"), "heldout_bce_after": mean("val_after", "bce"),
            "heldout_auroc_before": mean("val_before", "auroc"), "heldout_auroc_after": mean("val_after", "auroc")}


def summarize(episodes, fail_penalty=100) -> dict:
    out = {}
    for cond in ("noadapt", "adapt"):
        eps = condition_episodes(episodes, cond)
        if not eps:
            continue
        out[cond] = {"num_tasks": len({e["task_id"] for e in eps}), "num_episodes": len(eps),
                     "success_rate": success_rate(eps),
                     "success_rate_ci": list(bootstrap_ci(by_game(per_task(eps, "success_rate")).values())),
                     "time_to_completion": time_to_completion(eps, fail_penalty),
                     "time_to_completion_ci": list(bootstrap_ci(
                         by_game(per_task(eps, "time_to_completion", fail_penalty)).values())),
                     "num_games": len({e["task_id"].split("/")[0] for e in eps}),
                     "early_stop_rate": 100 * float(np.mean([bool(e.get("early_stop")) for e in eps]))}
        if any(e.get("sa_chunks") for e in eps):
            out[cond]["self_assessment"] = self_assessment(eps)
    adaptation = sa_adaptation(episodes)
    if adaptation:
        out["sa_adaptation"] = adaptation
    return out
