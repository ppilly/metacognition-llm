"""Reproduce Tables 2-5 and Fig. 9 from evaluation runs."""
import json
from pathlib import Path

from .metrics import by_game, condition_episodes, paired_comparison, per_task, self_assessment, summarize
from .runner import load_episodes

LABELS = {
    "react": "ReAct",
    "reflexion": "Reflexion",
    "reflexion_sl": "Reflexion with supervised learning",
    "muse_lowest": "MUSE with the lowest self-assessment",
    "muse_random": "MUSE with random self-assessment",
    "muse": "MUSE",
}
# The paper's values are shown only when the main run uses the paper's model (configs/default.yaml);
# other models' absolute rates are not comparable with them.
PAPER_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
PAPER = {  # reported values, for side-by-side comparison
    "table2": {"noadapt": (0.66, 60), "adapt": (0.93, 85)},
    "table3": {"react": (35, 35), "reflexion": (45, 51), "reflexion_sl": (62, 65), "muse_lowest": (59, 61),
               "muse_random": (68, 69), "muse": (84, 90)},
    "table4": {"react": (97, 97), "reflexion": (77, 66), "reflexion_sl": (61, 54), "muse_lowest": (65, 62),
               "muse_random": (53, 49), "muse": (43, 38)},
    "table5": {"react": (23, 4), "reflexion": (27, 9), "muse": (58, 55)},
}


def label(key: str) -> str:
    """'muse' -> 'MUSE'; an eval variant 'muse+paper_sa' (run.eval_variant) -> 'MUSE (paper_sa)'."""
    agent, _, variant = key.partition("+")
    return LABELS[agent] + (f" ({variant})" if variant else "")


def _protocol(run_dir: Path, agent: str):
    import yaml
    cfg = yaml.safe_load(open(run_dir / "eval" / agent / "config.yaml"))
    pr = cfg["protocol"]
    return pr["adapt_episodes"], pr["test_episodes"], pr["fail_penalty_steps"], cfg["model"]["name"]


TASK_SETS = {
    "all": "every task each agent finished",
    "common": "the tasks every agent finished",
    "solved": "the tasks every agent finished and at least one agent solved at least once (any episode)",
}


def select_tasks(episodes_by_agent: dict[str, list[dict]], which: str = "all") -> set[str] | None:
    """Task ids to score: None (all), the tasks every agent finished, or those of them that some agent
    solved in at least one episode. Restricting to common tasks keeps a stopped eval comparable."""
    if which == "all":
        return None
    common = set.intersection(*({e["task_id"] for e in eps} for eps in episodes_by_agent.values()))
    if which == "common":
        return common
    if which == "solved":
        return {e["task_id"] for eps in episodes_by_agent.values() for e in eps if e["success"]} & common
    raise ValueError(which)


def _load(run_dir: Path, which: str = "all"):
    raw = {}
    for agent in LABELS:      # each agent, then its variants (eval/<agent>+<variant>)
        dirs = [run_dir / "eval" / agent] + sorted((run_dir / "eval").glob(f"{agent}+*"))
        for d in dirs:
            if (d / "episodes.jsonl").exists():
                raw[d.name] = load_episodes(d / "episodes.jsonl")
    keep = select_tasks(raw, which) if raw else None
    runs = {}
    for agent, eps in raw.items():
        n_adapt, n_test, penalty, model = _protocol(run_dir, agent)
        if keep is not None:
            eps = [e for e in eps if e["task_id"] in keep]
        runs[agent] = {"episodes": eps, "n_adapt": n_adapt, "n_test": n_test, "model": model,
                       "penalty": penalty, "summary": summarize(eps, penalty)}
    return runs


def _fmt(v, pct=False):
    if v is None or v != v:
        return "–"
    return f"{v:.0f}%" if pct else f"{v:.0f}"


def _fmt_ci(v, ci, pct=False):
    """'75% [50–100]': value with its 95% bootstrap CI over tasks."""
    if not ci or ci[0] != ci[0]:
        return _fmt(v, pct)
    return f"{_fmt(v, pct)} [{ci[0]:.0f}–{ci[1]:.0f}]"


def _per_task(run, cond, metric):
    return by_game(per_task(condition_episodes(run["episodes"], cond), metric, run["penalty"]))


def write_report(run_dirs: list[Path], out_dir: Path, tasks: str = "all"):
    """`tasks` restricts the main run's tables to a task set (TASK_SETS); other runs are not filtered."""
    out_dir.mkdir(parents=True, exist_ok=True)
    main = _load(run_dirs[0], tasks)
    paper = bool(main) and next(iter(main.values()))["model"] == PAPER_MODEL
    lines = [f"# MUSE ALFWorld replication report\n\nMain run: `{run_dirs[0]}`"]
    if main:
        any_run = next(iter(main.values()))
        n_tasks = len({e["task_id"] for e in any_run["episodes"]})
        if tasks != "all":
            lines.append(f"Scored on {TASK_SETS[tasks]} (`--tasks {tasks}`): {n_tasks} tasks."
                         + (" Paper values are on all tasks and not directly comparable." if paper else "") + "\n")
        n_noadapt = round(sum(e["phase"] == "noadapt" for e in any_run["episodes"]) / max(n_tasks, 1))
        lines.append(f"Model: `{any_run['model']}`, {n_tasks} unseen tasks, "
                     f"{n_noadapt} No Adaptation, {any_run['n_adapt']} adaptation and "
                     f"{any_run['n_test']} test episodes per task.\n")
        lines.append("Rates are averaged over tasks. Values in brackets are 95% bootstrap confidence intervals, and "
                     "paired tests use games (scene + object + receptacle) as the independent units: trials of the "
                     "same game are averaged first.\n")

    # Table 2 + Fig. 9
    muses = [k for k in main if k.partition("+")[0] == "muse"]
    for key in muses:
        s = main[key]["summary"]
        variant = key.partition("+")[2]
        fig = f"fig9_type2_roc{'_' + variant if variant else ''}.png"
        lines += [f"## Table 2 – Self-assessment of {label(key)} on novel tasks", "",
                  "| Condition | AUROC2 (pooled) | AUROC2 (mean per task) | Metacognitive accuracy "
                  + ("| Paper AUROC2 | Paper accuracy " if paper else "") + "| # chunks |",
                  "|---|---|---|---|" + "---|" * (2 * paper) + "---|"]
        for cond, name in (("noadapt", "No Adaptation"), ("adapt", "Adaptation")):
            sa = s.get(cond, {}).get("self_assessment", {})
            pa, pacc = PAPER["table2"][cond]
            auroc = sa.get("auroc2")
            auroc = f"{auroc:.2f}" if auroc is not None and auroc == auroc else "–"
            per = sa.get("auroc2_per_task")
            n = sa.get("auroc2_n_tasks", 0)
            per = (f"{per:.2f} ({n} task{'s' * (n != 1)})" if per is not None and per == per
                   else "– (no task with both outcomes)")
            ref = f"{pa:.2f} | {pacc}% | " if paper else ""
            lines.append(f"| {name} | {auroc} | {per} | {_fmt(sa.get('meta_accuracy'), True)} | {ref}"
                         f"{sa.get('n_chunks', 0)} |")
        _plot_fig9(main[key], out_dir / fig)
        lines.append("\nPooled AUROC2 ranks chunks across tasks, each scored by its own copy of M_sa; the "
                     "per-task mean only compares chunks within a task (tasks with both outcomes).\n")
        lines.append(f"Fig. 9: `{fig}`\n")
        ad = s.get("sa_adaptation")
        if ad:
            lines += ["### M_sa adaptation (mean over tasks)", "",
                      "| | Before update | After update |", "|---|---|---|",
                      f"| BCE on the task's adaptation chunks | {ad['new_bce_before']:.3f} | {ad['new_bce_after']:.3f} |",
                      f"| BCE on held-out pre-deployment chunks | {ad['heldout_bce_before']:.3f} "
                      f"| {ad['heldout_bce_after']:.3f} |",
                      f"| AUROC on held-out pre-deployment chunks | {ad['heldout_auroc_before']:.2f} "
                      f"| {ad['heldout_auroc_after']:.2f} |", "",
                      "A rise in held-out BCE (or fall in AUROC) after the update indicates forgetting.\n"]

    # Tables 3 & 4
    for table, metric, title, pct in (("table3", "success_rate", "Table 3 – Success rate on novel tasks", True),
                                      ("table4", "time_to_completion",
                                       "Table 4 – Average time steps per episode (failures = 100)", False)):
        lines += [f"## {title}", "",
                  "| Agent | No Adaptation | Adaptation | # tasks (games) |"
                  + (" Paper (No Adapt.) | Paper (Adapt.) |" if paper else ""),
                  "|---|---|---|---|" + "---|" * (2 * paper)]
        rows = [k for a in LABELS for k in ([a] + [k for k in main if k.startswith(a + "+")])]
        for agent in rows:
            s = main.get(agent, {}).get("summary", {})
            p0, p1 = PAPER[table][agent.partition("+")[0]]
            unit = "%" if pct else ""
            cells = [_fmt_ci(s.get(c, {}).get(metric), s.get(c, {}).get(f"{metric}_ci"), pct)
                     for c in ("noadapt", "adapt")]
            c = s.get("adapt", s.get("noadapt", {}))
            n = f"{c['num_tasks']} ({c['num_games']})" if c else "–"
            ref = f" {p0}{unit} | {p1}{unit} |" if paper else ""
            lines.append(f"| {label(agent)} | {cells[0]} | {cells[1]} | {n} |{ref}")
        lines.append("")

    # Paired comparisons with MUSE
    for m in (muses if len(main) > 1 else []):
        lines += [f"## Paired comparisons: {label(m)} minus each agent, per game", "",
                  "Mean per-game difference over the games both agents ran, with its 95% bootstrap CI and a "
                  "two-sided sign-flip permutation p-value. Success in percentage points (positive favours "
                  "MUSE); time in steps (negative favours MUSE).", "",
                  "| Agent | Condition | Δ success | p | Δ time | p | # games |", "|---|---|---|---|---|---|---|"]
        for agent in main:
            if agent == m:
                continue
            for cond, cname in (("noadapt", "No Adaptation"), ("adapt", "Adaptation")):
                row = []
                for metric in ("success_rate", "time_to_completion"):
                    c = paired_comparison(_per_task(main[m], cond, metric), _per_task(main[agent], cond, metric))
                    row += [f"{c['mean_diff']:+.0f} [{c['ci'][0]:+.0f}, {c['ci'][1]:+.0f}]"
                            if c["ci"][0] == c["ci"][0] else f"{c['mean_diff']:+.0f}", f"{c['p']:.3f}"]
                lines.append(f"| {label(agent)} | {cname} | {row[0]} | {row[1]} | {row[2]} | {row[3]} | {c['n_tasks']} |")
        lines.append("")

    # Table 5
    if len(run_dirs) > 1:
        others = [(d, _load(d)) for d in run_dirs[1:]]
        names = [next(iter(r.values()))["model"] if r else d.name for d, r in others]
        lines += ["## Table 5 – Success rate with smaller / less capable LLMs (after adaptation)", "",
                  "| Agent | " + " | ".join(names) + " |" + (" Paper (OpenOrca / OpenELM) |" if paper else ""),
                  "|---" * (len(names) + 1 + paper) + "|"]
        for agent in ("react", "reflexion", "muse"):
            vals = [_fmt(r.get(agent, {}).get("summary", {}).get("adapt", {}).get("success_rate"), True)
                    for _, r in others]
            p = PAPER["table5"][agent]
            lines.append(f"| {LABELS[agent]} | " + " | ".join(vals) + " |" + (f" {p[0]}% / {p[1]}% |" if paper else ""))
        lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines))
    summaries = {str(d): {a: r["summary"] for a, r in (main if i == 0 else _load(d)).items()}
                 for i, d in enumerate(run_dirs)}
    for runs in summaries.values():
        for s in runs.values():
            for cond in s.values():
                cond.get("self_assessment", {}).pop("roc", None)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    print("\n".join(lines))


def _plot_fig9(run, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, (cond, title) in zip(axes, (("noadapt", "No Adaptation"), ("adapt", "Adaptation"))):
        sa = self_assessment(condition_episodes(run["episodes"], cond))
        ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
        if "roc" in sa:
            ax.plot(sa["roc"]["fpr"], sa["roc"]["tpr"], color="#1f5fbf", lw=2,
                    label=f"AUROC2 = {sa['auroc2']:.2f}")
            ax.legend(loc="lower right", frameon=False)
        ax.set_title(title)
        ax.set_xlabel("Type 2 false positive rate")
        ax.set_ylabel("Type 2 true positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
