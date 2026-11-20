"""Progress dashboard: which stage each agent's pipeline is in, for every model run.

Reads runs/<run>/status/*.json (written by muse.status) and renders a terminal table plus a
self-contained HTML page that reloads itself.
"""
import html
import time
from pathlib import Path

from .status import read_status

PRETRAIN = [("collect", "Collect"), ("train-sft", "SFT M_a"), ("train-dpo", "DPO M_rx"), ("train-sa", "Train M_sa")]
AGENTS = [
    ("react", "ReAct", []),
    ("reflexion", "Reflexion", []),
    ("reflexion_sl", "Reflexion + SL", ["collect", "train-sft", "train-dpo"]),
    ("muse_lowest", "MUSE (lowest SA)", ["collect", "train-sft", "train-dpo", "train-sa"]),
    ("muse_random", "MUSE (random SA)", ["collect", "train-sft", "train-dpo", "train-sa"]),
    ("muse", "MUSE", ["collect", "train-sft", "train-dpo", "train-sa"]),
]
STAGE_LABEL = dict(PRETRAIN)


def _fmt_duration(sec):
    if sec is None:
        return "–"
    sec = int(sec)
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m"
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def _eta(st):
    p = st.get("progress")
    if st["state"] != "running" or not p or not p["total"]:
        return None
    progressed = p["done"] - p.get("start", 0)
    if progressed <= 0:
        return None
    return (st["updated"] - st["started"]) / progressed * (p["total"] - p["done"])


def _pipeline(statuses, agent, deps):
    """[(stage, label, state)] for one agent: its pre-deployment dependencies, then deployment."""
    stages = [(d, STAGE_LABEL[d]) for d in deps] + [(f"eval-{agent}", "Deploy")]
    return [(s, label, statuses.get(s, {}).get("state", "pending")) for s, label in stages]


def _current(pipe):
    """The stage to report: the first one that is not done (or the last one if all are done)."""
    for stage in pipe:
        if stage[2] != "done":
            return stage
    return pipe[-1]


def collect_runs(runs_dir: Path) -> list[dict]:
    runs = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir() and (p / "status").is_dir()):
        statuses = read_status(run_dir)
        if not statuses:
            continue
        model = next((s["metrics"].get("model") for s in statuses.values() if s["metrics"].get("model")), "?")
        agents = []
        for key, label, deps in AGENTS:
            pipe = _pipeline(statuses, key, deps)
            stage, stage_label, state = _current(pipe)
            agents.append({"key": key, "label": label, "pipe": pipe, "stage": stage, "stage_label": stage_label,
                           "state": state, "status": statuses.get(stage), "eval": statuses.get(f"eval-{key}")})
        runs.append({"name": run_dir.name, "model": model, "statuses": statuses, "agents": agents})
    return runs


def _progress_text(st):
    if not st or not st.get("progress"):
        return ""
    p = st["progress"]
    return f"{p['done']}/{p['total']} {p['unit']}"


def _pct(v):
    return "–" if v is None else f"{v:.0f}%"


# ----------------------------------------------------------------------------- terminal

def render_text(runs) -> str:
    out = []
    for run in runs:
        out.append(f"\n{run['name']}  ({run['model']})")
        out.append(f"  {'agent':<18}{'stage':<12}{'state':<9}{'progress':<16}{'ETA':<8}{'no-adapt':<10}{'adapt':<8}")
        for a in run["agents"]:
            st, ev = a["status"], (a["eval"] or {}).get("metrics", {})
            out.append(f"  {a['label']:<18}{a['stage_label']:<12}{a['state']:<9}{_progress_text(st):<16}"
                       f"{_fmt_duration(_eta(st)) if st else '–':<8}{_pct(ev.get('success_noadapt')):<10}"
                       f"{_pct(ev.get('success_adapt')):<8}")
    return "\n".join(out) if out else "no runs with status files yet"


# ----------------------------------------------------------------------------- HTML

CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e4e3de;--track:#ecebe6;
--done:#2f7d4f;--run:#2563c9;--fail:#c2412d;--stop:#b7791f;--pend:#a3a29b}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;--bg:#151514;--card:#1e1e1c;
--ink:#ecebe6;--muted:#9d9c95;--line:#2f2f2c;--track:#2a2a27;--done:#4fae75;--run:#5b93f0;--fail:#e2664f;
--stop:#d9a03f;--pend:#6b6b66}}
:root[data-theme=dark]{color-scheme:dark;--bg:#151514;--card:#1e1e1c;--ink:#ecebe6;--muted:#9d9c95;--line:#2f2f2c;
--track:#2a2a27;--done:#4fae75;--run:#5b93f0;--fail:#e2664f;--stop:#d9a03f;--pend:#6b6b66}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;
padding:24px 16px}
main{max-width:1100px;margin:0 auto}
h1{font-size:20px;margin:0 0 2px}
.sub{color:var(--muted);margin:0 0 20px}
.run{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:18px}
.run h2{font-size:16px;margin:0}
.run .model{color:var(--muted);font-family:ui-monospace,Menlo,monospace;font-size:12.5px}
.scroll{overflow-x:auto;margin-top:12px}
table{border-collapse:collapse;width:100%;min-width:760px}
th{font-weight:500;color:var(--muted);text-align:left;font-size:12px;padding:6px 10px;border-bottom:1px solid var(--line)}
td{padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:middle}
tr:last-child td{border-bottom:none}
td.num{font-variant-numeric:tabular-nums;text-align:right}
th.num{text-align:right}
.pipe{display:flex;gap:4px;flex-wrap:wrap}
.chip{font-size:11.5px;padding:2px 7px;border-radius:999px;border:1px solid currentColor;white-space:nowrap}
.chip.done{color:var(--done)} .chip.running{color:var(--run);font-weight:600}
.chip.failed{color:var(--fail);font-weight:600} .chip.stopped{color:var(--stop)} .chip.pending{color:var(--pend)}
.chip.done::before{content:"✓ "} .chip.failed::before{content:"✕ "} .chip.stopped::before{content:"■ "}
.chip.running::before{content:"● "}
.bar{height:6px;background:var(--track);border-radius:3px;overflow:hidden;min-width:90px}
.bar>i{display:block;height:100%;background:var(--run)}
.bar.done>i{background:var(--done)}
.ptext{font-size:12px;color:var(--muted);margin-top:3px;font-variant-numeric:tabular-nums}
.err{color:var(--fail);font-size:12px}
.foot{color:var(--muted);font-size:12px}
"""


def _row(a):
    st = a["status"]
    pipe = "".join(f'<span class="chip {state}">{html.escape(label)}</span>' for _, label, state in a["pipe"])
    p = (st or {}).get("progress")
    frac = (p["done"] / p["total"]) if p and p["total"] else (1.0 if a["state"] == "done" else 0.0)
    detail = _progress_text(st)
    if st and a["state"] == "running" and _eta(st) is not None:
        detail += f" · ETA {_fmt_duration(_eta(st))}"
    m = (st or {}).get("metrics", {})
    if st and a["stage"].startswith("train") and "loss" in m:
        detail += f" · loss {m['loss']:.3f}"
    if st and a["stage"] == "train-sa" and "bce" in m:
        detail += f" · bce {m['bce']:.3f}"
    if st and a["state"] == "failed":
        detail += f'<div class="err">{html.escape(st.get("error", ""))[:200]}</div>'
    ev = (a["eval"] or {}).get("metrics", {})
    auroc = ev.get("auroc2_adapt")
    return (f"<tr><td><b>{html.escape(a['label'])}</b></td><td><div class=pipe>{pipe}</div></td>"
            f"<td><div class='bar {'done' if a['state'] == 'done' else ''}'><i style='width:{100 * frac:.1f}%'></i>"
            f"</div><div class=ptext>{detail}</div></td>"
            f"<td class=num>{_pct(ev.get('success_noadapt'))}</td><td class=num>{_pct(ev.get('success_adapt'))}</td>"
            f"<td class=num>{'–' if auroc is None or auroc != auroc else f'{auroc:.2f}'}</td></tr>")


def render_html(runs, refresh: float) -> str:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    meta = f'<meta http-equiv="refresh" content="{int(refresh)}">' if refresh else ""
    body = []
    for run in runs:
        rows = "".join(_row(a) for a in run["agents"])
        body.append(
            f'<section class=run><h2>{html.escape(run["name"])}</h2>'
            f'<div class=model>{html.escape(run["model"])}</div><div class=scroll><table>'
            "<tr><th>Agent</th><th>Pipeline</th><th>Current stage</th><th class=num>Success<br>no adapt.</th>"
            "<th class=num>Success<br>adapt.</th><th class=num>AUROC2</th></tr>"
            f"{rows}</table></div></section>")
    if not body:
        body.append('<p class=sub>No runs with status files yet. Start a stage with <code>python -m muse.cli</code>.</p>')
    note = f"Reloads every {int(refresh)}s · " if refresh else ""
    return (f"<!doctype html><html><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>{meta}"
            f"<title>MUSE run status</title><style>{CSS}</style></head><body><main>"
            f"<h1>MUSE run status</h1><p class=sub>Pre-deployment stages are shared by Reflexion + SL and the "
            f"MUSE variants. Success rates are live averages over finished tasks.</p>"
            f"{''.join(body)}<p class=foot>{note}updated {now}</p></main></body></html>")


def write_dashboard(runs_dir: Path, out: Path, refresh: float = 0) -> list[dict]:
    runs = collect_runs(runs_dir) if runs_dir.is_dir() else []
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(render_html(runs, refresh))
    tmp.replace(out)
    return runs


def watch(runs_dir: Path, out: Path, every: float):
    while True:
        runs = write_dashboard(runs_dir, out, every)
        print("\033[2J\033[H" if every else "", end="")
        print(render_text(runs))
        print(f"\ndashboard: {out.resolve()}")
        if not every:
            return
        time.sleep(every)
