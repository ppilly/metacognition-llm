"""Episode loop, per-task protocol and concurrent execution over tasks.

Per-task protocol (Alg. 3 / Alg. 4, Sec. 4.2), with phases recorded on every episode:
  'noadapt'  n_test independent episodes of the model as it is at the end of pre-deployment
             training: no reflections from this task and the pre-deployment M_sa (No Adaptation).
  'adapt'    n_adapt sequential episodes; reflections accumulate. Then `agent.after_adaptation`
             runs (MUSE updates this task's copy of g_eta; baselines do nothing). Tasks are
             independent: each starts from the end-of-pre-deployment models.
  'test'     n_test further episodes (Adaptation). With `protocol.test_independent`, each is independent
             and sees only the adaptation episodes' reflections; otherwise they run as a sequence
             and reflections keep accumulating.
Collection (Alg. 3) uses n_test = 0, i.e. only the 'adapt' phase. Raising the number of collection
episodes extends every finished task: it continues after its last episode with the reflections it
had built up, starting with the reflection on that episode, which was skipped as the last one.

Crash safety: a finished task is appended to the run file as one fsynced line. Until then, the task's
episodes and agent state are checkpointed after every episode in `<run file stem>.partial/`, so a
restart reruns at most the episode that was in progress.
"""
import hashlib
import json
import logging
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from .agents import EpisodeContext, Step, TaskState
from .env import ALFWorldEnv, Task, task_description

log = logging.getLogger(__name__)


def run_episode(env: ALFWorldEnv, agent, state: TaskState, max_steps: int,
                stop_on_repeat: bool = True) -> tuple[EpisodeContext, bool, bool]:
    """Returns (context, success, stopped early). Every actor output, `think:` included, is a step.
    As in Reflexion's ALFWorld loop, repeating the previous action ends the episode as a failure
    (unless that action wins)."""
    init_obs = env.reset(state.task)
    ctx = EpisodeContext(task=state.task, init_obs=init_obs, task_desc=task_description(init_obs),
                         memory=agent.memory_window(state), admissible=getattr(env, "admissible", None))
    for _ in range(max_steps):
        action, info = agent.act(ctx, state)
        if action.startswith("think:"):
            obs, won = "OK.", False
        else:
            obs, won = env.step(action)
            ctx.admissible = getattr(env, "admissible", None)
        repeated = bool(ctx.steps) and ctx.steps[-1].action == action
        ctx.steps.append(Step(action, obs, info))
        if won:
            return ctx, True, False
        if stop_on_repeat and repeated:
            return ctx, False, True
    return ctx, False, False


def run_task(env, agent, task: Task, cfg, n_adapt: int, n_test: int, resume: dict | None = None,
             on_episode=None) -> list[dict]:
    """`resume`: a checkpoint {'records', 'agent'} of this task's first episodes, to continue from.
    'agent' is None when extending a finished collection task (see `_extend_state`).
    `on_episode(records, agent_state)` is called after every episode, to checkpoint it."""
    state = agent.new_task_state(task)
    phases = ["noadapt"] * _n_noadapt(cfg, n_test) + ["adapt"] * n_adapt + ["test"] * n_test
    reflect_phases = ("adapt",) if cfg.protocol.test_independent else ("adapt", "test")
    records = []
    if resume:
        records = list(resume["records"])
        if resume["agent"] is None:
            _extend_state(agent, state, records, cfg)
        else:
            agent.restore_task_state(state, resume["agent"])
    for e, phase in enumerate(phases):
        if e < len(records):
            continue
        ctx, success, stopped = run_episode(env, agent, state, cfg.env.max_steps, cfg.agent.stop_on_repeat)
        rec = {
            "task_id": task.task_id, "task_type": task.task_type, "task_desc": ctx.task_desc,
            "init_obs": ctx.init_obs, "episode": e, "phase": phase,
            "memory": [{"text": m["text"], "success": m["success"]} for m in ctx.memory],
            "steps": [asdict(s) for s in ctx.steps], "success": success, "num_steps": len(ctx.steps),
            "early_stop": stopped, "plan": ctx.plan, "reflection": None, "sa_chunks": None,
        }
        if agent.uses_sa:
            rec["sa_chunks"] = agent.assess_episode(state, ctx)
        # No Adaptation (and independent test) episodes leave no reflection; one after the final
        # episode would never be used
        if phase in reflect_phases and e < len(phases) - 1:
            refl = agent.reflect(state, ctx, success)
            if refl is not None:
                rec["reflection"] = refl
                state.memory.append({"episode": e, "success": success, "text": refl["text"]})
        records.append(rec)
        log.info("%s episode %d (%s): %s in %d steps%s", task.task_id, e, phase, "success" if success else "fail",
                 len(ctx.steps), " (repeated action)" if stopped else "")
        if phase == "adapt" and e + 1 < len(phases) and phases[e + 1] == "test":
            adaptation = agent.after_adaptation(state, [r for r in records if r["phase"] == "adapt"])
            if adaptation is not None:
                rec["sa_adaptation"] = adaptation
        if on_episode:
            on_episode(records, agent.task_state(state))
    return records


def _n_noadapt(cfg, n_test: int) -> int:
    """No Adaptation episodes per task: n_test, or none with `protocol.skip_noadapt` (debugging runs)."""
    return 0 if cfg.protocol.skip_noadapt else n_test


def _extend_state(agent, state: TaskState, records: list[dict], cfg):
    """Rebuild a finished collection task's state from its records: the memory from the stored
    reflections, plus the reflection its last episode skipped."""
    state.memory = [{"episode": r["episode"], "success": r["success"], "text": r["reflection"]["text"]}
                    for r in records if r.get("reflection")]
    last = records[-1] = dict(records[-1])
    if last["reflection"] is None and last["phase"] == "adapt":
        ctx = EpisodeContext(task=state.task, init_obs=last["init_obs"], task_desc=last["task_desc"],
                             memory=last["memory"], steps=[Step(**s) for s in last["steps"]])
        refl = agent.reflect(state, ctx, last["success"])
        if refl is not None:
            last["reflection"] = refl
            state.memory.append({"episode": last["episode"], "success": last["success"], "text": refl["text"]})


def _read_jsonl(path: Path, repair: bool = False) -> list[dict]:
    """Rows of a run file. A last line cut short by a crash is dropped with a warning (with `repair`,
    also removed from the file, so that its task reruns); damage anywhere else is an error."""
    if not path.exists():
        return []
    data = path.read_bytes()
    rows, good = [], 0
    for line in data.splitlines(keepends=True):
        try:
            if not line.endswith(b"\n"):
                raise ValueError("no newline")
            if line.strip():
                rows.append(json.loads(line))
        except ValueError:
            break
        good += len(line)
    if good < len(data):
        if data[good:].rstrip(b"\n").count(b"\n"):
            raise ValueError(f"{path}: unreadable line at byte {good}, followed by more lines")
        log.warning("%s: dropping an incomplete last line (%d bytes) left by a crash%s", path, len(data) - good,
                    "" if not repair else "; its task reruns")
        if repair:
            with open(path, "r+b") as f:
                f.truncate(good)
                os.fsync(f.fileno())
    return rows


def _append_line(path: Path, row: dict):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _replace_line(path: Path, task_id: str, row: dict):
    """Swap one task's line for `row` by rewriting the file to a temporary one and renaming it over the
    original, so a crash leaves either the old or the new file."""
    rows = [row if r["task_id"] == task_id else r for r in _read_jsonl(path)]
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _save_atomic(obj, path: Path):
    """Write, fsync, then rename over the old file: a crash leaves either the old or the new checkpoint."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_checkpoint(path: Path, task_id: str) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
    except Exception:
        log.warning("%s: unreadable checkpoint, task %s restarts", path, task_id)
        return None
    return ckpt if ckpt.get("task_id") == task_id else None


def _live_metrics(records_by_task: list[list[dict]], n_adapt: int, n_test: int) -> dict:
    """Running success rates for the dashboard."""
    eps = [r for recs in records_by_task for r in recs]
    if n_test == 0:
        groups = {"success_rate": eps}
    else:
        groups = {"success_noadapt": [r for r in eps if r["phase"] == "noadapt"],
                  "success_adapt": [r for r in eps if r["phase"] == "test"]}
    return {k: 100 * sum(r["success"] for r in v) / len(v) for k, v in groups.items() if v}


def run_tasks(cfg, agent, tasks: list[Task], out_path: Path, n_adapt: int, n_test: int, status=None):
    """Run the protocol on every task concurrently; append one JSON line per task. Resumable: finished
    tasks are skipped, and unfinished ones continue from their last checkpointed episode. In collection
    (n_test = 0), a finished task with fewer than n_adapt episodes is extended and its line replaced."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(out_path, repair=True)
    n_episodes = n_adapt + n_test + _n_noadapt(cfg, n_test)
    short = {r["task_id"]: r["episodes"] for r in rows if n_test == 0 and len(r["episodes"]) < n_adapt}
    done = {r["task_id"] for r in rows} - set(short)
    for r in rows:
        if r["task_id"] in done and len(r["episodes"]) != n_episodes:
            log.warning("%s has %d episodes, not %d; kept as it is", r["task_id"], len(r["episodes"]), n_episodes)
    todo = [t for t in tasks if t.task_id not in done]
    log.info("%d tasks (%d already done%s) -> %s", len(tasks), len(done),
             f", {len(short)} to extend to {n_adapt} episodes" if short else "", out_path)
    finished = [r["episodes"] for r in rows if r["task_id"] in done]   # incl. tasks from an earlier run
    ckpt_dir = out_path.with_suffix(".partial")
    ckpt_dir.mkdir(exist_ok=True)

    def ckpt_path(task_id):
        return ckpt_dir / (hashlib.md5(task_id.encode()).hexdigest()[:16] + ".pkl")

    for task_id in done:   # a crash between writing a task's line and removing its checkpoint
        ckpt_path(task_id).unlink(missing_ok=True)
    if status:
        status.update(len(finished), len(tasks), "tasks", **_live_metrics(finished, n_adapt, n_test))
    if not todo:
        if not any(ckpt_dir.iterdir()):
            ckpt_dir.rmdir()
        return
    write_lock = threading.Lock()
    local = threading.local()
    envs = []

    def get_env():
        if not hasattr(local, "env"):
            local.env = ALFWorldEnv()
            envs.append(local.env)
        return local.env

    def work(task):
        t0 = time.time()
        ckpt = ckpt_path(task.task_id)
        resume = _load_checkpoint(ckpt, task.task_id)
        if resume:
            log.info("%s: resuming after %d checkpointed episodes", task.task_id, len(resume["records"]))
        elif task.task_id in short:
            resume = {"records": short[task.task_id], "agent": None}
            log.info("%s: extending from %d to %d episodes", task.task_id, len(short[task.task_id]), n_adapt)
        records = run_task(get_env(), agent, task, cfg, n_adapt, n_test, resume=resume,
                           on_episode=lambda recs, agent_state: _save_atomic(
                               {"task_id": task.task_id, "records": recs, "agent": agent_state}, ckpt))
        with write_lock:
            row = {"task_id": task.task_id, "episodes": records}
            if task.task_id in short:
                _replace_line(out_path, task.task_id, row)
            else:
                _append_line(out_path, row)
            ckpt.unlink(missing_ok=True)
            finished.append(records)
            if status:
                status.update(len(finished), len(tasks), "tasks",
                              **_live_metrics(finished, n_adapt, n_test))
        outcome = "".join("S" if r["success"] else "." for r in records)
        return task.task_id, outcome, time.time() - t0

    try:
        with ThreadPoolExecutor(max_workers=cfg.run.workers) as pool:
            futures = [pool.submit(work, t) for t in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                task_id, outcome, secs = fut.result()
                log.info("[%d/%d] %s  %s  (%.0fs)", i, len(todo), outcome, task_id, secs)
    finally:
        for env in envs:
            env.close()
    if not any(ckpt_dir.iterdir()):
        ckpt_dir.rmdir()


def load_episodes(path) -> list[dict]:
    """Flatten a run file into a list of episode records."""
    return [ep for row in _read_jsonl(Path(path)) for ep in row["episodes"]]
