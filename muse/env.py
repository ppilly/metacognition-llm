"""ALFWorld text environment: one game file per task, each game hosted in its own subprocess.

The TextWorld PDDL backend (fast-downward via ctypes) keeps global state, so games are run in
separate processes rather than threads.
"""
import json
import multiprocessing as mp
import re
from dataclasses import dataclass
from pathlib import Path

# ReAct's task-type mapping, game directory prefix -> few-shot prompt key
# (see muse/data/THIRD_PARTY_LICENSES).
TASK_PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


@dataclass(frozen=True)
class Task:
    task_id: str      # "<task dir>/<trial dir>", unique within a split
    gamefile: str
    task_type: str    # one of TASK_PREFIXES values

    @property
    def family(self) -> str:
        return self.task_id.split("/")[0]


def list_tasks(data_dir: str, split: str, num_tasks=None, task_ids=None, keep_order=False) -> list[Task]:
    """Collect solvable games exactly as alfworld.agents.environment.AlfredTWEnv does, sorted by id
    or, with `keep_order`, in the order of `task_ids`."""
    if "${" in str(data_dir):
        raise RuntimeError("Set ALFWORLD_DATA (or env.data_dir) to the ALFWorld data directory")
    root = Path(data_dir) / "json_2.1.1" / split
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found; run `alfworld-download` with ALFWORLD_DATA set")
    tasks = []
    for traj in sorted(root.glob("*/*/traj_data.json")):
        game_dir = traj.parent
        if "movable" in str(game_dir) or "Sliced" in str(game_dir):
            continue
        gamefile = game_dir / "game.tw-pddl"
        if not gamefile.exists():
            continue
        with open(gamefile) as f:
            if not json.load(f).get("solvable", False):
                continue
        task_id = f"{game_dir.parent.name}/{game_dir.name}"
        task_type = next((v for k, v in TASK_PREFIXES.items() if task_id.startswith(k)), None)
        if task_type is None:
            continue
        tasks.append(Task(task_id, str(gamefile), task_type))
    if task_ids:
        wanted = set(task_ids)
        tasks = [t for t in tasks if t.task_id in wanted]
        if keep_order:
            rank = {i: n for n, i in enumerate(dict.fromkeys(task_ids))}
            tasks.sort(key=lambda t: rank[t.task_id])
    if num_tasks:
        tasks = tasks[:num_tasks]
    return tasks


# ALFWorld >= 0.4 renamed "put {o} in/on {r}" to "move {o} to {r}". ReAct's prompts and the paper's
# transcripts use the original grammar, so translate in both directions. Only the exact original
# form is accepted ("put X in Y" stays invalid, as in the original game).
_PUT = re.compile(r"^put (.+?) in/on (.+)$")
_MOVED = re.compile(r"^You move the (.+?) to the (.+?)\.$")


def to_game_command(action: str) -> str:
    m = _PUT.match(action)
    return f"move {m.group(1)} to {m.group(2)}" if m else action


def from_game_feedback(obs: str) -> str:
    m = _MOVED.match(obs)
    return f"You put the {m.group(1)} in/on the {m.group(2)}." if m else obs


def process_obs(obs: str) -> str:
    """After ReAct's observation clean-up (see muse/data/THIRD_PARTY_LICENSES).

    Strip the arrival prefix ('You arrive at loc 5. ' in older ALFWorld, 'You arrive at drawer 1. '
    in 0.4.x) so observations match the few-shot examples ('The drawer 1 is closed.')."""
    if obs.startswith("You arrive at "):
        i = obs.find(". ")
        if i != -1:
            obs = obs[i + 2:]
    return obs


def process_initial_obs(obs: str) -> str:
    """Drop the '-= Welcome to TextWorld, ALFRED! =-' banner, as ReAct does (see
    muse/data/THIRD_PARTY_LICENSES)."""
    return "\n".join(obs.split("\n\n")[1:])


def _worker(conn):
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    env, loaded = None, None
    while True:
        cmd, arg = conn.recv()
        try:
            if cmd == "reset":
                if arg != loaded:
                    if env is not None:
                        env.close()
                    env = textworld.start(arg, request_infos=infos,
                                          wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
                    loaded = arg
                state = env.reset()
                conn.send(("ok", (state.feedback, bool(state.won), list(state.admissible_commands or []))))
            elif cmd == "step":
                state, _, done = env.step(arg)
                conn.send(("ok", (state.feedback, bool(state.won), list(state.admissible_commands or []))))
            elif cmd == "close":
                if env is not None:
                    env.close()
                conn.send(("ok", None))
                return
        except Exception as e:  # report to parent rather than dying silently
            conn.send(("error", repr(e)))


class ALFWorldEnv:
    """Runs one ALFWorld game at a time in a child process. `admissible` holds the game's valid commands
    after the last reset or step (in game form: to_game_command of an action is checked against it)."""

    def __init__(self):
        self.admissible: list[str] | None = None
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_worker, args=(child,), daemon=True)
        self._proc.start()

    def _call(self, cmd, arg=None):
        self._conn.send((cmd, arg))
        status, payload = self._conn.recv()
        if status != "ok":
            raise RuntimeError(f"ALFWorld env error on {cmd}({arg!r}): {payload}")
        return payload

    def reset(self, task: Task) -> str:
        feedback, _, self.admissible = self._call("reset", task.gamefile)
        return process_initial_obs(feedback)

    def step(self, action: str) -> tuple[str, bool]:
        feedback, won, self.admissible = self._call("step", to_game_command(action))
        return from_game_feedback(process_obs(feedback)), won

    def close(self):
        try:
            self._call("close")
        except Exception:
            pass
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.kill()


def task_description(init_obs: str) -> str:
    """The natural-language task I, e.g. 'put a clean bowl in cabinet.'"""
    marker = "Your task is to: "
    idx = init_obs.find(marker)
    return init_obs[idx + len(marker):].strip() if idx >= 0 else init_obs.strip()
