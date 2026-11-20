"""Per-stage progress files (runs/<run>/status/<stage>.json) read by `muse.cli status`."""
import json
import os
import time
from pathlib import Path


class StageStatus:
    def __init__(self, run_dir, stage: str):
        self.path = Path(run_dir) / "status" / f"{stage}.json"
        self.data = {"stage": stage, "state": "running", "pid": os.getpid(), "started": time.time(),
                     "updated": time.time(), "progress": None, "metrics": {}}
        self._write()

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data))
        tmp.replace(self.path)   # atomic, so the dashboard never reads a half-written file

    def update(self, done=None, total=None, unit=None, **metrics):
        if done is not None:
            # the first value seen is work already done before this process started (resumed runs)
            start = (self.data["progress"] or {}).get("start", done)
            self.data["progress"] = {"done": done, "total": total, "unit": unit, "start": start}
        self.data["metrics"].update(metrics)
        self.data["updated"] = time.time()
        self._write()

    def finish(self, state="done", error=None):
        self.data.update(state=state, updated=time.time(), finished=time.time())
        if error:
            self.data["error"] = error
        self._write()


def read_status(run_dir) -> dict[str, dict]:
    out = {}
    for p in sorted((Path(run_dir) / "status").glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if d["state"] == "running" and not _alive(d.get("pid")):
            d["state"] = "stopped"
        out[d["stage"]] = d
    return out


def _alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
