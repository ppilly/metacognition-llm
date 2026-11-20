"""Draw a seeded random subset of ALFWorld tasks (after AlfredTWEnv's filtering) as a task-id list.

  python scripts/sample_tasks.py --split valid_seen   --n 10 --seed 0 --out configs/tasks/local_train.json
  python scripts/sample_tasks.py --split valid_unseen --n 10 --seed 0 --out configs/tasks/local_eval.json
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from muse.env import list_tasks  # noqa: E402


def sample(data_dir, split, n, seed, prefix=None):
    ids = sorted(t.task_id for t in list_tasks(data_dir, split))
    if prefix:
        ids = [i for i in ids if i.startswith(prefix)]
    return sorted(random.Random(seed).sample(ids, n))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", help="only task ids starting with this, e.g. pick_and_place_simple")
    ap.add_argument("--data-dir", default=os.environ.get("ALFWORLD_DATA", "data/alfworld"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a run's task list is never redrawn")
    ids = sample(args.data_dir, args.split, args.n, args.seed, args.prefix)
    out.write_text(json.dumps(ids, indent=1) + "\n")
    print("\n".join(ids))


if __name__ == "__main__":
    main()
