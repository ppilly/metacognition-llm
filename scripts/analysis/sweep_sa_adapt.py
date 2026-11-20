"""Offline sweep of M_sa's deployment update (Sec. 4.2) on pre-deployment data only: no deployment task
is used. Each training task with both outcomes takes the place of a novel task in turn:

  1. g_eta is trained as train-sa trains it (self_assessment.*: chunk stride, class balance, held-out
     tasks for early stopping) on every other training task, whose chunks fill the replay buffer;
  2. a copy is adapted on the left-out task's first --adapt-episodes collection episodes (as on the
     adaptation episodes at deployment), once per self_assessment.adapt_new_fraction in --fractions;
  3. it scores the left-out task's remaining episodes, as the test episodes.

Reports within-task AUROC on those episodes' non-overlapping executed chunks, which must include both
outcomes, and, for forgetting, the BCE on the held-out chunks of step 1's early-stopping tasks. Collection episodes have no imagined rollouts, so adapt_rollouts cannot be compared here.
input: actor only; features come from compare_sa.py's cache when present.

  python scripts/analysis/sweep_sa_adapt.py -c configs/local.yaml [--fractions 0.5 0.3 0.2 0.1]
      [--adapt-episodes 5]
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_sa import actor_features  # noqa: E402

from muse.config import load_config  # noqa: E402
from muse.datasets import sa_examples, split_by_task  # noqa: E402
from muse.runner import load_episodes  # noqa: E402
from muse.self_assessment import ReplayBuffer, SelfAssessor  # noqa: E402


def auroc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.5, 0.3, 0.2, 0.1])
    ap.add_argument("--adapt-episodes", type=int, default=None, help="default: protocol.adapt_episodes")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    sc, size = cfg.self_assessment, cfg.muse.chunk_size
    assert sc.input == "actor", "the sweep reads actor features (self_assessment.input: actor)"
    n_adapt = args.adapt_episodes or cfg.protocol.adapt_episodes
    root = Path(cfg.run.out_dir)
    episodes = load_episodes(root / "collect" / "episodes.jsonl")

    d = sa_examples(episodes, size, stride=1)        # actor_features' row order
    x = actor_features(cfg, episodes, root / "sa_compare", root.name, [sc.layer])[sc.layer]
    y, tasks, starts = torch.tensor(d["labels"]).float(), np.array(d["task_ids"]), np.array(d["starts"])
    ep_idx, seen = [], {}
    for ep in episodes:                               # each row's episode index within its task
        k = seen[ep["task_id"]] = seen.get(ep["task_id"], -1) + 1
        ep_idx += [k] * len(sa_examples([ep], size, stride=1)["labels"])
    ep_idx = np.array(ep_idx)
    assert len(ep_idx) == len(y) == len(x)
    train_rows = starts % (sc.chunk_stride or size) == 0
    test_rows = starts % size == 0

    folds = [t for t in sorted(set(tasks))
             if len(set(y[(tasks == t) & (ep_idx >= n_adapt) & test_rows].tolist())) == 2]
    print(f"{len(episodes)} collection episodes; {len(folds)} training tasks with both outcomes after "
          f"episode {n_adapt}")
    settings = [("no update", None)] + [(f"{fr:.0%} new", fr) for fr in args.fractions]
    rows = []
    for t in folds:
        other = train_rows & (tasks != t)
        idx = np.flatnonzero(other)
        tr, va = split_by_task(tasks[idx].tolist(), sc.val_fraction, cfg.run.seed,
                               labels=y[idx].tolist() if sc.val_tasks == "mixed" else None)
        tr, va = idx[tr], idx[va]
        torch.manual_seed(cfg.run.seed)
        base = SelfAssessor(cfg, in_dim=x.shape[1])
        if sc.standardize:
            base.set_standardizer(x[tr])
        base.fit(x[tr], y[tr], sc.epochs, sc.lr,
                 val=(x[va], y[va], tasks[va].tolist()) if len(va) else None)
        base.replay = ReplayBuffer(x[tr], y[tr])
        new = train_rows & (tasks == t) & (ep_idx < n_adapt)
        test = test_rows & (tasks == t) & (ep_idx >= n_adapt)
        row = {"task": t.split("/")[0], "adapt_chunks": int(new.sum()), "adapt_success": int(y[new].sum()),
               "test_chunks": int(test.sum()), "test_success": int(y[test].sum())}
        for name, fr in settings:
            sa = base.clone()
            if fr is not None:
                sa.cfg = copy.copy(base.cfg)
                sa.cfg.adapt_new_fraction = fr
                torch.manual_seed(cfg.run.seed)
                sa.adapt(x[new], y[new])
            row[name] = {"auroc": auroc(y[test], sa.predict_features(x[test])),
                         "heldout_bce": sa.evaluate(x[va], y[va])["bce"] if len(va) else float("nan")}
        rows.append(row)
        print(f"{row['task'][:45]:45s} " + " ".join(f"{row[n]['auroc']:.3f}" for n, _ in settings), flush=True)

    names = [s[0] for s in settings]
    summary = {n: {k: float(np.nanmean([r[n][k] for r in rows])) for k in ("auroc", "heldout_bce")}
               for n in names}
    out = root / "sa_sweep"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"results_layer{sc.layer}_adapt{n_adapt}.json"
    path.write_text(json.dumps({"folds": rows, "summary": summary}, indent=2))
    print(f"{'mean within-task AUROC':45s} " + " ".join(f"{summary[n]['auroc']:.3f}" for n in names))
    print(f"{'held-out BCE of the other tasks (forgetting)':45s} "
          + " ".join(f"{summary[n]['heldout_bce']:.3f}" for n in names))
    print(f"({' | '.join(names)}; results in {path})")


if __name__ == "__main__":
    main()
