"""Offline comparison of M_sa inputs on collected episodes, leaving out one task at a time.

  paper   : the paper's M_sa, Z = E(I) and S = E(P_e, chunk) from the sentence encoder (Eq. 6)
  actor   : extension, the actor LLM's final hidden states given its full context (actor_context_features)
  actor@k : the same from decoder layer k (--layers)
  index   : baseline, the chunk's position in its episode (later chunks score lower)
  q@k     : with --question, the hidden state of decoder layer k (q@final: final layer) at the last
            token of QUESTION, appended after the chunk's last observation
  gap     : with --question, log p(" Yes") - log p(" No") for the next token after QUESTION, as a 1-d
            input; gap0 scores the raw gap with no training (zero-shot)
  actor@k+q@k+gap : the three concatenated

Each learned input is fit once per --balance option (self_assessment.class_balance), with features
standardized on the fold's training chunks when self_assessment.standardize is true.

Each fold trains a fresh g_eta (self_assessment.* settings) on every other task and scores the left-out
task, which must have both successful and failed episodes. Training chunks start every
self_assessment.chunk_stride steps (1: every step); test chunks are always the non-overlapping ones.
Within-task AUROC is what MUSE's choice among rollouts relies on. The actor features come from the
first config's actor for every run's episodes.

  python scripts/analysis/compare_sa.py -c configs/local.yaml [--also <run>] [--layers final 7 14 21]
      [--balance none weight sample] [--question [--check]] [self_assessment.chunk_stride=1]
      [self_assessment.standardize=true]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from muse import prompts  # noqa: E402
from muse.config import Config, load_config  # noqa: E402
from muse.datasets import sa_examples  # noqa: E402
from muse.runner import load_episodes  # noqa: E402
from muse.self_assessment import MLP, SelfAssessor, _decode, actor_context_features  # noqa: E402

QUESTION = "\nQuestion: Based on everything above, will this attempt complete the task? Answer:"


def load_actor(cfg):
    from transformers import AutoTokenizer

    from muse.llm import load_causal_lm, merged_checkpoint

    m = cfg.model
    actor = Path(cfg.run.out_dir) / "models" / "actor"
    model = load_causal_lm(Config({**m, "name": merged_checkpoint(m, actor)})).eval()
    tok = AutoTokenizer.from_pretrained(m.tokenizer or m.name, trust_remote_code=m.trust_remote_code)
    return model, tok


def _transcript(ep):
    """The World Model transcript (as actor_context_text builds it) and each step's observation end."""
    from types import SimpleNamespace

    text = prompts.actor_header(ep["task_type"], ep["init_obs"], ep["memory"], terminal=True)
    ends = []
    for s in ep["steps"]:
        text += prompts.render_steps([SimpleNamespace(**s)])
        ends.append(len(text) - 2)                       # through the observation, not its "\n>"
    return text, ends


def _question_at(model, tok, text, ids, offs, end, cache, layers, yes_no):
    """Feed QUESTION after text[:end] (whose tokens ids[:k] are on `cache`); return each layer's hidden
    state at its last token and the Yes/No gap. The caller crops `cache` back to k tokens."""
    k = sum(1 for _, e in offs if e <= end)
    q = tok(text[offs[k - 1][1]:end] + QUESTION, add_special_tokens=False)["input_ids"]
    out, hs = _decode(model.get_decoder(), torch.tensor([q], device=model.device), cache, layers + [None])
    logp = torch.log_softmax(model.get_output_embeddings()(hs[-1][-1:]).float(), -1)[0]
    return [h[-1].float().cpu() for h in hs[:-1]], float(logp[yes_no[0]] - logp[yes_no[1]]), k


def question_features(cfg, episodes, out: Path, tag: str, layers: list, check=False) -> tuple[dict, torch.Tensor]:
    """Per chunk (sa_examples' stride-1 order), features at the last token of QUESTION appended after the
    chunk's last observation: {layer: hidden states} and the Yes/No gap. Computed once per step on the
    episode's KV cache, then cropped back; a chunk takes the values at its last step. Cached."""
    path = {k: out / f"question_features_{tag}_{'final' if k is None else f'layer{k}'}.pt" for k in layers}
    gap_path = out / f"question_features_{tag}_gap.pt"
    if not check and gap_path.exists() and all(p.exists() for p in path.values()):
        return {k: torch.load(path[k]) for k in layers}, torch.load(gap_path)
    model, tok = load_actor(cfg)
    yes_no = [tok(w, add_special_tokens=False)["input_ids"] for w in (" Yes", " No")]
    assert all(len(t) == 1 for t in yes_no), yes_no
    yes_no = [t[0] for t in yes_no]
    size, room = cfg.muse.chunk_size, cfg.model.max_model_len
    feats, gaps = {k: [] for k in layers}, []
    with torch.no_grad():
        for n_ep, ep in enumerate(episodes[:1] if check else episodes):
            text, ends = _transcript(ep)
            enc = tok(text, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            assert len(ids) < room - 64, f"episode of {len(ids)} tokens exceeds the context"
            cache, fed, step_h, step_g = None, 0, [], []
            for end in ends:
                k = sum(1 for _, e in offs if e <= end)
                if k > fed:
                    o, _ = _decode(model.get_decoder(), torch.tensor([ids[fed:k]], device=model.device), cache, [])
                    cache, fed = o.past_key_values, k
                h, g, _ = _question_at(model, tok, text, ids, offs, end, cache, layers, yes_no)
                cache.crop(fed)
                step_h.append(h)
                step_g.append(g)
                if check:     # the same tokens without the cache
                    q = tok(text[offs[k - 1][1]:end] + QUESTION, add_special_tokens=False)["input_ids"]
                    _, hf = _decode(model.get_decoder(), torch.tensor([ids[:k] + q], device=model.device), None,
                                    layers, use_cache=False)
                    for a, b in zip(h, hf):
                        cos = torch.nn.functional.cosine_similarity(a, b[-1].float().cpu(), dim=0)
                        assert cos > 0.999, f"cached and full features differ: cosine {cos:.4f}"
            n = len(ends)
            for i in range(n):                               # chunk i (stride 1) ends at step min(i+size, n) - 1
                j = min(i + size, n) - 1
                for li, k in enumerate(layers):
                    feats[k].append(step_h[j][li])
                gaps.append(step_g[j])
            if n_ep % 25 == 0:
                print(f"question features: episode {n_ep}/{len(episodes)}", flush=True)
    del model
    torch.cuda.empty_cache()
    feats, gap = {k: torch.stack(v) for k, v in feats.items()}, torch.tensor(gaps).unsqueeze(1)
    if check:
        print(f"--check passed: cached and full features agree on {len(ends)} steps of one episode")
        return feats, gap
    out.mkdir(parents=True, exist_ok=True)
    for k in layers:
        torch.save(feats[k], path[k])
    torch.save(gap, gap_path)
    return feats, gap


def actor_features(cfg, episodes, out: Path, tag: str, layers: list) -> dict:
    """Features of the chunks starting at every step for each of `layers` (None: final), cached per layer;
    the missing layers come from one pass over the episodes."""
    path = {k: out / (f"actor_features_{tag}.pt" if k is None else f"actor_features_{tag}_layer{k}.pt")
            for k in layers}
    feats = {k: torch.load(path[k]) for k in layers if path[k].exists()}
    missing = [k for k in layers if k not in feats]
    if missing:
        model, tok = load_actor(cfg)
        xs = actor_context_features(model, tok, episodes, cfg.muse.chunk_size, cfg.model.max_model_len, stride=1,
                                    layer=missing)
        del model
        torch.cuda.empty_cache()
        out.mkdir(parents=True, exist_ok=True)
        for k, x in zip(missing, xs):
            torch.save(x, path[k])
            feats[k] = x
    return feats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--also", nargs="*", default=[], help="names of other runs under runs/ to pool episodes from")
    ap.add_argument("--out", help="default: <run>/sa_compare")
    ap.add_argument("--layers", nargs="+", default=["final"], help="actor layers: final and/or decoder layer numbers")
    ap.add_argument("--balance", nargs="+", default=None, help="class_balance options (default: the config's)")
    ap.add_argument("--question", action="store_true", help="add the question-suffix features (q@k, gap)")
    ap.add_argument("--check", action="store_true", help="with --question: check cached against full features "
                    "on one episode, then stop")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.check:
        runs = [Path(cfg.run.out_dir)]
        eps = load_episodes(runs[0] / "collect" / "episodes.jsonl")
        layers = [None if k == "final" else int(k) for k in args.layers]
        question_features(cfg, eps, runs[0] / "sa_compare", "check", sorted({k for k in layers if k}) + [None],
                          check=True)
        return
    from sklearn.metrics import roc_auc_score

    sc, size = cfg.self_assessment, cfg.muse.chunk_size
    runs = [Path(cfg.run.out_dir)] + [Path(cfg.run.out_dir).parent / r for r in args.also]
    out = Path(args.out or runs[0] / "sa_compare")
    episodes = [ep for r in runs for ep in load_episodes(r / "collect" / "episodes.jsonl")]
    d = sa_examples(episodes, size, stride=1)
    y, tasks, starts = torch.tensor(d["labels"]), np.array(d["task_ids"]), np.array(d["starts"])
    cfg.self_assessment.input = "paper"            # its g_eta is refit for each input below
    sa = SelfAssessor(cfg)
    layers = [None if k == "final" else int(k) for k in args.layers]
    actor = actor_features(cfg, episodes, out, "+".join(r.name for r in runs), layers)
    feats = {"paper": sa.features(d["descs"], d["plans"], d["texts"]),
             **{"actor" if k is None else f"actor@{k}": actor[k] for k in layers}}
    gap = None
    if args.question:
        q_layers = sorted({k for k in layers if k is not None}) + [None]
        q, gap = question_features(cfg, episodes, out, "+".join(r.name for r in runs), q_layers)
        feats.update({("q@final" if k is None else f"q@{k}"): q[k] for k in q_layers})
        feats["gap"] = gap
        k = next((k for k in layers if k is not None), None)
        if k is not None:
            feats[f"actor@{k}+q@{k}+gap"] = torch.cat([actor[k], q[k], gap], 1)
    balances = args.balance or [sc.class_balance]
    assert all(len(x) == len(y) for x in feats.values())
    train_rows = starts % (sc.chunk_stride or size) == 0
    test_rows = starts % size == 0
    print(f"runs {[r.name for r in runs]}: {len(episodes)} episodes; chunk_stride {sc.chunk_stride or size}: "
          f"{int(train_rows.sum())} training chunks in all, {int(test_rows.sum())} test chunks")

    folds = [t for t in sorted(set(tasks)) if len(set(y[tasks == t].tolist())) == 2]
    names = ([f"{n}/{b}" if len(balances) > 1 else n for n in feats for b in balances] + ["index"]
             + (["gap0"] if gap is not None else []))
    rows, pooled = [], {k: ([], []) for k in names}
    for t in folds:
        tr, te = train_rows & (tasks != t), test_rows & (tasks == t)
        row = {"task": t.split("/")[0], "chunks": int(te.sum()), "success_chunks": int(y[te].sum())}
        for feat, x in feats.items():
            for b in balances:
                name = f"{feat}/{b}" if len(balances) > 1 else feat
                sc.class_balance = b
                torch.manual_seed(cfg.run.seed)
                sa.mlp = MLP(x.shape[1], sc.hidden, sc.dropout).to(sa.device)
                sa.mean = sa.std = None
                if sc.standardize:
                    sa.set_standardizer(x[tr])
                sa.fit(x[tr], y[tr], sc.epochs, sc.lr)
                row[name] = sa.evaluate(x[te], y[te])
                pooled[name][0].extend(y[te].tolist())
                pooled[name][1].extend(sa.predict_features(x[te]).tolist())
        row["index"] = {"auroc": float(roc_auc_score(y[te], -starts[te]))}
        pooled["index"][0].extend(y[te].tolist()), pooled["index"][1].extend((-starts[te]).tolist())
        if gap is not None:
            g = gap[te, 0].numpy()
            row["gap0"] = {"auroc": float(roc_auc_score(y[te], g))}
            pooled["gap0"][0].extend(y[te].tolist()), pooled["gap0"][1].extend(g.tolist())
        rows.append(row)

    summary = {n: {"mean_auroc": float(np.mean([r[n]["auroc"] for r in rows])),
                   "pooled_auroc": float(roc_auc_score(*pooled[n]))} for n in names}
    out.mkdir(parents=True, exist_ok=True)
    tag = (f"{'+'.join(r.name for r in runs)}_stride{sc.chunk_stride or size}"
           f"{'_std' if sc.standardize else ''}_{'-'.join(args.layers)}_{'-'.join(balances)}{'_q' if args.question else ''}")
    (out / f"results_{tag}.json").write_text(json.dumps({"folds": rows, "summary": summary}, indent=2))
    w = max(7, *(len(n) for n in names))
    print(f"{'left-out task':45s} {'chunks':>6s} {'succ':>5s} " + " ".join(f"{n:>{w}s}" for n in names))
    for r in rows:
        print(f"{r['task'][:45]:45s} {r['chunks']:6d} {r['success_chunks']:5d} "
              + " ".join(f"{r[n]['auroc']:{w}.3f}" for n in names))
    for k in ("mean_auroc", "pooled_auroc"):
        print(f"{k:58s}" + " ".join(f"{summary[n][k]:{w}.3f}" for n in names))
    print(f"(within-task AUROC; results in {out / f'results_{tag}.json'})")


if __name__ == "__main__":
    main()
