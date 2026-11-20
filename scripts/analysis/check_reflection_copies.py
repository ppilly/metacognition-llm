"""Offline check of how often the Reflector copies an earlier reflection, by sampling temperature.

Replays reflection prompts from collected (pre-deployment) episodes whose memory already holds at least
one reflection, decodes new reflections at each temperature, and counts outputs that repeat a reflection
in that memory exactly, or nearly (difflib ratio > 0.9). Both the untrained model (as in collection) and
a trained reflector adapter (as in deployment). Writes <out>/reflection_copies.json.

  python scripts/analysis/check_reflection_copies.py -c configs/local.yaml \\
      --episodes runs/local/collect/episodes.jsonl --per-task 3 --temps 0 0.5 0.7 1.0 --samples 3
"""
import argparse
import difflib
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from muse.config import Config, load_config  # noqa: E402


def prompts_with_memory(path, per_task, seed):
    rng = random.Random(seed)
    out = []
    for line in open(path):
        rec = json.loads(line)
        cands = [e for e in rec["episodes"] if e.get("reflection") and e["memory"]]
        for e in rng.sample(cands, min(per_task, len(cands))):
            out.append({"task": rec["task_id"].split("/")[0], "episode": e["episode"], "query": e["reflection"]["query"],
                        "memory": [m["text"].strip() for m in e["memory"]], "logged": e["reflection"]["text"].strip()})
    return out


def copy_kind(text, memory):
    if text in memory:
        return "exact"
    if any(difflib.SequenceMatcher(None, text, m).ratio() > 0.9 for m in memory):
        return "near"
    return "new"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--per-task", type=int, default=3)
    ap.add_argument("--temps", nargs="+", type=float, default=[0.0, 0.5, 0.7, 1.0])
    ap.add_argument("--samples", type=int, default=3, help="samples per prompt at temperature > 0")
    ap.add_argument("--models", nargs="+", default=["base", "reflector"], help="base and/or adapter dirs under models/")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    from transformers import AutoTokenizer

    from muse.llm import load_causal_lm, merged_checkpoint

    m, a = cfg.model, cfg.agent
    items = prompts_with_memory(args.episodes, args.per_task, cfg.run.seed)
    logged = [copy_kind(it["logged"], it["memory"]) for it in items]
    print(f"{len(items)} prompts; as logged in collection: "
          + ", ".join(f"{k} {logged.count(k)}" for k in ("exact", "near", "new")), flush=True)
    tok = AutoTokenizer.from_pretrained(m.tokenizer or m.name, trust_remote_code=m.trust_remote_code)
    results = {}
    for name in args.models:
        cm = m if name == "base" else Config({**m, "name": merged_checkpoint(m, Path(cfg.run.out_dir) / "models" / name)})
        model = load_causal_lm(cm).eval()
        torch.manual_seed(cfg.run.seed)
        rows = []
        for t in args.temps:
            n = 1 if t == 0 else args.samples
            for it in items:
                ids = tok(it["query"], return_tensors="pt").input_ids[:, -(m.max_model_len - a.reflection_max_tokens):]
                ids = ids.to(model.device)
                kw = dict(do_sample=True, temperature=t) if t > 0 else dict(do_sample=False)
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=a.reflection_max_tokens, num_return_sequences=n,
                                         pad_token_id=tok.eos_token_id, **kw)
                for o in out:
                    text = tok.decode(o[ids.shape[1]:], skip_special_tokens=True).split("\n\n")[0].strip()
                    rows.append({"temp": t, "task": it["task"], "episode": it["episode"], "text": text,
                                 "kind": copy_kind(text, it["memory"])})
            rs = [r for r in rows if r["temp"] == t]
            print(f"{name:10s} T={t:.1f}  n={len(rs):3d}  exact {sum(r['kind'] == 'exact' for r in rs) / len(rs):5.0%}"
                  f"  near {sum(r['kind'] == 'near' for r in rs) / len(rs):5.0%}"
                  f"  new {sum(r['kind'] == 'new' for r in rs) / len(rs):5.0%}"
                  f"  empty {sum(not r['text'] for r in rs) / len(rs):4.0%}"
                  f"  mean chars {sum(len(r['text']) for r in rs) / len(rs):5.0f}", flush=True)
        results[name] = rows
        del model
        torch.cuda.empty_cache()
    out = Path(args.out or Path(cfg.run.out_dir) / "reflection_copies.json")
    out.write_text(json.dumps({"prompts": items, "results": results}, indent=1))
    print(f"(rows in {out})")


if __name__ == "__main__":
    main()
