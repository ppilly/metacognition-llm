"""Offline probe of World Model adapters on real steps: given the true history and the action taken,
does the adapter predict the observation the environment returned? Steps are split into actions the
environment rejected ("Nothing happens.") and accepted ones. Greedy decoding of one observation line.

To choose between adapters, probe the pre-deployment tasks they held out (train-wm logs them), never
deployment tasks: those stand for tasks unseen until deployment.

  python scripts/analysis/probe_world_model.py -c configs/local.yaml --episodes runs/local/collect/episodes.jsonl \\
      --tasks Plate-None-Cabinet-7 --adapters actor world_model
"""
import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from muse import prompts  # noqa: E402
from muse.config import Config, load_config  # noqa: E402

REJECTED = "Nothing happens."


def steps_to_probe(path, tasks, per_kind, seed):
    """(episode, step index) pairs, per_kind rejected and per_kind accepted actions per task."""
    rng = random.Random(seed)
    out = []
    for line in open(path):
        rec = json.loads(line)
        if not any(t in rec["task_id"] for t in tasks):
            continue
        rej, acc = [], []
        for ep in rec["episodes"]:
            for i, s in enumerate(ep["steps"]):
                if s["action"].startswith("think"):
                    continue
                (rej if s["observation"].startswith(REJECTED) else acc).append((ep, i))
        out += [(rec["task_id"], "rejected", *x) for x in rng.sample(rej, min(per_kind, len(rej)))]
        out += [(rec["task_id"], "accepted", *x) for x in rng.sample(acc, min(per_kind, len(acc)))]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--episodes", required=True, help="an episodes.jsonl (collected or eval)")
    ap.add_argument("--tasks", nargs="+", required=True, help="substrings of task ids")
    ap.add_argument("--adapters", nargs="+", required=True, help="adapter dirs under the config's models/")
    ap.add_argument("--per-kind", type=int, default=40)
    ap.add_argument("--unmerged", action="store_true",
                    help="serve each adapter on the quantized base instead of merging it in (as training evaluates)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    from transformers import AutoTokenizer

    from muse.llm import load_causal_lm, merged_checkpoint

    probes = steps_to_probe(args.episodes, args.tasks, args.per_kind, cfg.run.seed)
    m = cfg.model
    tok = AutoTokenizer.from_pretrained(m.tokenizer or m.name, trust_remote_code=m.trust_remote_code)
    results = {}
    for name in args.adapters:
        adapter = Path(cfg.run.out_dir) / "models" / name
        if args.unmerged:
            from peft import PeftModel
            model = PeftModel.from_pretrained(load_causal_lm(m), str(adapter)).eval()
        else:
            model = load_causal_lm(Config({**m, "name": merged_checkpoint(m, adapter)})).eval()
        rows = []
        for task, kind, ep, i in probes:
            steps = [SimpleNamespace(action=s["action"], observation=s["observation"]) for s in ep["steps"]]
            prompt = prompts.world_model_prompt(ep["task_type"], ep["init_obs"], ep["memory"], steps[:i])
            prompt += f" {steps[i].action}\n"
            ids = tok(prompt, return_tensors="pt").input_ids[:, -m.max_model_len + 64:].to(model.device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
            pred = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).split("\n")[0].strip()
            real = steps[i].observation.strip()
            rows.append({"task": task.split("/")[0], "kind": kind, "action": steps[i].action, "real": real,
                         "pred": pred, "exact": pred == real,
                         "pred_rejected": pred.startswith(REJECTED), "real_rejected": real.startswith(REJECTED)})
        results[name] = rows
        del model
        torch.cuda.empty_cache()
        print(f"\n== {name}")
        for task in sorted({r["task"] for r in rows}):
            for kind in ("rejected", "accepted"):
                rs = [r for r in rows if r["task"] == task and r["kind"] == kind]
                if rs:
                    print(f"{task[:45]:45s} {kind:8s} n={len(rs):3d}  exact {sum(r['exact'] for r in rs) / len(rs):5.0%}"
                          f"  predicts 'Nothing happens' {sum(r['pred_rejected'] for r in rs) / len(rs):5.0%}")
    out = Path(cfg.run.out_dir) / "wm_probe.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\n(rows in {out})")


if __name__ == "__main__":
    main()
