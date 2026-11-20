"""Offline check of success-reflection prompts (agent.success_reflection) on the reflections they lead to.
Replays successful episodes from a collection: each episode's success-reflection query is rebuilt from its
transcript and the task's earlier reflections (the run's memory layout), once per prompt, and the
reflector writes new reflections from each. No game is played. Pre-deployment data only.

Scored per reflection:
  right_place  names a receptacle the task's object was actually taken from in that episode
  rejected     repeats a rejected action ("take X from Y" phrase or the whole action) as if it worked
  invented     names a numbered object or place ("countertop 2") absent from the episode
  chars        length

  python scripts/analysis/check_success_reflection.py -c configs/local.yaml \\
      --episodes runs/local/collect/episodes.jsonl --prompts plan plan_grounded plan_strategy \\
      --models base reflector --samples 3 --out runs/local/success_reflection_check.json
"""
import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_memory_layout import THING, Tokens, task_object  # noqa: E402

from muse import prompts  # noqa: E402
from muse.agents import ReflexionAgent  # noqa: E402
from muse.config import Config, load_config  # noqa: E402
from muse.env import Task  # noqa: E402

REJECTED = "Nothing happens"


def successes(path):
    """Successful episodes that wrote a reflection, with the task's earlier reflections as memory."""
    out = []
    for line in open(path):
        rec = json.loads(line)
        eps = rec["episodes"]
        for e in eps:
            if e["success"] and e.get("reflection"):
                earlier = [{"episode": p["episode"], "success": p["success"], "text": p["reflection"]["text"]}
                           for p in eps[:e["episode"]] if p.get("reflection")]
                out.append({"task_id": rec["task_id"], "task_type": e["task_type"], "ep": e, "memory": earlier})
    return out


def score(text, item):
    ep, obj, t = item["ep"], task_object(item["task_id"]), text.lower()
    acts = [s for s in ep["steps"] if not s["action"].startswith("think")]
    taken = {m.group(1) for s in acts if not s["observation"].startswith(REJECTED)
             for m in [re.match(rf"take {obj} \d+ from (.+)", s["action"])] if m}
    rejected = [s["action"].lower() for s in acts if s["observation"].startswith(REJECTED)]
    phrases = [m.group(1) for a in rejected for m in [re.match(r"take (.+ from .+)", a)] if m]
    seen = (ep["init_obs"] + " " + " ".join(s["observation"] + " " + s["action"] for s in ep["steps"])).lower()
    named = {m.group(0) for m in THING.finditer(t)}
    return {"right_place": any(r in t for r in taken) if taken else None,
            "rejected": any(a in t for a in rejected) or any(p in t for p in phrases),
            "invented": bool({n for n in named if n not in seen}),
            "chars": len(text)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--episodes", required=True, help="a collection's episodes.jsonl")
    ap.add_argument("--prompts", nargs="+", default=["plan", "plan_grounded", "plan_strategy"])
    ap.add_argument("--models", nargs="+", default=["base", "reflector"], help="base and/or adapter dirs under models/")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    from transformers import AutoTokenizer

    from muse.llm import load_causal_lm, merged_checkpoint

    m, a = cfg.model, cfg.agent
    tok = AutoTokenizer.from_pretrained(m.tokenizer or m.name, trust_remote_code=m.trust_remote_code)
    agent = ReflexionAgent(cfg, Tokens(tok))
    items = successes(args.episodes)
    same = 0
    for it in items:
        steps = [SimpleNamespace(action=s["action"], observation=s["observation"]) for s in it["ep"]["steps"]]
        state = agent.new_task_state(Task(it["task_id"], "", it["task_type"]))
        state.memory = it["memory"]
        window = agent.memory_window(state)
        it["queries"] = {p: prompts.reflection_query(it["ep"]["init_obs"], steps, True, window, success_prompt=p)
                         for p in args.prompts}
        same += it["queries"].get(a.success_reflection) == it["ep"]["reflection"]["query"]
    print(f"{len(items)} successful episodes; the run's own prompt ({a.success_reflection}) reproduces the logged "
          f"query for {same}", flush=True)

    rows = []
    for name in args.models:
        cm = m if name == "base" else Config({**m, "name": merged_checkpoint(m, Path(cfg.run.out_dir) / "models" / name)})
        model = load_causal_lm(cm).eval()
        torch.manual_seed(cfg.run.seed)
        for it in items:
            for p, q in it["queries"].items():
                ids = tok(q, return_tensors="pt").input_ids.to(model.device)
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=a.reflection_max_tokens, num_return_sequences=args.samples,
                                         do_sample=True, temperature=a.reflection_temperature,
                                         pad_token_id=tok.eos_token_id)
                for o in out:
                    text = tok.decode(o[ids.shape[1]:], skip_special_tokens=True).split("\n\n")[0].strip()
                    rows.append({"model": name, "prompt": p, "task": it["task_id"].split("/")[0],
                                 "episode": it["ep"]["episode"], "text": text, **score(text, it)})
            print(f"{name}: done {it['task_id'].split('/')[0][:40]} episode {it['ep']['episode']}", flush=True)
        del model
        torch.cuda.empty_cache()

    print(f"\n{'model':10s} {'prompt':15s} {'n':>4s} {'right_place':>12s} {'rejected':>9s} {'invented':>9s} {'chars':>6s}")
    for name in args.models:
        for p in args.prompts:
            rs = [r for r in rows if r["model"] == name and r["prompt"] == p]
            rp = [r["right_place"] for r in rs if r["right_place"] is not None]
            print(f"{name:10s} {p:15s} {len(rs):4d} {sum(rp) / len(rp):12.0%} {sum(r['rejected'] for r in rs) / len(rs):9.0%}"
                  f" {sum(r['invented'] for r in rs) / len(rs):9.0%} {sum(r['chars'] for r in rs) / len(rs):6.0f}")
    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"(rows in {args.out})")


if __name__ == "__main__":
    main()
