"""Offline check of the memory layout options (agent.max_failure_memory, agent.successes_last) on the
reflections they lead to. Replays failed episodes from a collection: each episode's reflection prompt is
rebuilt from its transcript and the task's earlier reflections, once with the run's own layout and once
per variant, and the reflector writes new reflections from each. No game is played.

Scored per reflection:
  overlap     highest difflib similarity to an earlier reflection of the task (template copying)
  grounded    names a location (e.g. "drawer 10") or object (e.g. "creditcard 4") seen in that episode
  invented    names a location that the room does not have
  take_fix    in episodes where the task object was seen but never taken: names "take" and the object
and the prompt length in tokens.

  python scripts/analysis/check_memory_layout.py -c configs/local.yaml \\
      --episodes runs/local/collect/episodes.jsonl --per-task 3 --samples 3 \\
      --variant max_failure_memory=3 successes_last=true
"""
import argparse
import difflib
import json
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from muse import prompts  # noqa: E402
from muse.agents import ReflexionAgent  # noqa: E402
from muse.config import Config, load_config  # noqa: E402
from muse.env import Task  # noqa: E402

THING = re.compile(r"\b([a-z]+) (\d+)\b")


class Tokens:
    """What ReflexionAgent.memory_window needs from an LLM."""

    def __init__(self, tok):
        self.tok = tok

    def num_tokens(self, text):
        return len(self.tok(text).input_ids)


def task_object(task_id):
    """'pick_two_obj_and_place-CreditCard-None-Safe-323/...' -> 'creditcard'."""
    return task_id.split("-")[1].lower()


def pick(path, per_task, min_memory, seed):
    """Failed episodes whose task already has at least min_memory reflections, per_task of them per task."""
    rng = random.Random(seed)
    out = []
    for line in open(path):
        rec = json.loads(line)
        eps = rec["episodes"]
        cands = [e for e in eps if not e["success"] and e.get("reflection")
                 and sum(1 for p in eps[:e["episode"]] if p.get("reflection")) >= min_memory]
        for e in rng.sample(cands, min(per_task, len(cands))):
            earlier = [{"episode": p["episode"], "success": p["success"], "text": p["reflection"]["text"]}
                       for p in eps[:e["episode"]] if p.get("reflection")]
            out.append({"task_id": rec["task_id"], "task_type": e["task_type"], "ep": e, "memory": earlier})
    return out


def score(text, item):
    ep, obj = item["ep"], task_object(item["task_id"])
    seen = " ".join(s["observation"] + " " + s["action"] for s in ep["steps"]).lower()
    room = set(m.group(0) for m in THING.finditer(ep["init_obs"].split("Your task")[0].lower()))
    named = set(m.group(0) for m in THING.finditer(text.lower()))
    places = {n for n in named if not n.startswith(obj)}
    took = any(s["action"].startswith(f"take {obj}") and not s["observation"].startswith("Nothing")
               for s in ep["steps"])
    saw = obj in " ".join(s["observation"] for s in ep["steps"]).lower()
    earlier = [m["text"].strip() for m in item["memory"]]
    return {"overlap": max((difflib.SequenceMatcher(None, text, e).ratio() for e in earlier), default=0.0),
            "grounded": any(n in seen for n in named),
            "invented": bool(places - room),
            "take_fix": ("take" in text.lower() and obj in text.lower()) if (saw and not took) else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--episodes", required=True, help="a collection's episodes.jsonl")
    ap.add_argument("--per-task", type=int, default=3)
    ap.add_argument("--min-memory", type=int, default=6, help="earlier reflections a replayed episode needs")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--variant", nargs="+", default=["max_failure_memory=3", "successes_last=true"],
                    help="agent.* overrides that define the variant layout")
    ap.add_argument("--adapter", default=None, help="reflector adapter under models/ (default: untrained model)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = load_config(args.config)
    variant = load_config(args.config, [f"agent.{v}" for v in args.variant])
    from transformers import AutoTokenizer

    from muse.llm import load_causal_lm, merged_checkpoint

    m, a = base.model, base.agent
    tok = AutoTokenizer.from_pretrained(m.tokenizer or m.name, trust_remote_code=m.trust_remote_code)
    items = pick(args.episodes, args.per_task, args.min_memory, base.run.seed)
    layouts = {"current": ReflexionAgent(base, Tokens(tok)), "variant": ReflexionAgent(variant, Tokens(tok))}
    for it in items:      # the current layout must reproduce the logged prompt
        st = SimpleNamespace(memory=it["memory"])
        steps = [SimpleNamespace(**{k: s[k] for k in ("action", "observation")}) for s in it["ep"]["steps"]]
        it["queries"] = {}
        for name, agent in layouts.items():
            st2 = agent.new_task_state(Task(it["task_id"], "", it["task_type"]))
            st2.memory = st.memory
            it["queries"][name] = prompts.reflection_query(it["ep"]["init_obs"], steps, False,
                                                           agent.memory_window(st2), success_prompt=a.success_reflection)
    same = sum(it["queries"]["current"] == it["ep"]["reflection"]["query"] for it in items)
    print(f"{len(items)} failed episodes; current layout reproduces the logged prompt for {same}", flush=True)

    cm = m if args.adapter is None else Config({**m, "name": merged_checkpoint(m, Path(base.run.out_dir) / "models" / args.adapter)})
    model = load_causal_lm(cm).eval()
    torch.manual_seed(base.run.seed)
    rows = []
    for it in items:
        for name, q in it["queries"].items():
            ids = tok(q, return_tensors="pt").input_ids.to(model.device)
            kw = dict(do_sample=True, temperature=a.reflection_temperature) if a.reflection_temperature > 0 else {}
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=a.reflection_max_tokens, num_return_sequences=args.samples,
                                     pad_token_id=tok.eos_token_id, **kw)
            for o in out:
                text = tok.decode(o[ids.shape[1]:], skip_special_tokens=True).split("\n\n")[0].strip()
                rows.append({"layout": name, "task": it["task_id"].split("/")[0], "episode": it["ep"]["episode"],
                             "prompt_tokens": ids.shape[1], "text": text, **score(text, it)})
        print(f"done {it['task_id'].split('/')[0][:40]} episode {it['ep']['episode']}", flush=True)

    print(f"\n{'layout':8s} {'n':>4s} {'tokens':>7s} {'overlap':>8s} {'>0.9':>6s} {'grounded':>9s} {'invented':>9s} {'take_fix':>9s}")
    for name in layouts:
        rs = [r for r in rows if r["layout"] == name]
        tf = [r["take_fix"] for r in rs if r["take_fix"] is not None]
        print(f"{name:8s} {len(rs):4d} {sum(r['prompt_tokens'] for r in rs) / len(rs):7.0f}"
              f" {sum(r['overlap'] for r in rs) / len(rs):8.2f} {sum(r['overlap'] > 0.9 for r in rs) / len(rs):6.0%}"
              f" {sum(r['grounded'] for r in rs) / len(rs):9.0%} {sum(r['invented'] for r in rs) / len(rs):9.0%}"
              f" {(sum(tf) / len(tf)) if tf else float('nan'):9.0%}  (take_fix over {len(tf)})")
    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"(rows in {args.out})")


if __name__ == "__main__":
    main()
