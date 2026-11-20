"""Command line entry point.

  python -m muse.cli collect   -c configs/default.yaml      # Alg. 3: Reflexion on the 140 seen tasks
  python -m muse.cli train-sft -c ...                       # Actor M_a  (LoRA SFT)
  python -m muse.cli train-dpo -c ...                       # Reflector M_rx (LoRA DPO)
  python -m muse.cli train-sa  -c ...                       # Self-Assessment Model M_sa
  python -m muse.cli eval      -c ... --agent muse          # Alg. 4 on the 134 unseen tasks
  python -m muse.cli report    --runs runs/mistral7b runs/openorca7b runs/openelm3b
  python -m muse.cli status    --watch 15                   # progress dashboard -> runs/dashboard.html

Any config value can be overridden with trailing key=value pairs, e.g. `env.num_tasks=10`.
"""
import argparse
import json
import logging
from pathlib import Path

import yaml

from .config import load_config

log = logging.getLogger("muse")

AGENTS = ["react", "reflexion", "reflexion_sl", "muse", "muse_lowest", "muse_random"]


def _paths(cfg):
    root = Path(cfg.run.out_dir)
    return {
        "root": root,
        "collect": root / "collect" / "episodes.jsonl",
        "actor": root / "models" / "actor",
        "reflector": root / "models" / "reflector",
        "world_model": root / "models" / cfg.world_model.name,
        "sa": root / "models" / ("sa_actor" if cfg.self_assessment.input == "actor" else "sa"),
        "eval": root / "eval",
    }


def eval_name(cfg, agent: str) -> str:
    """The eval directory under eval/: the agent, or '<agent>+<run.eval_variant>' for a variant of it."""
    return f"{agent}+{cfg.run.eval_variant}" if cfg.run.eval_variant else agent


def _save_config(cfg, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(json.loads(json.dumps(cfg)), f, sort_keys=False)


def _tasks(cfg, split):
    from .env import list_tasks
    return list_tasks(cfg.env.data_dir, split, cfg.env.num_tasks, cfg.env.task_ids, cfg.env.keep_task_order)


def cmd_collect(cfg, args, status):
    from .agents import ReflexionAgent
    from .llm import build_llm
    from .runner import run_tasks

    p = _paths(cfg)
    tasks = _tasks(cfg, cfg.env.train_split)
    llm = build_llm(cfg)
    agent = ReflexionAgent(cfg, llm)
    _save_config(cfg, p["collect"].parent / "config.yaml")
    run_tasks(cfg, agent, tasks, p["collect"], n_adapt=cfg.collect.num_episodes, n_test=0, status=status)


def _split(items, val_fraction, seed):
    """Hold out whole training tasks to check that a model learns something that generalizes."""
    from .datasets import split_by_task
    train, val = split_by_task([it["task_id"] for it in items], val_fraction, seed)
    return [items[i] for i in train], [items[i] for i in val]


def cmd_train_sft(cfg, args, status):
    from .datasets import sft_examples
    from .runner import load_episodes
    from .train_lora import train_sft

    p = _paths(cfg)
    examples = sft_examples(load_episodes(p["collect"]), cfg.sft.include_fewshot)
    if not examples:
        raise SystemExit("no successful training episodes to fine-tune the actor on")
    train, val = _split(examples, cfg.sft.val_fraction, cfg.run.seed)
    train_sft(cfg, train, p["actor"], status, val)


def cmd_train_dpo(cfg, args, status):
    from .datasets import dpo_pairs
    from .runner import load_episodes
    from .train_lora import train_dpo

    p = _paths(cfg)
    d = cfg.dpo
    pairs = dpo_pairs(load_episodes(p["collect"]), d.fallback, d.max_pairs_per_task, cfg.run.seed)
    if not pairs:
        raise SystemExit("no reflection preference pairs could be built from the collected data: no task has "
                         "both a positive (fail->success) and a negative (success->fail) reflection"
                         + ("" if d.fallback else "; dpo.fallback=true also uses success->success / fail->fail"))
    train, val = _split(pairs, d.val_fraction, cfg.run.seed)
    train_dpo(cfg, train, p["reflector"], status, val)


def cmd_train_wm(cfg, args, status):
    """Extension: the World Model adapter for MUSE's rollouts (train_lora.train_world_model)."""
    import random

    from .datasets import split_by_task, wm_examples
    from .runner import load_episodes
    from .train_lora import train_world_model

    p = _paths(cfg)
    wc = cfg.world_model
    examples = wm_examples(load_episodes(p["collect"]), wc.max_failed, cfg.run.seed, wc.max_label_steps)
    # held-out tasks with both outcomes: the action-loss guard needs successful held-out episodes
    tr, va = split_by_task([e["task_id"] for e in examples], wc.val_fraction, cfg.run.seed,
                           labels=[e["success"] for e in examples])
    train, val = [examples[i] for i in tr], [examples[i] for i in va]
    if not any(e["success"] for e in val):
        raise SystemExit("world_model: no held-out task has a successful episode to check the action loss on")
    if wc.val_max and len(val) > wc.val_max:    # held-out checks run several times per epoch
        ok, bad = [e for e in val if e["success"]], [e for e in val if not e["success"]]
        rng = random.Random(cfg.run.seed)
        n_ok = min(len(ok), wc.val_max // 2)
        val = rng.sample(ok, n_ok) + rng.sample(bad, min(len(bad), wc.val_max - n_ok))
    train_world_model(cfg, train, p["world_model"], p["actor"], status, val)


def _sa_feature_adapter(cfg):
    """The adapter whose hidden states M_sa reads (input: actor): the one that writes MUSE's rollouts."""
    p = _paths(cfg)
    return p["world_model"] if cfg.world_model.enabled else p["actor"]


def _actor_context_features(cfg, actor_dir, episodes):
    """Chunk features from the trained actor as MUSE serves it (merged, quantized like the base model)."""
    import torch
    from transformers import AutoTokenizer

    from .config import Config
    from .llm import load_causal_lm, merged_checkpoint
    from .self_assessment import actor_context_features

    m = cfg.model
    log.info("M_sa input: the actor's hidden states (layer %s) given its full context (extension)",
             cfg.self_assessment.layer or "final")
    model = load_causal_lm(Config({**m, "name": merged_checkpoint(m, actor_dir)})).eval()
    tok = AutoTokenizer.from_pretrained(m.tokenizer or m.name, trust_remote_code=m.trust_remote_code)
    x = actor_context_features(model, tok, episodes, cfg.muse.chunk_size, m.max_model_len,
                               cfg.self_assessment.chunk_stride, cfg.self_assessment.layer)
    del model
    torch.cuda.empty_cache()
    return x


def cmd_train_sa(cfg, args, status):
    import torch

    from .datasets import split_by_task, sa_examples
    from .runner import load_episodes
    from .self_assessment import ReplayBuffer, SelfAssessor
    from .train_lora import save_curves

    p = _paths(cfg)
    sc = cfg.self_assessment
    episodes = load_episodes(p["collect"])
    data = sa_examples(episodes, cfg.muse.chunk_size, sc.chunk_stride)
    log.info("M_sa training chunks: %d (%.0f%% from successful episodes)",
             len(data["labels"]), 100 * sum(data["labels"]) / max(1, len(data["labels"])))
    if sc.input == "actor":
        x = _actor_context_features(cfg, _sa_feature_adapter(cfg), episodes)
        sa = SelfAssessor(cfg, in_dim=x.shape[1])
    else:
        sa = SelfAssessor(cfg)
        n_trunc = sum(sa.truncated(data["plans"], data["texts"]))
        (log.warning if n_trunc else log.info)(
            "%d of %d M_sa inputs exceed the encoder's %d-token limit and lose the end of their chunk",
            n_trunc, len(data["texts"]), sa.encoder.model.max_seq_length)
        status.update(truncated_inputs=n_trunc)
        x = sa.features(data["descs"], data["plans"], data["texts"])
    y = torch.tensor(data["labels"])
    if sc.val_tasks not in ("random", "mixed"):
        raise ValueError(f"self_assessment.val_tasks {sc.val_tasks!r}: random | mixed")
    tr, va = split_by_task(data["task_ids"], sc.val_fraction, cfg.run.seed,
                           labels=data["labels"] if sc.val_tasks == "mixed" else None)
    x_tr, y_tr, x_va, y_va = x[tr], y[tr], x[va], y[va]
    t_va = [data["task_ids"][i] for i in va]
    log.info("held-out tasks: %s", sorted({t.split("/")[0] for t in t_va}))
    log.info("%d training chunks (%d successful), %d held-out chunks (%d successful) from unseen training tasks; "
             "class balance: %s", len(tr), int(y_tr.sum()), len(va), int(y_va.sum()), sc.class_balance)
    if sc.standardize:
        sa.set_standardizer(x_tr)
    history = {"train": [], "val": [], "x_label": "epoch"}

    def log_fn(e, bce):
        row = {"epoch": e + 1, "step": e + 1, "bce": bce}
        history["train"].append(row)
        if len(va):
            val = sa.evaluate(x_va, y_va)
            history["val"].append({"epoch": e + 1, "step": e + 1, "bce": val["bce"], "acc": val["acc"],
                                   "auroc": val["auroc"]})
            status.update(val_bce=val["bce"], val_auroc=val["auroc"])
        status.update(e + 1, sc.epochs, "epochs", bce=bce)
        if e % 5 == 0 or e + 1 == sc.epochs:
            log.info("epoch %d bce=%.4f%s", e + 1, bce, f" held-out bce={val['bce']:.4f} auroc={val['auroc']:.3f}"
                     if len(va) else "")

    stopped = sa.fit(x_tr, y_tr, sc.epochs, sc.lr, log_fn=log_fn, val=(x_va, y_va, t_va) if len(va) else None)
    if stopped["best_epoch"]:
        log.info("early stopping: kept epoch %d, held-out within-task AUROC %.3f",
                 stopped["best_epoch"], stopped["best_val_auroc"])
        status.update(best_epoch=stopped["best_epoch"])
    sa.replay = ReplayBuffer(x_tr, y_tr)
    if len(va):
        sa.val_x, sa.val_y = x_va, y_va
    for name, (xs, ys) in (("train", (x_tr, y_tr)), ("held-out", (x_va, y_va))):
        if len(xs):
            m = sa.evaluate(xs, ys)
            log.info("%s: accuracy %.3f  AUROC %.3f  BCE %.4f", name, m["acc"], m["auroc"], m["bce"])
            status.update(**{f"{name.replace('-', '')}_auroc": m["auroc"]})
    sa.save(p["sa"])
    save_curves(p["sa"], history, "Self-assessment M_sa")


def build_agent(cfg, name):
    from .agents import MuseAgent, ReActAgent, ReflexionAgent
    from .llm import build_llm

    p = _paths(cfg)
    if name == "react":
        return ReActAgent(cfg, build_llm(cfg))
    if name == "reflexion":
        return ReflexionAgent(cfg, build_llm(cfg))
    wm = name.startswith("muse") and cfg.world_model.enabled
    adapters = {("world_model" if wm else "actor"): str(p["world_model"] if wm else p["actor"]),
                "reflector": str(p["reflector"])}
    for k, v in adapters.items():
        if not Path(v).exists():
            raise SystemExit(f"missing {k} adapter at {v}; run the pre-deployment training first")
    llm = build_llm(cfg, adapters)
    if name == "reflexion_sl":
        return ReflexionAgent(cfg, llm, actor_adapter="actor", reflector_adapter="reflector")
    from .self_assessment import SelfAssessor
    cfg.muse.selection = {"muse": "best", "muse_lowest": "lowest", "muse_random": "random"}[name]
    # with world_model.enabled the World Model adapter writes the rollouts and also serves the rare
    # actor fallback, so MUSE still holds two models on the GPU
    agent = MuseAgent(cfg, llm, SelfAssessor.load(cfg, p["sa"]), actor_adapter="world_model" if wm else "actor",
                      reflector_adapter="reflector")
    agent.name = name
    return agent


def cmd_eval(cfg, args, status):
    from .metrics import summarize
    from .runner import load_episodes, run_tasks

    pr = cfg.protocol
    out_dir = _paths(cfg)["eval"] / eval_name(cfg, args.agent)
    agent = build_agent(cfg, args.agent)
    _save_config(cfg, out_dir / "config.yaml")
    run_tasks(cfg, agent, _tasks(cfg, cfg.env.eval_split), out_dir / "episodes.jsonl",
              n_adapt=pr.adapt_episodes, n_test=pr.test_episodes, status=status)
    summary = summarize(load_episodes(out_dir / "episodes.jsonl"), pr.fail_penalty_steps)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    adaptation = summary.pop("sa_adaptation", None)
    for cond, s in summary.items():
        status.update(**{f"time_{cond}": s["time_to_completion"]})
        if "self_assessment" in s:
            status.update(**{f"auroc2_{cond}": s["self_assessment"].get("auroc2"),
                             f"meta_acc_{cond}": s["self_assessment"].get("meta_accuracy")})
    for cond, s in summary.items():
        extra = ""
        if "self_assessment" in s:
            sa = s["self_assessment"]
            extra = (f"  AUROC2={sa.get('auroc2', float('nan')):.2f}  meta-acc={sa.get('meta_accuracy', 0):.0f}%"
                     f"  M_sa inputs truncated: chunks {sa.get('truncated_chunks', 0):.1f}%"
                     f" rollouts {sa.get('truncated_rollouts', 0):.1f}%")
        log.info("%s %-8s success=%.0f%%  time=%.0f  early stops=%.0f%%%s", args.agent, cond, s["success_rate"],
                 s["time_to_completion"], s["early_stop_rate"], extra)
    if adaptation:
        log.info("M_sa update: held-out pre-deployment BCE %.3f -> %.3f, AUROC %.2f -> %.2f (forgetting check)",
                 adaptation["heldout_bce_before"], adaptation["heldout_bce_after"],
                 adaptation["heldout_auroc_before"], adaptation["heldout_auroc_after"])


def cmd_status(cfg, args):
    from .dashboard import watch
    watch(Path(args.runs_dir), Path(args.out), args.watch)


def cmd_report(cfg, args):
    from .report import write_report
    write_report([Path(r) for r in args.runs], Path(args.out), args.tasks)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="muse", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["collect", "train-sft", "train-dpo", "train-wm", "train-sa", "eval"]:
        sp = sub.add_parser(name)
        sp.add_argument("-c", "--config", default="configs/default.yaml")
        sp.add_argument("overrides", nargs="*", help="key=value config overrides")
        if name == "eval":
            sp.add_argument("--agent", required=True, choices=AGENTS)
    sp = sub.add_parser("report")
    sp.add_argument("--runs", nargs="+", required=True, help="run dirs; the first is the main (Mistral-7B) run")
    sp.add_argument("--out", default="runs/report")
    sp.add_argument("--tasks", choices=["all", "common", "solved"], default="all",
                    help="main run only: score every task, the tasks all agents finished, or those of them some "
                         "agent solved at least once")
    sp = sub.add_parser("status", help="print pipeline progress and write an HTML dashboard")
    sp.add_argument("--runs-dir", default="runs")
    sp.add_argument("--out", default="runs/dashboard.html")
    sp.add_argument("--watch", type=float, default=0, help="refresh every N seconds (0 = once)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    for noisy in ("httpx", "urllib3", "filelock", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if args.cmd == "report":
        return cmd_report(None, args)
    if args.cmd == "status":
        return cmd_status(None, args)
    cfg = load_config(args.config, args.overrides)
    import random
    import numpy as np
    import torch
    random.seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)
    torch.manual_seed(cfg.run.seed)
    from .provenance import record
    from .status import StageStatus
    fn = {"collect": cmd_collect, "train-sft": cmd_train_sft, "train-dpo": cmd_train_dpo,
          "train-wm": cmd_train_wm, "train-sa": cmd_train_sa, "eval": cmd_eval}[args.cmd]
    stage = f"eval-{eval_name(cfg, args.agent)}" if args.cmd == "eval" else args.cmd
    rec = record(cfg, stage)
    if rec["git"] and rec["git"]["dirty"]:
        log.warning("uncommitted changes to the code or configs; the diff is saved in provenance.jsonl")
    status = StageStatus(cfg.run.out_dir, stage)
    status.update(model=cfg.model.name)
    try:
        fn(cfg, args, status)
    except BaseException as e:
        status.finish("failed", repr(e))
        raise
    status.finish()


if __name__ == "__main__":
    main()
