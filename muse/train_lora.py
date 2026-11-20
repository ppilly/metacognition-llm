"""LoRA fine-tuning of the Actor (SFT) and Reflection LLM (DPO), Sec. 4.1.3."""
import json
import logging
import math
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .llm import load_causal_lm

log = logging.getLogger(__name__)


def _setup(cfg, init_adapter=None):
    """Base model with a trainable LoRA adapter: new (cfg.lora), or continued from `init_adapter`."""
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer

    m = cfg.model
    tok = AutoTokenizer.from_pretrained(m.tokenizer or m.name, trust_remote_code=m.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_causal_lm(m)
    model.config.use_cache = False
    if m.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lc = cfg.lora
    if init_adapter:
        model = PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=lc.r, lora_alpha=lc.alpha, lora_dropout=lc.dropout,
                                                 target_modules=lc.target_modules, task_type="CAUSAL_LM"))
    for p in model.parameters():          # train adapters in fp32 under bf16 autocast
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    return tok, model


def _optimizer(model, tc, num_items):
    from transformers import get_cosine_schedule_with_warmup

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc.lr, weight_decay=0.0)
    steps = max(1, math.ceil(num_items * tc.epochs / tc.grad_accum))
    sched = get_cosine_schedule_with_warmup(opt, int(tc.warmup_ratio * steps), steps)
    return params, opt, sched, steps


def _evaluate(model, items, loss_fn) -> dict | None:
    """Mean of each loss_fn stat over the held-out items that report it, without gradients."""
    if not items:
        return None
    model.eval()
    sums, counts = {}, {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for item in items:
            for k, v in loss_fn(item)[1].items():
                sums[k] = sums.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1
    model.train()
    return {k: sums[k] / counts[k] for k in sums}


def _loop(model, items, tc, loss_fn, seed, status=None, val_items=None, val_every=None, select=None) -> dict:
    """Train; returns the learning curves: running train stats every 10 optimizer steps and
    held-out stats before training, after every epoch and every `val_every` optimizer steps. With
    `select(row, first_row)` -> score (None: not eligible), the adapter ends at the held-out check with
    the highest score; the untrained start (step 0) is a candidate."""
    params, opt, sched, total_steps = _optimizer(model, tc, len(items))
    rng = random.Random(seed)
    history = {"train": [], "val": []}
    best = {"score": None, "state": None, "step": None}

    def validate(epoch, step):
        val = _evaluate(model, val_items, loss_fn)
        if val:
            history["val"].append({"epoch": epoch, "step": step, **val})
            log.info("validation after %s epochs (step %d): %s", epoch, step,
                     " ".join(f"{k}={v:.4f}" for k, v in val.items()))
            if status:
                status.update(**{f"val_{k}": v for k, v in val.items()})
            if select:
                score = select(history["val"][-1], history["val"][0])
                if score is not None and (best["score"] is None or score > best["score"]):
                    best.update(score=score, step=step, state={n: p.detach().cpu().clone()
                                                               for n, p in model.named_parameters() if p.requires_grad})

    model.train()
    step, running = 0, []
    validate(0, 0)
    for epoch in range(tc.epochs):
        order = list(range(len(items)))
        rng.shuffle(order)
        for i, idx in enumerate(order, 1):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, stats = loss_fn(items[idx])
            (loss / tc.grad_accum).backward()
            running.append(stats)
            if i % tc.grad_accum == 0 or i == len(order):
                lr = sched.get_last_lr()[0]       # the rate this optimizer step uses
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or i == len(order):
                    keys = dict.fromkeys(k for r in running for k in r)   # a stat may be missing from some items
                    avg = {k: sum(r[k] for r in running if k in r) / sum(k in r for r in running) for k in keys}
                    history["train"].append({"epoch": epoch, "step": step, "lr": lr, **avg})
                    log.info("epoch %d step %d %s lr=%.2e", epoch, step,
                             " ".join(f"{k}={v:.4f}" for k, v in avg.items()), lr)
                    if status:
                        status.update(step, total_steps, "steps", epoch=epoch, **avg)
                    running = []
                if val_every and step % val_every == 0 and i != len(order):
                    validate(round(step * tc.epochs / total_steps, 2), step)
        validate(epoch + 1, step)
    if select and best["state"] is not None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in best["state"]:
                    p.copy_(best["state"][n].to(p.device))
        history["selected_step"] = best["step"]
        log.info("kept the adapter from step %s (held-out score %.4f)", best["step"], best["score"])
    return history


def save_curves(out_dir, history: dict, title: str):
    """Write the learning curves as JSON and as a plot (one panel per metric)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "curves.json").write_text(json.dumps(history, indent=2))
    skip = {"epoch", "step", "lr"}
    rows = history["train"] + history["val"]
    metrics = list(dict.fromkeys(k for r in rows for k in r if k not in skip))
    if not metrics:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 3.5), squeeze=False)
    for ax, k in zip(axes[0], metrics):
        for rows_, style, label in ((history["train"], "-", "train"), (history["val"], "o-", "held-out tasks")):
            pts = [(r["step"], r[k]) for r in rows_ if k in r]       # a stat can be missing from some rows
            if pts:
                ax.plot(*zip(*pts), style, label=label)
        ax.set_xlabel(history.get("x_label", "optimizer step"))
        ax.set_title(k)
        ax.legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=120)
    plt.close(fig)


def _save(model, tok, out_dir, meta):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out_dir / "merged", ignore_errors=True)    # stale merge of a previous adapter (llm.py)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))


# ----------------------------------------------------------------------------- SFT

def _span_labels(ids, offsets, spans):
    """Token ids inside the character spans (sorted, non-overlapping), -100 elsewhere."""
    labels, j = [], 0
    for tid, (start, end) in zip(ids, offsets):
        while j < len(spans) and spans[j][1] <= start:
            j += 1
        inside = j < len(spans) and spans[j][0] <= start < spans[j][1] and end > start
        labels.append(tid if inside else -100)
    return labels


def _tokenize_sft(tok, ex, max_len):
    enc = tok(ex["text"], return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    labels = _span_labels(ids, offsets, ex["spans"])
    if len(ids) > max_len:                 # keep the end of the episode; drop the oldest context
        ids, labels = ids[-max_len:], labels[-max_len:]
    return torch.tensor(ids), torch.tensor(labels)


def _sft_loss(model, ids, labels):
    """Mean next-token loss on the labelled (actor) tokens, as model(labels=...) computes it, but with
    logits only at the positions that predict them: full-sequence logits over a large vocabulary
    dominate memory for long episodes."""
    ids, labels = ids.unsqueeze(0).to(model.device), labels.to(model.device)
    keep = (labels[1:] != -100).nonzero().squeeze(-1)       # position t predicts token t + 1
    logits = model(input_ids=ids, logits_to_keep=keep).logits[0].float()
    return F.cross_entropy(logits, labels[1:][keep])


def train_sft(cfg, examples: list[dict], out_dir, status=None, val_examples=None):
    tok, model = _setup(cfg)
    tc = cfg.sft

    def tokenize(exs):
        data = [_tokenize_sft(tok, ex, tc.max_len) for ex in exs or []]
        return [d for d in data if (d[1] != -100).any()]

    data, val = tokenize(examples), tokenize(val_examples)
    log.info("SFT on %d successful episodes (max %d tokens), %d held out", len(data),
             max(len(d[0]) for d in data), len(val))

    def loss_fn(item):
        loss = _sft_loss(model, *item)
        return loss, {"loss": loss.item()}

    history = _loop(model, data, tc, loss_fn, cfg.run.seed, status, val)
    _save(model, tok, out_dir, {"kind": "sft", "num_examples": len(data), "num_val": len(val),
                                "base": cfg.model.name})
    save_curves(out_dir, history, "Actor M_a: SFT")


# ----------------------------------------------------------------------------- World Model

def _tokenize_wm(tok, ex, max_len):
    enc = tok(ex["text"], return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    labels = [_span_labels(ids, offsets, ex[k]) for k in ("action_spans", "obs_spans", "anchor_spans")]
    if len(ids) > max_len:
        ids, labels = ids[-max_len:], [l[-max_len:] for l in labels]
    return (torch.tensor(ids), *map(torch.tensor, labels))


def _chunk_logp(head, h, target):
    return -F.cross_entropy(head(h).float(), target, reduction="none")


def _token_logps(model, ids, keep, target, chunk=512):
    """log p(target) at positions `keep` of `ids`. The output layer runs on `chunk` positions at a time
    and is recomputed in the backward pass, so the logits (positions x vocabulary) never sit in memory
    together: those of a 50-step episode's labels overflow a 4 GB GPU."""
    from torch.utils.checkpoint import checkpoint

    inner = model.get_base_model() if hasattr(model, "get_base_model") else model   # LoRA layers stay active
    h = inner.model(input_ids=ids).last_hidden_state[0][keep]
    head = inner.get_output_embeddings()
    return torch.cat([checkpoint(_chunk_logp, head, h[i:i + chunk], target[i:i + chunk], use_reentrant=False)
                      for i in range(0, len(keep), chunk)]), h, head


def _wm_loss(model, ids, act, obs, anchor, obs_weight, action_term="ce", ref=None, anchor_weight=1.0):
    """Extension: obs_weight x the mean next-token loss on the observation tokens, plus an action term,
    each averaged over its own tokens so the many observation tokens do not swamp the actions.
    action_term ce: the mean loss on the action tokens of successful episodes (as SFT). anchor: the mean
    squared change of the log-probability of every episode's action tokens from `ref`, the starting
    adapter's (_ref_logps), which holds the policy in place without retraining it on the same successes.
    The action loss is reported either way. One forward pass; the output layer runs in chunks (_token_logps)."""
    dev = model.device
    ids, act, obs, anchor = ids.unsqueeze(0).to(dev), act.to(dev), obs.to(dev), anchor.to(dev)
    a, o, n = act[1:] != -100, obs[1:] != -100, anchor[1:] != -100
    keep = (a | o | n).nonzero().squeeze(-1)
    target = torch.where(o, obs[1:], anchor[1:])[keep]          # anchor spans include every action span
    tok_logp, h, head = _token_logps(model, ids, keep, target)
    is_act, is_obs, is_anchor = a[keep], o[keep], n[keep]
    stats, loss = {}, tok_logp.new_zeros(())
    if is_act.any():
        la = -tok_logp[is_act].mean()
        stats["action_loss"] = la.item()
        if action_term == "ce":
            loss = loss + la
    if action_term == "anchor" and is_anchor.any():
        drift = ((tok_logp[is_anchor] - ref.to(dev)) ** 2).mean()
        loss = loss + anchor_weight * drift
        stats["anchor_drift"] = drift.item()
    if is_obs.any():
        lo = -tok_logp[is_obs].mean()
        loss = loss + obs_weight * lo
        stats["obs_loss"] = lo.item()
        with torch.no_grad():
            hits = sum((head(h[is_obs][i:i + 512]).argmax(-1) == target[is_obs][i:i + 512]).sum().item()
                       for i in range(0, int(is_obs.sum()), 512))
            stats["obs_token_acc"] = hits / int(is_obs.sum())
    return loss, stats


@torch.no_grad()
def _ref_logps(model, ids, anchor):
    """Log-probabilities the starting adapter gives each anchor (action) token, for the anchor term."""
    ids, anchor = ids.unsqueeze(0).to(model.device), anchor.to(model.device)
    keep = (anchor[1:] != -100).nonzero().squeeze(-1)
    if not len(keep):
        return torch.zeros(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return _token_logps(model, ids, keep, anchor[1:][keep])[0].float().cpu()


def train_world_model(cfg, examples: list[dict], out_dir, init_adapter, status=None, val_examples=None):
    """Extension: the World Model adapter used for MUSE's rollouts, continued from the actor adapter with
    a weighted observation loss; keeps the held-out check with the lowest observation loss whose action
    loss is within world_model.action_tolerance of the actor's."""
    tok, model = _setup(cfg, init_adapter)
    wc, tc = cfg.world_model, cfg.sft
    train_cfg = SimpleNamespace(lr=wc.lr, epochs=wc.epochs, grad_accum=tc.grad_accum, warmup_ratio=tc.warmup_ratio)

    def tokenize(exs):
        return [_tokenize_wm(tok, ex, tc.max_len) for ex in exs or []]

    data, val = tokenize(examples), tokenize(val_examples)
    log.info("World Model on %d episodes (%d successful; max %d tokens), %d held out, obs_weight %.2f",
             len(data), sum(ex["success"] for ex in examples), max(len(d[0]) for d in data), len(val),
             wc.obs_weight)

    if wc.action_term not in ("ce", "anchor"):
        raise ValueError(f"world_model.action_term {wc.action_term!r}: ce | anchor")
    refs = {}
    if wc.action_term == "anchor":              # the starting adapter's action log-probs, before any update
        model.eval()
        for i, item in enumerate(data):
            refs[i] = _ref_logps(model, item[0], item[3])
        model.train()
        data = [(*item, i) for i, item in enumerate(data)]
        log.info("anchor: reference log-probs of %d action tokens", sum(len(r) for r in refs.values()))

    def loss_fn(item):
        if wc.action_term == "anchor" and len(item) == 5:
            return _wm_loss(model, *item[:4], wc.obs_weight, "anchor", refs[item[4]], wc.anchor_weight)
        return _wm_loss(model, *item[:4], wc.obs_weight, "ce" if wc.action_term == "ce" else "none")

    def select(row, first):
        if row["action_loss"] > first["action_loss"] * (1 + wc.action_tolerance):
            return None
        return -row["obs_loss"]

    history = _loop(model, data, train_cfg, loss_fn, cfg.run.seed, status, val, wc.val_every, select)
    _save(model, tok, out_dir, {"kind": "world_model", "num_examples": len(data), "num_val": len(val),
                                "obs_weight": wc.obs_weight, "action_term": wc.action_term,
                                "anchor_weight": wc.anchor_weight, "init": str(init_adapter),
                                "selected_step": history.get("selected_step"), "base": cfg.model.name})
    save_curves(out_dir, history, f"World Model adapter (obs_weight {wc.obs_weight})")


# ----------------------------------------------------------------------------- DPO

def _tokenize_pair(tok, prompt, response, max_len):
    p_ids = tok(prompt)["input_ids"]
    r_ids = tok(response, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (p_ids + r_ids)[-max_len:]
    n_resp = min(len(r_ids), len(ids) - 1)
    return torch.tensor(ids), n_resp


def _seq_logp(model, ids, n_resp):
    """Summed log-prob of the last n_resp tokens. Only the logits for those positions are computed:
    full-sequence logits over a large vocabulary dominate memory for long reflection prompts."""
    ids = ids.unsqueeze(0).to(model.device)
    logits = model(input_ids=ids, logits_to_keep=n_resp + 1).logits[:, :-1].float()
    logp = torch.log_softmax(logits, -1).gather(-1, ids[:, -n_resp:].unsqueeze(-1)).squeeze(-1)
    return logp.sum()


def train_dpo(cfg, pairs: list[dict], out_dir, status=None, val_pairs=None):
    """DPO-like objective: each reflection is scored under its own prompt, i.e. the margin is
    beta * ([log pi - log pi_ref](chosen | prompt_c) - [log pi - log pi_ref](rejected | prompt_r))."""
    tok, model = _setup(cfg)
    tc = cfg.dpo

    def prepare(ps):
        data = [{"chosen": _tokenize_pair(tok, p["prompt_chosen"], p["chosen"], tc.max_len),
                 "rejected": _tokenize_pair(tok, p["prompt_rejected"], p["rejected"], tc.max_len)}
                for p in ps or []]
        model.eval()
        with torch.no_grad(), model.disable_adapter(), torch.autocast("cuda", dtype=torch.bfloat16):
            for d in data:
                d["ref_c"] = _seq_logp(model, *d["chosen"]).item()
                d["ref_r"] = _seq_logp(model, *d["rejected"]).item()
        return data

    log.info("DPO on %d preference pairs (%d held out); computing reference log-probs",
             len(pairs), len(val_pairs or []))
    data, val = prepare(pairs), prepare(val_pairs)

    def loss_fn(d):
        pc = _seq_logp(model, *d["chosen"])
        pr = _seq_logp(model, *d["rejected"])
        margin = tc.beta * ((pc - d["ref_c"]) - (pr - d["ref_r"]))
        loss = -F.logsigmoid(margin)
        return loss, {"loss": loss.item(), "reward_acc": float(margin.item() > 0), "margin": margin.item()}

    history = _loop(model, data, tc, loss_fn, cfg.run.seed, status, val)
    _save(model, tok, out_dir, {"kind": "dpo", "num_pairs": len(data), "num_val": len(val),
                                "base": cfg.model.name})
    save_curves(out_dir, history, "Reflection LLM M_rx: DPO")
