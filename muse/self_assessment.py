"""Self-Assessment Model M_sa (Sec. 4.1.3, Alg. 2).

A frozen sentence-transformer encoder E embeds the task description I (Z = E(I)) and the
plan + trajectory chunk (S = E(tau, P_e)); an MLP g_eta maps concat(S, Z) to P(task success).
Only g_eta is trained, both before deployment and during adaptation. Its input can be standardized
with statistics of the pre-deployment training set, and its loss balanced between successful and
failed chunks (self_assessment.standardize / class_balance; both are implementation choices).
"""
import copy
import threading
from pathlib import Path

import numpy as np
import torch
from torch import nn


def format_pairs(pairs, status: str | None = None) -> str:
    """Text form of action-observation pairs, matching the supplement's imagined trajectories:
    'go to cabinet 1.The cabinet 1 is closed..> open cabinet 1.You open the cabinet 1. ...'
    A terminal signal, if any, follows as one more '> STATUS: ...' item."""
    text = "> ".join(f"{a}.{o}." for a, o in pairs)
    return f"{text}> {status}" if status else text


def chunk_starts(num_steps: int, size: int, stride: int | None = None) -> list[int]:
    """First step of each chunk. stride None (= size): non-overlapping, as in the paper; 1: a chunk
    starting at every step (extension), the last ones shorter than `size`."""
    return list(range(0, num_steps, stride or size))


def chunk_steps(steps, size: int, stride: int | None = None) -> list[list[tuple[str, str]]]:
    """Chunks of up to `size` action-observation pairs: (a1..a4), (a5..a8), ... by default."""
    pairs = [(s.action, s.observation) for s in steps]
    return [pairs[i:i + size] for i in chunk_starts(len(pairs), size, stride)]


def chunk_texts(steps, size: int, stride: int | None = None) -> list[str]:
    """Texts of an executed episode's chunks, without a terminal signal: it would give away the
    outcome that M_sa is trained and evaluated to predict."""
    return [format_pairs(c) for c in chunk_steps(steps, size, stride)]


def actor_context_text(task_type, init_obs, memory, steps, chunk_size: int, stride: int | None = None):
    """The World Model prompt (examples, memory, task) through the last step, and each chunk's character
    span, from its first action through its last observation."""
    from . import prompts

    text = prompts.actor_header(task_type, init_obs, memory, terminal=True)
    bounds = []
    for s in steps:
        start = len(text)
        text += prompts.render_steps([s])
        bounds.append((start, len(text) - 2))           # through the observation, not its "\n>"
    spans = [(bounds[i][0], bounds[min(i + chunk_size, len(steps)) - 1][1])
             for i in chunk_starts(len(steps), chunk_size, stride)]
    return text, spans


def _pool(h, offsets, spans) -> list[torch.Tensor]:
    """Mean of the hidden states over each character span's tokens, then the span's last token."""
    h = h.float().cpu()
    starts = torch.tensor([o[0] for o in offsets])
    ends = torch.tensor([o[1] for o in offsets])
    out = []
    for s, e in spans:
        idx = ((ends > s) & (starts < e)).nonzero().squeeze(-1)
        if not len(idx) or starts[0] > s:
            raise ValueError(f"span {s}-{e} lost to the context limit")
        out.append(torch.cat([h[idx].mean(0), h[idx[-1]]]))
    return out


def _decode(decoder, ids, cache, layers, use_cache=True):
    """Run the decoder on `ids` after `cache`; return its output and the hidden states of each of
    `layers` (None: the final layer, after the model's last norm; k: the output of decoder layer k,
    1-based, before any norm)."""
    n = len(decoder.layers)
    for k in layers:
        if k is not None and not 1 <= k <= n:
            raise ValueError(f"self_assessment.layer {k} outside 1..{n}")
    grabbed = {}

    def hook(k):
        return lambda mod, inp, out: grabbed.__setitem__(k, out[0] if isinstance(out, tuple) else out)

    handles = [decoder.layers[k - 1].register_forward_hook(hook(k)) for k in set(layers) if k is not None]
    try:
        out = decoder(input_ids=ids, past_key_values=cache, use_cache=use_cache)
    finally:
        for h in handles:
            h.remove()
    return out, [out.last_hidden_state[0] if k is None else grabbed[k][0] for k in layers]


def span_features(model, tok, prefix: str, spans, max_len: int, continuations: list[str] | None = None,
                  layer=None):
    """Hidden-state features (_pool) of character spans at `layer` (None: the final layer; k: the output
    of decoder layer k; a list: one feature tensor per layer, from the same passes). Without
    `continuations` the spans are in `prefix`. With them, `prefix` is run once and each continuation
    after it on the prefix's KV cache; spans[i] is then continuation i's span in prefix + continuation."""
    layers = layer if isinstance(layer, list) else [layer]
    decoder = model.get_decoder()
    enc = tok(prefix, return_offsets_mapping=True)
    room = max_len
    if continuations:
        room -= max(len(tok(c, add_special_tokens=False)["input_ids"]) for c in continuations)
    ids, offsets = enc["input_ids"][-room:], enc["offset_mapping"][-room:]
    with torch.no_grad():
        out, hs = _decode(decoder, torch.tensor([ids], device=model.device), None, layers, bool(continuations))
        if not continuations:
            res = [torch.stack(_pool(h, offsets, spans)) for h in hs]
        else:
            cache, feats = out.past_key_values, [[] for _ in layers]
            for cont, span in zip(continuations, spans):
                c = tok(cont, add_special_tokens=False, return_offsets_mapping=True)
                _, hs = _decode(decoder, torch.tensor([c["input_ids"]], device=model.device), cache, layers)
                cache.crop(len(ids))
                offs = [(a + len(prefix), b + len(prefix)) for a, b in c["offset_mapping"]]
                for f, h in zip(feats, hs):
                    f += _pool(h, offs, [span])
            res = [torch.stack(f) for f in feats]
    return res if isinstance(layer, list) else res[0]


def actor_context_features(model, tok, episodes, chunk_size: int, max_len: int,
                           stride: int | None = None, layer=None):
    """Extension, not in the paper (whose M_sa sees only I, P_e and the chunk, Eq. 6): features of each
    executed chunk from the actor LLM given the actor's full context, i.e. the World Model prompt
    (examples, memory, task) and every step through the chunk's last observation. The model is causal,
    so one forward pass per episode gives every chunk's features. Feature: `layer`'s hidden states
    (span_features) averaged over the chunk's tokens, then their value at the chunk's last token. Rows
    follow sa_examples' order. A list of layers gives a list of feature tensors, from the same passes."""
    from types import SimpleNamespace

    layers = layer if isinstance(layer, list) else [layer]
    feats = [[] for _ in layers]
    for ep in episodes:
        steps = [SimpleNamespace(**s) for s in ep["steps"]]
        text, spans = actor_context_text(ep["task_type"], ep["init_obs"], ep["memory"], steps, chunk_size, stride)
        if spans:
            for f, x in zip(feats, span_features(model, tok, text, spans, max_len, layer=layers)):
                f.append(x)
    out = [torch.cat(f) for f in feats]
    return out if isinstance(layer, list) else out[0]


def sa_input_text(plan: str, chunk_text: str) -> str:
    """S = E(tau, P_e): the plan, then the chunk. Encoder truncation would cut the chunk's end, so
    truncated inputs are counted (`SelfAssessor.truncated`) and reported."""
    return f"Plan: {plan}\nTrajectory: {chunk_text}"


class _Encoder:
    """Shared, thread-safe, caching wrapper around the sentence-transformer."""

    _instances: dict = {}
    _lock = threading.Lock()

    def __init__(self, name, device, max_seq_length=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device=device)
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()
        self.cache: dict[str, np.ndarray] = {}
        self.lock = threading.Lock()

    @classmethod
    def get(cls, name, device, max_seq_length=None):
        with cls._lock:
            key = (name, device, max_seq_length)
            if key not in cls._instances:
                cls._instances[key] = cls(name, device, max_seq_length)
            return cls._instances[key]

    def encode(self, texts: list[str]) -> np.ndarray:
        with self.lock:
            missing = list({t for t in texts if t not in self.cache})
            if missing:
                vecs = self.model.encode(missing, batch_size=128, convert_to_numpy=True,
                                         normalize_embeddings=False, show_progress_bar=False)
                self.cache.update(zip(missing, vecs))
            return np.stack([self.cache[t] for t in texts])

    def num_tokens(self, texts: list[str]) -> list[int]:
        """Encoder tokens per text, special tokens included; beyond max_seq_length the end is cut."""
        return [len(ids) for ids in self.model.tokenizer(texts)["input_ids"]]


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ReplayBuffer:
    """Chunk features replayed during adaptation (Alg. 4's memory buffer): the pre-deployment training
    set. Each novel task gets its own copy, so nothing carries from one task to another."""

    def __init__(self, x: torch.Tensor | None = None, y: torch.Tensor | None = None):
        self.xs, self.ys = ([x], [y]) if x is not None else ([], [])
        self.lock = threading.Lock()

    def add(self, x: torch.Tensor, y: torch.Tensor):
        with self.lock:
            self.xs.append(x.cpu())
            self.ys.append(y.float().cpu())

    def get(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        with self.lock:
            if not self.xs:
                return None, None
            return torch.cat(self.xs), torch.cat(self.ys)

    def __len__(self):
        with self.lock:
            return sum(len(x) for x in self.xs)

    def copy(self) -> "ReplayBuffer":
        new = ReplayBuffer()
        with self.lock:
            new.xs, new.ys = list(self.xs), list(self.ys)   # tensors are never modified in place
        return new


def within_task_auroc(y, p, task_ids) -> float:
    """Mean over tasks with both labels of the AUROC within each task (what MUSE's choice among one
    step's rollouts relies on); nan if no task has both."""
    from sklearn.metrics import roc_auc_score

    y, p, t = np.asarray(y, dtype=float), np.asarray(p, dtype=float), np.asarray(task_ids)
    vals = [roc_auc_score(y[t == k], p[t == k]) for k in sorted(set(t.tolist())) if len(set(y[t == k])) == 2]
    return float(np.mean(vals)) if vals else float("nan")


class SelfAssessor:
    """g_eta on features of the paper's input (the sentence encoder's Z and S) or, with
    self_assessment.input: actor, on actor_context_features of dimension `in_dim` (computed by the caller)."""

    def __init__(self, cfg, device=None, in_dim: int | None = None):
        self.cfg = cfg.self_assessment
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.cfg.input == "actor":
            self.encoder = None
        else:
            self.encoder = _Encoder.get(self.cfg.encoder, self.device, self.cfg.max_seq_length)
            in_dim = 2 * self.encoder.dim
        self.in_dim = in_dim
        self.mlp = MLP(in_dim, self.cfg.hidden, self.cfg.dropout).to(self.device)
        self.replay = ReplayBuffer()
        # held-out pre-deployment chunks: never trained on, used to measure forgetting after adaptation
        self.val_x: torch.Tensor | None = None
        self.val_y: torch.Tensor | None = None
        # feature standardization (set_standardizer), fixed after pre-deployment training; None: identity
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.lock = threading.RLock()   # re-entered when fit()'s log_fn evaluates

    # ----- features -----
    def truncated(self, plans: list[str] | str, chunk_texts: list[str]) -> list[bool]:
        """Whether each plan + chunk input is too long for the encoder and loses the chunk's end."""
        if isinstance(plans, str):
            plans = [plans] * len(chunk_texts)
        limit = self.encoder.model.max_seq_length
        return [n > limit for n in self.encoder.num_tokens([sa_input_text(p, c)
                                                            for p, c in zip(plans, chunk_texts)])]

    def features(self, task_descs: list[str], plans: list[str], chunk_texts: list[str]) -> torch.Tensor:
        z = self.encoder.encode(task_descs)
        s = self.encoder.encode([sa_input_text(p, c) for p, c in zip(plans, chunk_texts)])
        return torch.from_numpy(np.concatenate([s, z], axis=1)).float()

    # ----- inference -----
    @torch.no_grad()
    def predict_features(self, x: torch.Tensor) -> np.ndarray:
        with self.lock:
            self.mlp.eval()
            return torch.sigmoid(self.mlp(self._prep(x.to(self.device)))).cpu().numpy()

    def score(self, task_desc: str, plans: list[str] | str, chunk_texts: list[str]) -> np.ndarray:
        if isinstance(plans, str):
            plans = [plans] * len(chunk_texts)
        return self.predict_features(self.features([task_desc] * len(chunk_texts), plans, chunk_texts))

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        """BCE, accuracy at 0.5 and AUROC of g_eta on a labelled set of chunk features."""
        from sklearn.metrics import roc_auc_score

        p = np.clip(self.predict_features(x), 1e-7, 1 - 1e-7)
        y = y.numpy()
        out = {"bce": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
               "acc": float(np.mean((p >= 0.5) == y))}
        out["auroc"] = float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else float("nan")
        return out

    def set_standardizer(self, x: torch.Tensor):
        """Standardize every input feature with the mean and std of `x` (the pre-deployment training
        chunks). The actor's middle layers have a few dimensions hundreds of times larger than the rest."""
        self.mean, self.std = x.float().mean(0), x.float().std(0).clamp_min(1e-6)

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return x if self.mean is None else (x - self.mean.to(x.device)) / self.std.to(x.device)

    # ----- training -----
    def fit(self, x: torch.Tensor, y: torch.Tensor, epochs: int, lr: float, log_fn=None,
            replay: tuple[torch.Tensor, torch.Tensor] | None = None, new_fraction: float = 0.5,
            val: tuple | None = None) -> dict:
        """BCE training of g_eta. With `replay` = (x_old, y_old), each epoch is one pass over (x, y) plus
        replayed chunks sampled so that the new chunks make up `new_fraction` of the epoch. Class
        balance (self_assessment.class_balance) applies to each epoch's chunks, replay included:
        weight scales the loss of successful chunks, sample redraws successful chunks with replacement
        until they match the failed ones. With self_assessment.early_stop and `val` = (x, y, task_ids),
        g_eta ends at the epoch with the best within-task AUROC on `val` (pre-deployment training only)."""
        cfg = self.cfg
        stop = bool(cfg.early_stop and val is not None
                    and any(len(set(val[1][np.asarray(val[2]) == t].tolist())) == 2 for t in set(val[2])))
        best = {"epoch": None, "auroc": -1.0, "state": None}
        balance = cfg.class_balance
        if balance not in ("none", "weight", "sample"):
            raise ValueError(f"self_assessment.class_balance {balance!r}: none | weight | sample")
        with self.lock:
            opt = torch.optim.AdamW(self.mlp.parameters(), lr=lr, weight_decay=cfg.weight_decay)
            gen = torch.Generator(device="cpu").manual_seed(0)
            n_old = 0
            if replay is not None and replay[0] is not None and len(replay[0]):
                n_old = round(len(x) * (1 - new_fraction) / new_fraction)
            self.mlp.train()
            for epoch in range(epochs):
                xe, ye = x, y.float()
                if n_old:
                    x_old, y_old = replay
                    idx = (torch.randperm(len(x_old), generator=gen)[:n_old] if n_old <= len(x_old)
                           else torch.randint(len(x_old), (n_old,), generator=gen))
                    xe, ye = torch.cat([x, x_old[idx]]), torch.cat([ye, y_old[idx].float()])
                n_pos = int(ye.sum())
                n_neg = len(ye) - n_pos
                pos_weight = None
                if balance == "weight" and n_pos and n_neg:
                    pos_weight = torch.tensor(n_neg / n_pos, device=self.device)
                elif balance == "sample" and 0 < n_pos < n_neg:
                    pos = ye.nonzero().squeeze(-1)
                    extra = pos[torch.randint(len(pos), (n_neg - n_pos,), generator=gen)]
                    xe, ye = torch.cat([xe, xe[extra]]), torch.cat([ye, ye[extra]])
                xe, ye = self._prep(xe).to(self.device), ye.to(self.device)
                perm = torch.randperm(len(xe), generator=gen).to(self.device)
                total = 0.0
                for i in range(0, len(xe), cfg.batch_size):
                    b = perm[i:i + cfg.batch_size]
                    loss = nn.functional.binary_cross_entropy_with_logits(self.mlp(xe[b]), ye[b],
                                                                          pos_weight=pos_weight)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total += loss.item() * len(b)
                if stop:
                    a = within_task_auroc(val[1], self.predict_features(val[0]), val[2])
                    if a > best["auroc"]:
                        best = {"epoch": epoch + 1, "auroc": a, "state": copy.deepcopy(self.mlp.state_dict())}
                    self.mlp.train()
                if log_fn:
                    self.mlp.eval()
                    log_fn(epoch, total / len(xe))
                    self.mlp.train()
            if stop:
                self.mlp.load_state_dict(best["state"])
            self.mlp.eval()
            return {"best_epoch": best["epoch"], "best_val_auroc": best["auroc"] if stop else None}

    def adapt(self, x_new: torch.Tensor, y_new: torch.Tensor) -> dict:
        """Deployment update of g_eta on the new task's chunks mixed with replayed chunks (Sec. 4.2).
        Returns held-out pre-deployment metrics before and after, to show any forgetting."""
        cfg = self.cfg
        y_new = y_new.float()
        before = self.evaluate(self.val_x, self.val_y) if self.val_x is not None else None
        new_before = self.evaluate(x_new, y_new)
        old = self.replay.get()
        self.fit(x_new, y_new, cfg.adapt_epochs or cfg.epochs, cfg.adapt_lr or cfg.lr,
                 replay=old, new_fraction=cfg.adapt_new_fraction)
        return {"n_new": len(x_new), "n_replay": 0 if old[0] is None else len(old[0]),
                "new_before": new_before, "new_after": self.evaluate(x_new, y_new),
                "val_before": before,
                "val_after": self.evaluate(self.val_x, self.val_y) if self.val_x is not None else None}

    def clone(self) -> "SelfAssessor":
        """Fresh copy of g_eta and the replay buffer for one novel task (per-task reset to the end of
        pre-deployment training, Alg. 4 line 4); shares the frozen encoder and the held-out set."""
        new = copy.copy(self)
        new.mlp = copy.deepcopy(self.mlp)
        new.replay = self.replay.copy()
        new.lock = threading.RLock()
        return new

    # ----- persistence -----
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        replay_x, replay_y = self.replay.get()
        torch.save({"mlp": self.mlp.state_dict(), "replay_x": replay_x, "replay_y": replay_y,
                    "val_x": self.val_x, "val_y": self.val_y, "in_dim": self.in_dim,
                    "mean": self.mean, "std": self.std}, path / "sa.pt")

    @classmethod
    def load(cls, cfg, path, device=None) -> "SelfAssessor":
        state = torch.load(Path(path) / "sa.pt", map_location="cpu", weights_only=True)
        sa = cls(cfg, device, in_dim=state.get("in_dim"))
        sa.mlp.load_state_dict(state["mlp"])
        sa.replay = ReplayBuffer(state["replay_x"], state["replay_y"])
        sa.val_x, sa.val_y = state.get("val_x"), state.get("val_y")
        sa.mean, sa.std = state.get("mean"), state.get("std")      # absent in models trained before
        sa.mlp.eval()
        return sa
