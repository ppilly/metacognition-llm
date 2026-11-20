"""LLM backends with request batching across concurrently running tasks.

Agents call `llm.generate(...)` from many worker threads; a background thread groups pending
requests into batches. LoRA adapters (actor M_a, reflector M_rx) are addressed by name; adapter=None
means the pre-trained base weights.
"""
import contextlib
import logging
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from .config import Config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplingKey:
    n: int
    temperature: float
    max_tokens: int | None      # None: generate until the context window is full
    stop: tuple
    adapter: str | None
    max_turns: int | None = None    # stop once this many transcript turns ("\n>") are complete
    max_line_chars: int | None = None   # stop a sequence whose current line runs past this (a loop)


@dataclass
class Request:
    prompt: str | None
    key: SamplingKey | None
    future: Future
    fn: Callable | None = None      # other GPU work, run on the generation thread (BatchedLLM.call)


TURN = "\n>"     # in the ReAct transcript, each action-observation pair ends where the next action begins


def truncate_at_stop(text: str, stop, max_turns: int | None = None) -> str:
    cut = len(text)
    for s in stop:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    if max_turns:
        i = -1
        for _ in range(max_turns):
            i = text.find(TURN, i + 1)
            if i == -1:
                break
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


class BatchedLLM:
    def __init__(self, max_batch_size: int = 64, batch_wait_ms: int = 20, chat_tokenizer=None):
        self.max_batch_size = max_batch_size
        self.chat_tokenizer = chat_tokenizer
        self.batch_wait = batch_wait_ms / 1000
        self._queue: queue.Queue[Request] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def generate(self, prompt: str, *, n=1, temperature=0.0, max_tokens=128, stop=(), adapter=None,
                 max_turns=None, max_line_chars=None) -> list[str]:
        key = SamplingKey(n, float(temperature), None if max_tokens is None else int(max_tokens), tuple(stop),
                          adapter, max_turns, max_line_chars)
        if self.chat_tokenizer is not None:
            prompt = self.chat_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        fut = Future()
        self._queue.put(Request(prompt, key, fut))
        return fut.result()

    def call(self, fn):
        """fn() on the generation thread, between batches, so it never shares the GPU with one."""
        fut = Future()
        self._queue.put(Request(None, None, fut, fn))
        return fut.result()

    def span_features(self, prefix: str, spans, adapter: str | None, continuations: list[str] | None = None,
                      layer: int | None = None):
        """self_assessment.span_features with the model serving `adapter` (actor-context M_sa)."""
        raise NotImplementedError(f"{type(self).__name__} has no hidden-state access; use the hf backend")

    def _loop(self):
        while True:
            reqs = [self._queue.get()]
            deadline = time.monotonic() + self.batch_wait
            while len(reqs) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    reqs.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break
            for r in [r for r in reqs if r.fn is not None]:
                try:
                    r.future.set_result(r.fn())
                except Exception as e:
                    log.exception("call failed")
                    r.future.set_exception(e)
            reqs = [r for r in reqs if r.fn is None]
            if not reqs:
                continue
            try:
                outs = self._run(reqs)
                for r, o in zip(reqs, outs):
                    r.future.set_result([truncate_at_stop(t, r.key.stop, r.key.max_turns) for t in o])
            except Exception as e:
                log.exception("generation failed")
                for r in reqs:
                    if not r.future.done():
                        r.future.set_exception(e)

    def _run(self, reqs: list[Request]) -> list[list[str]]:
        raise NotImplementedError

    def num_tokens(self, text: str) -> int:
        """Tokens of `text` in the model's tokenizer, without special tokens."""
        return len(self.tokenizer().encode(text, add_special_tokens=False))

    def tokenizer(self):
        raise NotImplementedError


class HFLLM(BatchedLLM):
    """transformers + peft. Used for local development and for models vLLM does not support."""

    def __init__(self, cfg, adapters: dict[str, str] | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        m = cfg.model
        self.torch = torch
        self.max_len = m.max_model_len
        self.max_batch_tokens = m.max_batch_tokens
        self.tok =AutoTokenizer.from_pretrained(m.tokenizer or m.name, trust_remote_code=m.trust_remote_code)
        self.tok.padding_side = "left"
        self.tok.truncation_side = "left"   # keep the most recent context if a prompt is too long
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.adapters = adapters or {}
        self.merged = {}
        if self.adapters and m.merge_adapters:
            # one plain model per adapter (no base model: agents with adapters never use the base weights)
            for name, path in self.adapters.items():
                self.merged[name] = load_causal_lm(Config({**m, "name": merged_checkpoint(m, path)}),
                                                   AutoModelForCausalLM).eval()
            self.model = None
        else:
            self.model = load_causal_lm(m, AutoModelForCausalLM)
            if self.adapters:
                from peft import PeftModel
                names = list(self.adapters)
                self.model = PeftModel.from_pretrained(self.model, self.adapters[names[0]], adapter_name=names[0])
                for name in names[1:]:
                    self.model.load_adapter(self.adapters[name], adapter_name=name)
            self.model.eval()
        super().__init__(m.max_batch_size, m.batch_wait_ms, self.tok if m.chat_template else None)

    def tokenizer(self):
        return self.tok

    def _run(self, reqs):
        groups: dict[SamplingKey, list[int]] = {}
        for i, r in enumerate(reqs):
            groups.setdefault(r.key, []).append(i)
        outs: list[list[str]] = [None] * len(reqs)
        for key, idxs in groups.items():
            per_call = max(1, self.max_batch_size // key.n)
            for j in range(0, len(idxs), per_call):
                chunk = idxs[j:j + per_call]
                texts = self._generate([reqs[i].prompt for i in chunk], key)
                for i, t in zip(chunk, texts):
                    outs[i] = t
        return outs

    def span_features(self, prefix, spans, adapter, continuations=None, layer=None):
        from .self_assessment import span_features

        def run():
            model = self.merged[adapter] if self.merged else self.model
            with self.torch.inference_mode(), (contextlib.nullcontext() if self.merged else self._adapter_ctx(adapter)):
                return span_features(model, self.tok, prefix, spans, self.max_len, continuations, layer)
        return self.call(run)

    def _adapter_ctx(self, adapter):
        if not self.adapters:
            if adapter is not None:
                raise ValueError(f"adapter {adapter!r} requested but no adapters loaded")
            return contextlib.nullcontext()
        if adapter is None:
            return self.model.disable_adapter()
        self.model.set_adapter(adapter)
        return contextlib.nullcontext()

    def _generate(self, prompts, key: SamplingKey, split=True) -> list[list[str]]:
        torch = self.torch
        budget = self.max_len - (key.max_tokens or 1)
        # add_special_tokens=True (BOS) matches vLLM's default and the training tokenization
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=budget)
        if enc.input_ids.shape[1] >= budget:
            log.warning("prompt truncated to %d tokens", budget)
        max_tokens = self.max_batch_tokens if split else None
        if max_tokens and len(prompts) * key.n * enc.input_ids.shape[1] > max_tokens:
            # split the call so that activations and KV cache stay within GPU memory
            if len(prompts) > 1:
                return [o for p in prompts for o in self._generate([p], key)]
            per_call = max(1, max_tokens // enc.input_ids.shape[1])
            outs = []
            for j in range(0, key.n, per_call):
                outs += self._generate(prompts, replace(key, n=min(per_call, key.n - j)), split=False)[0]
            return [outs]
        if self.merged:
            if key.adapter not in self.merged:
                raise ValueError(f"no merged model for adapter {key.adapter!r}")
            model = self.merged[key.adapter]
        else:
            model = self.model
        enc = enc.to(model.device)
        max_new = key.max_tokens or self.max_len - enc.input_ids.shape[1]
        kwargs = dict(max_new_tokens=max_new, num_return_sequences=key.n,
                      pad_token_id=self.tok.pad_token_id)
        if key.temperature > 0:
            kwargs.update(do_sample=True, temperature=key.temperature, top_p=1.0)
        else:
            kwargs.update(do_sample=False)
        if key.stop:
            kwargs.update(stop_strings=list(key.stop), tokenizer=self.tok)
        if key.max_turns or key.max_line_chars:
            kwargs.update(stopping_criteria=[_turn_stop(self.tok, enc.input_ids.shape[1], key.max_turns,
                                                        key.max_line_chars)])
        with torch.inference_mode(), (contextlib.nullcontext() if self.merged else self._adapter_ctx(key.adapter)):
            out = model.generate(**enc, **kwargs)
        gen = out[:, enc.input_ids.shape[1]:]
        texts = self.tok.batch_decode(gen, skip_special_tokens=True)
        return [texts[i * key.n:(i + 1) * key.n] for i in range(len(prompts))]


def _turn_stop(tok, prompt_len, turns=None, max_line_chars=None):
    """Per-sequence stopping criterion: a sequence is done once its generated text completes `turns`
    transcript turns, or its current line exceeds `max_line_chars` (a degenerate loop such as an endless
    list of objects). vLLM has no equivalent; there the text is only cut afterwards, at the turn limit."""
    import torch
    from transformers import StoppingCriteria

    class TurnStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            texts = tok.batch_decode(input_ids[:, prompt_len:], skip_special_tokens=True)
            return torch.tensor([bool(turns and t.count(TURN) >= turns)
                                 or bool(max_line_chars and len(t.rsplit("\n", 1)[-1]) > max_line_chars)
                                 for t in texts], device=input_ids.device)

    return TurnStop()


class VLLMLLM(BatchedLLM):
    """vLLM offline engine with multi-LoRA serving and prefix caching (AWS runs)."""

    def __init__(self, cfg, adapters: dict[str, str] | None = None):
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        m = cfg.model
        self.adapters = adapters or {}
        self.lora = {name: LoRARequest(name, i + 1, path) for i, (name, path) in enumerate(self.adapters.items())}
        self.engine = LLM(
            model=m.name, tokenizer=m.tokenizer or m.name, dtype=m.dtype, trust_remote_code=m.trust_remote_code,
            max_model_len=m.max_model_len, gpu_memory_utilization=m.gpu_memory_utilization,
            enable_prefix_caching=True, enable_lora=bool(self.adapters), max_lora_rank=m.max_lora_rank,
            max_loras=max(1, len(self.adapters)), seed=cfg.run.seed,
        )
        # vLLM schedules internally, so hand it everything that is pending at once.
        super().__init__(max_batch_size=4096, batch_wait_ms=m.batch_wait_ms,
                         chat_tokenizer=self.engine.get_tokenizer() if m.chat_template else None)

    def tokenizer(self):
        return self.engine.get_tokenizer()

    def _run(self, reqs):
        from vllm import SamplingParams

        params = [SamplingParams(n=r.key.n, temperature=r.key.temperature, max_tokens=r.key.max_tokens,
                                 stop=list(r.key.stop) or None) for r in reqs]
        loras = [self.lora[r.key.adapter] if r.key.adapter else None for r in reqs]
        results = self.engine.generate([r.prompt for r in reqs], params, lora_request=loras, use_tqdm=False)
        return [[o.text for o in res.outputs] for res in results]


def load_causal_lm(m, auto_cls=None):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    auto_cls = auto_cls or AutoModelForCausalLM
    dtype = getattr(torch, m.dtype)
    kwargs = dict(dtype=dtype, trust_remote_code=m.trust_remote_code, device_map={"": 0})
    if m.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True)
    return auto_cls.from_pretrained(m.name, **kwargs)


def merged_checkpoint(m, adapter_dir) -> str:
    """The base model with the LoRA adapter merged in, saved once next to the adapter. Merging is done in
    full precision; the result is quantized on load like the base model. (On the local actor this kept the
    SFT loss: base 0.139, base + adapter 0.126, merged and re-quantized 0.126.)"""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    out = Path(adapter_dir) / "merged"
    if not (out / "config.json").exists():
        log.info("merging %s into %s -> %s", adapter_dir, m.name, out)
        base = AutoModelForCausalLM.from_pretrained(m.name, dtype=getattr(torch, m.dtype),
                                                    trust_remote_code=m.trust_remote_code, device_map={"": "cpu"})
        PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload().save_pretrained(out)
    return str(out)


def build_llm(cfg, adapters: dict[str, str] | None = None) -> BatchedLLM:
    if cfg.model.backend == "vllm":
        return VLLMLLM(cfg, adapters)
    if cfg.model.backend == "hf":
        return HFLLM(cfg, adapters)
    raise ValueError(f"unknown backend {cfg.model.backend}")
