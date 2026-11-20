"""ReAct, Reflexion and MUSE agents for ALFWorld (Sec. 4.1)."""
import difflib
import random
from collections import Counter
from dataclasses import dataclass, field

from . import prompts
from .env import Task, to_game_command
from .self_assessment import SelfAssessor, actor_context_text, chunk_texts, format_pairs

ACTOR_STOP = ("\n",)
# Imagined trajectories continue the transcript; stop where it would leave the current episode.
# A predicted "STATUS: ..." line is kept: parse_rollout ends the trajectory there.
ROLLOUT_STOP = ("\n\n", "Here is the task", "Your task is to", "You are in the middle")


@dataclass
class Step:
    action: str
    observation: str
    info: dict = field(default_factory=dict)


@dataclass
class TaskState:
    """Per-task state that persists across episodes (long-term memory, per-task M_sa copy)."""
    task: Task
    memory: list = field(default_factory=list)      # [{'episode', 'success', 'text'}]
    sa: SelfAssessor | None = None


@dataclass
class EpisodeContext:
    task: Task
    init_obs: str
    task_desc: str
    memory: list                                     # memory window shown to the agent this episode
    steps: list = field(default_factory=list)
    admissible: list | None = None                   # the game's valid commands now (agent.admissible_filter)

    @property
    def plan(self) -> str:
        """Initial plan P_e: the first reasoning step of the episode ("think: To solve the task, ...")."""
        return first_think(self.steps)


def first_think(steps) -> str:
    for s in steps:
        if s.action.startswith("think:"):
            return s.action
    return ""


def clean_action(text: str) -> str:
    a = text.strip().split("\n")[0].strip()
    if a.startswith(">"):
        a = a[1:].strip()
    if a.lower().startswith("action:"):
        a = a[len("action:"):].strip()
    return a


def parse_rollout(text: str) -> tuple[list[Step], str | None]:
    """Parse an imagined continuation ' a1\\no1\\n> a2\\no2\\nSTATUS: OK' into steps and the predicted
    terminal signal (None if the rollout did not reach the end of the episode)."""
    lines = text.split("\n")
    steps, action, obs, status = [], clean_action(lines[0]), [], None
    for line in lines[1:]:
        if line.strip().startswith("STATUS:"):
            status = prompts.STATUS_OK if "OK" in line else prompts.STATUS_FAIL
            break
        if line.startswith(">"):
            steps.append(Step(action, " ".join(obs).strip()))
            action, obs = clean_action(line), []
        elif line.strip():
            obs.append(line.strip())
    steps.append(Step(action, " ".join(obs).strip()))
    return [s for s in steps if s.action], status


def admissible(action: str, commands: list) -> bool:
    """Whether the environment would accept `action`: think steps are not game commands."""
    return action.startswith("think:") or to_game_command(action) in set(commands)


class ReActAgent:
    """ReAct (Yao et al., 2022): 2-shot reason+act prompting, context reset every episode."""

    name = "react"
    uses_memory = False
    uses_sa = False

    def __init__(self, cfg, llm, actor_adapter=None, reflector_adapter=None):
        self.cfg = cfg
        self.llm = llm
        self.actor_adapter = actor_adapter
        self.reflector_adapter = reflector_adapter

    def new_task_state(self, task: Task) -> TaskState:
        return TaskState(task=task)

    def task_state(self, state: TaskState) -> dict:
        """What a task needs to continue after a crash, besides its episode records (picklable).
        Randomness comes from the process-wide generators seeded once from run.seed, so none is kept."""
        return {"memory": list(state.memory)}

    def restore_task_state(self, state: TaskState, saved: dict):
        state.memory = list(saved["memory"])

    def memory_window(self, state: TaskState) -> list:
        """The reflections the actor and reflector see: the last max_memory; of the failure reflections,
        only the last max_failure_memory; with successes_last, the failures first and the successes after
        them (each in episode order), nearest the end of the prompt. Then, from the front, dropped until
        the memory block fits max_memory_tokens LLM tokens."""
        if not self.uses_memory or not state.memory:
            return []
        a = self.cfg.agent
        window = state.memory[-a.max_memory:] if a.max_memory else list(state.memory)
        if a.max_failure_memory is not None:
            fails = [m for m in window if not m["success"]]
            keep = {id(m) for m in fails[len(fails) - a.max_failure_memory:]} if a.max_failure_memory else set()
            window = [m for m in window if m["success"] or id(m) in keep]
        if a.successes_last:
            window = [m for m in window if not m["success"]] + [m for m in window if m["success"]]
        if a.max_memory_tokens:
            while window and self.llm.num_tokens(prompts.memory_block(window)) > a.max_memory_tokens:
                window = window[1:]
        return window

    def prompt(self, ctx: EpisodeContext) -> str:
        return prompts.actor_prompt(ctx.task.task_type, ctx.init_obs, ctx.memory, ctx.steps)

    def plan_prefix(self, ctx: EpisodeContext) -> str:
        """Prefilled start of the first step, so that every episode opens with a plan."""
        return prompts.PLAN_PREFIX if self.cfg.agent.force_plan and not ctx.steps else ""

    def act(self, ctx: EpisodeContext, state: TaskState) -> tuple[str, dict]:
        a = self.cfg.agent
        if a.action_votes > 1:
            return self._vote(ctx)
        action, info = self._sample_action(ctx)
        if not a.admissible_filter or ctx.admissible is None:
            return action, info
        # extension: resample an action the environment would reject, as MUSE drops such rollouts;
        # if every sample is rejected, keep the first
        first, resamples = (action, info), 0
        while not admissible(action, ctx.admissible) and resamples < a.admissible_resamples:
            (action, info), resamples = self._sample_action(ctx), resamples + 1
        if not admissible(action, ctx.admissible):
            action, info = first
        return action, {**info, "inadmissible_resamples": resamples}

    def _vote(self, ctx: EpisodeContext) -> tuple[str, dict]:
        """Extension (agent.action_votes): sample N actions in one call, drop those the environment would
        reject (admissible_filter; all kept if every one is), and take the most frequent. Think steps count
        as one choice; ties go to the earliest sample. No samples left: one action as without voting."""
        a = self.cfg.agent
        prefix = self.plan_prefix(ctx)
        prompt = self.prompt(ctx) + (" " + prefix if prefix else "")
        outs = self.llm.generate(prompt, n=a.action_votes, temperature=a.actor_temperature,
                                 max_tokens=a.actor_max_tokens, stop=ACTOR_STOP, adapter=self.actor_adapter)
        cands = [c for c in (clean_action(prefix + o) for o in outs if len(o.strip()) >= a.actor_min_chars) if c]
        if not cands:
            return self._sample_action(ctx)
        dropped = 0
        if a.admissible_filter and ctx.admissible is not None:
            kept = [c for c in cands if admissible(c, ctx.admissible)]
            dropped = len(cands) - len(kept) if kept else 0
            cands = kept or cands
        key = [("think" if c.startswith("think:") else to_game_command(c)) for c in cands]
        counts = Counter(key)
        top = max(counts.values())
        action = cands[next(i for i, k in enumerate(key) if counts[k] == top)]
        return action, {"votes": top, "candidates": len(cands), "dropped_inadmissible": dropped}

    def _sample_action(self, ctx: EpisodeContext) -> tuple[str, dict]:
        a = self.cfg.agent
        prefix = self.plan_prefix(ctx)
        prompt = self.prompt(ctx) + (" " + prefix if prefix else "")
        # Reflexion's +0.2 per retry, added to the actor temperature (muse/data/THIRD_PARTY_LICENSES)
        for attempt in range(a.actor_retries + 1):
            temp = a.actor_temperature + 0.2 * attempt
            out = self.llm.generate(prompt, n=1, temperature=temp, max_tokens=a.actor_max_tokens,
                                    stop=ACTOR_STOP, adapter=self.actor_adapter)[0]
            action = clean_action(prefix + out) if len(out.strip()) >= a.actor_min_chars else ""
            if action:
                return action, {}
        return "look", {"empty_output": True}

    def reflect(self, state: TaskState, ctx: EpisodeContext, success: bool) -> dict | None:
        return None

    def after_adaptation(self, state: TaskState, episodes: list[dict]) -> dict | None:
        return None


def is_copy(text: str, earlier: list[str], ratio=0.9) -> bool:
    """Whether a reflection repeats an earlier one exactly or nearly (difflib similarity > ratio)."""
    t = text.strip()
    return any(t == e.strip() or difflib.SequenceMatcher(None, t, e.strip()).ratio() > ratio for e in earlier)


class ReflexionAgent(ReActAgent):
    """Reflexion (Shinn et al., 2023) with separate actor/reflector models and success reflections."""

    name = "reflexion"
    uses_memory = True

    def reflect(self, state, ctx, success):
        a = self.cfg.agent
        if success and not a.reflect_on_success:
            return None
        query = prompts.reflection_query(ctx.init_obs, ctx.steps, success, ctx.memory,
                                         success_prompt=a.success_reflection)
        def sample(temperature):
            return self.llm.generate(query, n=1, temperature=temperature, max_tokens=a.reflection_max_tokens,
                                     stop=("\n\n",), adapter=self.reflector_adapter)[0]

        text, resamples = sample(a.reflection_temperature), 0
        earlier = [m["text"] for m in state.memory]
        while resamples < a.reflection_resamples and is_copy(text, earlier):
            text, resamples = sample(a.reflection_resample_temperature), resamples + 1
        out = {"query": query, "text": text}
        if resamples:
            out["resamples"] = resamples
        return out


class MuseAgent(ReflexionAgent):
    """MUSE: World Model/Actor rollouts, scored by M_sa, with competence-aware action selection."""

    name = "muse"
    uses_sa = True

    def __init__(self, cfg, llm, sa: SelfAssessor, actor_adapter=None, reflector_adapter=None,
                 world_model_adapter=None):
        super().__init__(cfg, llm, actor_adapter, reflector_adapter)
        self.sa_template = sa
        self.selection = cfg.muse.selection
        self.actor_sa = cfg.self_assessment.input == "actor"     # extension: M_sa on the actor's context
        # M_w uses the same LLM as M_a (Sec. 4.1.3)
        self.world_model_adapter = world_model_adapter if world_model_adapter is not None else actor_adapter

    def new_task_state(self, task):
        """Novel tasks are evaluated in isolation: each starts from the end-of-pre-deployment M_sa
        (g_eta and replay buffer), so No Adaptation episodes use those weights and one task's
        update never reaches another."""
        state = super().new_task_state(task)
        state.sa = self.sa_template.clone()
        return state

    def task_state(self, state):
        """Adds this task's g_eta, which differs from the template once adaptation has run."""
        return {**super().task_state(state),
                "sa_mlp": {k: v.detach().cpu().clone() for k, v in state.sa.mlp.state_dict().items()}}

    def restore_task_state(self, state, saved):
        super().restore_task_state(state, saved)
        state.sa.mlp.load_state_dict(saved["sa_mlp"])

    def act(self, ctx, state):
        """Every action, the first included, is the first action of the best of N imagined rollouts."""
        m = self.cfg.muse
        prefix = self.plan_prefix(ctx)     # the first step's rollouts are alternative plans
        prompt = prompts.world_model_prompt(ctx.task.task_type, ctx.init_obs, ctx.memory, ctx.steps)
        outs = self.llm.generate(prompt + (" " + prefix if prefix else ""), n=m.num_rollouts,
                                 temperature=m.rollout_temperature, max_tokens=m.rollout_max_tokens,
                                 stop=ROLLOUT_STOP, adapter=self.world_model_adapter,
                                 max_turns=m.chunk_size, max_line_chars=m.rollout_max_line_chars)
        rollouts = [(r, st) for r, st in (parse_rollout(prefix + o) for o in outs if o.strip()) if r]
        dropped = 0
        filtering = self.cfg.agent.admissible_filter and ctx.admissible
        if filtering:
            # extension: drop rollouts whose first action the environment would reject. If none is
            # left, keep them all rather than fall back
            kept = [x for x in rollouts if admissible(x[0][0].action, ctx.admissible)]
            dropped = len(rollouts) - len(kept)
            if kept:
                rollouts = kept
            else:
                dropped = 0
        if not rollouts:     # act as the Reflexion + SL actor (same trained adapter), with its decoding
            action, info = super().act(ctx, state)
            return action, {**info, "fallback": True}
        plans = [ctx.plan or first_think(r) for r, _ in rollouts]
        # Each rollout looks chunk_size (4) steps ahead: anything generated past that is dropped, and a
        # predicted terminal signal counts only if it falls within those steps
        texts = [format_pairs([(s.action, s.observation) for s in r[:m.chunk_size]],
                              st if len(r) <= m.chunk_size else None) for r, st in rollouts]
        conts = None
        if self.actor_sa:
            # each rollout's steps as the transcript would show them, after the prompt (no STATUS line)
            conts = [prompts.render_steps(r[:m.chunk_size]) for r, _ in rollouts]
            spans = [(len(prompt), len(prompt) + len(c) - 2) for c in conts]
            scores = state.sa.predict_features(
                self.llm.span_features(prompt, spans, self.world_model_adapter, continuations=conts,
                                       layer=self.cfg.self_assessment.layer))
            truncated = [False] * len(texts)
        else:
            scores = state.sa.score(ctx.task_desc, plans, texts)
            truncated = state.sa.truncated(plans, texts)
        if self.selection == "best":
            idx = int(scores.argmax())
        elif self.selection == "lowest":
            idx = int(scores.argmin())
        elif self.selection == "random":
            idx = random.randrange(len(rollouts))     # the global generator, seeded from run.seed
        else:
            raise ValueError(self.selection)
        # plan and continuation are what M_sa scored, kept so after_adaptation can train on the rollouts
        info = {"candidates": [{"rollout": t, "score": float(s), "truncated": tr, "plan": p,
                                **({"continuation": conts[i]} if conts else {})}
                               for i, (t, s, tr, p) in enumerate(zip(texts, scores, truncated, plans))],
                "selected": idx, **({"dropped_inadmissible": dropped} if self.cfg.agent.admissible_filter else {})}
        return rollouts[idx][0][0].action, info

    def _features(self, sa, episodes):
        import torch

        from .datasets import sa_examples
        size, stride = self.cfg.muse.chunk_size, self.cfg.self_assessment.chunk_stride
        d = sa_examples(episodes, size, stride)
        if not d["texts"]:
            return None, None
        if self.actor_sa:
            x = torch.cat([self._chunk_features(ep["task_type"], ep["init_obs"], ep["memory"],
                                                [Step(s["action"], s["observation"]) for s in ep["steps"]], stride)
                           for ep in episodes if ep["steps"]])
        else:
            x = sa.features(d["descs"], d["plans"], d["texts"])
        return x, torch.tensor(d["labels"])

    def _chunk_features(self, task_type, init_obs, memory, steps, stride=None):
        text, spans = actor_context_text(task_type, init_obs, memory, steps, self.cfg.muse.chunk_size, stride)
        return self.llm.span_features(text, spans, self.world_model_adapter, layer=self.cfg.self_assessment.layer)

    def _rollout_features(self, sa, episodes):
        """Extension (self_assessment.adapt_rollouts): M_sa's input for the imagined rollouts of the
        adaptation episodes, each whole 4-step rollout (imagined observations included) exactly as it was
        scored, labelled with its episode's outcome. selected: the rollout whose first action was taken;
        all: every candidate. M_sa is trained on executed chunks but scores imagined ones."""
        import torch

        which = self.cfg.self_assessment.adapt_rollouts
        if which not in ("none", "selected", "all"):
            raise ValueError(f"self_assessment.adapt_rollouts {which!r}: none | selected | all")
        if which == "none":
            return None, None
        xs, ys = [], []
        for ep in episodes:
            steps = [Step(s["action"], s["observation"]) for s in ep["steps"]]
            for i, s in enumerate(ep["steps"]):
                info = s.get("info") or {}
                cands = info.get("candidates")
                if not cands or info.get("fallback"):
                    continue
                cands = [cands[info["selected"]]] if which == "selected" else cands
                if self.actor_sa:
                    prompt = prompts.world_model_prompt(ep["task_type"], ep["init_obs"], ep["memory"], steps[:i])
                    conts = [c["continuation"] for c in cands]
                    spans = [(len(prompt), len(prompt) + len(c) - 2) for c in conts]
                    x = self.llm.span_features(prompt, spans, self.world_model_adapter, continuations=conts,
                                               layer=self.cfg.self_assessment.layer)
                else:
                    x = sa.features([ep["task_desc"]] * len(cands), [c["plan"] for c in cands],
                                    [c["rollout"] for c in cands])
                xs.append(x)
                ys += [float(ep["success"])] * len(cands)
        return (torch.cat(xs), torch.tensor(ys)) if xs else (None, None)

    def after_adaptation(self, state, episodes):
        """Update only the MLP g_eta of this task's M_sa with its adaptation episodes plus replay
        (Sec. 4.2): their executed chunks and, with self_assessment.adapt_rollouts, their imagined rollouts."""
        import torch

        x, y = self._features(state.sa, episodes)
        xr, yr = self._rollout_features(state.sa, episodes)
        n_rollouts = 0 if xr is None else len(xr)
        if xr is not None:
            x, y = (xr, yr) if x is None else (torch.cat([x, xr]), torch.cat([y.float(), yr]))
        if x is None:
            return None
        return {**state.sa.adapt(x, y), "n_rollouts": n_rollouts}

    def assess_episode(self, state, ctx) -> list[dict]:
        """Chunk-wise self-assessment of an executed episode (for metacognitive accuracy/AUROC2).
        No terminal signal is appended here: it would reveal the outcome being predicted."""
        texts = chunk_texts(ctx.steps, self.cfg.muse.chunk_size)
        if not texts:
            return []
        if self.actor_sa:
            preds = state.sa.predict_features(self._chunk_features(ctx.task.task_type, ctx.init_obs, ctx.memory,
                                                                   ctx.steps))
            truncated = [False] * len(texts)
        else:
            preds = state.sa.score(ctx.task_desc, ctx.plan, texts)
            truncated = state.sa.truncated(ctx.plan, texts)
        return [{"text": t, "pred": float(p), "truncated": tr} for t, p, tr in zip(texts, preds, truncated)]
