"""Prompts for the ALFWorld agents.

Actor prompts follow ReAct (Yao et al., 2022) with the 2-shot task-type-specific examples shown in
Supplementary §1.1; memory and reflection queries follow Reflexion (Shinn et al., 2023), extended
with a success variant (Sec. 4.1.2).
"""
import json
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / "data"

# From ReAct (see muse/data/THIRD_PARTY_LICENSES)
ACTOR_INSTRUCTION = "Interact with a household to solve a task. Here are two examples.\n"

# Terminal signal at the end of a trajectory, in Reflexion's log format. The World Model prompt shows
# it after each example, so imagined trajectories can end with a predicted episode outcome.
STATUS_OK = "STATUS: OK"
STATUS_FAIL = "STATUS: FAIL"
# Opening of every ReAct example's first step; prefilled so each episode starts with a plan P_e
PLAN_PREFIX = "think: To solve the task,"

# Reflexion's failure query, verbatim (see muse/data/THIRD_PARTY_LICENSES)
FAIL_REFLECTION = (
    "You will be given the history of a past experience in which you were placed in an environment "
    "and given a task to complete. You were unsuccessful in completing the task. Do not summarize your "
    "environment, but rather think about the strategy and path you took to attempt to complete the "
    "task. Devise a concise, new plan of action that accounts for your mistake with reference to "
    "specific actions that you should have taken. For example, if you tried A and B but forgot C, then "
    "devise a plan to achieve C with environment-specific actions. You will need this later when you "
    'are solving the same task. Give your plan after "Plan". Here are two examples:'
)

# Success queries with no examples, by agent.success_reflection. plan (the default): the decisive decisions
# and actions, distilled into a reusable plan naming the essential actions in order
SUCCESS_REFLECTION = (
    "You will be given the history of a past experience in which you were placed in an environment "
    "and given a task to complete. You were successful in completing the task. Do not summarize your "
    "environment. Instead, identify the specific decisions and actions that made the difference, and "
    "distill them into a concise, reusable plan for solving this task again. Name the concrete actions "
    "you took in order that were essential. You will need this later when you are solving the same task. "
    'Give your plan after "Plan".'
)

# plan_strategy: the strategy behind the steps that led to success
SUCCESS_REFLECTION_STRATEGY = (
    "You will be given the history of a past experience in which you were placed in an environment "
    "and given a task to complete. You were successful in completing the task. Think about the path "
    "that led to success. Summarize concisely the strategy underlying the steps that led to success, with "
    "reference to specific actions in the order you did them. You will "
    'need this later when you are solving the same task. Give your plan after "Plan".'
)

# plan_all_actions: every essential action, in order through the one that completed the task,
# worded as in the history and drawn only from it
SUCCESS_REFLECTION_ALL_ACTIONS = (
    "You will be given the history of a past experience in which you were placed in an environment "
    "and given a task to complete. You were successful in completing the task. Do not summarize your "
    "environment. Instead, identify the specific decisions and actions that made the difference, and "
    "distill them into a concise, reusable plan for solving this task again. List in order every "
    "essential action you took, from the first to the one that completed the task, including any action "
    "that a later action depended on. Write each action exactly as it appears in the history, and base "
    "the plan only on this history. You will need this later when you are solving the same task. "
    'Give your plan after "Plan".'
)

# plan_grounded: why the episode succeeded, the strategy and its reasoning, drawn only from what happened:
# the relevant observations and the actions that worked, none that failed or had no effect, and nothing
# absent from the history (the 1.5B reflector's plan-style reflections recorded rejected steps as facts)
SUCCESS_REFLECTION_GROUNDED = (
    "You will be given the history of a past experience in which you were placed in an environment "
    "and given a task to complete. You were successful in completing the task. Explain briefly why you "
    "succeeded: the strategy that worked and the reasoning behind it. Refer only to what actually happened "
    "in the history: state the relevant observations and which actions worked. Leave out any action that "
    "failed or had no effect, and do not mention anything that does not appear in the history. If you "
    "wasted steps, say what to skip next time. You will need this later when you are solving the same "
    'task. Give your plan after "Plan".'
)

# Reflexion's failure query rewritten for success, followed by its two (failure) examples
# (agent.success_reflection: reflexion; see muse/data/THIRD_PARTY_LICENSES)
SUCCESS_REFLECTION_REFLEXION = (
    "You will be given the history of a past experience in which you were placed in an environment "
    "and given a task to complete. You were successful in completing the task. Do not summarize your "
    "environment, but rather think about the strategy and path you took to complete the task. Devise a "
    "concise plan of action that keeps the steps that led to success and removes any redundant or "
    "unnecessary actions, with reference to specific environment actions. You will need this later "
    'when you are solving the same task. Give your plan after "Plan". Here are two examples of '
    "reflections on unsuccessful attempts:"
)


@lru_cache(maxsize=None)
def fewshot_prompts() -> dict:
    with open(DATA / "alfworld_3prompts.json") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def reflection_examples() -> str:
    return (DATA / "reflexion_few_shot_examples.txt").read_text()


def actor_base(task_type: str, terminal: bool = False) -> str:
    """ReAct's 2-shot prefix. terminal=True ends each (successful) example with STATUS_OK, for the
    World Model prompt."""
    d = fewshot_prompts()
    examples = [d[f"react_{task_type}_1"], d[f"react_{task_type}_0"]]
    if terminal:
        examples = [ex + STATUS_OK + "\n" for ex in examples]
    return ACTOR_INSTRUCTION + "".join(examples)


def memory_block(memory: list[dict]) -> str:
    """memory: [{'text': reflection, 'success': bool}, ...] (already truncated to max_memory)."""
    if not memory:
        return ""
    s = "\n\nYour memory for the task below:"
    for i, m in enumerate(memory):
        outcome = "SUCCESS" if m["success"] else "FAIL"
        s += f"\nTrial {i} ({outcome}):\n{m['text'].strip()}"
    return s


def actor_header(task_type: str, init_obs: str, memory: list[dict], fewshot: bool = True,
                 terminal: bool = False) -> str:
    base = actor_base(task_type, terminal) if fewshot else ACTOR_INSTRUCTION.split(" Here are")[0] + "\n"
    return base + memory_block(memory) + "\nHere is the task:\n" + init_obs + "\n>"


def render_steps(steps) -> str:
    """Continuation after the header: ' a1\\no1\\n> a2\\no2\\n>' (ReAct format)."""
    return "".join(f" {s.action}\n{s.observation}\n>" for s in steps)


def actor_prompt(task_type, init_obs, memory, steps, fewshot=True) -> str:
    return actor_header(task_type, init_obs, memory, fewshot) + render_steps(steps)


def world_model_prompt(task_type, init_obs, memory, steps) -> str:
    """M_w is the actor LLM prompted to continue the transcript with actions, observations and a
    terminal STATUS line (Sec. 4.1.3); only the examples differ from the actor prompt."""
    return actor_header(task_type, init_obs, memory, terminal=True) + render_steps(steps)


def actor_text_with_action_spans(task_type, init_obs, memory, steps, fewshot=True):
    """Full episode transcript plus character spans of the actor's outputs (for SFT loss masking).

    Each span covers ' <action>\\n', exactly what the actor generates after '>' at inference time.
    """
    text = actor_header(task_type, init_obs, memory, fewshot)
    spans = []
    for s in steps:
        start = len(text)
        text += f" {s.action}\n"
        spans.append((start, len(text)))
        text += f"{s.observation}\n>"
    return text, spans


def world_model_text_with_spans(task_type, init_obs, memory, steps):
    """World Model transcript (world_model_prompt's header) plus character spans of each action
    (' <action>\n') and each observation ('<observation>\n'), what a rollout generates."""
    text = actor_header(task_type, init_obs, memory, terminal=True)
    actions, observations = [], []
    for s in steps:
        start = len(text)
        text += f" {s.action}\n"
        actions.append((start, len(text)))
        start = len(text)
        text += f"{s.observation}\n"
        observations.append((start, len(text)))
        text += ">"
    return text, actions, observations


def scenario(init_obs: str, steps) -> str:
    lines = [init_obs] + [f"> {s.action}\n{s.observation}" for s in steps]
    return "\n".join(lines)


def reflection_query(init_obs, steps, success: bool, memory: list[dict], success_prompt: str = "plan") -> str:
    """success_prompt picks the success query: 'plan', 'plan_strategy', 'plan_all_actions' or 'plan_grounded' (no examples),
    or 'reflexion' (with Reflexion's)."""
    plans = {"plan": SUCCESS_REFLECTION, "plan_strategy": SUCCESS_REFLECTION_STRATEGY,
             "plan_all_actions": SUCCESS_REFLECTION_ALL_ACTIONS, "plan_grounded": SUCCESS_REFLECTION_GROUNDED}
    if not success:
        head = f"{FAIL_REFLECTION}\n\n{reflection_examples()}"
    elif success_prompt == "reflexion":
        head = f"{SUCCESS_REFLECTION_REFLEXION}\n\n{reflection_examples()}"
    elif success_prompt in plans:
        head = plans[success_prompt]
    else:
        raise ValueError(f"unknown success_prompt {success_prompt!r}")
    q = f"{head}\n\n{scenario(init_obs, steps)}\n{STATUS_OK if success else STATUS_FAIL}"
    if memory:
        q += "\n\nPlans from past attempts:\n"
        for i, m in enumerate(memory):
            outcome = "SUCCESS" if m["success"] else "FAIL"
            q += f"Trial #{i} ({outcome}): {m['text'].strip()}\n"
    q += "\n\nNew plan:"
    return q
