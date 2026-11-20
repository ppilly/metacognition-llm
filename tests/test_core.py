"""Fast tests of the protocol, parsing, data building and metrics with a fake LLM and environment."""
import json

import numpy as np
import pytest
import torch

from muse import prompts
from muse.agents import ROLLOUT_STOP, EpisodeContext, MuseAgent, ReActAgent, ReflexionAgent, Step, parse_rollout
from muse.config import load_config
from muse.datasets import dpo_pairs, sa_examples, sft_examples, split_by_task
from muse.env import Task, from_game_feedback, process_obs, task_description, to_game_command
from muse.metrics import condition_episodes, self_assessment, success_rate, summarize, time_to_completion
from muse import runner
from muse.runner import load_episodes, run_task, run_tasks
from muse.self_assessment import MLP, ReplayBuffer, SelfAssessor, chunk_steps, format_pairs, sa_input_text

INIT_OBS = ("You are in the middle of a room. Looking quickly around you, you see a cabinet 1, and a sinkbasin 1.\n"
            "Your task is to: put a clean bowl in cabinet.")
TASK = Task("pick_clean_then_place_in_recep-Bowl-None-Cabinet-1/trial_1", "unused", "clean")


@pytest.fixture
def cfg():
    c = load_config("configs/default.yaml", ["env.max_steps=6", "agent.max_memory=2", "agent.force_plan=false"])
    return c


class FakeEnv:
    """Wins when the agent puts the bowl in the cabinet."""

    def reset(self, task):
        return INIT_OBS

    def step(self, action):
        if action == "put bowl 1 in/on cabinet 1":
            return "You put the bowl 1 in/on the cabinet 1.", True
        return "Nothing happens.", False


class ScriptedLLM:
    """Returns the scripted action for actor calls (stop='\\n'), rollouts for n>1, text otherwise."""

    def __init__(self, actions, rollouts=None):
        self.actions = list(actions)
        self.rollouts = rollouts
        self.calls = []

    def generate(self, prompt, *, n=1, temperature=0.0, max_tokens=128, stop=(), adapter=None, **limits):
        self.calls.append({"prompt": prompt, "n": n, "temperature": temperature, "adapter": adapter, "stop": stop,
                           **limits})
        if n > 1:
            return list(self.rollouts)
        if stop == ("\n",):
            return [" " + self.actions.pop(0)] if self.actions else [" look"]
        return [" I should put the bowl in the cabinet."]


class StubSA:
    def __init__(self, scores):
        self.scores = np.array(scores)
        self.adapted = None
        self.feature_inputs = []
        self.mlp = torch.nn.Linear(2, 1)

    def clone(self):
        return self

    def score(self, task_desc, plans, texts):
        return self.scores[:len(texts)]

    def truncated(self, plans, texts):
        return [False] * len(texts)

    def features(self, descs, plans, texts):
        self.feature_inputs.append((plans, texts))
        return torch.zeros(len(texts), 2)

    def predict_features(self, x):
        return self.scores[:len(x)]

    def adapt(self, x, y):
        self.adapted = (x, y)
        torch.nn.init.constant_(self.mlp.weight, 1.0)
        return {"n_new": len(y)}


def test_prompt_format_matches_react():
    steps = [Step("think: To solve the task, I need to find a bowl.", "OK."), Step("go to cabinet 1", "Closed.")]
    p = prompts.actor_prompt("clean", INIT_OBS, [], steps)
    assert p.startswith("Interact with a household to solve a task. Here are two examples.\nYou are in")
    assert p.endswith("Here is the task:\n" + INIT_OBS + "\n> think: To solve the task, I need to find a bowl.\nOK.\n"
                      "> go to cabinet 1\nClosed.\n>")
    text, spans = prompts.actor_text_with_action_spans("clean", INIT_OBS, [], steps)
    assert [text[a:b] for a, b in spans] == [" think: To solve the task, I need to find a bowl.\n", " go to cabinet 1\n"]
    assert p.startswith(text[: len(text) - 1])


def test_memory_block_includes_outcome():
    p = prompts.actor_prompt("clean", INIT_OBS, [{"text": " plan A ", "success": False}], [])
    assert "Your memory for the task below:\nTrial 0 (FAIL):\nplan A\nHere is the task:" in p


def test_parse_rollout_and_chunk_format():
    text = (" go to cabinet 1\nThe cabinet 1 is closed.\n> open cabinet 1\nYou open the cabinet 1.\n"
            "> think: Now I find a bowl (1).\nOK.\n> take bowl 1 from cabinet 1\nYou pick\n> go to")
    r, status = parse_rollout(text)
    assert status is None
    assert [s.action for s in r] == ["go to cabinet 1", "open cabinet 1", "think: Now I find a bowl (1).",
                                     "take bowl 1 from cabinet 1", "go to"]
    assert r[0].observation == "The cabinet 1 is closed."
    assert format_pairs([(s.action, s.observation) for s in r[:2]]) == \
        "go to cabinet 1.The cabinet 1 is closed..> open cabinet 1.You open the cabinet 1.."
    assert [len(c) for c in chunk_steps(r * 2, 4)] == [4, 4, 2]


def test_rollout_terminal_signal():
    r, status = parse_rollout(" put bowl 1 in/on cabinet 1\nYou put the bowl 1 in/on the cabinet 1.\nSTATUS: OK\n> look")
    assert [s.action for s in r] == ["put bowl 1 in/on cabinet 1"] and status == "STATUS: OK"
    assert format_pairs([("a", "o")], status) == "a.o.> STATUS: OK"
    p = prompts.world_model_prompt("clean", INIT_OBS, [], [])
    assert p.count("STATUS: OK\n") == 2 and p.endswith("Here is the task:\n" + INIT_OBS + "\n>")
    assert "STATUS" not in prompts.actor_prompt("clean", INIT_OBS, [], [])


def test_process_obs_strips_arrival():
    assert process_obs("You arrive at drawer 1. The drawer 1 is closed.") == "The drawer 1 is closed."
    assert process_obs("You arrive at loc 5. On the desk 1, you see a pen 1.") == "On the desk 1, you see a pen 1."
    assert process_obs("You arrive at drawer 1.") == "You arrive at drawer 1."


def test_put_command_compat():
    assert to_game_command("put book 3 in/on sidetable 1") == "move book 3 to sidetable 1"
    assert to_game_command("put watch 1 in safe 1") == "put watch 1 in safe 1"      # invalid, as originally
    assert to_game_command("take book 3 from diningtable 1") == "take book 3 from diningtable 1"
    assert from_game_feedback("You move the book 3 to the sidetable 1.") == "You put the book 3 in/on the sidetable 1."


def test_task_description():
    assert task_description(INIT_OBS) == "put a clean bowl in cabinet."


def test_react_episode_and_protocol(cfg):
    llm = ScriptedLLM(["think: plan", "go to cabinet 1", "put bowl 1 in/on cabinet 1"] * 10)
    agent = ReActAgent(cfg, llm)
    recs = run_task(FakeEnv(), agent, TASK, cfg, n_adapt=2, n_test=2)
    assert [r["phase"] for r in recs] == ["noadapt", "noadapt", "adapt", "adapt", "test", "test"]
    assert all(r["success"] and r["num_steps"] == 3 and not r["early_stop"] for r in recs)
    assert recs[0]["steps"][0]["observation"] == "OK."   # think actions are not sent to the env
    assert recs[0]["plan"] == "think: plan"
    assert all(r["reflection"] is None for r in recs)


def test_reflexion_memory_accumulates_and_is_windowed(cfg):
    cfg.agent.stop_on_repeat = False
    llm = ScriptedLLM(["look"] * 100)          # always fails
    agent = ReflexionAgent(cfg, llm)
    recs = run_task(FakeEnv(), agent, TASK, cfg, n_adapt=3, n_test=1)
    # No Adaptation episode: no memory, no reflection; then memory accumulates (max_memory=2)
    assert [len(r["memory"]) for r in recs] == [0, 0, 1, 2, 2]
    assert recs[0]["reflection"] is None and recs[-1]["reflection"] is None
    assert all(not r["success"] and r["num_steps"] == 6 for r in recs)
    assert "STATUS: FAIL" in recs[1]["reflection"]["query"]
    assert "Trial #0 (FAIL)" in recs[2]["reflection"]["query"]    # reflector sees prior reflections + outcomes


def test_independent_test_episodes_see_only_adaptation_reflections(cfg):
    cfg.agent.stop_on_repeat = False
    cfg.agent.max_memory = None
    assert cfg.protocol.test_independent
    recs = run_task(FakeEnv(), ReflexionAgent(cfg, ScriptedLLM(["look"] * 200)), TASK, cfg, n_adapt=3, n_test=3)
    assert [r["phase"] for r in recs] == ["noadapt"] * 3 + ["adapt"] * 3 + ["test"] * 3
    assert [len(r["memory"]) for r in recs] == [0, 0, 0, 0, 1, 2, 3, 3, 3]
    assert all(r["reflection"] for r in recs if r["phase"] == "adapt")
    assert all(r["reflection"] is None for r in recs if r["phase"] != "adapt")
    # As a sequence (the local runs), test episodes keep reflecting and the memory keeps growing
    cfg.protocol.test_independent = False
    recs = run_task(FakeEnv(), ReflexionAgent(cfg, ScriptedLLM(["look"] * 200)), TASK, cfg, n_adapt=3, n_test=3)
    assert [len(r["memory"]) for r in recs] == [0, 0, 0, 0, 1, 2, 3, 4, 5]


def test_skip_noadapt(cfg):
    cfg.agent.stop_on_repeat = False
    cfg.protocol.skip_noadapt = True
    recs = run_task(FakeEnv(), ReflexionAgent(cfg, ScriptedLLM(["look"] * 200)), TASK, cfg, n_adapt=2, n_test=2)
    assert [r["phase"] for r in recs] == ["adapt"] * 2 + ["test"] * 2


def test_success_reflection_prompt(cfg):
    cfg.agent.stop_on_repeat = False
    put = "put bowl 1 in/on cabinet 1"
    examples = prompts.reflection_examples()
    for prompt, shown in (("reflexion", True), ("plan", False)):
        cfg.agent.success_reflection = prompt
        script = ["look"] * 36 + [put]         # adapt episodes: fail, success, then any
        recs = run_task(FakeEnv(), ReflexionAgent(cfg, ScriptedLLM(script)), TASK, cfg, n_adapt=3, n_test=0)
        fail, succ = recs[0]["reflection"]["query"], recs[1]["reflection"]["query"]
        assert [r["success"] for r in recs[:2]] == [False, True]
        assert examples in fail                                    # failure queries always show them
        assert (examples in succ) == shown and ("unsuccessful attempts" in succ) == shown
        assert succ.endswith("New plan:") and "STATUS: OK" in succ and "Trial #0 (FAIL)" in succ
        assert ("decisions and actions that made the difference" in succ) == (not shown)
        assert ("think about the strategy and path" in succ) == shown
    cfg.agent.success_reflection = "fewshot"
    with pytest.raises(ValueError, match="fewshot"):
        prompts.reflection_query(INIT_OBS, [], True, [], success_prompt=cfg.agent.success_reflection)
    assert load_config("configs/local.yaml").agent.success_reflection == "plan"
    for prompt, words in (("plan_strategy", "strategy underlying the steps"),
                          ("plan", "Name the concrete actions"),
                          ("plan_all_actions", "exactly as it appears in the history")):
        assert words in prompts.reflection_query(INIT_OBS, [], True, [], success_prompt=prompt)
    grounded = prompts.reflection_query(INIT_OBS, [], True, [], success_prompt="plan_grounded")
    assert "Refer only to what actually happened" in grounded and examples not in grounded
    assert "Nothing happens" not in grounded.split("Here is the task")[0]      # generic, no ALFWorld wording


def test_actor_decoding_follows_reflexion(cfg):
    # The default samples at 0.5, then +0.2 per retry until the output has 5 characters ("look" has 4)
    llm = ScriptedLLM(["look", "go to cabinet 1"])
    ReActAgent(cfg, llm).act(EpisodeContext(TASK, INIT_OBS, "", []), ReActAgent(cfg, llm).new_task_state(TASK))
    assert [c["temperature"] for c in llm.calls] == pytest.approx([0.5, 0.7])
    # Reflexion's greedy start
    cfg.agent.actor_temperature = 0.0
    llm = ScriptedLLM(["go", "look", "go to cabinet 1"])
    agent = ReActAgent(cfg, llm)
    action, info = agent.act(EpisodeContext(TASK, INIT_OBS, "", []), agent.new_task_state(TASK))
    assert action == "go to cabinet 1" and info == {}
    assert [c["temperature"] for c in llm.calls] == pytest.approx([0.0, 0.2, 0.4])
    action, info = agent.act(EpisodeContext(TASK, INIT_OBS, "", []), agent.new_task_state(TASK))
    assert (action, info) == ("look", {"empty_output": True}) and len(llm.calls) == 3 + 6
    # MUSE falls back to the Reflexion actor, greedy, when no rollout parses
    llm = ScriptedLLM(["go to cabinet 1"], rollouts=[" ", ""])
    muse = MuseAgent(cfg, llm, StubSA([0.5]), actor_adapter="actor")
    action, info = muse.act(EpisodeContext(TASK, INIT_OBS, "", [], [Step("go to sinkbasin 1", "OK.")]),
                            muse.new_task_state(TASK))
    assert action == "go to cabinet 1" and info == {"fallback": True}
    assert llm.calls[0]["temperature"] == cfg.muse.rollout_temperature == 0.5
    assert llm.calls[1]["temperature"] == 0.0 and llm.calls[1]["adapter"] == "actor"
    # The earlier local runs keep their settings
    local = load_config("configs/local.yaml").agent
    assert (local.actor_temperature, local.actor_max_tokens, local.actor_min_chars) == (0.5, 256, 5)


def test_repeated_action_ends_episode(cfg):
    agent = ReActAgent(cfg, ScriptedLLM(["go to cabinet 1", "look", "look", "look"]))
    recs = run_task(FakeEnv(), agent, TASK, cfg, n_adapt=1, n_test=0)
    assert recs[0]["num_steps"] == 3 and recs[0]["early_stop"] and not recs[0]["success"]


def test_muse_selects_best_and_adapts_once(cfg):
    cfg.self_assessment.adapt_rollouts = "selected"      # extension, as in configs/local.yaml
    rollouts = [" go to sinkbasin 1\nOn the sinkbasin 1.", " put bowl 1 in/on cabinet 1\nYou put", " look\nYou are"]
    sa = StubSA([0.2, 0.9, 0.5])
    agent = MuseAgent(cfg, ScriptedLLM([], rollouts), sa, actor_adapter="actor", reflector_adapter="reflector")
    recs = run_task(FakeEnv(), agent, TASK, cfg, n_adapt=1, n_test=1)
    assert [r["phase"] for r in recs] == ["noadapt", "adapt", "test"]
    assert all(r["success"] and r["num_steps"] == 1 for r in recs)
    assert recs[0]["steps"][0]["info"]["selected"] == 1
    call = agent.llm.calls[0]
    assert call["n"] == cfg.muse.num_rollouts and call["adapter"] == "actor" and "STATUS: OK" in call["prompt"]
    assert call["max_turns"] == cfg.muse.chunk_size and call["max_line_chars"] == cfg.muse.rollout_max_line_chars
    # adaptation episode only: its executed chunk, then its imagined rollout, whole, as it was scored
    assert sa.adapted is not None and sa.adapted[1].tolist() == [1.0, 1.0]
    assert recs[1]["sa_adaptation"] == {"n_new": 2, "n_rollouts": 1}
    chosen = recs[1]["steps"][0]["info"]["candidates"][1]
    assert sa.feature_inputs[-1] == ([chosen["plan"]], [chosen["rollout"]])
    assert chosen["rollout"] == "put bowl 1 in/on cabinet 1.You put."
    cfg.self_assessment.adapt_rollouts = "all"
    sa = StubSA([0.2, 0.9, 0.5])
    recs = run_task(FakeEnv(), MuseAgent(cfg, ScriptedLLM([], rollouts), sa), TASK, cfg, n_adapt=1, n_test=1)
    assert recs[1]["sa_adaptation"]["n_rollouts"] == 3
    cfg.self_assessment.adapt_rollouts = "none"
    sa = StubSA([0.2, 0.9, 0.5])
    recs = run_task(FakeEnv(), MuseAgent(cfg, ScriptedLLM([], rollouts), sa), TASK, cfg, n_adapt=1, n_test=1)
    assert recs[1]["sa_adaptation"] == {"n_new": 1, "n_rollouts": 0}
    assert recs[0]["sa_chunks"][0]["pred"] == pytest.approx(0.2)

    cfg.muse.selection = "lowest"
    agent = MuseAgent(cfg, ScriptedLLM([], rollouts), StubSA([0.2, 0.9, 0.5]))
    recs = run_task(FakeEnv(), agent, TASK, cfg, n_adapt=1, n_test=0)
    assert recs[0]["steps"][0]["action"] == "go to sinkbasin 1" and not recs[0]["success"]


def test_muse_admissible_filter_drops_rejected_first_actions(cfg):
    class AdmissibleEnv(FakeEnv):
        admissible = ["go to sinkbasin 1", "move bowl 1 to cabinet 1"]

    rollouts = [" go to countertop 2\nOn the countertop 2.", " go to sinkbasin 1\nOn the sinkbasin 1.",
                " put bowl 1 in/on cabinet 1\nYou put\nSTATUS: OK"]
    cfg.agent.admissible_filter = True
    agent = MuseAgent(cfg, ScriptedLLM([], rollouts), StubSA([0.9, 0.2, 0.5]))   # scores the kept rollouts
    recs = run_task(AdmissibleEnv(), agent, TASK, cfg, n_adapt=1, n_test=0)
    info = recs[0]["steps"][0]["info"]
    assert info["dropped_inadmissible"] == 1 and len(info["candidates"]) == 2
    assert recs[0]["steps"][0]["action"] == "go to sinkbasin 1"         # countertop 2 is not in the room
    cfg.agent.admissible_filter = False
    recs = run_task(AdmissibleEnv(), MuseAgent(cfg, ScriptedLLM([], rollouts), StubSA([0.9, 0.2, 0.5])),
                    TASK, cfg, n_adapt=1, n_test=0)
    assert recs[0]["steps"][0]["action"] == "go to countertop 2" and "dropped_inadmissible" not in recs[0]["steps"][0]["info"]


def test_reflexion_admissible_filter_resamples_rejected_actions(cfg):
    class AdmissibleEnv(FakeEnv):
        admissible = ["go to sinkbasin 1"]

    cfg.agent.stop_on_repeat = False
    cfg.agent.admissible_filter = True
    script = ["think: I need a bowl.", "go to countertop 2", "go to countertop 3", "go to sinkbasin 1"]
    recs = run_task(AdmissibleEnv(), ReflexionAgent(cfg, ScriptedLLM(script)), TASK, cfg, n_adapt=1, n_test=0)
    steps = recs[0]["steps"]
    assert steps[0]["action"].startswith("think:") and steps[0]["info"]["inadmissible_resamples"] == 0
    assert steps[1]["action"] == "go to sinkbasin 1" and steps[1]["info"]["inadmissible_resamples"] == 2
    cfg.agent.admissible_resamples = 1      # every sample rejected: the first is kept
    recs = run_task(AdmissibleEnv(), ReflexionAgent(cfg, ScriptedLLM(script)), TASK, cfg, n_adapt=1, n_test=0)
    assert recs[0]["steps"][1]["action"] == "go to countertop 2"
    assert recs[0]["steps"][1]["info"]["inadmissible_resamples"] == 1


def test_react_action_votes_takes_majority_of_admissible_samples(cfg):
    class AdmissibleEnv(FakeEnv):
        admissible = ["go to sinkbasin 1", "go to cabinet 1"]

    cfg.agent.stop_on_repeat = False
    cfg.agent.admissible_filter = True
    cfg.agent.action_votes = 5
    votes = [" go to countertop 2", " go to cabinet 1", " go to sinkbasin 1", " go to countertop 2", " go to sinkbasin 1"]
    llm = ScriptedLLM([], votes)
    recs = run_task(AdmissibleEnv(), ReActAgent(cfg, llm), TASK, cfg, n_adapt=1, n_test=0)
    step = recs[0]["steps"][0]
    assert step["action"] == "go to sinkbasin 1"       # countertop 2 has 2 votes but is rejected
    assert step["info"] == {"votes": 2, "candidates": 3, "dropped_inadmissible": 2}
    assert llm.calls[0]["n"] == 5 and llm.calls[0]["temperature"] == cfg.agent.actor_temperature
    cfg.agent.admissible_filter = False                 # no filter: the tie goes to the earliest sample
    recs = run_task(FakeEnv(), ReActAgent(cfg, ScriptedLLM([], votes)), TASK, cfg, n_adapt=1, n_test=0)
    assert recs[0]["steps"][0]["action"] == "go to countertop 2"


def test_muse_scores_rollouts_on_actor_context(cfg):
    cfg.self_assessment.adapt_rollouts = "selected"      # extension, as in configs/local.yaml
    class FeatureLLM(ScriptedLLM):
        def span_features(self, prefix, spans, adapter, continuations=None, layer=None):
            assert layer == 14
            self.feature_calls.append((prefix, spans, adapter, continuations))
            return torch.zeros(len(spans), 4)

    cfg.self_assessment.input, cfg.self_assessment.layer = "actor", 14
    rollouts = [" go to sinkbasin 1\nOn the sinkbasin 1.", " put bowl 1 in/on cabinet 1\nYou put\nSTATUS: OK"]
    llm = FeatureLLM([], rollouts)
    llm.feature_calls = []
    agent = MuseAgent(cfg, llm, StubSA([0.2, 0.9]), actor_adapter="actor")
    recs = run_task(FakeEnv(), agent, TASK, cfg, n_adapt=1, n_test=0)
    assert recs[0]["success"] and recs[0]["steps"][0]["info"]["selected"] == 1
    prefix, spans, adapter, conts = llm.feature_calls[0]
    assert prefix == llm.calls[0]["prompt"] and adapter == "actor"
    assert conts == [" go to sinkbasin 1\nOn the sinkbasin 1.\n>", " put bowl 1 in/on cabinet 1\nYou put\n>"]
    assert [(prefix + c)[s:e] for c, (s, e) in zip(conts, spans)] == [c[:-2] for c in conts]
    text, chunk_spans, _, none = llm.feature_calls[1]        # the executed episode's chunks, for sa_chunks
    assert none is None and text.startswith(prefix) and len(chunk_spans) == 1
    assert recs[0]["sa_chunks"][0]["pred"] == pytest.approx(0.2)
    # adaptation trains on the taken rollout, rebuilt exactly as it was scored: same prompt, same text
    llm = FeatureLLM([], rollouts)
    llm.feature_calls = []
    agent = MuseAgent(cfg, llm, StubSA([0.2, 0.9]), actor_adapter="actor")
    recs = run_task(FakeEnv(), agent, TASK, cfg, n_adapt=1, n_test=1)
    adapt_ep = next(r for r in recs if r["phase"] == "adapt")
    chosen = adapt_ep["steps"][0]["info"]["candidates"][1]
    prefix, spans, _, conts = next(c for c in llm.feature_calls if c[3] == [chosen["continuation"]])
    assert prefix in [c["prompt"] for c in llm.calls] and prefix.endswith("Your task is to: put a clean bowl in cabinet.\n>")
    assert adapt_ep["sa_adaptation"]["n_rollouts"] == 1


def test_randomness_is_global(cfg):
    import random
    rollouts = [" go to sinkbasin 1\nOK.", " go to cabinet 1\nOK.", " look\nOK.", " go to fridge 1\nOK.",
                " go to drawer 1\nOK."]
    cfg.muse.selection = "random"

    def picks():
        random.seed(cfg.run.seed)
        agent = MuseAgent(cfg, ScriptedLLM([], rollouts), StubSA([0.5] * 5))
        state = agent.new_task_state(TASK)
        return [agent.act(EpisodeContext(TASK, INIT_OBS, "", []), state)[1]["selected"] for _ in range(20)]

    assert picks() == picks() and len(set(picks())) > 1     # reproducible from the global seed alone
    agent = ReflexionAgent(cfg, ScriptedLLM([]))
    state = agent.new_task_state(TASK)
    assert agent.task_state(state) == {"memory": []}         # checkpoints keep no generator
    agent.restore_task_state(state, {"memory": [{"text": "x"}], "rng": (3, (), None)})   # older checkpoint
    assert state.memory == [{"text": "x"}]


def test_run_tasks_resumes_mid_task_after_crash(cfg, tmp_path, monkeypatch):
    cfg.self_assessment.adapt_rollouts = "selected"      # extension, as in configs/local.yaml
    class CrashingEnv(FakeEnv):
        resets = 0

        def reset(self, task):
            CrashingEnv.resets += 1
            if CrashingEnv.resets == 4:
                raise RuntimeError("machine froze")
            return INIT_OBS

        def close(self):
            pass

    monkeypatch.setattr(runner, "ALFWorldEnv", CrashingEnv)
    out = tmp_path / "episodes.jsonl"
    rollouts = [" look\nNothing happens."]

    def agent():
        return MuseAgent(cfg, ScriptedLLM([], rollouts), StubSA([0.5]))

    with pytest.raises(RuntimeError, match="froze"):     # during the test episode, after adaptation
        run_tasks(cfg, agent(), [TASK], out, n_adapt=2, n_test=1)
    assert not out.exists() and len(list(tmp_path.glob("episodes.partial/*.pkl"))) == 1

    resumed = agent()
    run_tasks(cfg, resumed, [TASK], out, n_adapt=2, n_test=1)
    recs = load_episodes(out)
    assert [r["episode"] for r in recs] == [0, 1, 2, 3] and CrashingEnv.resets == 5   # only episode 3 reran
    assert len(recs[3]["memory"]) == 2                    # reflections from before the crash
    assert recs[2]["sa_adaptation"] == {"n_new": 6, "n_rollouts": 4} and resumed.sa_template.adapted is None
    assert resumed.sa_template.mlp.weight.eq(1).all()      # the task's adapted g_eta was restored
    assert not (tmp_path / "episodes.partial").exists()


def test_run_file_with_cut_off_last_line(tmp_path):
    path = tmp_path / "episodes.jsonl"
    good = "".join(json.dumps({"task_id": t, "episodes": [{"task_id": t}]}) + "\n" for t in "ab")
    path.write_text(good + '{"task_id": "c", "epis')
    assert [e["task_id"] for e in load_episodes(path)] == ["a", "b"]
    assert path.read_text().endswith('"epis')                          # reading alone leaves the file
    assert len(runner._read_jsonl(path, repair=True)) == 2 and path.read_text() == good
    path.write_text('{"task_id": "a", "epis\n' + good)
    with pytest.raises(ValueError, match="followed by more lines"):
        load_episodes(path)


def test_force_plan_prefills_first_step(cfg):
    cfg.agent.force_plan = True
    agent = ReActAgent(cfg, ScriptedLLM(["I need to find a bowl.", "go to cabinet 1"]))
    recs = run_task(FakeEnv(), agent, TASK, cfg, n_adapt=1, n_test=0)
    assert agent.llm.calls[0]["prompt"].endswith("\n> think: To solve the task,")
    assert not agent.llm.calls[1]["prompt"].endswith("To solve the task,")         # first step only
    assert recs[0]["plan"] == "think: To solve the task, I need to find a bowl."
    assert recs[0]["steps"][1]["action"] == "go to cabinet 1"

    rollouts = [" I need a bowl.\nOK.\n> go to cabinet 1\nClosed.", " I need a mug.\nOK.\n> look\nNothing."]
    agent = MuseAgent(cfg, ScriptedLLM([], rollouts), StubSA([0.2, 0.9]))
    ctx = EpisodeContext(task=TASK, init_obs=INIT_OBS, task_desc="d", memory=[])
    action, info = agent.act(ctx, agent.new_task_state(TASK))
    assert action == "think: To solve the task, I need a mug."                     # rollouts are plans
    assert info["candidates"][1]["rollout"].startswith("think: To solve the task, I need a mug..OK..> look")


def _ep(task, e, success, refl="r", steps=5, task_type="clean", phase="adapt"):
    return {"task_id": task, "task_type": task_type, "task_desc": "d", "init_obs": INIT_OBS, "episode": e,
            "phase": phase, "memory": [], "plan": "think: p", "success": success, "num_steps": steps,
            "steps": [{"action": f"a{i}", "observation": f"o{i}"} for i in range(steps)],
            "reflection": {"query": f"q-{task}-{e}", "text": f"{refl}-{task}-{e}"}}


def _pair(task, c, r):
    return {"task_id": task, "prompt_chosen": f"q-{task}-{c}", "chosen": f"r-{task}-{c}",
            "prompt_rejected": f"q-{task}-{r}", "rejected": f"r-{task}-{r}"}


def test_dpo_pairs_within_task_all_combinations():
    # outcomes F S F S F: positives r0, r2 (fail->success); negatives r1, r3 (success->fail)
    eps = [_ep("t1", e, e % 2 == 1) for e in range(5)]
    assert dpo_pairs(eps, fallback=False) == [_pair("t1", 0, 1), _pair("t1", 0, 3),
                                              _pair("t1", 2, 1), _pair("t1", 2, 3)]
    assert len(dpo_pairs(eps, max_pairs_per_task=2)) == 2
    # never pairs across tasks
    assert dpo_pairs(eps[:2] + [_ep("t2", 0, True), _ep("t2", 1, False)], fallback=False) == []


def test_dpo_pairs_fallback():
    eps = [_ep("t1", 0, False), _ep("t1", 1, True), _ep("t1", 2, True)]   # r0 f->s, r1 s->s: no negative
    assert dpo_pairs(eps) == []
    eps += [_ep("t2", 0, False), _ep("t2", 1, False), _ep("t2", 2, True)]   # r0 f->f, r1 f->s
    assert dpo_pairs(eps, fallback=False) == []
    assert dpo_pairs(eps) == [_pair("t2", 1, 0)]                          # fail->fail as negative
    eps = [_ep("t3", 0, True), _ep("t3", 1, True), _ep("t3", 2, False)]   # r0 s->s, r1 s->f
    assert dpo_pairs(eps) == [_pair("t3", 0, 1)]                          # success->success as positive


def test_split_by_task_holds_out_whole_tasks():
    ids = ["a", "a", "b", "c", "c", "d"]
    tr, va = split_by_task(ids, 0.25, seed=0)
    assert sorted(tr + va) == list(range(6)) and va
    assert not {ids[i] for i in tr} & {ids[i] for i in va}
    assert split_by_task(ids, 0.0)[1] == []
    # mixed: held-out tasks only from those with both outcomes, val_fraction of them (at least one)
    ids = ["a", "a", "b", "b", "c", "c", "d", "d"]
    labels = [0, 1, 0, 0, 1, 0, 1, 1]
    for seed in range(5):
        tr, va = split_by_task(ids, 0.2, seed, labels=labels)
        assert {ids[i] for i in va} in ({"a"}, {"c"}) and len(tr) == 6


def test_sa_input_puts_plan_first():
    assert sa_input_text("think: p", "a0.o0.") == "Plan: think: p\nTrajectory: a0.o0."


def test_sa_adaptation_mixes_replay_and_resets_per_task(cfg):
    import threading

    import torch

    cfg.self_assessment.epochs = 3
    sa = SelfAssessor.__new__(SelfAssessor)          # no sentence-transformer: random features
    sa.cfg, sa.device, sa.lock = cfg.self_assessment, "cpu", threading.RLock()
    sa.mlp, sa.mean, sa.std = MLP(4, [8], 0.0), None, None
    x = torch.randn(40, 4, generator=torch.Generator().manual_seed(0))
    y = (x[:, 0] > 0).float()
    sa.replay, sa.val_x, sa.val_y = ReplayBuffer(x[:30], y[:30]), x[30:], y[30:]

    epochs = []
    sa.fit(torch.randn(4, 4), torch.ones(4), 2, 1e-3, replay=sa.replay.get(),
           log_fn=lambda e, l: epochs.append((e, sa.evaluate(sa.val_x, sa.val_y)["bce"])))   # re-enters the lock
    assert [e for e, _ in epochs] == [0, 1]
    a, b = sa.clone(), sa.clone()
    w0 = [p.clone() for p in sa.mlp.parameters()]
    info = a.adapt(torch.randn(6, 4), torch.ones(6))
    assert info["n_new"] == 6 and info["n_replay"] == 30
    assert set(info["val_before"]) == {"bce", "acc", "auroc"} and info["val_after"]["bce"] > 0
    # the next task starts from the end-of-pre-deployment g_eta and buffer
    assert all(torch.equal(p, q) for p, q in zip(b.mlp.parameters(), w0))
    assert not all(torch.equal(p, q) for p, q in zip(a.mlp.parameters(), w0))
    a.replay.add(torch.randn(3, 4), torch.ones(3))
    assert len(sa.replay) == len(b.replay) == 30
    assert b.adapt(torch.randn(2, 4), torch.zeros(2))["n_replay"] == 30


def _bare_sa(cfg, in_dim):
    import threading

    sa = SelfAssessor.__new__(SelfAssessor)          # no sentence-transformer
    sa.cfg, sa.device, sa.lock = cfg.self_assessment, "cpu", threading.RLock()
    sa.in_dim, sa.mean, sa.std, sa.val_x, sa.val_y = in_dim, None, None, None, None
    sa.replay = ReplayBuffer()
    torch.manual_seed(0)
    sa.mlp = MLP(in_dim, [8], 0.0)
    return sa


def test_sa_class_balance_raises_scores_of_rare_successes(cfg):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(400, 4, generator=g)
    y = ((x[:, 0] + 0.8 * torch.randn(400, generator=g)) > 1.8).float()     # ~6% successes, noisy
    means = {}
    for balance in ("none", "weight", "sample"):
        cfg.self_assessment.class_balance = balance
        sa = _bare_sa(cfg, 4)
        sa.fit(x, y, 20, 1e-2)
        means[balance] = sa.predict_features(x[y == 1]).mean()
    assert means["weight"] > means["none"] + 0.2 and means["sample"] > means["none"] + 0.2
    cfg.self_assessment.class_balance = "both"
    with pytest.raises(ValueError):
        _bare_sa(cfg, 4).fit(x, y, 1, 1e-2)


def test_sa_early_stopping_keeps_best_held_out_epoch(cfg):
    from muse.self_assessment import within_task_auroc

    g = torch.Generator().manual_seed(0)
    x = torch.randn(300, 4, generator=g)
    y = (x[:, 0] > 0.5).float()
    xv = torch.randn(60, 4, generator=g)
    yv = (xv[:, 0] > 0.5).float()
    tv = ["t1"] * 30 + ["t2"] * 30
    cfg.self_assessment.class_balance, cfg.self_assessment.early_stop = "none", True
    sa, aurocs = _bare_sa(cfg, 4), []
    out = sa.fit(x, y, 8, 1e-2, val=(xv, yv, tv),
                 log_fn=lambda e, l: aurocs.append(within_task_auroc(yv, sa.predict_features(xv), tv)))
    assert out["best_epoch"] == int(np.argmax(aurocs)) + 1 and out["best_val_auroc"] == pytest.approx(max(aurocs))
    assert within_task_auroc(yv, sa.predict_features(xv), tv) == pytest.approx(max(aurocs))
    # no held-out task with both outcomes: all epochs, as in adaptation
    assert _bare_sa(cfg, 4).fit(x, y, 2, 1e-2, val=(xv, torch.zeros(60), tv))["best_epoch"] is None
    assert np.isnan(within_task_auroc([0, 0, 1], [0.1, 0.2, 0.3], ["a", "a", "b"]))


def test_sa_standardizer_is_applied_and_saved(cfg, tmp_path):
    x = torch.randn(50, 3, generator=torch.Generator().manual_seed(0)) * torch.tensor([1.0, 500.0, 0.01]) + 7
    sa = _bare_sa(cfg, 3)
    raw = sa.predict_features(x)
    sa.set_standardizer(x)
    z = (x - x.mean(0)) / x.std(0)
    assert torch.allclose(sa._prep(x).mean(0), torch.zeros(3), atol=1e-4)
    assert sa.predict_features(x) == pytest.approx(torch.sigmoid(sa.mlp(z)).detach().numpy(), abs=1e-5)
    assert not sa.predict_features(x) == pytest.approx(raw, abs=1e-3)
    sa.save(tmp_path)
    cfg.self_assessment.input, cfg.self_assessment.hidden = "actor", [8]
    loaded = SelfAssessor.load(cfg, tmp_path, device="cpu")
    assert loaded.predict_features(x) == pytest.approx(sa.predict_features(x), abs=1e-6)


def test_sft_and_sa_examples():
    eps = [_ep("t1", 0, False, steps=9), _ep("t1", 1, True, steps=4)]
    sft = sft_examples(eps)
    assert len(sft) == 1 and len(sft[0]["spans"]) == 4
    sa = sa_examples(eps, 4)
    assert sa["labels"] == [0.0, 0.0, 0.0, 1.0] and sa["texts"][-1].startswith("a0.o0.> a1.o1.")
    assert sa["texts"][2] == "a8.o8." and sa["texts"][-1].endswith("a3.o3.")       # no terminal signal
    assert not any("STATUS" in t for t in sa["texts"])
    every = sa_examples(eps, 4, stride=1)                  # a chunk from every step, the last ones shorter
    assert every["starts"] == list(range(9)) + list(range(4)) and every["texts"][6] == "a6.o6.> a7.o7.> a8.o8."
    assert [t for t, s in zip(every["texts"], every["starts"]) if s % 4 == 0] == sa["texts"]
    assert sa["texts"][0].endswith("a3.o3.") and sa["task_ids"] == ["t1"] * 4


def test_world_model_examples_and_loss():
    from muse.datasets import wm_examples
    from muse.train_lora import _ref_logps, _span_labels, _wm_loss

    eps = [_ep("t1", 0, False, steps=3), _ep("t1", 1, True, steps=2), _ep("t2", 0, False, steps=2)]
    ex = wm_examples(eps)
    assert [len(e["action_spans"]) for e in ex] == [0, 2, 0] and [len(e["obs_spans"]) for e in ex] == [3, 2, 2]
    assert [len(e["anchor_spans"]) for e in ex] == [3, 2, 2]          # every episode's actions
    e = ex[1]
    assert [e["text"][s:t] for s, t in e["action_spans"]] == [" a0\n", " a1\n"]
    assert [e["text"][s:t] for s, t in e["obs_spans"]] == ["o0\n", "o1\n"]
    assert e["text"].startswith(prompts.world_model_prompt(eps[1]["task_type"], eps[1]["init_obs"], [], []))
    assert len(wm_examples(eps, max_failed=1)) == 2 and sum(x["success"] for x in wm_examples(eps, max_failed=0)) == 1
    parts = wm_examples(eps[:1], max_label_steps=2)                     # 3 steps: items of 2 and 1 steps
    assert [len(x["obs_spans"]) for x in parts] == [2, 1] and parts[1]["text"] == ex[0]["text"]
    assert parts[0]["text"] == ex[0]["text"][:ex[0]["obs_spans"][1][1]] and parts[0]["text"].endswith("o1\n")

    # the loss averages actions and observations separately: obs_weight scales only the latter
    model, _ = _tiny_lm()
    torch.manual_seed(0)
    ids = torch.randint(0, 50, (12,))
    pos = [(i, i + 1) for i in range(12)]
    act = torch.tensor(_span_labels(ids.tolist(), pos, [(3, 5)]))
    obs = torch.tensor(_span_labels(ids.tolist(), pos, [(6, 10)]))
    anchor = torch.tensor(_span_labels(ids.tolist(), pos, [(1, 2), (3, 5)]))
    l0, s0 = _wm_loss(model, ids, act, obs, anchor, 0.0)
    l1, s1 = _wm_loss(model, ids, act, obs, anchor, 1.0)
    logits = model(input_ids=ids[None]).logits[0, :-1].float()
    ce = torch.nn.functional.cross_entropy(logits, ids[1:], reduction="none")
    assert s0["action_loss"] == pytest.approx(ce[2:4].mean().item(), rel=1e-4)
    assert s0["obs_loss"] == pytest.approx(ce[5:9].mean().item(), rel=1e-4)
    assert l1.item() == pytest.approx(s0["action_loss"] + s0["obs_loss"], rel=1e-4)
    assert l0.item() == pytest.approx(s0["action_loss"], rel=1e-4)
    none = torch.full((12,), -100)
    l2, s2 = _wm_loss(model, ids, none, obs, none, 0.5)      # a failed episode: observations only
    assert "action_loss" not in s2 and l2.item() == pytest.approx(0.5 * s0["obs_loss"], rel=1e-4)
    # anchor: no action loss in the objective; zero drift against the model's own log-probs, which are
    # those of anchor tokens 1, 3, 4 (predicted from positions 0, 2, 3)
    ref = _ref_logps(model, ids, anchor)
    assert len(ref) == 3 and torch.allclose(ref, -ce[[0, 2, 3]].detach(), atol=1e-4)
    l3, s3 = _wm_loss(model, ids, act, obs, anchor, 1.0, "anchor", ref, 2.0)
    assert s3["anchor_drift"] == pytest.approx(0.0, abs=1e-8) and l3.item() == pytest.approx(s0["obs_loss"], rel=1e-4)
    l4, s4 = _wm_loss(model, ids, act, obs, anchor, 1.0, "anchor", ref + 0.5, 2.0)
    assert s4["anchor_drift"] == pytest.approx(0.25, rel=1e-3) and l4.item() == pytest.approx(s0["obs_loss"] + 0.5, rel=1e-3)
    assert "action_loss" in s4


def test_training_loop_mixed_stats_and_checkpoint_selection(monkeypatch):
    from types import SimpleNamespace

    import muse.train_lora as T

    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1)

    def loss_fn(i):                          # failed episodes report no action loss
        loss = (model(torch.ones(2)).sum() - 3) ** 2
        return loss, ({"action_loss": loss.item(), "obs_loss": 1.0} if i % 2 else {"obs_loss": 2.0})

    checks = iter([{"obs_loss": 0.5, "action_loss": 1.0}, {"obs_loss": 0.2, "action_loss": 1.0},
                   {"obs_loss": 0.1, "action_loss": 9.0}, {"obs_loss": 0.3, "action_loss": 1.0}])
    weights = []
    monkeypatch.setattr(T, "_evaluate", lambda m, items, fn: (weights.append(m.weight.detach().clone()), next(checks))[1])
    tc = SimpleNamespace(lr=1e-2, epochs=1, grad_accum=1, warmup_ratio=0.0)
    select = lambda row, first: None if row["action_loss"] > first["action_loss"] * 1.05 else -row["obs_loss"]
    h = T._loop(model, list(range(6)), tc, loss_fn, 0, None, [0], 2, select)
    assert h["train"][0]["obs_loss"] == pytest.approx(1.5) and "action_loss" in h["train"][0]
    # checks at steps 0, 2, 4, 6: step 4 has the lowest obs loss but breaks the action-loss guard
    assert h["selected_step"] == 2 and torch.equal(model.weight, weights[1])


def test_world_model_adapter_serves_muse_and_m_sa(cfg, tmp_path, monkeypatch):
    import muse.cli as cli
    cfg.run.out_dir = str(tmp_path)
    assert cfg.world_model.enabled is False and cli._sa_feature_adapter(cfg).name == "actor"
    cfg.world_model.enabled = True
    assert cli._sa_feature_adapter(cfg) == tmp_path / "models" / "world_model"
    seen = {}
    monkeypatch.setattr("muse.llm.build_llm", lambda c, adapters=None: seen.setdefault("adapters", adapters))
    monkeypatch.setattr("muse.self_assessment.SelfAssessor.load", lambda *a, **k: StubSA([0.5]))
    for d in ("world_model", "reflector"):
        (tmp_path / "models" / d).mkdir(parents=True)
    agent = cli.build_agent(cfg, "muse")
    assert set(seen["adapters"]) == {"world_model", "reflector"}
    assert agent.world_model_adapter == agent.actor_adapter == "world_model"


def test_sft_loss_matches_full_logits():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from muse.train_lora import _sft_loss
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                         num_attention_heads=2, num_key_value_heads=1)).eval()
    ids = torch.randint(0, 50, (12,))
    labels = torch.full((12,), -100)
    labels[[4, 5, 9, 11]] = ids[[4, 5, 9, 11]]
    full = model(input_ids=ids[None], labels=labels[None]).loss
    assert _sft_loss(model, ids, labels).item() == pytest.approx(full.item(), rel=1e-5)


def _tiny_lm():
    """A random 1-layer Qwen2 and a character-level tokenizer with offsets."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    def tok(text, return_offsets_mapping=False, add_special_tokens=True):
        return {"input_ids": [ord(c) % 50 for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                         num_attention_heads=2, num_key_value_heads=1)).eval()
    return model, tok


def test_span_features_on_cached_prefix_match_full_text():
    from muse.self_assessment import span_features

    model, tok = _tiny_lm()
    prefix, conts = "examples, memory, task\n>", [" go to desk 1\nOn the desk 1.\n>", " look\nNothing.\n>"]
    spans = [(len(prefix), len(prefix) + len(c) - 2) for c in conts]
    x = span_features(model, tok, prefix, spans, 1000, continuations=conts)
    full = torch.cat([span_features(model, tok, prefix + c, [s], 1000) for c, s in zip(conts, spans)])
    assert x.shape == (2, 32) and torch.allclose(x, full, atol=1e-5)


def test_span_features_from_a_middle_layer():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from muse.self_assessment import span_features

    _, tok = _tiny_lm()
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                         num_attention_heads=2, num_key_value_heads=1)).eval()
    prefix, conts = "examples, memory, task\n>", [" go to desk 1\nOn the desk 1.\n>", " look\nNothing.\n>"]
    spans = [(len(prefix), len(prefix) + len(c) - 2) for c in conts]
    final, mid = span_features(model, tok, prefix, spans, 1000, continuations=conts, layer=[None, 1])
    assert torch.equal(final, span_features(model, tok, prefix, spans, 1000, continuations=conts))
    full = torch.cat([span_features(model, tok, prefix + c, [s], 1000, layer=1) for c, s in zip(conts, spans)])
    assert torch.allclose(mid, full, atol=1e-5) and not torch.allclose(mid, final, atol=1e-3)
    # layer k is decoder layer k's output: the same as output_hidden_states[k]
    ids = torch.tensor([tok(prefix + conts[0])["input_ids"]])
    h = model.get_decoder()(input_ids=ids, output_hidden_states=True).hidden_states[1][0]
    s, e = spans[0]
    assert torch.allclose(mid[0], torch.cat([h[s:e].mean(0), h[e - 1]]), atol=1e-5)
    with pytest.raises(ValueError):
        span_features(model, tok, prefix, spans, 1000, layer=3)


def test_actor_context_features_one_row_per_chunk_and_causal():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from muse.self_assessment import actor_context_features

    model, tok = _tiny_lm()
    eps = [_ep("t1", 0, False, steps=9), _ep("t1", 1, True, steps=4)]
    x = actor_context_features(model, tok, eps, 4, 100_000)
    assert x.shape == (len(sa_examples(eps, 4)["labels"]), 32)
    eps[0]["steps"][8]["observation"] = "changed"          # last chunk only: earlier chunks see no later step
    x2 = actor_context_features(model, tok, eps, 4, 100_000)
    assert torch.allclose(x[:2], x2[:2]) and not torch.allclose(x[2], x2[2])
    every = actor_context_features(model, tok, eps, 4, 100_000, stride=1)
    starts = sa_examples(eps, 4, stride=1)["starts"]
    assert len(every) == len(starts) and torch.allclose(every[[i for i, s in enumerate(starts) if s % 4 == 0]], x2)


def test_metrics():
    phases = ["noadapt", "noadapt", "adapt", "adapt", "test", "test"]
    eps = ([_ep("t1", e, e >= 4, steps=10, phase=ph) for e, ph in enumerate(phases)]
           + [_ep("t2", e, False, steps=50, phase=ph) for e, ph in enumerate(phases)])
    assert success_rate(condition_episodes(eps, "noadapt")) == 0
    assert success_rate(condition_episodes(eps, "adapt")) == 50
    assert time_to_completion(condition_episodes(eps, "adapt")) == pytest.approx((10 + 100) / 2)
    for ep in eps:
        ep["sa_chunks"] = [{"pred": 0.8 if ep["success"] else 0.3}]
    sa = self_assessment(eps)
    assert sa["auroc2"] == 1.0 and sa["meta_accuracy"] == 100
    eps[3]["sa_adaptation"] = {"new_before": {"bce": 0.9}, "new_after": {"bce": 0.5},
                               "val_before": {"bce": 0.4, "auroc": 0.8}, "val_after": {"bce": 0.45, "auroc": 0.78}}
    s = summarize(eps)
    assert set(s) == {"noadapt", "adapt", "sa_adaptation"}
    assert s["sa_adaptation"]["heldout_bce_after"] == pytest.approx(0.45)


def test_config_extends_and_overrides():
    c = load_config("configs/local.yaml", ["muse.num_rollouts=3"])
    assert c.model.backend == "hf" and c.muse.num_rollouts == 3 and c.env.max_steps == 50
    assert load_config("configs/default.yaml").agent.max_memory is None   # cumulative reflections
    assert c.run.out_dir == "runs/local"
    d = load_config("configs/default.yaml")
    assert (d.agent.actor_temperature, d.protocol.test_independent) == (0.5, True)
    assert d.collect.num_episodes == 30
    assert (d.self_assessment.epochs, d.self_assessment.adapt_epochs, d.self_assessment.adapt_lr) == (60, 30, 1e-3)
    assert c.protocol.test_independent is True
    assert (d.self_assessment.standardize, d.self_assessment.class_balance, d.self_assessment.layer) == \
        (False, "weight", None)
    # The paper's M_sa by default; the local config uses the extensions, which three keys switch off
    assert (d.self_assessment.input, d.self_assessment.chunk_stride, d.self_assessment.adapt_rollouts) == \
        ("paper", None, "none")
    sc = c.self_assessment
    assert (sc.input, sc.layer, sc.chunk_stride, sc.adapt_rollouts) == ("actor", 14, 1, "selected")
    paper = load_config("configs/local.yaml", ["self_assessment.input=paper", "self_assessment.chunk_stride=null",
                                               "self_assessment.adapt_rollouts=none"]).self_assessment
    assert {k: paper[k] for k in ("input", "chunk_stride", "adapt_rollouts")} == \
        {k: d.self_assessment[k] for k in ("input", "chunk_stride", "adapt_rollouts")}
    assert (d.self_assessment.lr, d.self_assessment.val_tasks, d.self_assessment.early_stop) == (3e-4, "mixed", True)


def test_dashboard_stages(tmp_path):
    import json
    import os

    from muse.dashboard import collect_runs, render_html, render_text
    from muse.status import StageStatus

    run = tmp_path / "mistral7b"
    s = StageStatus(run, "collect")
    s.update(140, 140, "tasks", success_rate=40.0, model="mistralai/Mistral-7B-Instruct-v0.2")
    s.finish()
    s = StageStatus(run, "train-sft")
    s.update(30, 120, "steps", loss=0.8)
    s = StageStatus(run, "eval-react")
    s.update(10, 134, "tasks", success_noadapt=30.0, success_adapt=31.0)
    dead = json.loads((run / "status" / "eval-react.json").read_text())
    dead["pid"] = 2 ** 22 + 12345      # not a live process -> reported as stopped
    (run / "status" / "eval-react.json").write_text(json.dumps(dead))

    runs = collect_runs(tmp_path)
    agents = {a["key"]: a for a in runs[0]["agents"]}
    assert runs[0]["model"] == "mistralai/Mistral-7B-Instruct-v0.2"
    assert agents["react"]["stage"] == "eval-react" and agents["react"]["state"] == "stopped"
    assert agents["reflexion"]["state"] == "pending"
    assert agents["muse"]["stage"] == "train-sft" and agents["muse"]["state"] == "running"
    assert [st for _, _, st in agents["muse"]["pipe"]] == ["done", "running", "pending", "pending", "pending"]
    assert "30%" in render_text(runs) and "MUSE" in render_html(runs, 15)
    from muse.dashboard import _eta
    resumed = StageStatus(tmp_path / "other", "collect")
    resumed.update(100, 140, "tasks")
    resumed.update(100, 140, "tasks")
    assert _eta(resumed.data) is None          # no progress yet in this process: no ETA
    resumed.data["started"] -= 100
    resumed.update(110, 140, "tasks")
    assert 280 <= _eta(resumed.data) <= 320    # 10 tasks per ~100 s, 30 left
    assert os.getpid() == json.loads((run / "status" / "train-sft.json").read_text())["pid"]


def test_report_table2_and_fig9(tmp_path):
    import json

    import yaml

    from muse.report import write_report

    eval_dir = tmp_path / "run" / "eval" / "muse"
    eval_dir.mkdir(parents=True)
    (eval_dir / "config.yaml").write_text(yaml.safe_dump(
        {"protocol": {"adapt_episodes": 2, "test_episodes": 2, "fail_penalty_steps": 100},
         "model": {"name": "test-model"}}))
    with open(eval_dir / "episodes.jsonl", "w") as f:
        for t in range(3):
            eps = []
            for e, ph in enumerate(["noadapt", "noadapt", "adapt", "adapt", "test", "test"]):
                ok = (t + e) % 2 == 0
                ep = _ep(f"task{t}", e, ok, steps=12, phase=ph)
                ep["sa_chunks"] = [{"pred": 0.7 if ok else 0.4}, {"pred": 0.6 if ok else 0.55}]
                eps.append(ep)
            f.write(json.dumps({"task_id": f"task{t}", "episodes": eps}) + "\n")
    out = tmp_path / "report"
    write_report([tmp_path / "run"], out)
    md = (out / "report.md").read_text()
    assert "Table 2" in md and "| Adaptation | 1.00 |" in md and "Paper" not in md
    assert (out / "fig9_type2_roc.png").stat().st_size > 0
    from muse.report import PAPER_MODEL
    (eval_dir / "config.yaml").write_text(yaml.safe_dump(
        {"protocol": {"adapt_episodes": 2, "test_episodes": 2, "fail_penalty_steps": 100},
         "model": {"name": PAPER_MODEL}}))
    write_report([tmp_path / "run"], out)    # the paper's model: its values are shown alongside
    md = (out / "report.md").read_text()
    assert "| Paper AUROC2 |" in md and "| 84% | 90% |" in md


def test_rollout_stops_after_chunk_size_turns():
    from muse.llm import truncate_at_stop
    text = " a1\no1\n> a2\no2\n> a3\no3\n> a4\no4\n> a5\no5"
    cut = truncate_at_stop(text, ROLLOUT_STOP, max_turns=4)
    assert cut == " a1\no1\n> a2\no2\n> a3\no3\n> a4\no4"
    assert [s.action for s in parse_rollout(cut)[0]] == ["a1", "a2", "a3", "a4"]
    assert truncate_at_stop(" a1\no1\nSTATUS: OK\n\n> a2", ROLLOUT_STOP, max_turns=4) == " a1\no1\nSTATUS: OK"


def test_paired_comparison_and_per_task_auroc2():
    from muse.metrics import auroc2_per_task, paired_comparison
    a, b = {f"t{i}": 100.0 for i in range(8)}, {f"t{i}": 0.0 for i in range(8)}
    c = paired_comparison(a, b)
    assert c["n_tasks"] == 8 and c["mean_diff"] == 100 and c["p"] == pytest.approx(2 / 2 ** 8)
    assert paired_comparison({"x": 1, "y": -1}, {"x": 0, "y": 0})["p"] == 1.0
    # two tasks scored on different scales: pooled AUROC2 is 0, within each task it is 1
    eps = [{**_ep(t, e, ok), "sa_chunks": [{"pred": p}]}
           for t, e, ok, p in (("t1", 0, True, 0.02), ("t1", 1, False, 0.01),
                               ("t2", 0, True, 0.40), ("t2", 1, False, 0.30))]
    assert auroc2_per_task(eps) == {"auroc2_per_task": 1.0, "auroc2_n_tasks": 2}


def test_provenance_appends_one_record_per_launch(tmp_path):
    from muse.provenance import record
    c = load_config("configs/local.yaml", [f"run.out_dir={tmp_path}"])
    first = record(c, "eval-muse", argv=["eval"])
    sa = tmp_path / "models" / "sa" / "sa.pt"
    sa.parent.mkdir(parents=True)
    sa.write_bytes(b"weights")
    second = record(c, "eval-muse", argv=["eval"])   # a restart appends, never overwrites
    lines = [json.loads(l) for l in (tmp_path / "provenance.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in lines] == ["eval-muse", "eval-muse"]
    assert first["trained_models"] == {} and list(second["trained_models"]) == ["models/sa/sa.pt"]
    assert lines[1]["config"]["model"]["max_batch_tokens"] == 10000
    assert lines[1]["packages"]["torch"] and lines[1]["model"]["name"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert lines[1]["git"] is None or len(lines[1]["git"]["commit"]) == 40


def test_local_config():
    c = load_config("configs/local.yaml")
    assert c.run.out_dir == "runs/local" and c.model.name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert c.env.max_steps == 50 and (c.protocol.adapt_episodes, c.protocol.test_episodes) == (5, 5)
    assert c.collect.num_episodes == 30 and c.agent.max_memory is None and c.agent.admissible_filter
    assert (c.self_assessment.input, c.self_assessment.layer, c.protocol.skip_noadapt) == ("actor", 14, False)
    assert c.sft.epochs == 2 and c.sft.grad_accum == 2 and c.dpo.max_pairs_per_task == 8


def test_memory_token_budget_is_fifo():
    """agent.max_memory_tokens: the oldest reflections are dropped first until the memory block fits;
    the actor and the reflector see the same window."""
    class WordLLM(ScriptedLLM):
        def num_tokens(self, text):
            return len(text.split())

    memory = [{"episode": i, "success": i == 0, "text": f"reflection {i} " + "word " * 8} for i in range(4)]
    block = len(prompts.memory_block(memory[-2:]).split())
    c = load_config("configs/default.yaml", [f"agent.max_memory_tokens={block}"])
    agent = ReflexionAgent(c, WordLLM([]))
    state = agent.new_task_state(TASK)
    state.memory = list(memory)
    assert agent.memory_window(state) == memory[-2:]      # the first success plan goes first too
    assert ReflexionAgent(load_config("configs/default.yaml"), WordLLM([])).memory_window(state) == memory
    local = load_config("configs/local.yaml")
    assert local.agent.max_memory is None and local.agent.max_memory_tokens == 7500 and local.model.max_model_len == 16384


def test_sample_tasks_is_seeded(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("sample_tasks", "scripts/sample_tasks.py")
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    pool = [Task(f"{p}-X-None-Y-1/trial_{i}", "unused", "put")
            for p in ("pick_and_place_simple", "pick_two_obj_and_place") for i in range(20)]
    monkeypatch.setattr(st, "list_tasks", lambda data_dir, split: pool)
    a = st.sample("unused", "valid_seen", 10, seed=0)
    assert a == st.sample("unused", "valid_seen", 10, seed=0) != st.sample("unused", "valid_seen", 10, seed=1)
    assert len(set(a)) == 10 and a == sorted(a)
    assert all(i.startswith("pick_and") for i in st.sample("unused", "valid_seen", 5, 0, "pick_and_place_simple"))


def test_collection_extends_finished_tasks(cfg, tmp_path, monkeypatch):
    class CountingEnv(FakeEnv):
        resets = 0

        def reset(self, task):
            CountingEnv.resets += 1
            if CountingEnv.resets == crash_at[0]:
                raise RuntimeError("machine froze")
            return INIT_OBS

        def close(self):
            pass

    crash_at = [None]
    monkeypatch.setattr(runner, "ALFWorldEnv", CountingEnv)
    cfg.agent.stop_on_repeat = False

    def agent():
        return ReflexionAgent(cfg, ScriptedLLM(["look"] * 1000))   # always fails, reflects every episode

    straight = tmp_path / "straight.jsonl"
    run_tasks(cfg, agent(), [TASK], straight, n_adapt=4, n_test=0)

    out = tmp_path / "episodes.jsonl"
    run_tasks(cfg, agent(), [TASK], out, n_adapt=2, n_test=0)
    assert load_episodes(out)[-1]["reflection"] is None          # the last episode is not reflected on
    CountingEnv.resets, crash_at[0] = 0, 2
    with pytest.raises(RuntimeError, match="froze"):              # crash in the 2nd extension episode
        run_tasks(cfg, agent(), [TASK], out, n_adapt=4, n_test=0)
    assert len(load_episodes(out)) == 2                          # the old line is kept until replaced
    crash_at[0] = None
    ext = agent()
    run_tasks(cfg, ext, [TASK], out, n_adapt=4, n_test=0)
    assert CountingEnv.resets == 3                                 # resumed from the checkpoint: only episode 3 reran
    assert len(out.read_text().splitlines()) == 1 and not (tmp_path / "episodes.partial").exists()
    recs, ref = load_episodes(out), load_episodes(straight)
    assert [r["episode"] for r in recs] == [0, 1, 2, 3]
    assert [r["reflection"] is not None for r in recs] == [r["reflection"] is not None for r in ref] \
        == [True, True, True, False]
    assert [len(r["memory"]) for r in recs] == [len(r["memory"]) for r in ref] == [0, 1, 2, 2]
    assert "Trial #1 (FAIL)" in recs[2]["reflection"]["query"]  # episode 1's reflection reached memory

    run_tasks(cfg, agent(), [TASK], out, n_adapt=4, n_test=0)    # already at 4: nothing reruns
    assert load_episodes(out) == recs
    ev = tmp_path / "eval.jsonl"
    run_tasks(cfg, agent(), [TASK], ev, n_adapt=1, n_test=1)
    run_tasks(cfg, agent(), [TASK], ev, n_adapt=2, n_test=1)       # eval protocols are never extended
    assert len(load_episodes(ev)) == 3


def test_list_tasks_keeps_given_order(tmp_path):
    from muse.env import list_tasks
    ids = ["pick_heat_then_place_in_recep-Mug-None-Cabinet-10/trial_1",
           "look_at_obj_in_light-CD-None-DeskLamp-308/trial_1",
           "pick_clean_then_place_in_recep-SoapBar-None-Cabinet-424/trial_1"]
    for i in ids:
        d = tmp_path / "json_2.1.1" / "valid_unseen" / i
        d.mkdir(parents=True)
        (d / "traj_data.json").write_text("{}")
        (d / "game.tw-pddl").write_text('{"solvable": true}')
    assert [t.task_id for t in list_tasks(tmp_path, "valid_unseen", task_ids=ids)] == sorted(ids)
    assert [t.task_id for t in list_tasks(tmp_path, "valid_unseen", task_ids=ids, keep_order=True)] == ids
    assert load_config("configs/default.yaml").env.keep_task_order is False
    assert load_config("configs/local.yaml").env.keep_task_order is True


def test_report_task_sets():
    from muse.report import select_tasks
    ep = lambda t, s: {"task_id": t, "success": s}
    eps = {"react": [ep("a", False), ep("b", True), ep("c", False), ep("d", False)],
           "muse": [ep("a", False), ep("b", False), ep("c", True)]}          # stopped before task d
    assert select_tasks(eps, "all") is None
    assert select_tasks(eps, "common") == {"a", "b", "c"}
    assert select_tasks(eps, "solved") == {"b", "c"}                       # a: nobody solved it; d: not common


def test_eval_variant_names(cfg):
    """run.eval_variant writes an agent's eval to eval/<agent>+<variant>, shown as its own report row."""
    from muse.cli import eval_name
    from muse.report import label
    assert eval_name(cfg, "muse") == "muse"
    cfg.run.eval_variant = "paper_sa"
    assert eval_name(cfg, "muse") == "muse+paper_sa"
    assert (label("muse"), label("muse+paper_sa"), label("muse_random")) == \
        ("MUSE", "MUSE (paper_sa)", "MUSE with random self-assessment")


def test_reflection_resampled_when_it_copies(cfg):
    from muse.agents import is_copy
    assert is_copy(" I should go to cabinet 1. ", ["I should go to cabinet 1."])
    assert is_copy("I should go to cabinet 1 first.", ["I should go to cabinet 1 first!"])
    assert not is_copy("Search the drawers next.", ["I should go to cabinet 1."])
    cfg.agent.stop_on_repeat = False
    cfg.agent.reflection_temperature = 0.7
    cfg.agent.reflection_resamples = 2
    llm = ScriptedLLM(["look"] * 200)       # its reflection is always the same text
    recs = run_task(FakeEnv(), ReflexionAgent(cfg, llm), TASK, cfg, n_adapt=3, n_test=0)
    assert [r["reflection"].get("resamples") for r in recs[:2]] == [None, 2]
    temps = [c["temperature"] for c in llm.calls if c["stop"] == ("\n\n",)]
    assert temps == [0.7, 0.7, 1.0, 1.0]


def test_memory_window_caps_failures_and_puts_successes_last(cfg):
    cfg.agent.max_memory = None
    cfg.agent.max_memory_tokens = None
    agent = ReflexionAgent(cfg, ScriptedLLM([]))
    state = agent.new_task_state(TASK)
    outcomes = [False, True, False, False, True, False, False]
    state.memory = [{"episode": e, "success": s, "text": f"r{e}"} for e, s in enumerate(outcomes)]
    assert [m["episode"] for m in agent.memory_window(state)] == list(range(7))     # defaults: unchanged
    cfg.agent.max_failure_memory = 3
    assert [m["episode"] for m in agent.memory_window(state)] == [1, 3, 4, 5, 6]
    cfg.agent.successes_last = True
    assert [m["episode"] for m in agent.memory_window(state)] == [3, 5, 6, 1, 4]
    cfg.agent.max_failure_memory = 0
    assert [m["episode"] for m in agent.memory_window(state)] == [1, 4]
