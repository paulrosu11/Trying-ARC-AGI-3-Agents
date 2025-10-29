from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm
from .templates.langgraph_functional_agent import LangGraphFunc, LangGraphTextOnly
from .templates.langgraph_random_agent import LangGraphRandom
from .templates.langgraph_thinking import LangGraphThinking
from .templates.llm_agents import LLM, FastLLM, GuidedLLM, ReasoningLLM
from .templates.random_agent import Random
from .templates.reasoning_agent import ReasoningAgent
from .templates.smolagents import SmolCodingAgent, SmolVisionAgent
from .templates.manual_script_runner import ManualScriptText, ManualScriptVision
from .templates.llm_agents import BimodalGuidedChecklistLLM

from .templates.as66 import (
    AS66GuidedAgent,
    AS66VisualGuidedAgent,
    AS66ManualScriptText,
    AS66ManualScriptVision,
    AS66GuidedAgent64,
    AS66MemoryAgent,
)

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

# add all the recording files as valid agent names
#for rec in Recorder.list():
#    AVAILABLE_AGENTS[rec] = Playback

# update the agent dictionary to include subclasses of LLM class
AVAILABLE_AGENTS["reasoningagent"] = ReasoningAgent

# LS20 manual runners: friendly aliases
AVAILABLE_AGENTS.update({
    "manualscripttext": ManualScriptText,
    "manualscriptvision": ManualScriptVision,
    "manual-script-text": ManualScriptText,
    "manual-script-vision": ManualScriptVision,
    "manual_script_text": ManualScriptText,
    "manual_script_vision": ManualScriptVision,
})

# AS66 runners: friendly aliases
AVAILABLE_AGENTS.update({
    "as66manualscripttext": AS66ManualScriptText,
    "as66manualscriptvision": AS66ManualScriptVision,
    "as66-manual-text": AS66ManualScriptText,
    "as66-manual-vision": AS66ManualScriptVision,

    "as66guidedagent": AS66GuidedAgent,
    "as66-visual-guided": AS66VisualGuidedAgent,
    "as66visualguidedagent": AS66VisualGuidedAgent,
    "as66-guided": AS66GuidedAgent,
    "as66guidedagent64": AS66GuidedAgent64,
    "as66memoryagent": AS66MemoryAgent,
    "as66-memory": AS66MemoryAgent,
})

__all__ = [
    "Swarm",
    "Random",
    "LangGraphFunc",
    "LangGraphTextOnly",
    "LangGraphThinking",
    "LangGraphRandom",
    "LLM",
    "FastLLM",
    "ReasoningLLM",
    "GuidedLLM",
    "ReasoningAgent",
    "SmolCodingAgent",
    "SmolVisionAgent",
    "Agent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
]