# agents/templates/manual_script_runner.py
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from PIL import Image
from openai import OpenAI

# --- Avoid circular/package import traps: load env first, then import agents bits ---
log = logging.getLogger(__name__)

def _load_env() -> None:
    """Load .env (and .env.example) from repo root and strip stray quotes/spaces."""
    try_root = Path(__file__).resolve().parents[2]
    candidates = [try_root / ".env.example", try_root / ".env"]
    for p in candidates:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip()
            if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                v = v[1:-1]
            os.environ.setdefault(k, v)

    for key in ("OPENAI_API_KEY", "ARC_API_KEY", "AGENTOPS_API_KEY"):
        v = os.getenv(key)
        if v:
            v = v.strip().strip("'").strip('"')
            os.environ[key] = v

_load_env()

# Now import the project classes (package __init__ may import other templates)
from ..agent import Agent
from ..structs import FrameData, GameAction, GameState
from .llm_agents import GuidedLLM, VisualGuidedLLM


# ---------- utilities ----------

def _ts_dir() -> Path:
    """Base transcripts dir (mirrors style from LLM agents)."""
    base = Path(os.getenv("TRANSCRIPTS_DIR", "transcripts")).resolve()
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    d = base / stamp
    d.mkdir(parents=True, exist_ok=True)
    (d / "images").mkdir(exist_ok=True)
    return d

def _pretty_print_3d(array_3d: list[list[list[Any]]]) -> str:
    lines: list[str] = []
    for i, block in enumerate(array_3d):
        lines.append(f"Grid {i}:")
        for row in block:
            lines.append("  " + str(row))
        lines.append("")
    return "\n".join(lines)

def _grid_to_png_bytes(grid: list[list[int]]) -> bytes:
    """
    Convert a single 2D grid to a compact PNG using the canonical KEY_COLORS palette.
    Reuses VisualGuidedLLM.KEY_COLORS so colors match the rest of the codebase.
    """
    # Use the same mapping as VisualGuidedLLM
    try:
        from .llm_agents import VisualGuidedLLM
        key_colors = VisualGuidedLLM.KEY_COLORS  # dict[int, "#RRGGBB"]
    except Exception:
        key_colors = {
            0: "#FFFFFF", 1: "#CCCCCC", 2: "#999999",
            3: "#666666", 4: "#333333", 5: "#000000",
            6: "#E53AA3", 7: "#FF7BCC", 8: "#F93C31",
            9: "#1E93FF", 10: "#88D8F1", 11: "#FFDC00",
            12: "#FF851B", 13: "#921231", 14: "#4FCC30",
            15: "#A356D6",
        }

    def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
        hex_str = hex_str.strip()
        if not hex_str.startswith("#") or len(hex_str) != 7:
            return (136, 136, 136)  # default gray if malformed
        return (int(hex_str[1:3], 16), int(hex_str[3:5], 16), int(hex_str[5:7], 16))

    h = len(grid)
    w = len(grid[0]) if h else 0
    im = Image.new("RGB", (w, h), (0, 0, 0))
    px = im.load()

    for y, row in enumerate(grid):
        for x, val in enumerate(row):
            rgb = _hex_to_rgb(key_colors.get(val & 15, "#888888"))
            px[x, y] = rgb

    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


# ---------- Manual traces (edit these in THIS file) ----------

# Two simple examples for each level; add your own here.
# Use plain directions: "Up", "Down", "Left", "Right".
MANUAL_TRACES: dict[int, dict[str, list[str]]] = {
    1: {
        # Example A (Level 1)
        "l1_demo_a": ["Up", "Down", "Down", "Left", "Right"],
        # Example B (Level 1)
        "l1_demo_b": ["Right", "Up", "Left", "Down", "Down"],
    },
    2: {
        # Example A (Level 2) -- you MUST provide a Level-1 prelude to reach L2 (see env below)
        "l2_demo_a": ["Up", "Up", "Right", "Down", "Left", "Left"],
        # Example B (Level 2)
        "l2_demo_b": ["Down", "Right", "Right", "Up", "Left", "Up"],
    },
}

def _load_l2_prelude() -> list[str]:
    raw = os.getenv("L2_PRELUDE_MOVES_JSON", "").strip()
    if not raw:
        return []
    try:
        arr = json.loads(raw)
        if not isinstance(arr, list) or not all(isinstance(x, str) for x in arr):
            raise ValueError
        return arr
    except Exception as e:
        raise RuntimeError(
            "L2_PRELUDE_MOVES_JSON must be a JSON array of moves, e.g. "
            '["Up","Up","Right","Right","Down","Left"]'
        ) from e

def _dir_to_action(d: str) -> GameAction:
    k = d.strip().lower()
    if k in ("u", "up"): return GameAction.ACTION1
    if k in ("d", "down"): return GameAction.ACTION2
    if k in ("l", "left"): return GameAction.ACTION3
    if k in ("r", "right"): return GameAction.ACTION4
    raise ValueError(f"Unknown direction '{d}'")

def _action_to_dir_name(action_name: str) -> str:
    """Map enum name to human direction."""
    k = action_name.upper()
    if k.endswith("ACTION1"): return "Up"
    if k.endswith("ACTION2"): return "Down"
    if k.endswith("ACTION3"): return "Left"
    if k.endswith("ACTION4"): return "Right"
    if k.endswith("RESET") or k.endswith("ACTION5"): return "Reset"
    return action_name

def _selected_trace() -> tuple[int, str, list[str]]:
    """
    Decide which trace to run.
    Priority:
      1) LS20_OVERRIDE_MOVES_JSON — JSON array of moves, returns ("override")
      2) LS20_SELECTED_TRACE — "l1:<name>" or "l2:<name>"
      3) default — l1:l1_demo_a
    For L2, an L1 prelude must be provided via L2_PRELUDE_MOVES_JSON.
    """
    override_json = os.getenv("LS20_OVERRIDE_MOVES_JSON", "").strip()
    if override_json:
        try:
            arr = json.loads(override_json)
            if not isinstance(arr, list) or not arr:
                raise ValueError
            return (0, "override", [str(x) for x in arr])
        except Exception as e:
            raise RuntimeError(
                "LS20_OVERRIDE_MOVES_JSON must be a JSON array of moves (strings)."
            ) from e

    sel = os.getenv("LS20_SELECTED_TRACE", "").strip()
    if not sel:
        sel = "l1:l1_demo_a"

    if ":" not in sel:
        raise RuntimeError("LS20_SELECTED_TRACE must look like 'l1:<name>' or 'l2:<name>'")

    level_tag, name = sel.split(":", 1)
    level = 1 if level_tag.lower() == "l1" else 2 if level_tag.lower() == "l2" else None
    if level is None:
        raise RuntimeError("LS20_SELECTED_TRACE must start with 'l1:' or 'l2:'")

    choices = MANUAL_TRACES.get(level, {})
    if name not in choices:
        avail = ", ".join(sorted(choices.keys()))
        raise RuntimeError(f"Trace '{name}' not found for level {level}. Available: {avail or '(none)'}")

    moves = list(choices[name])
    if level == 2:
        prelude = _load_l2_prelude()
        if not prelude:
            raise RuntimeError(
                "You selected a Level‑2 trace but did not set L2_PRELUDE_MOVES_JSON. "
                "Provide a JSON array of moves that solves L1 for your current card/game."
            )
        moves = list(prelude) + moves

    return (level, name, moves)

def ls20_script_moves() -> list[GameAction]:
    """
    Your exact move list for this run, chosen by env.
    - LS20_OVERRIDE_MOVES_JSON: JSON array of moves  -> takes precedence
    - LS20_SELECTED_TRACE: "l1:<name>" | "l2:<name>"
    """
    _, _, moves = _selected_trace()
    try:
        return [_dir_to_action(x) for x in moves]
    except Exception as e:
        raise RuntimeError(f"Invalid move found in selected trace: {e}") from e


# ---------- transcript records ----------

@dataclass
class StepRecord:
    step: int
    trace_name: str
    level: int
    input: str                # enum name e.g. ACTION1
    state_before: str         # GameState name BEFORE the move
    state_after: str          # GameState name AFTER the move
    score: int
    grid_text: Optional[str] = None
    image_path: Optional[str] = None
    image_data_url: Optional[str] = None


# ---------- annotation prompts (omniscient: whole transcript available) ----------

def _build_text_annotation_prompt(
    guided_context: str,
    all_records: list[StepRecord],
    focus_step: int,
) -> list[dict[str, Any]]:
    """
    Provide the ENTIRE transcript as alternating input→state pairs so the model can
    "cheat" with future steps, but require it to explain ONLY the focus move.
    Return strictly valid JSON with minimal SFT-friendly fields.
    """
    lines: list[str] = []
    for r in all_records:
        dir_name = _action_to_dir_name(r.input)
        lines.append(f"- Move #{r.step:02d}: INPUT={dir_name}")
        lines.append(f"  STATE: {r.state_before} -> {r.state_after} | SCORE={r.score}")
        if r.grid_text:
            lines.append("  Grid:")
            lines.append("  " + "\n  ".join(r.grid_text.splitlines()))
    transcript_text = "\n".join(lines)

    system = (
        guided_context
        + "\n\nYou are NOT choosing an action. You are annotating a past move."
        " Return JSON only. No extra text."
    )
    user = f"""
# Full transcript (input→state pairs)
{transcript_text}

Task:
Explain the rationale for the SINGLE move with index {focus_step}.
You MAY use what happens later in the transcript to craft a better explanation,
but phrase it as if it was a reasonable plan at that time (do NOT say you saw the future).

Return STRICT JSON with:
- "step": integer (the move index)
- "action": string (one of Up/Down/Left/Right)
- "cot": string (2–5 sentences; concise chain-of-thought for that move)
- "observed_effect": string (what changed or didn’t in the state)
- "progress_assessment": string; one of "closer", "neutral/no-change", "worse"
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_vision_annotation_prompt(
    visual_context: str,
    all_records: list[StepRecord],
    focus_step: int,
) -> list[dict[str, Any]]:
    """Vision version: attach an image per state if available, keep the same JSON schema."""
    content: list[dict[str, Any]] = [{"type": "text", "text": f"# Explain move #{focus_step} only (omniscient transcript)"}]
    for r in all_records:
        dir_name = _action_to_dir_name(r.input)
        content.append({"type": "text", "text": f"Move #{r.step:02d}: INPUT={dir_name}"})
        content.append({"type": "text", "text": f"STATE: {r.state_before} -> {r.state_after} | SCORE={r.score}"})
        if r.image_data_url:
            content.append({"type": "image_url", "image_url": {"url": r.image_data_url, "detail": "high"}})

    system = visual_context + "\nReturn JSON only. Do not choose future actions."
    user_tail = """
Return STRICT JSON with:
- "step": integer
- "action": string (Up/Down/Left/Right)
- "cot": string (2–5 sentences)
- "observed_effect": string
- "progress_assessment": "closer" | "neutral/no-change" | "worse"
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content + [{"type": "text", "text": user_tail}]},
    ]


# ---------- manual-script agents ----------

class _ManualScriptBase(Agent):
    """
    Drives a fixed script of moves through the API and writes rich transcripts.
    After running the script, performs an annotation pass with a GPT‑5 class model.

    The script to run is selected by:
      - LS20_OVERRIDE_MOVES_JSON (JSON array) OR
      - LS20_SELECTED_TRACE = "l1:<name>" | "l2:<name>" (+ L2_PRELUDE_MOVES_JSON for L2)
    """

    MAX_ACTIONS = 10_000

    # annotation controls (default: omniscient)
    ANNOTATE: bool = True
    ANNOTATE_MODE: Literal["omniscient", "sliding", "full"] = "omniscient"
    WINDOW_SIZE: int = 6  # used only in sliding

    # models
    TEXT_MODEL: str = "gpt-5"
    VISION_MODEL: str = "gpt-5"
    REASONING_EFFORT: Optional[str] = "low"

    # toggled by child class
    USE_IMAGES: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        arc_key = os.getenv("ARC_API_KEY", "").strip()
        if not arc_key:
            raise RuntimeError("ARC_API_KEY not set. Put it (without quotes/spaces) in the repo-root .env")
        super().__init__(*args, **kwargs)

        # Resolve which trace is requested
        level, name, _moves = _selected_trace()
        self._trace_name = f"L{level}:{name}"

        # Load LS20 script
        self._script: list[GameAction] = ls20_script_moves()
        self._ptr: int = 0
        self._level: int = 1
        self._last_score: int = 0

        # transcript sinks
        self._out_dir = _ts_dir()
        self._text_path = self._out_dir / "ls20.manual.text.jsonl"
        self._vision_path = self._out_dir / "ls20.manual.vision.jsonl"
        self._annot_text_path = self._out_dir / "ls20.manual.text.annot.jsonl"
        self._annot_vision_path = self._out_dir / "ls20.manual.vision.annot.jsonl"
        self._sft_path = self._out_dir / "ls20.manual.sft.jsonl"

        # memo of per-step transcript rows
        self._records: list[StepRecord] = []

        # OpenAI client (lazy in case annotation disabled)
        self._client: Optional[OpenAI] = None
        self._annot_ran = False

        # allow env overrides
        mode = os.getenv("ANNOTATE_MODE", "").strip().lower()
        if mode in ("omniscient", "full", "sliding"):
            self.ANNOTATE_MODE = mode  # type: ignore[assignment]
        self.ANNOTATE = os.getenv("ANNOTATE", "true").strip().lower() != "false"
        ws = os.getenv("WINDOW_SIZE", "").strip()
        if ws.isdigit():
            self.WINDOW_SIZE = max(1, int(ws))

    # ----- Agent loop plumbing -----

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Stop when script is exhausted or the game is won.
        return (self._ptr >= len(self._script)) or (latest_frame.state is GameState.WIN)

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        # Always RESET to start or after GAME_OVER
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        if self._ptr >= len(self._script):
            # No-op safety; return ACTION5 but main loop will stop via is_done()
            return GameAction.ACTION5

        act = self._script[self._ptr]
        self._ptr += 1

        # include a breadcrumb marking this as a scripted run
        act.reasoning = {
            "source": "manual_script_runner",
            "script_index": self._ptr,
            "level_hint": self._level,
            "trace": self._trace_name,
        }
        return act

    def append_frame(self, frame: FrameData) -> None:
        """Augment: detect level changes & write running transcript rows as we go."""
        super().append_frame(frame)

        # Skip the initial empty frame 0
        if len(self.frames) <= 1:
            self._last_score = frame.score
            return

        # Heuristic: a score bump with NOT_FINISHED usually marks a level gate; full_reset also relevant
        if (frame.score > self._last_score) and (frame.state != GameState.WIN):
            self._level += 1
        self._last_score = frame.score

        # Build one record for this step
        step_idx = len(self.frames) - 1  # step number aligned to appended output
        action_name = frame.action_input.id.name if frame.action_input else "UNKNOWN"
        state_after_name = frame.state.name
        state_before_name = self._records[-1].state_after if self._records else "NOT_PLAYED"
        grid_text = _pretty_print_3d(frame.frame) if not self.USE_IMAGES else None

        img_path = None
        data_url = None
        if self.USE_IMAGES and frame.frame:
            grid = frame.frame[-1] if frame.frame else []
            if grid and len(grid) and len(grid[0]):
                png = _grid_to_png_bytes(grid)
                img_path = str(self._out_dir / "images" / f"{frame.score:02d}-{step_idx:04d}.png")
                with open(img_path, "wb") as f:
                    f.write(png)
                data_url = f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}"

        rec = StepRecord(
            step=step_idx,
            trace_name=self._trace_name,
            level=self._level,
            input=action_name,
            state_before=state_before_name,
            state_after=state_after_name,
            score=frame.score,
            grid_text=grid_text,
            image_path=img_path,
            image_data_url=data_url,
        )
        self._records.append(rec)

        # Stream to per‑modality transcripts
        if grid_text is not None:
            with open(self._text_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
        if img_path is not None:
            with open(self._vision_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")

    # ----- annotation pass -----

    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set. Put it (without quotes) in your .env at repo root."
                )
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _annotate_text(self) -> None:
        """
        Textual annotation: reuse GuidedLLM textual game context.
        Model sees the ENTIRE transcript (omniscient) and explains only the focus move.
        """
        if not self._records:
            raise RuntimeError("No records collected; nothing to annotate.")

        # Create a GuidedLLM instance to access its prompt text
        try:
            guided = GuidedLLM(
                card_id=self.card_id, game_id=self.game_id, agent_name="manual-annot-text",
                ROOT_URL=self.ROOT_URL, record=False
            )
            # Use the current/latest frame to build the context
            guided_context = guided.build_user_prompt(self.frames[-1])
        except Exception:
            guided_context = (
                "You are analyzing a LockSmith game transcript. "
                "Do not choose actions; only explain the rationale for a single move using the state."
            )

        client = self._ensure_client()
        out = open(self._annot_text_path, "a", encoding="utf-8")
        sft = open(self._sft_path, "a", encoding="utf-8")

        mode = self.ANNOTATE_MODE
        if mode == "omniscient":
            # Whole transcript each time
            for rec in self._records:
                messages = _build_text_annotation_prompt(guided_context, self._records, rec.step)
                resp = client.chat.completions.create(
                    model=self.TEXT_MODEL,
                    messages=messages,
                    reasoning_effort=self.REASONING_EFFORT,
                )
                txt = (resp.choices[0].message.content or "").strip()
                try:
                    data = json.loads(txt)
                except Exception as e:
                    raise RuntimeError(
                        f"Non‑JSON annotation for step {rec.step}: {txt!r}"
                    ) from e
                data.update({"mode": "omniscient"})
                out.write(json.dumps(data, ensure_ascii=False) + "\n")

                # Minimal SFT row
                dir_name = _action_to_dir_name(rec.input)
                cot = data.get("cot") or data.get("rationale") or ""
                sft.write(json.dumps({
                    "move": dir_name,
                    "cot": cot,
                    "state_before": rec.state_before,
                    "state_after": rec.state_after,
                    "level": rec.level,
                    "trace": rec.trace_name,
                    "index": rec.step,
                }, ensure_ascii=False) + "\n")

        elif mode == "full":
            acc: list[StepRecord] = []
            for rec in self._records:
                acc.append(rec)
                messages = _build_text_annotation_prompt(guided_context, acc, rec.step)
                resp = client.chat.completions.create(
                    model=self.TEXT_MODEL,
                    messages=messages,
                    reasoning_effort=self.REASONING_EFFORT,
                )
                txt = (resp.choices[0].message.content or "").strip()
                try:
                    data = json.loads(txt)
                except Exception as e:
                    raise RuntimeError(f"Non‑JSON annotation for step {rec.step}: {txt!r}") from e
                data.update({"mode": "full"})
                out.write(json.dumps(data, ensure_ascii=False) + "\n")

                dir_name = _action_to_dir_name(rec.input)
                cot = data.get("cot") or data.get("rationale") or ""
                sft.write(json.dumps({
                    "move": dir_name,
                    "cot": cot,
                    "state_before": rec.state_before,
                    "state_after": rec.state_after,
                    "level": rec.level,
                    "trace": rec.trace_name,
                    "index": rec.step,
                }, ensure_ascii=False) + "\n")

        else:  # sliding
            k = max(1, int(self.WINDOW_SIZE))
            for i, rec in enumerate(self._records):
                start = max(0, i - (k - 1))
                win = self._records[start : i + 1]
                messages = _build_text_annotation_prompt(guided_context, win, rec.step)
                resp = client.chat.completions.create(
                    model=self.TEXT_MODEL,
                    messages=messages,
                    reasoning_effort=self.REASONING_EFFORT,
                )
                txt = (resp.choices[0].message.content or "").strip()
                try:
                    data = json.loads(txt)
                except Exception as e:
                    raise RuntimeError(f"Non‑JSON annotation for step {rec.step}: {txt!r}") from e
                data.update({"mode": "sliding", "window_size": k})
                out.write(json.dumps(data, ensure_ascii=False) + "\n")

                dir_name = _action_to_dir_name(rec.input)
                cot = data.get("cot") or data.get("rationale") or ""
                sft.write(json.dumps({
                    "move": dir_name,
                    "cot": cot,
                    "state_before": rec.state_before,
                    "state_after": rec.state_after,
                    "level": rec.level,
                    "trace": rec.trace_name,
                    "index": rec.step,
                }, ensure_ascii=False) + "\n")

        out.close()
        sft.close()

    def _annotate_vision(self) -> None:
        """
        Vision annotation: reuse VisualGuidedLLM's shared game context.
        """
        if not self._records:
            raise RuntimeError("No records collected; nothing to annotate.")

        # Instantiate to fetch the context string
        try:
            visual = VisualGuidedLLM(
                card_id=self.card_id, game_id=self.game_id, agent_name="manual-annot-vision",
                ROOT_URL=self.ROOT_URL, record=False
            )
            visual_context = visual.build_game_context_prompt()
        except Exception:
            visual_context = (
                "You are analyzing a LockSmith game via images. "
                "Annotate the specified move given the small window of frames."
            )

        client = self._ensure_client()
        out = open(self._annot_vision_path, "a", encoding="utf-8")

        if self.ANNOTATE_MODE == "omniscient":
            records = self._records
            for rec in records:
                msgs = _build_vision_annotation_prompt(visual_context, records, rec.step)
                resp = client.chat.completions.create(
                    model=self.VISION_MODEL,
                    messages=msgs,
                    reasoning_effort=self.REASONING_EFFORT,
                )
                txt = (resp.choices[0].message.content or "").strip()
                try:
                    data = json.loads(txt)
                except Exception as e:
                    raise RuntimeError(f"Non‑JSON vision annotation for step {rec.step}: {txt!r}") from e
                data.update({"mode": "omniscient"})
                out.write(json.dumps(data, ensure_ascii=False) + "\n")

        elif self.ANNOTATE_MODE == "full":
            acc: list[StepRecord] = []
            for rec in self._records:
                acc.append(rec)
                msgs = _build_vision_annotation_prompt(visual_context, acc, rec.step)
                resp = client.chat.completions.create(
                    model=self.VISION_MODEL,
                    messages=msgs,
                    reasoning_effort=self.REASONING_EFFORT,
                )
                txt = (resp.choices[0].message.content or "").strip()
                try:
                    data = json.loads(txt)
                except Exception as e:
                    raise RuntimeError(f"Non‑JSON vision annotation for step {rec.step}: {txt!r}") from e
                data.update({"mode": "full"})
                out.write(json.dumps(data, ensure_ascii=False) + "\n")
        else:  # sliding
            k = max(1, int(self.WINDOW_SIZE))
            for i, rec in enumerate(self._records):
                start = max(0, i - (k - 1))
                win = self._records[start : i + 1]
                msgs = _build_vision_annotation_prompt(visual_context, win, rec.step)
                resp = client.chat.completions.create(
                    model=self.VISION_MODEL,
                    messages=msgs,
                    reasoning_effort=self.REASONING_EFFORT,
                )
                txt = (resp.choices[0].message.content or "").strip()
                try:
                    data = json.loads(txt)
                except Exception as e:
                    raise RuntimeError(f"Non‑JSON vision annotation for step {rec.step}: {txt!r}") from e
                data.update({"mode": "sliding", "window_size": k})
                out.write(json.dumps(data, ensure_ascii=False) + "\n")

        out.close()

    # ----- cleanup -----
    def cleanup(self, scorecard: Optional[Any] = None) -> None:
        super().cleanup(scorecard)
        if self._annot_ran or not self.ANNOTATE or scorecard is not None:
            return
        try:
            if self.USE_IMAGES:
                self._annotate_vision()
            else:
                self._annotate_text()
        finally:
            self._annot_ran = True


class ManualScriptText(_ManualScriptBase):
    """Runs the LS20 manual script and produces a textual transcript + textual annotation (omniscient by default)."""
    USE_IMAGES = False


class ManualScriptVision(_ManualScriptBase):
    """Runs the LS20 manual script and produces a visual transcript + vision-based annotation (omniscient by default)."""
    USE_IMAGES = True


if __name__ == "__main__":
    # Running this file directly will not have proper package context.
    raise SystemExit(
        "Do not run this file directly. Use your project's agent runner to run the "
        "ManualScriptText or ManualScriptVision class (module path: agents.templates.manual_script_runner). "
        "Also set LS20_SELECTED_TRACE (e.g., 'l1:l1_demo_a')."
    )
