# agents/templates/manual_script_runner.py

from __future__ import annotations

import base64
import io
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from PIL import Image, ImageDraw

from openai import OpenAI

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState
from .llm_agents import GuidedLLM, VisualGuidedLLM

log = logging.getLogger(__name__)


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
        # Fallback: hardcode the canonical map
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

def _save_step_image(img_bytes: bytes, out_dir: Path, score: int, step_idx: int) -> str:
    name = f"{score:02d}-{step_idx:04d}.png"
    path = out_dir / "images" / name
    with open(path, "wb") as f:
        f.write(img_bytes)
    return str(path)


# ---------- LS20 manual scripts ----------

def _dir_to_action(d: str) -> GameAction:
    k = d.strip().lower()
    if k in ("u", "up"): return GameAction.ACTION1
    if k in ("d", "down"): return GameAction.ACTION2
    if k in ("l", "left"): return GameAction.ACTION3
    if k in ("r", "right"): return GameAction.ACTION4
    raise ValueError(f"Unknown direction '{d}'")

def ls20_script_moves() -> list[GameAction]:
    """
    Your exact level-1 and level-2 move lists.
    L1: Up Up Left Up Up
    L2: Down Right Right Right Up Up Up Up Left Left Down Up Down Up Down Up Down Up Up Left Left Left Down Down Down Down
    """
    level1 = ["Up", "Up", "Left", "Up", "Up"]
    level2 = [
        "Down",
        "Right", "Right", "Right",
        "Up", "Up", "Up", "Up",
        "Left", "Left",
        "Down", "Up", "Down", "Up", "Down", "Up", "Down", "Up",
        "Up",
        "Left", "Left", "Left",
        "Down", "Down", "Down", "Down",

    ]
    return [_dir_to_action(x) for x in (level1 + level2)]
    #return [_dir_to_action(x) for x in (level1)]


# ---------- transcript records ----------

@dataclass
class StepRecord:
    step: int
    level: int
    input: str
    state: str           # GameState
    score: int
    grid_text: Optional[str] = None
    image_path: Optional[str] = None
    image_data_url: Optional[str] = None


# ---------- annotation prompts ----------

def _build_text_annotation_prompt(
    guided_context: str,
    window_records: list[StepRecord],
    focus_step: int,
) -> list[dict[str, Any]]:
    """
    Reuse the textual GuidedLLM game instructions (guided_context) but ask strictly
    for rationale about a single move. Provide the window as alternating input/state pairs.
    """
    # Emit the transcript window exactly as input/state alternating
    lines: list[str] = []
    for r in window_records:
        lines.append(f"- Move #{r.step}: INPUT={r.input}")
        lines.append(f"  → STATE={r.state} | SCORE={r.score}")
        if r.grid_text:
            lines.append("  Grid:")
            lines.append("  " + "\n  ".join(r.grid_text.splitlines()))
    window_text = "\n".join(lines)

    system = (
        guided_context
        + "\n\nYou are NOT choosing an action. You are annotating a past move."
        " Provide a concise but concrete rationale for the specified move, grounded in the visible state."
    )
    user = f"""
We will analyze exactly one move in the following transcript window.
Transcript (as input→state pairs):
{window_text}

Task:
Explain the rationale for the SINGLE move with step index {focus_step}.
Explain from the perspective of the player: what the move tried to accomplish,
what changed in the state (if visible), and how/why this advances the goal of winning.
If the move looks suboptimal or illegal (no change), say why and what alternative would have been better.

Return JSON with the following fields:
- "step": integer (the move index you are explaining)
- "action": string (the move name)
- "rationale": string (2–5 sentences)
- "observed_effect": string (what changed or didn’t)
- "progress_assessment": string ("closer", "neutral/no-change", or "worse")
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_vision_annotation_prompt(
    visual_context: str,
    window_records: list[StepRecord],
    focus_step: int,
) -> list[dict[str, Any]]:
    """
    Vision version: use VisualGuidedLLM's shared context, then attach the image for each state
    in the window. We still only explain the first move in the window (focus_step).
    """
    # Build a multimodal user content with alternating text + images
    content: list[dict[str, Any]] = [{"type": "text", "text": f"# Window (explain move #{focus_step})"}]
    for r in window_records:
        # Input line
        content.append({"type": "text", "text": f"Move #{r.step}: INPUT={r.input}"})
        # State line
        state_line = f"→ STATE={r.state} | SCORE={r.score}"
        content.append({"type": "text", "text": state_line})
        # Attach image if present
        if r.image_data_url:
            content.append({"type": "image_url", "image_url": {"url": r.image_data_url, "detail": "high"}})

    system = visual_context + "\nYou are NOT choosing an action. Annotate the specified move only."
    user_tail = f"""
Explain the rationale for move #{focus_step} only. Provide:
- "step": integer
- "action": string
- "rationale": string (2–5 sentences, reference objects in the image if needed)
- "observed_effect": string (what changed or didn’t)
- "progress_assessment": "closer" | "neutral/no-change" | "worse"
Return strictly valid JSON.
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content + [{"type": "text", "text": user_tail}]},
    ]


# ---------- manual-script agents ----------

class _ManualScriptBase(Agent):
    """
    Drives a fixed script of moves through the API and writes rich transcripts.
    After running the script, optionally performs an annotation pass with a GPT‑5 class model.
    """

    

    MAX_ACTIONS = 10_000

    # annotation controls (override in child or pass via env)
    ANNOTATE: bool = True
    ANNOTATE_MODE: Literal["sliding", "full"] = "full"
    WINDOW_SIZE: int = 6

    # models
    TEXT_MODEL: str = "gpt-5"   # low reasoning textual
    VISION_MODEL: str = "gpt-5" # low reasoning multimodal
    REASONING_EFFORT: Optional[str] = "low"

    # toggled by child class
    USE_IMAGES: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Load LS20 script
        self._script: list[GameAction] = ls20_script_moves()
        self._ptr: int = 0
        self._level: int = 1
        self._last_score: int = 0

        # transcript sink
        self._out_dir = _ts_dir()
        self._text_path = self._out_dir / "ls20.manual.text.jsonl"
        self._vision_path = self._out_dir / "ls20.manual.vision.jsonl"
        self._annot_text_path = self._out_dir / "ls20.manual.text.annot.jsonl"
        self._annot_vision_path = self._out_dir / "ls20.manual.vision.annot.jsonl"

        # memo of per-step transcript rows
        self._records: list[StepRecord] = []

        # OpenAI client (lazy in case annotation disabled)
        self._client: Optional[OpenAI] = None
        self._annot_ran = False

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

        # include a small reasoning breadcrumb marking this as a scripted run
        act.reasoning = {
            "source": "manual_script_runner",
            "script_index": self._ptr,
            "level_hint": self._level,
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
        state_name = frame.state.name
        grid_text = _pretty_print_3d(frame.frame) if not self.USE_IMAGES else None

        img_path = None
        data_url = None
        if self.USE_IMAGES and frame.frame:
            grid = frame.frame[-1] if frame.frame else []
            if grid and len(grid) and len(grid[0]):
                png = _grid_to_png_bytes(grid)
                img_path = _save_step_image(png, self._out_dir, frame.score, step_idx)
                data_url = f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}"

        rec = StepRecord(
            step=step_idx,
            level=self._level,
            input=action_name,
            state=state_name,
            score=frame.score,
            grid_text=grid_text,
            image_path=img_path,
            image_data_url=data_url,
        )
        self._records.append(rec)

        # Also stream to per‑modality transcripts
        if grid_text is not None:
            with open(self._text_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
        if img_path is not None:
            with open(self._vision_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")

    # ----- annotation pass -----

    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        return self._client

    def _annotate_text(self) -> None:
        """
        Textual annotation: reuse the exact textual guidance from GuidedLLM.
        We instantiate GuidedLLM only to obtain its game-context text.
        """
        if not self._records:
            return

        # Create a GuidedLLM instance to access its prompt text
        try:
            guided = GuidedLLM(
                card_id=self.card_id, game_id=self.game_id, agent_name="manual-annot-text",
                ROOT_URL=self.ROOT_URL, record=False
            )
        except Exception:
            guided = None

        def guided_context_for(frame: FrameData) -> str:
            # Fallback to a minimal context if instantiation failed
            if guided:
                try:
                    return guided.build_user_prompt(frame)
                except Exception:
                    pass
            return (
                "You are analyzing a LockSmith game transcript. Do not choose actions; "
                "only explain the rationale for the given move using the state."
            )

        client = self._ensure_client()
        out = open(self._annot_text_path, "a", encoding="utf-8")

        if self.ANNOTATE_MODE == "full":
            # Accumulate all prior records and their prior reasoning
            acc: list[StepRecord] = []
            for rec in self._records:
                acc.append(rec)
                frame = self.frames[rec.step] if rec.step < len(self.frames) else self.frames[-1]
                messages = _build_text_annotation_prompt(
                    guided_context_for(frame), acc, rec.step
                )
                resp = client.chat.completions.create(
                    model=self.TEXT_MODEL,
                    messages=messages,
                    reasoning_effort=self.REASONING_EFFORT,
                    
                )
                txt = resp.choices[0].message.content or "{}"
                try:
                    data = json.loads(txt)
                except Exception:
                    data = {"raw": txt}
                data.update({"mode": "full"})
                out.write(json.dumps(data, ensure_ascii=False) + "\n")

        else:  # sliding
            k = max(1, int(self.WINDOW_SIZE))
            for i, rec in enumerate(self._records):
                start = max(0, i - (k - 1))
                win = self._records[start : i + 1]
                frame = self.frames[rec.step] if rec.step < len(self.frames) else self.frames[-1]
                messages = _build_text_annotation_prompt(
                    guided_context_for(frame), win, rec.step
                )
                resp = client.chat.completions.create(
                    model=self.TEXT_MODEL,
                    messages=messages,
                    reasoning_effort=self.REASONING_EFFORT,
                    
                )
                txt = resp.choices[0].message.content or "{}"
                try:
                    data = json.loads(txt)
                except Exception:
                    data = {"raw": txt}
                data.update({"mode": "sliding", "window_size": k})
                out.write(json.dumps(data, ensure_ascii=False) + "\n")

        out.close()

    def _annotate_vision(self) -> None:
        """
        Vision annotation: reuse VisualGuidedLLM's shared game context.
        """
        if not self._records:
            return

        # Instantiate to fetch the context string
        try:
            visual = VisualGuidedLLM(
                card_id=self.card_id, game_id=self.game_id, agent_name="manual-annot-vision",
                ROOT_URL=self.ROOT_URL, record=False
            )
            visual_context = visual.build_game_context_prompt()
        except Exception:
            visual = None
            visual_context = (
                "You are analyzing a LockSmith game via images. Do not choose actions; "
                "explain the rationale for a specified move given a small window of frames."
            )

        client = self._ensure_client()
        out = open(self._annot_vision_path, "a", encoding="utf-8")

        if self.ANNOTATE_MODE == "full":
            acc: list[StepRecord] = []
            for rec in self._records:
                acc.append(rec)
                msgs = _build_vision_annotation_prompt(visual_context, acc, rec.step)
                resp = client.chat.completions.create(
                    model=self.VISION_MODEL,
                    messages=msgs,
                    reasoning_effort=self.REASONING_EFFORT,
                    
                )
                txt = resp.choices[0].message.content or "{}"
                try:
                    data = json.loads(txt)
                except Exception:
                    data = {"raw": txt}
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
                txt = resp.choices[0].message.content or "{}"
                try:
                    data = json.loads(txt)
                except Exception:
                    data = {"raw": txt}
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
    """Runs the LS20 manual script and produces a textual transcript + textual annotation."""
    USE_IMAGES = False


class ManualScriptVision(_ManualScriptBase):
    """Runs the LS20 manual script and produces a visual transcript + vision-based annotation."""
    USE_IMAGES = True
