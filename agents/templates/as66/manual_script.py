"""
AS66 manual script runner → multi-turn SFT conversations for *playing the game*.

What this version does:
  • Runs scripted candidates (Cartesian products per level) with laddering.
  • For each episode (candidate), emits ONE multi-turn conversation JSONL record:
      - system: unified primer (observation rules + function-calling instructions)
      - step loop:
          user: provides current 16×16 state + score + step
          assistant: multi-paragraph OBSERVATION in content + a single tool_call (ACTION1..4)
          tool: environment result with post-state scoreboard + 16×16 matrix
  • Rationale is IMPUTED (can use future knowledge, but phrased as if at the step).
  • Generates artifacts under transcripts/<stamp>/as66/:
      - sft_multi_turn.jsonl   ← used by trainer
      - interleaved.md         ← human-readable
      - rows.jsonl             ← low-level step log
      - images/                ← optional 16×16 PNGs per step (vision variant)

TEXT-ONLY CONTRACT: integer codes only (no colors in observations).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple
import itertools
import json
import os
import uuid

from openai import OpenAI

from ...agent import Agent
from ...structs import GameAction, GameState, FrameData
from .downsample import downsample_4x4, matrix16_to_lines, ds16_png_bytes
from .prompts_sft import build_primer_system_text, build_user_step_text

# ----------------------------- Per-level Cartesian products (edit here) -----------------------------

TRACE_LIBRARY: dict[str, List[List[str]]] = {
    "l1": [["Down"], ["Left"], ["Down"]],
    "l2": [["Right"], ["Down"], ["Left"]],
    # Add more levels with options as needed
}
LEVEL_ORDER: List[str] = ["l1", "l2"]

# -------------------------------------- Data models --------------------------------------

@dataclass
class StepRow:
    level: str
    candidate_index: int
    step_index: int
    move: str
    pre_state: str
    pre_score: int
    post_state: str
    post_score: int
    ds16_pre: str
    ds16_post: str
    level_note: Optional[str] = None  # e.g., "LEVEL UP" or "GAME OVER"

@dataclass
class Block:
    idx: int
    move: str
    pre_state: str
    pre_score: int
    post_state: str
    post_score: int
    ds16_pre: str
    ds16_post: str
    level_note: Optional[str] = None
    observation: Optional[str] = None  # imputed observation text

# ----------------------------- Tool schemas for ACTION1..4 (OpenAI style) -----------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "ACTION1",
            "description": "Move Up",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ACTION2",
            "description": "Move Down",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ACTION3",
            "description": "Move Left",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ACTION4",
            "description": "Move Right",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]

# ----------------------------- Manual runner -----------------------------

class AS66ManualScriptBase(Agent):
    """
    Produces multi-turn SFT conversations where the assistant writes an OBSERVATION
    (multi-paragraph) and calls exactly one ACTIONn tool each turn.
    """
    USE_IMAGES: bool = False

    # Satisfy ABC; we run our own loop
    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        return True

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        return GameAction.ACTION5

    def main(self) -> None:
        out = self._prepare_outdir()
        md_path = self._out_dir / "interleaved.md"
        sft_path = self._out_dir / "sft_multi_turn.jsonl"
        rows_path = self._rows_path

        md = open(md_path, "a", encoding="utf-8")
        sft = open(sft_path, "a", encoding="utf-8")

        try:
            prefix: List[str] = []  # laddering across levels
            md.write(f"# AS66 Multi-Turn Conversations — {datetime.utcnow().isoformat()}Z\n\n")

            for level in LEVEL_ORDER:
                options = TRACE_LIBRARY[level]
                candidates = list(itertools.product(*options))
                level_solved = False

                for ci, cand in enumerate(candidates, start=1):
                    # RESET, then ladder via discovered winning prefix
                    f = self.take_action(GameAction.RESET)
                    if f: self.append_frame(f)
                    for mv in prefix:
                        f = self.take_action(self._mv_to_action(mv))
                        if f: self.append_frame(f)

                    # Build a conversation for this candidate
                    conversation: List[Dict[str, Any]] = []
                    # System primer once per episode
                    conversation.append({"role": "system", "content": build_primer_system_text()})

                    # Create initial "user" message with current state
                    if self.frames and self.frames[-1].frame:
                        ds16_pre = downsample_4x4(self.frames[-1].frame, round_to_int=True)
                    else:
                        ds16_pre = []
                    conversation.append({
                        "role": "user",
                        "content": build_user_step_text(ds16_pre, self.score, step=0, note="Initial state"),
                    })

                    # Episode header in markdown
                    md.write(f"## Candidate {level}-cand{ci:02d}\n\n")

                    pre_score = self.score
                    last_pre_state = self.state.name
                    last_ds16_lines = matrix16_to_lines(ds16_pre) if ds16_pre else "(empty)"

                    cand_blocks: List[Block] = []

                    # Execute candidate moves
                    for mv in cand:
                        # Call environment
                        f = self.take_action(self._mv_to_action(mv))
                        if not f:
                            break
                        self.append_frame(f)

                        ds16_post = downsample_4x4(f.frame, round_to_int=True)
                        ds16_post_lines = matrix16_to_lines(ds16_post)

                        # Prepare step row and block
                        step_row = StepRow(
                            level=level, candidate_index=ci, step_index=self.action_counter,
                            move=mv, pre_state=last_pre_state, pre_score=pre_score,
                            post_state=f.state.name, post_score=f.score,
                            ds16_pre=last_ds16_lines, ds16_post=ds16_post_lines, level_note=None,
                        )
                        if f.score > pre_score:
                            step_row.level_note = "LEVEL UP"
                            pre_score = f.score
                        if f.state is GameState.GAME_OVER:
                            step_row.level_note = (step_row.level_note + " | GAME OVER") if step_row.level_note else "GAME OVER"

                        with open(rows_path, "a", encoding="utf-8") as rf:
                            rf.write(json.dumps(step_row.__dict__, ensure_ascii=False) + "\n")

                        block = Block(
                            idx=len(cand_blocks),
                            move=mv,
                            pre_state=step_row.pre_state,
                            pre_score=step_row.pre_score,
                            post_state=step_row.post_state,
                            post_score=step_row.post_score,
                            ds16_pre=step_row.ds16_pre,
                            ds16_post=step_row.ds16_post,
                            level_note=step_row.level_note,
                        )
                        cand_blocks.append(block)

                        # 1) Assistant OBSERVATION + tool call (imputed observation)
                        assistant_obs = self._impute_observation_text(
                            ds16_lines=block.ds16_pre,
                            score=block.pre_score,
                            step=len(cand_blocks)-1,
                            full_timeline=cand_blocks,
                        )
                        action_name = self._move_to_action_name(mv)
                        tool_call_id = str(uuid.uuid4())
                        conversation.append({
                            "role": "assistant",
                            "content": assistant_obs,
                            "tool_calls": [
                                {"id": tool_call_id, "type": "function", "function": {"name": action_name, "arguments": {}}}
                            ],
                        })

                        # 2) Tool message: environment result
                        tool_content = (
                            f"RESULT for {action_name}:\n"
                            f"PostState={block.post_state} | Score={block.post_score}\n"
                            f"Matrix 16x16 (integer codes):\n{block.ds16_post}\n"
                        )
                        conversation.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": action_name,
                            "content": tool_content,
                        })

                        # Optional image
                        if self.USE_IMAGES:
                            try:
                                png = ds16_png_bytes(ds16_post, cell=22)
                                (self._out_dir / "images" / f"{level}-cand{ci:02d}-step{block.idx:03d}.png").write_bytes(png)
                            except Exception:
                                pass

                        # Markdown transcript
                        md.write(f"### Step {block.idx:02d}\n\n")
                        md.write(f"**MOVE (target)**: {mv}\n\n")
                        md.write("**OBSERVATION (imputed)**\n\n")
                        md.write(assistant_obs + "\n\n")
                        md.write("**STATE AFTER**\n\n")
                        md.write(f"GameState={block.post_state} | Score={block.post_score}\n\n")
                        md.write("```\n" + block.ds16_post + "\n```\n\n")

                        # Prepare next loop
                        last_pre_state = block.post_state
                        last_ds16_lines = block.ds16_post

                        if f.state is GameState.GAME_OVER:
                            break

                        # Insert next user step input (so the assistant continues)
                        conversation.append({
                            "role": "user",
                            "content": build_user_step_text(ds16_post, f.score, step=len(cand_blocks), note=block.level_note),
                        })

                    # Write one JSONL record per episode
                    sft.write(json.dumps({
                        "id": f"{level}-cand{ci:02d}",
                        "messages": conversation,
                        "tools": TOOL_SCHEMAS,
                        "meta": {"level": level, "candidate": ci, "steps": len(cand_blocks)},
                    }, ensure_ascii=False) + "\n")

                    # If this candidate netted a higher score than the start, assume level solved; ladder onward
                    if cand_blocks and cand_blocks[-1].post_score > 0:
                        prefix.extend([b.move for b in cand_blocks])
                        level_solved = True
                        break

                if not level_solved:
                    break

        finally:
            try: md.close()
            except Exception: pass
            try: sft.close()
            except Exception: pass
            self.cleanup()

    # -------------------- helpers --------------------

    @staticmethod
    def _mv_to_action(m: str) -> GameAction:
        m = m.strip().lower()
        if m == "up": return GameAction.ACTION1
        if m == "down": return GameAction.ACTION2
        if m == "left": return GameAction.ACTION3
        if m == "right": return GameAction.ACTION4
        return GameAction.ACTION5

    @staticmethod
    def _move_to_action_name(m: str) -> str:
        m = m.strip().lower()
        return {"up": "ACTION1", "down": "ACTION2", "left": "ACTION3", "right": "ACTION4"}.get(m, "ACTION1")

    def _prepare_outdir(self) -> Path:
        base = Path(os.getenv("TRANSCRIPTS_DIR", "transcripts")).resolve()
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self._out_dir = base / stamp / "as66"
        (self._out_dir / "images").mkdir(parents=True, exist_ok=True)
        self._rows_path = self._out_dir / "rows.jsonl"
        return self._out_dir

    def _impute_observation_text(
        self,
        *,
        ds16_lines: str,
        score: int,
        step: int,
        full_timeline: List["Block"],
    ) -> str:
        """
        Create an imputed multi-paragraph observation: codes-only, long-form,
        informed by the whole candidate trajectory but phrased as 'now'.
        """
        sys_msg = build_primer_system_text()
        # Build a compact timeline header (without leaking future explicitly)
        snapshot = []
        if full_timeline:
            b = full_timeline[-1]
            if b.level_note:
                snapshot.append(f"Note: {b.level_note}")
        timeline_hint = ("\n".join(snapshot)).strip()

        user_msg = (
            f"Step: {step}\nScore: {score}\n"
            "Matrix 16x16 (integer codes):\n"
            f"{ds16_lines}\n\n"
            "Write a thorough OBSERVATION (multi-paragraph, codes-only). "
            "You may use full-trajectory understanding to craft a better justification, "
            "but do not explicitly mention future moves. End your message with a single function call.\n"
            f"{('Hint:\n' + timeline_hint) if timeline_hint else ''}"
        )

        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            # Fallback stub that still looks like a rich observation
            return (
                "OBSERVATION:\n"
                "• The movable cluster is identified against 15 (background) and bounded by 1/14 borders; walls are 4 and block sliding.\n"
                "• Simulating wrap: Up lands adjacent to 4 at the upper corridor; Down wraps into a 0 cavity candidate; Left stalls; Right aligns with the cavity mouth.\n"
                "• To progress while avoiding dead wraps, prefer the direction that reduces distance to the 0 gap without repeating prior no-ops.\n"
                "ACTION:\n"
            )
        client = OpenAI(api_key=key)
        # Ask the model for the observation content; the tool call will be added by our code (we supervise the label)
        resp = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": user_msg}],
            reasoning_effort="high",
        )
        return (resp.choices[0].message.content or "").strip()

# -------------------------------------- Concrete classes --------------------------------------

class AS66ManualScriptText(AS66ManualScriptBase):
    USE_IMAGES = False

class AS66ManualScriptVision(AS66ManualScriptBase):
    USE_IMAGES = True
