"""
AS66 manual script runner with built-in Cartesian-product traces and text-only annotation.

Rules:
- Traces are defined HERE (no environment variables). Each level is expressed as turn-option lists.
- The runner enumerates the Cartesian product for the current level, but before testing a candidate for
  level N it REPLAYS the already-discovered winning sequences for levels < N to reach the correct state.
- Annotation is textual and codes-only (16×16), producing short rationales per move.

Outputs:
- Records are written to the project recorder JSONL (RECORDINGS_DIR).
- Additionally creates a transcripts folder with ds16 snapshots and annotations.
"""

from __future__ import annotations
from typing import Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json
import base64
import os

from ...agent import Agent
from ...structs import GameAction, GameState, FrameData
from .downsample import downsample_4x4, matrix16_to_lines, ds16_png_bytes
from .prompts_text import build_observation_system_text


# ---------------- Traces (Cartesian products) ----------------
# Each level is a list of "turn option lists". Product across turns = candidate sequence.
TRACE_LIBRARY: dict[str, List[List[str]]] = {
    # Level 1: your known winning example
    "l1": [["Down"], ["Left"], ["Down"]],
    # Level 2: your example starting point (expand by adding more options per turn as needed)
    "l2": [["Right"], ["Down"], ["Left"]],
    # "l3": [[...], [...], ...],
}
LEVEL_ORDER: List[str] = ["l1", "l2"]  # extend as needed


def _mv_to_action(name: str) -> GameAction:
    k = name.strip().lower()
    if k == "up": return GameAction.ACTION1
    if k == "down": return GameAction.ACTION2
    if k == "left": return GameAction.ACTION3
    if k == "right": return GameAction.ACTION4
    raise ValueError(f"Unknown move: {name}")


@dataclass
class StepRow:
    level: str
    candidate_idx: int
    step_index: int
    move: str
    state_before: str
    state_after: str
    score_after: int
    ds16_lines: str


class _ManualBase(Agent):
    """
    Base class that overrides main() to run Cartesian products across levels with laddering.
    """
    USE_IMAGES: bool = False  # Vision variant can set this True

    def main(self) -> None:
        # Start fresh
        self._ensure_dir()
        rows: List[StepRow] = []
        winning_prefix: List[str] = []

        for lvl in LEVEL_ORDER:
            turn_options = TRACE_LIBRARY[lvl]
            # Cartesian product
            candidates = [[]]
            for opts in turn_options:
                candidates = [c + [o] for c in candidates for o in opts]

            level_solved = False
            for idx, cand in enumerate(candidates, start=1):
                # RESET
                frame = self.take_action(GameAction.RESET)
                if frame: self.append_frame(frame)

                # Replay previously discovered winning prefix to reach current level
                for mv in winning_prefix:
                    frame = self.take_action(_mv_to_action(mv))
                    if frame: self.append_frame(frame)

                # Before candidate moves, log ds16
                if self.frames and self.frames[-1].frame:
                    ds16 = downsample_4x4(self.frames[-1].frame, round_to_int=True)
                    self._log_snapshot(lvl, idx, len(self.frames)-1, "(pre)", self.frames[-1], ds16, rows)

                pre_score = self.score

                # Execute this candidate sequence
                ok = True
                for mv in cand:
                    frame = self.take_action(_mv_to_action(mv))
                    if not frame:
                        ok = False
                        break
                    self.append_frame(frame)
                    # Per-step ds16 snapshot
                    ds16 = downsample_4x4(frame.frame, round_to_int=True)
                    self._log_snapshot(lvl, idx, len(self.frames)-1, mv, frame, ds16, rows)
                    if frame.state is GameState.GAME_OVER:
                        ok = False
                        break

                post_score = self.score

                # Heuristic: each solved level increments score by +1 (consistent with ARC servers)
                if ok and (post_score == pre_score + 1):
                    winning_prefix.extend(cand)
                    level_solved = True
                    break  # stop trying more candidates for this level

            if not level_solved:
                break  # stop laddering if we couldn't solve this level

        # Optional: annotate each row (codes-only, concise)
        try:
            self._annotate(rows)
        except Exception:
            pass

        # Normal cleanup/scorecard record
        self.cleanup()

    # ---- helpers ----

    def _ensure_dir(self) -> Path:
        base = Path(os.getenv("TRANSCRIPTS_DIR", "transcripts")).resolve()
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self._out_dir = base / stamp / "as66"
        (self._out_dir / "images").mkdir(parents=True, exist_ok=True)
        self._rows_path = self._out_dir / "rows.jsonl"
        self._annot_path = self._out_dir / "annot.jsonl"
        return self._out_dir

    def _log_snapshot(
        self,
        level: str,
        cand_idx: int,
        step_index: int,
        move_name: str,
        frame: FrameData,
        ds16: List[List[int]],
        rows: List[StepRow],
    ) -> None:
        lines = matrix16_to_lines(ds16)
        # recorder JSONL
        self.recorder.record({
            "phase": "as66.manual",
            "level": level,
            "candidate": cand_idx,
            "step_index": step_index,
            "move": move_name,
            "state": frame.state.name,
            "score": frame.score,
            "ds16": ds16,
        })
        # transcripts jsonl
        row = StepRow(
            level=level,
            candidate_idx=cand_idx,
            step_index=step_index,
            move=move_name,
            state_before=self.frames[-2].state.name if len(self.frames) >= 2 else "NOT_PLAYED",
            state_after=frame.state.name,
            score_after=frame.score,
            ds16_lines=lines,
        )
        rows.append(row)
        with open(self._rows_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")

        if self.USE_IMAGES:
            try:
                png = ds16_png_bytes(ds16, cell=22)
                p = self._out_dir / "images" / f"{level}-{cand_idx:02d}-{step_index:03d}.png"
                p.write_bytes(png)
            except Exception:
                pass

    def _annotate(self, rows: List[StepRow]) -> None:
        """
        Lightweight textual annotation (codes-only). Uses the short guidance in build_observation_system_text().
        """
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            return
        client = OpenAI(api_key=key)

        sys = build_observation_system_text()
        for r in rows:
            user = (
                f"[Level {r.level} | Candidate {r.candidate_idx} | Step {r.step_index}] "
                f"State before → after: {r.state_before} → {r.state_after} (score {r.score_after})\n"
                "Matrix 16x16 (codes only):\n"
                f"{r.ds16_lines}\n\n"
                "Write 2–4 sentences of codes-only rationale for the move that was taken."
            )
            resp = client.chat.completions.create(
                model="gpt-5",
                messages=[{"role":"system","content":sys},
                          {"role":"user","content":user}],
                reasoning_effort="low",
            )
            text = (resp.choices[0].message.content or "").strip()
            with open(self._annot_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "level": r.level,
                    "candidate": r.candidate_idx,
                    "step_index": r.step_index,
                    "move": r.move,
                    "rationale": text,
                }, ensure_ascii=False) + "\n")


class AS66ManualScriptText(_ManualBase):
    """Manual runner that stores numeric 16×16 snapshots and textual annotations (codes only)."""
    USE_IMAGES = False


class AS66ManualScriptVision(_ManualBase):
    """Manual runner that additionally saves per-step 16×16 PNGs (still annotates textually)."""
    USE_IMAGES = True
