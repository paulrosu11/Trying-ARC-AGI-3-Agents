"""
AS66 manual script runner with built-in Cartesian-product traces, laddering,
and LLM-powered per-step rationales suitable for fine-tuning.

What this file does:
  • Executes scripted moves formed by Cartesian products per level (TRACE_LIBRARY).
  • Replays the discovered winning prefix to reach deeper levels (laddering).
  • Logs each step with 16×16 (4×4-avg) numeric matrices (codes-only).
  • Generates per-step rationales with a very long, detailed system prompt:
      - For step k: feed the entire timeline of (state→action→new state),
        with level notes (LEVEL UP / GAME OVER), plus all prior rationales,
        then ask for rationale for the NEXT step only (as-if-now).
  • Produces training artifacts:
      - rows.jsonl        (raw step log with ds16)
      - rationales.jsonl  (flat per-step rationale)
      - dialog.jsonl      (messages array: system/user -> assistant; SFT-ready)
      - interleaved.md    (readable transcript)
      - images/           (optional 16×16 PNGs if using Vision subclass)

TEXT-ONLY CONTRACT:
  - All prompts and outputs must remain in the INTEGER codes domain (no colors).
  - 16×16 matrices are printed as space-separated integers rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple
import itertools
import json
import os
import sys

from openai import OpenAI

from ...agent import Agent
from ...structs import GameAction, GameState, FrameData
from .downsample import downsample_4x4, matrix16_to_lines, ds16_png_bytes


# ----------------------------- Per-level Cartesian products (edit here) -----------------------------

TRACE_LIBRARY: dict[str, List[List[str]]] = {
    # Level 1: your known winning example
    "l1": [["Down"], ["Left"], ["Down"]],
    # Level 2: your example (expand options per turn to broaden search)
    "l2": [["Right"], ["Down"], ["Left"]],
    # "l3": [["Up","Right","Down","Left"], ["Up","Right","Down","Left"], ...],
}
LEVEL_ORDER: List[str] = ["l1", "l2"]  # extend as needed


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
    """One step in a candidate timeline (for rationale building)."""
    idx: int
    move: str
    pre_state: str
    pre_score: int
    post_state: str
    post_score: int
    ds16_pre: str
    ds16_post: str
    level_note: Optional[str] = None
    rationale: Optional[str] = None  # filled by LLM later


# -------------------------------------- Prompts (very long, instructive) --------------------------------------

def _rationale_system_prompt() -> str:
    """
    A very long, detailed, instructive prompt for per-step rationalization,
    aligned with the 16×16 observation rules but extended for timeline context.
    TEXT-ONLY: strictly integer codes, no color words.
    """
    return (
        "ROLE & GOAL:\n"
        "You are an expert analyst for the AS66 game. You will be given a FULL timeline of a hand-authored trace: "
        "a sequence of (state BEFORE → CHOSEN MOVE → state AFTER) pairs. Each state is represented by a 16×16 matrix "
        "of integer codes (0–15), where each code is a semantic class. You must produce a succinct, codes-only "
        "rationale for WHY a single specified move (the current step) made sense at that moment.\n\n"
        "STRICT TEXT-ONLY DOMAIN (CODES ONLY):\n"
        "• DO NOT mention or infer colors. Reason only about integer codes and grid structure.\n"
        "• Matrices are down-sampled to 16×16 via 4×4 averaging, then rounded to integers.\n"
        "• Your reasoning should reference codes (e.g., 4=walls, 0=goal cells, 15=background, 1/14=borders, 6=move/lives hints), "
        "  but DO NOT use any color names.\n\n"
        "GAME MECHANICS (observation-aligned but condensed):\n"
        "• The movable character(s) appear as the 'odd one out' contiguous code-cluster(s) distinct from background (often 8 in early levels, "
        "  but DO NOT assume a fixed code; infer structurally from context).\n"
        "• A chosen direction (Up/Down/Left/Right) makes each mover SLIDE as far as possible, WRAPPING across edges if unobstructed, "
        "  stopping just before it would overlap a 4 (wall). Multiple movers, if present, move together under the same command.\n"
        "• 0 cells form target cavities (e.g., a U-shaped cluster with a single non-zero gap); the objective is to land a mover into "
        "  the cavity gap to complete the shape. The 0 region interrupts sliding if it becomes the immediate stop.\n"
        "• 15 denotes general background play area. 1 and 14 typically fence the playable field. The distribution of 6 may correlate with moves/lives.\n\n"
        "TIMELINE INPUT FORMAT (codes-only):\n"
        "You will receive a concatenated summary that always preserves temporal order:\n"
        "  • INITIAL STATE: GameState=… | Score=…  (and its 16×16 matrix)\n"
        "  • For each step i: \n"
        "      - MOVE i: <Up|Down|Left|Right>\n"
        "      - STATE AFTER i: GameState=… | Score=…  (and its 16×16 matrix)\n"
        "      - LEVEL NOTE (optional): 'LEVEL UP' when score increases by +1; 'GAME OVER' if applicable\n"
        "  • Additionally, for previous steps j<i, their already-computed rationales MAY be included as 'Rationale (past, j): …'.\n\n"
        "YOUR TASK (for exactly ONE step k):\n"
        "Given the entire timeline (past and future are visible) and optional past rationales, write a short rationale (3–7 sentences) "
        "for WHY the move at step k made sense AT THAT MOMENT. You may use the presence of later success/failure only to refine your reasoning, "
        "but DO NOT write as if you know the future. Frame your explanation as if *you are the player at that time*, relying on the BEFORE/AFTER "
        "state and rules:\n"
        "  • Identify the movable cluster(s) by code distinctness and contiguity vs. 15 background.\n"
        "  • Simulate full sliding with wrap for all directions at that time; identify blocking 4s and where each direction would stop.\n"
        "  • Compare which direction best aligns with reaching/occupying the correct 0 cavity gap, and whether risk exists (e.g., forced traps).\n"
        "  • If multiple movers exist, ensure the chosen direction advances the plan for mapping movers to their intended 0 regions.\n"
        "  • If the move was exploratory or hypothesis-testing, justify it (e.g., clearing alignment, probing wrap, avoiding dead ends).\n"
        "  • Explain succinctly in codes-only terms (no colors) and reference local substructures (e.g., 'the 0 cavity at rows 10–12, "
        "    with a gap at row 11, col 7').\n\n"
        "FORMAT & CONSTRAINTS:\n"
        "• Output only the rationale paragraph (no JSON). 3–7 sentences. No tool calls. No action names.\n"
        "• DO NOT restate the future steps explicitly. DO NOT leak future knowledge ('we win next').\n"
        "• DO NOT mention colors; use codes and structural descriptions only.\n"
        "• Be concrete: reason about wrap stopping conditions adjacent to code 4; use the 16×16 matrices.\n"
        "• Keep it crisp and useful for training a model that must choose good actions with the same rules.\n"
    )


def _timeline_for_candidate(blocks: List[Block], focus_idx: int) -> str:
    """
    Build the full (past+future) timeline string for a candidate, including
    matrices and any past rationales, then mark the focus step.
    """
    lines: List[str] = []
    if blocks:
        b0 = blocks[0]
        lines.append("## INITIAL STATE")
        lines.append(f"GameState={b0.pre_state} | Score={b0.pre_score}")
        lines.append("```\n" + b0.ds16_pre + "\n```")
        lines.append("")

    for b in blocks:
        lines.append(f"### Step {b.idx:02d}")
        if b.rationale and b.idx < focus_idx:
            lines.append("Rationale (past):")
            lines.append(b.rationale)
            lines.append("")
        lines.append(f"MOVE: {b.move}")
        lines.append(f"STATE AFTER: GameState={b.post_state} | Score={b.post_score}")
        if b.level_note:
            lines.append(f">>> {b.level_note}")
        lines.append("```\n" + b.ds16_post + "\n```")
        lines.append("")

    lines.append(f"### Focus: Step {focus_idx:02d} (write rationale for THIS step only; do not reveal future steps)")
    return "\n".join(lines)


# -------------------------------------- Manual runner base --------------------------------------

class AS66ManualScriptBase(Agent):
    """
    Agent subclass that:
      • Drives its own main() (does not use Agent.main loop).
      • Executes Cartesian-product traces with laddering by replaying 'winning prefix'.
      • Generates per-step rationales using a very long, instructive system prompt.
    """
    USE_IMAGES: bool = False  # Vision variant saves ds16 PNGs; LLM remains text-only

    # Satisfy ABC; unused by our custom main()
    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        return True

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        return GameAction.ACTION5

    # -------------------- driver --------------------

    def main(self) -> None:
        out = self._prepare_outdir()
        rows_path = self._rows_path
        md_path = self._out_dir / "interleaved.md"
        rationales_path = self._out_dir / "rationales.jsonl"
        dialog_path = self._out_dir / "dialog.jsonl"

        # open files
        md = open(md_path, "a", encoding="utf-8")
        rj = open(rationales_path, "a", encoding="utf-8")
        dj = open(dialog_path, "a", encoding="utf-8")

        try:
            prefix: List[str] = []  # discovered winning sequence across prior levels

            md.write(f"# AS66 Interleaved Transcript — {datetime.utcnow().isoformat()}Z\n\n")

            for level in LEVEL_ORDER:
                options = TRACE_LIBRARY[level]
                candidates = list(itertools.product(*options))
                level_solved = False

                for ci, cand in enumerate(candidates, start=1):
                    # RESET to clean state
                    f = self.take_action(GameAction.RESET)
                    if f: self.append_frame(f)

                    # Replay prior winning prefix to reach this level
                    for mv in prefix:
                        f = self.take_action(self._mv_to_action(mv))
                        if f: self.append_frame(f)

                    pre_score = self.score

                    # Build candidate blocks
                    cand_blocks: List[Block] = []
                    # Initial pre snapshot (before first move)
                    if self.frames and self.frames[-1].frame:
                        ds16_pre0 = matrix16_to_lines(
                            downsample_4x4(self.frames[-1].frame, round_to_int=True)
                        )
                    else:
                        ds16_pre0 = "(empty)"

                    # Log the 'initial state' row for candidate with step_index set after rows size
                    init_pre_state = self.state.name
                    init_pre_score = self.score

                    # Execute moves in candidate
                    last_pre_ds16 = ds16_pre0
                    last_pre_state = init_pre_state
                    last_pre_score = init_pre_score
                    level_note: Optional[str] = None

                    for mv in cand:
                        # Action
                        f = self.take_action(self._mv_to_action(mv))
                        if not f:
                            break
                        self.append_frame(f)

                        # Build snapshots
                        ds16_post = matrix16_to_lines(downsample_4x4(f.frame, round_to_int=True))
                        step_row = StepRow(
                            level=level,
                            candidate_index=ci,
                            step_index=self.action_counter,  # monotone during run
                            move=mv,
                            pre_state=last_pre_state,
                            pre_score=last_pre_score,
                            post_state=f.state.name,
                            post_score=f.score,
                            ds16_pre=last_pre_ds16,
                            ds16_post=ds16_post,
                            level_note=None,
                        )

                        # Level-up heuristic: +1 score implies next level
                        if f.score > pre_score:
                            step_row.level_note = "LEVEL UP"
                            pre_score = f.score
                        if f.state is GameState.GAME_OVER:
                            step_row.level_note = (step_row.level_note + " | GAME OVER") if step_row.level_note else "GAME OVER"

                        # Persist raw row
                        with open(rows_path, "a", encoding="utf-8") as rf:
                            rf.write(json.dumps(step_row.__dict__, ensure_ascii=False) + "\n")

                        # Update blocks list for this candidate
                        cand_blocks.append(
                            Block(
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
                        )

                        # Optional 16×16 image
                        if self.USE_IMAGES:
                            try:
                                png = ds16_png_bytes(downsample_4x4(f.frame, round_to_int=True), cell=22)
                                (self._out_dir / "images" / f"{level}-cand{ci:02d}-step{len(cand_blocks)-1:03d}.png").write_bytes(png)
                            except Exception:
                                pass

                        # Prepare for next iter
                        last_pre_ds16 = step_row.ds16_post
                        last_pre_state = step_row.post_state
                        last_pre_score = step_row.post_score

                        if f.state is GameState.GAME_OVER:
                            break

                    # Generate rationales per step for THIS candidate (even if it failed)
                    self._annotate_candidate(cand_blocks, rj, dj, md, level, ci)

                    # If the candidate solved this level (+1 score overall), keep prefix and advance
                    if cand_blocks and cand_blocks[-1].post_score > init_pre_score:
                        prefix.extend([b.move for b in cand_blocks])  # discovered winning subseq
                        level_solved = True
                        break  # next level

                if not level_solved:
                    # stop laddering if we couldn't solve this level
                    break
        finally:
            md.close()
            rj.close()
            dj.close()
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

    def _prepare_outdir(self) -> Path:
        base = Path(os.getenv("TRANSCRIPTS_DIR", "transcripts")).resolve()
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self._out_dir = base / stamp / "as66"
        (self._out_dir / "images").mkdir(parents=True, exist_ok=True)
        self._rows_path = self._out_dir / "rows.jsonl"
        self._annot_path = self._out_dir / "rationales.jsonl"
        return self._out_dir

    # ---- LLM call ----

    def _call_openai(self, system_text: str, user_text: str) -> str:
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            # If no key, fall back to a short local stub so the pipeline still runs
            stub = (
                "Rationale: Adopted the chosen direction because its wrap-stopping condition adjacent to code 4 "
                "created alignment to the 0 cavity while avoiding immediate traps, given the 16×16 matrices.\n"
            )
            return stub
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            reasoning_effort="low",
        )
        return (resp.choices[0].message.content or "").strip()

    def _annotate_candidate(
        self,
        blocks: List[Block],
        rj_file,
        dj_file,
        md_file,
        level: str,
        cand_idx: int,
    ) -> None:
        """
        For a given candidate timeline:
          - Iterate k = 0..len(blocks)-1
          - Build a full timeline string including all steps (past+future)
          - Include any prior rationales (past only)
          - Ask the LLM for a rationale for step k
          - Write: rationales.jsonl, dialog.jsonl, and append to interleaved.md
        """
        if not blocks:
            return

        sys_text = _rationale_system_prompt()

        md_file.write(f"## Candidate {level}-cand{cand_idx:02d}\n\n")

        for k in range(len(blocks)):
            timeline = _timeline_for_candidate(blocks, k)

            # USER message for step k: include the 'focus step' and instructions
            user_text = (
                timeline
                + "\n\n# TASK\n"
                + "Write a single rationale paragraph (3–7 sentences, codes-only) for why the move at the focus step made sense at that time.\n"
                + "Do not include any tool calls or action names. Do not leak future knowledge explicitly."
            )

            out_text = self._call_openai(sys_text, user_text)
            blocks[k].rationale = out_text

            # Save flat rationale row
            rrow = {
                "level": level,
                "candidate": cand_idx,
                "index": blocks[k].idx,
                "move": blocks[k].move,
                "pre_state": blocks[k].pre_state,
                "pre_score": blocks[k].pre_score,
                "post_state": blocks[k].post_state,
                "post_score": blocks[k].post_score,
                "rationale": out_text,
            }
            rj_file.write(json.dumps(rrow, ensure_ascii=False) + "\n")

            # Save dialog (SFT-ready): system+user -> assistant
            dj_file.write(json.dumps({
                "messages": [
                    {"role": "system", "content": sys_text},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": out_text},
                ],
                "meta": {
                    "level": level,
                    "candidate": cand_idx,
                    "index": blocks[k].idx,
                    "move": blocks[k].move,
                },
            }, ensure_ascii=False) + "\n")

            # Append to readable markdown
            md_file.write(f"### Step {blocks[k].idx:02d}\n\n")
            md_file.write("**MOVE**\n\n")
            md_file.write(blocks[k].move + "\n\n")
            md_file.write("**RATIONALE**\n\n")
            md_file.write(out_text + "\n\n")
            md_file.write("**STATE AFTER**\n\n")
            md_file.write(f"GameState={blocks[k].post_state} | Score={blocks[k].post_score}\n\n")
            md_file.write("```\n" + blocks[k].ds16_post + "\n```\n\n")


# -------------------------------------- Concrete classes --------------------------------------

class AS66ManualScriptText(AS66ManualScriptBase):
    """Text-only transcript + rationales (codes only)."""
    USE_IMAGES = False


class AS66ManualScriptVision(AS66ManualScriptBase):
    """Same as text-only, but also saves per-step 16×16 PNG images."""
    USE_IMAGES = True
