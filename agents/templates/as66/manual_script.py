"""
AS66 manual script runner → multi-turn SFT conversations for *playing the game*.

Goal (this version):
  • Generate the GLOBAL CARTESIAN PRODUCT of vetted ways across levels.
    If L1 has A ways, L2 has B ways, L3 has C ways, ... → produce A×B×C×... distinct
    episodes. Each episode starts from RESET and executes the chosen way for L1,
    then chosen way for L2, etc., producing an independent annotated conversation.

Parallelism:
  • Controlled by env:
      MANUAL_MAX_WORKERS=25    (default 25)
      MANUAL_PARALLEL=1        (enable parallel execution; default ON here)
  • Each combo runs independently with its own HTTP session/guid.
  • File writes are serialized with a lock.

Outputs per run under transcripts/<stamp>/as66/:
  - sft_multi_turn.jsonl   ← one line per combo (episode)
  - interleaved.md         ← readable transcript for all combos
  - rows.jsonl             ← low-level step log for all combos

OBSERVATION text:
  - Uses build_primer_system_text()/build_user_step_text() via prompts_sft.py
  - Annotator model: gpt-5 (if OPENAI_API_KEY set); otherwise stub text.
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
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests

from openai import OpenAI

from ...agent import Agent
from ...structs import GameAction, GameState, FrameData
from .downsample import downsample_4x4, matrix16_to_lines
from .prompts_sft import build_primer_system_text, build_user_step_text

# ----------------------------- Vetted WAYS per level -----------------------------
# Each level key maps to a LIST OF WAYS; each way is a LIST OF MOVES (strings).
# Edit these to your vetted sequences. (I folded your l2a..l2e etc. under "l2"/"l3"/"l4".)

WAYS_BY_LEVEL: dict[str, List[List[str]]] = {
    "l1": [
        ["Down", "Left", "Down"],

    ],
    "l2": [
        ["Right","Down","Left"],                                        # l2a
       ["Down","Left","Up","Right","Down","Left"],                    # l2b
        ["Right","Up","Down","Left"],
        ["Down","Left","Up","Right","Down","Left"],
        ["Up","Down","Left","Up","Right","Up","Down","Left"],
    ],
    "l3": [
        ["Right","Down","Right","Down","Left","Right","Left","Up"],    # l3a
        ["Down","Left","Up","Right","Left","Right","Up"],
        ["Right","Down","Right","Down","Left","Right","Left","Up"],
        ["Down","Right","Left","Down","Up","Right","Down","Up"],
        ["Right","Down","Right","Up","Left","Up","Left","Up","Right","Down","Right","Down","Right","Left","Right","Left","Up"],


    ],
    "l4": [
        ["Up","Left","Down","Left","Up"],              
        ["Up","Left","Down","Left","Up"],
        ["Right","Up","Left","Up"],
        ["Up","Right","Left","Down","Left","Up"],


 
    ],
}

LEVEL_ORDER: List[str] = ["l1", "l2", "l3", "l4"]

# ----------------------------- Parallel config -----------------------------

PARALLEL: bool = os.getenv("MANUAL_PARALLEL", "1").strip().lower() in ("1", "true", "yes", "on")
MAX_WORKERS: int = int(os.getenv("MANUAL_MAX_WORKERS", "25"))
WRITE_LOCK = threading.Lock()  # serialize file writes

# -------------------------------------- Data models --------------------------------------

@dataclass
class StepRow:
    level: str
    combo_id: str
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
    level: str
    move: str
    pre_state: str
    pre_score: int
    post_state: str
    post_score: int
    ds16_pre: str
    ds16_post: str
    level_note: Optional[str] = None
    observation: Optional[str] = None  # imputed observation text

@dataclass
class EpisodeResult:
    combo_key: str            # e.g., "l1w1_l2w3_l3w2_l4w1"
    conversation: List[Dict[str, Any]]
    blocks: List[Block]
    rows: List[StepRow]

# ----------------------------- Tool schemas for ACTION1..4 -----------------------------

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "ACTION1", "description": "Move Up",    "parameters": {"type":"object","properties":{},"additionalProperties": False}}},
    {"type": "function", "function": {"name": "ACTION2", "description": "Move Down",  "parameters": {"type":"object","properties":{},"additionalProperties": False}}},
    {"type": "function", "function": {"name": "ACTION3", "description": "Move Left",  "parameters": {"type":"object","properties":{},"additionalProperties": False}}},
    {"type": "function", "function": {"name": "ACTION4", "description": "Move Right", "parameters": {"type":"object","properties":{},"additionalProperties": False}}},
]

# ----------------------------- Isolated env client (one per episode) -----------------------------

class _EnvClient:
    def __init__(self, root_url: str, game_id: str, card_id: str, headers: Dict[str, str], cookies) -> None:
        self.root = root_url
        self.game_id = game_id
        self.card_id = card_id
        self.sess = requests.Session()
        self.sess.headers.update(headers)
        self.sess.cookies = deepcopy(cookies)
        self.guid: Optional[str] = None

    def _post(self, cmd: str, payload: Dict[str, Any]) -> FrameData:
        r = self.sess.post(f"{self.root}/api/cmd/{cmd}", json=payload, timeout=60)
        data = r.json()
        if "guid" in data and data["guid"]:
            self.guid = data["guid"]
        return FrameData.model_validate(data)

    def reset(self) -> FrameData:
        return self._post("RESET", {"card_id": self.card_id, "game_id": self.game_id})

    def act(self, action_name: str) -> FrameData:
        payload: Dict[str, Any] = {"game_id": self.game_id}
        if self.guid:
            payload["guid"] = self.guid
        return self._post(action_name, payload)

# ----------------------------- Manual runner (Cartesian across levels) -----------------------------

class AS66ManualScriptBase(Agent):
    """
    Produces multi-turn SFT conversations where the assistant writes an OBSERVATION
    (multi-paragraph) and calls exactly one ACTIONn tool each turn.

    NEW: Generates global Cartesian product across WAYS_BY_LEVEL[LEVEL_ORDER].
    Each combo is an independent episode from RESET.
    """
    USE_IMAGES: bool = False

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        # We don't "play" in this Agent; we orchestrate offline episode generation.
        return True

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        return GameAction.ACTION5

    # ------------------------ main entry ------------------------

    def main(self) -> None:
        self._prepare_outdir()
        self._md_path = self._out_dir / "interleaved.md"
        self._sft_path = self._out_dir / "sft_multi_turn.jsonl"
        self._rows_path = self._rows_path

        # open files once; writes are serialized with WRITE_LOCK
        self._md = open(self._md_path, "a", encoding="utf-8")
        self._sft = open(self._sft_path, "a", encoding="utf-8")

        try:
            self._run_all_combos()
        finally:
            try: self._md.close()
            except Exception: pass
            try: self._sft.close()
            except Exception: pass
            self.cleanup()

    # ------------------------ build global Cartesian ------------------------

    def _iter_global_combos(self) -> List[Tuple[str, List[Tuple[str, int, List[str]]]]]:
        """
        Returns list of (combo_key, [(level_name, way_index, moves), ...]) for all global combos.
        combo_key example: "l1w1_l2w3_l3w2_l4w1"
        """
        level_ways = [WAYS_BY_LEVEL[lvl] for lvl in LEVEL_ORDER]
        index_products = itertools.product(*[range(len(ws)) for ws in level_ways])
        combos: List[Tuple[str, List[Tuple[str, int, List[str]]]]] = []
        for idx_tuple in index_products:
            parts = []
            spec: List[Tuple[str, int, List[str]]] = []
            for lvl, w_idx in zip(LEVEL_ORDER, idx_tuple):
                spec.append((lvl, w_idx + 1, WAYS_BY_LEVEL[lvl][w_idx]))
                parts.append(f"{lvl}w{w_idx+1}")
            key = "_".join(parts)
            combos.append((key, spec))
        return combos

    # ------------------------ driver (parallel or sequential) ------------------------

    def _run_all_combos(self) -> None:
        combos = self._iter_global_combos()
        total = len(combos)
        headers = {"X-API-Key": os.getenv("ARC_API_KEY", ""), "Accept": "application/json"}
        cookies = self._session.cookies

        with WRITE_LOCK:
            self._md.write(f"# AS66 Multi-Turn Conversations — GLOBAL CARTESIAN ({total} episodes)\n")
            self._md.write(f"Generated: {datetime.utcnow().isoformat()}Z  |  Parallel={PARALLEL}  |  MaxWorkers={MAX_WORKERS}\n\n")

        if PARALLEL and MAX_WORKERS > 1:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futs = [pool.submit(self._run_one_combo, key, spec, headers, cookies) for (key, spec) in combos]
                for fut in as_completed(futs):
                    res = fut.result()
                    self._write_episode(res)
        else:
            for key, spec in combos:
                res = self._run_one_combo(key, spec, headers, cookies)
                self._write_episode(res)

    # ------------------------ run one episode (one global combo) ------------------------

    def _run_one_combo(
        self,
        combo_key: str,
        combo_spec: List[Tuple[str, int, List[str]]],
        base_headers: Dict[str, str],
        base_cookies,
    ) -> EpisodeResult:
        """
        combo_spec: list of (level_name, way_index (1-based), moves)
        """
        env = _EnvClient(self.ROOT_URL, self.game_id, self.card_id, base_headers, base_cookies)

        conversation: List[Dict[str, Any]] = []
        conversation.append({"role": "system", "content": build_primer_system_text()})

        # RESET to start
        f = env.reset()
        frames: List[FrameData] = [f]

        # First user message with initial state
        ds16_pre = downsample_4x4(frames[-1].frame, round_to_int=True) if frames[-1].frame else []
        conversation.append({"role": "user", "content": build_user_step_text(ds16_pre, frames[-1].score, step=0, note="Initial state")})

        # Episode bookkeeping
        step_counter = 0
        last_pre_state = frames[-1].state.name
        last_ds16_lines = matrix16_to_lines(ds16_pre) if ds16_pre else "(empty)"
        pre_score = frames[-1].score

        all_blocks: List[Block] = []
        all_rows: List[StepRow] = []

        # Execute all levels' ways in order
        for level_name, way_idx, moves in combo_spec:
            for mv in moves:
                action_name = self._move_to_action_name(mv)

                # Perform action
                f = env.act(action_name)
                frames.append(f)
                step_counter += 1

                ds16_post = downsample_4x4(f.frame, round_to_int=True)
                ds16_post_lines = matrix16_to_lines(ds16_post)

                # Row + block
                row = StepRow(
                    level=level_name,
                    combo_id=combo_key,
                    step_index=step_counter,
                    move=mv,
                    pre_state=last_pre_state,
                    pre_score=pre_score,
                    post_state=f.state.name,
                    post_score=f.score,
                    ds16_pre=last_ds16_lines,
                    ds16_post=ds16_post_lines,
                    level_note=None,
                )
                if f.score > pre_score:
                    row.level_note = "LEVEL UP"
                    pre_score = f.score
                if f.state is GameState.GAME_OVER:
                    row.level_note = (row.level_note + " | GAME_OVER") if row.level_note else "GAME_OVER"

                all_rows.append(row)

                blk = Block(
                    idx=len(all_blocks),
                    level=level_name,
                    move=mv,
                    pre_state=row.pre_state,
                    pre_score=row.pre_score,
                    post_state=row.post_state,
                    post_score=row.post_score,
                    ds16_pre=row.ds16_pre,
                    ds16_post=row.ds16_post,
                    level_note=row.level_note,
                )

                # Impute observation text (gpt-5 if key present)
                blk.observation = self._impute_observation_text(
                    ds16_lines=blk.ds16_pre,
                    score=blk.pre_score,
                    step=blk.idx,
                    full_timeline=all_blocks,
                )
                tool_call_id = str(uuid.uuid4())
                conversation.append({
                    "role": "assistant",
                    "content": blk.observation,
                    "tool_calls": [
                        {"id": tool_call_id, "type": "function", "function": {"name": action_name, "arguments": {}}}
                    ],
                })

                # Tool result
                tool_content = (
                    f"RESULT for {action_name}:\n"
                    f"PostState={blk.post_state} | Score={blk.post_score}\n"
                    f"Matrix 16x16 (integer codes):\n{blk.ds16_post}\n"
                )
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": action_name,
                    "content": tool_content,
                })

                all_blocks.append(blk)

                if f.state is GameState.GAME_OVER:
                    break

                # Next user message
                conversation.append({
                    "role": "user",
                    "content": build_user_step_text(ds16_post, f.score, step=step_counter, note=blk.level_note),
                })

            # optional: early stop if game over
            if frames[-1].state is GameState.GAME_OVER:
                break

        return EpisodeResult(
            combo_key=combo_key,
            conversation=conversation,
            blocks=all_blocks,
            rows=all_rows,
        )

    # ------------------------ write one episode safely ------------------------

    def _write_episode(self, res: EpisodeResult) -> None:
        with WRITE_LOCK:
            # Markdown
            self._md.write(f"## Combo {res.combo_key}\n\n")
            for blk in res.blocks:
                self._md.write(f"### Step {blk.idx:02d} [{blk.level}]\n\n")
                self._md.write(f"**MOVE (target)**: {blk.move}\n\n")
                self._md.write("**OBSERVATION (imputed)**\n\n")
                self._md.write((blk.observation or "(missing)") + "\n\n")
                self._md.write("**STATE AFTER**\n\n")
                self._md.write(f"GameState={blk.post_state} | Score={blk.post_score}\n\n")
                self._md.write("```\n" + blk.ds16_post + "\n```\n\n")

            # JSONL (one record per combo)
            self._sft.write(json.dumps({
                "id": f"combo-{res.combo_key}",
                "messages": res.conversation,
                "tools": TOOL_SCHEMAS,
                "meta": {"combo": res.combo_key, "levels": LEVEL_ORDER, "moves": len(res.blocks)},
            }, ensure_ascii=False) + "\n")

            # Rows
            with open(self._rows_path, "a", encoding="utf-8") as rf:
                for r in res.rows:
                    rf.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")

    # -------------------- helpers --------------------

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
        sys_msg = build_primer_system_text()
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
            return (
                "OBSERVATION:\n"
                "• Identify the movable cluster against background (15) and boundaries (1/14); walls are 4.\n"
                "• Simulate wrap/blocks for the four directions; avoid no-op repeats; advance toward target zeros.\n"
                "ACTION:\n"
            )
        client = OpenAI(api_key=key)
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
    USE_IMAGES = False  # keep off for parallel speed; re-enable if needed
