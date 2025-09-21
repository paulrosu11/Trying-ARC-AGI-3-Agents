# agents/templates/manual_trace_processing.py
# Text-only LS20 L1×L2 combinations → ARC states → interleaved rationales (no state repetition).
# Run: uv run python agents/templates/manual_trace_processing.py
#
# Folder layout:
#   transcripts/<UTC>/annotated_manual_traces/
#     └── combos/<L1_name>__x__<L2_name>/
#         ├── arc_frames.jsonl     # raw ARC frames (proof of API)
#         ├── interleaved.jsonl    # one row per step (pre/post state + COT)
#         ├── interleaved.md       # INITIAL STATE, then (RATIONALE → MOVE → STATE AFTER)* with level markers
#         └── sft.jsonl (optional) # flat {move,cot} rows if --flat-sft

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---- LLM backends (OpenAI + Gemini) ----
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

# google-generativeai is optional; import lazily when used
_GENAI = None  # module cache


# ----------------------------- fail-fast helpers -----------------------------

def _die(msg: str) -> None:
    raise RuntimeError(msg)


# ----------------------------- env loading -----------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

def _strip(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = v.strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1]
    return s

def _load_env() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        load_dotenv = None

    for p in (REPO_ROOT / ".env.example", REPO_ROOT / ".env"):
        if p.exists():
            if load_dotenv:
                load_dotenv(p, override=True)
            else:
                for line in p.read_text(encoding="utf-8").splitlines():
                    s = line.strip()
                    if not s or s.startswith("#") or "=" not in s:
                        continue
                    k, v = s.split("=", 1)
                    os.environ.setdefault(k.strip(), _strip(v) or "")

    for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ARC_API_KEY", "SCHEME", "HOST", "PORT", "LLM_BACKEND"):
        v = _strip(os.getenv(k))
        if v is not None:
            os.environ[k] = v

_load_env()

# ----------------------------- ARC REST helpers -----------------------------

def _root_url() -> str:
    scheme = os.getenv("SCHEME", "https")
    host = os.getenv("HOST", "three.arcprize.org")
    port = os.getenv("PORT", "443")
    if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"

def _headers() -> Dict[str, str]:
    key = os.getenv("ARC_API_KEY", "").strip()
    if not key:
        _die("ARC_API_KEY is missing. Put it in .env at the repo root (no quotes).")
    return {"X-API-Key": key, "Accept": "application/json"}

class ARCClient:
    def __init__(self, game_id: str, out_dir: Path, print_frame_json: bool) -> None:
        self.game_id = game_id
        self.session = requests.Session()
        self.session.headers.update(_headers())
        self.card_id: Optional[str] = None
        self.guid: Optional[str] = None
        self.out_dir = out_dir
        self.print_frame_json = print_frame_json
        self.frames_path = self.out_dir / "arc_frames.jsonl"

    def list_games(self) -> List[str]:
        url = f"{_root_url()}/api/games"
        r = self.session.get(url, timeout=20)
        try:
            arr = r.json()
        except Exception:
            _die(f"GET /api/games failed: {r.status_code} {r.text[:200]}")
        gids = [g.get("game_id") for g in (arr or []) if isinstance(g.get("game_id"), str)]
        print(f"[ARC] games → {gids}")
        return gids

    def open_scorecard(self, tags: List[str]) -> str:
        url = f"{_root_url()}/api/scorecard/open"
        r = self.session.post(url, json={"tags": tags}, timeout=20)
        try:
            data = r.json()
        except Exception:
            _die(f"open_scorecard failed: {r.status_code} {r.text[:200]}")
        print(f"[ARC] open_scorecard → {json.dumps(data, ensure_ascii=False)}")
        if not r.ok or "card_id" not in data:
            _die(f"Scorecard open error: {data}")
        self.card_id = str(data["card_id"])
        return self.card_id

    def close_scorecard(self) -> Dict[str, Any]:
        if not self.card_id:
            return {}
        url = f"{_root_url()}/api/scorecard/close"
        r = self.session.post(url, json={"card_id": self.card_id}, timeout=20)
        try:
            data = r.json()
        except Exception:
            _die(f"close_scorecard failed: {r.status_code} {r.text[:200]}")
        print(f"[ARC] close_scorecard → {json.dumps(data, ensure_ascii=False)[:2000]}")
        self.card_id = None
        return data

    def _post_cmd(self, action_name: str, payload: Dict[str, Any], *, allow_error: bool = False) -> Dict[str, Any]:
        url = f"{_root_url()}/api/cmd/{action_name}"
        r = self.session.post(url, json=payload, timeout=30)
        try:
            data = r.json()
        except Exception:
            _die(f"cmd {action_name} failed: {r.status_code} {r.text[:200]}")
        # proof/logging:
        print(f"[ARC] POST /api/cmd/{action_name} payload={json.dumps(payload, ensure_ascii=False)}")
        preview = json.dumps(data, ensure_ascii=False)
        print(f"[ARC]   ↳ Response ({len(preview)} chars) state={data.get('state')} score={data.get('score')} guid={data.get('guid')}")
        if self.print_frame_json:
            print(f"[ARC]   ↳ Frame JSON preview:\n{preview[:2000]}{'…' if len(preview)>2000 else ''}")
        with open(self.frames_path, "a", encoding="utf-8") as f:
            f.write(preview + "\n")
        if ("error" in data) and not allow_error:
            _die(f"ARC error for {action_name}: {data['error']}")
        return data

    def reset(self) -> Dict[str, Any]:
        if not self.card_id:
            _die("reset() called without open scorecard.")
        # Try payload variants; server implementations can differ subtly.
        attempts = [
            {"card_id": self.card_id, "game_id": self.game_id},
            {"card_id": self.card_id},
            {"game_id": self.game_id},
        ]
        last_err = None
        for pay in attempts:
            res = self._post_cmd("RESET", pay, allow_error=True)
            if "error" not in res:
                self.guid = res.get("guid")
                return res
            last_err = res["error"]
            print(f"[ARC] RESET attempt with payload={pay} failed: {last_err}")
        _die(f"All RESET attempts failed: {last_err}")
        return {}

    def move(self, action_name: str) -> Dict[str, Any]:
        if not self.guid:
            _die("move() called without guid (RESET must succeed first).")
        payload: Dict[str, Any] = {"game_id": self.game_id, "guid": self.guid}
        res = self._post_cmd(action_name, payload)
        self.guid = res.get("guid", self.guid)
        return res

# ----------------------------- LS20 textual context (from your code) -----------------------------

LLM_AGENTS_PY = Path(__file__).resolve().parent / "llm_agents.py"

def _extract_ls20_context() -> str:
    if not LLM_AGENTS_PY.exists():
        _die(f"Missing {LLM_AGENTS_PY}. Cannot extract LS20 text context.")
    src = LLM_AGENTS_PY.read_text(encoding="utf-8")
    m_class = re.search(r"\bclass\s+GuidedLLM\b", src)
    if not m_class:
        _die("GuidedLLM not found in llm_agents.py.")
    region = src[m_class.start():]
    m_def = re.search(r"def\s+build_user_prompt\s*\(", region)
    if not m_def:
        _die("build_user_prompt not found in GuidedLLM.")
    sub = region[m_def.end():]
    m_q1 = re.search(r'("""|\'\'\')', sub, re.DOTALL)
    if not m_q1:
        _die("No triple-quoted text in build_user_prompt.")
    q = m_q1.group(1); start = m_q1.end()
    m_q2 = re.search(re.escape(q), sub[start:], re.DOTALL)
    if not m_q2:
        _die("Unterminated triple-quoted block in build_user_prompt.")
    end = start + m_q2.start()
    block = sub[start:end].strip()
    if not block:
        _die("Extracted LS20 context is empty.")
    return block.strip()

TEXT_CONTEXT = _extract_ls20_context()

# ----------------------------- training format utilities -----------------------------

INCLUDE_GRIDS = True         # include numeric matrices
RATIONAL_PREVIEW_CHARS = 500 # live preview length

def _pretty_grid_3d(arr3d: List[List[List[int]]]) -> str:
    lines: List[str] = []
    for i, block in enumerate(arr3d or []):
        lines.append(f"Grid {i}:")
        for row in block:
            lines.append("  " + " ".join(str(x) for x in row))
        lines.append("")
    return "\n".join(lines).strip() or "(empty grid)"

def _norm_move(m: str) -> str:
    k = m.strip().lower()
    if k in ("u", "up"): return "Up"
    if k in ("d", "down"): return "Down"
    if k in ("l", "left"): return "Left"
    if k in ("r", "right"): return "Right"
    _die(f"Unknown move string '{m}'. Use Up/Down/Left/Right."); return "Up"

def _to_action_name(m: str) -> str:
    k = m.lower()
    if k in ("u", "up"): return "ACTION1"
    if k in ("d", "down"): return "ACTION2"
    if k in ("l", "left"): return "ACTION3"
    if k in ("r", "right"): return "ACTION4"
    _die(f"Unknown direction '{m}'."); return "ACTION1"


@dataclass
class Block:
    idx: int
    move: str
    pre_state: str
    pre_score: int
    pre_grid: str
    post_state: str
    post_score: int
    post_grid: str
    level_note: Optional[str] = None  # e.g., "LEVEL 1 COMPLETED → starting LEVEL 2"
    cot: Optional[str] = None         # LLM rationale (longer planning/analysis)

# ----------------------------- example traces -----------------------------
# Default: combinations L1×L2. We assume L1 is successful to unlock L2.
# Use your successful L1 example twice (two names, same moves)
TRACES_L1: List[Dict[str, Any]] = [
    {"name": "l1_success_a", "moves": ["Up", "Up", "Left", "Up", "Up"]},
    {"name": "l1_success_b", "moves": ["Up", "Up", "Left", "Up", "Up"]},
]
# Two L2 examples (may or may not win)
TRACES_L2: List[Dict[str, Any]] = [
    {"name": "l2_variant_a", "moves": ["Right", "Right", "Up", "Left", "Down"]},
    {"name": "l2_variant_b", "moves": ["Down", "Right", "Up", "Right", "Left"]},
]

# ----------------------------- context builders (no duplication) -----------------------------

def _build_prev_chain(blocks: List[Block]) -> str:
    """
    Continuous chain with NO repeated 'current state' at the start of each step.
    Layout:
      INITIAL STATE
      (for each completed step)
        RATIONALE
        MOVE
        STATE AFTER (and optional LEVEL NOTE)
    """
    parts: List[str] = []
    if blocks:
        b0 = blocks[0]
        parts.append("## INITIAL STATE")
        parts.append(f"GameState={b0.pre_state} | Score={b0.pre_score}")
        if INCLUDE_GRIDS: parts.append("```\n" + b0.pre_grid + "\n```")
        parts.append("")
    for b in blocks:
        parts.append(f"### Step {b.idx:02d}")
        parts.append("RATIONALE:")
        parts.append((b.cot or "(not available)"))
        parts.append("")
        parts.append("MOVE:")
        parts.append(b.move)
        parts.append("")
        parts.append("STATE AFTER:")
        parts.append(f"GameState={b.post_state} | Score={b.post_score}")
        if b.level_note:
            parts.append(f">>> {b.level_note}")
        if INCLUDE_GRIDS: parts.append("```\n" + b.post_grid + "\n```")
        parts.append("")
    return "\n".join(parts).strip()

def _build_messages_for_step(history_blocks: List[Block], current_block: Block) -> Tuple[str, str]:
    """
    Returns (system_text, user_text)
    Model sees a continuous chain up to the *previous* step (no duplicate 'current state' line),
    then we append the CURRENT step's MOVE and STATE AFTER, and ask for a long, in-the-moment rationale
    (≈8–12 sentences, or bullets of comparable length) justifying that move from the player’s perspective.
    """
    prev_text = _build_prev_chain(history_blocks)  # includes INITIAL STATE and prior steps (with their rationales)

    sys_text = (
        TEXT_CONTEXT
        + "\n\nYou are writing natural reasoning notes for an LS20 gameplay transcript. "
          "Do not output JSON or code. Use natural prose (or concise bullets). "
          "Write from the player’s perspective at that moment: analyze the current state, "
          "what matches vs. not yet, planned subgoals, obstacles (walls/paths), "
          "and why the given MOVE is the right next step in this sequence."
          "Keep in mind that if you won a level sometime in the future the key must've been matching so use that to verify if it is indeed matching with the distributions of 0s at the given level"
          "Think deeply about this, if you're about to win in 2/3 moves, your key probably is matching "
    )

    b = current_block
    cur_parts: List[str] = []
    if prev_text:
        cur_parts += [prev_text, ""]
    cur_parts += [
        f"### Step {b.idx:02d}",
        "MOVE:",
        b.move,
        "",
        "STATE AFTER:",
        f"GameState={b.post_state} | Score={b.post_score}",
    ]
    if b.level_note:
        cur_parts.append(f">>> {b.level_note}")
    if INCLUDE_GRIDS:
        cur_parts += ["```\n" + b.post_grid + "\n```"]
    cur_parts += [
        "",
        "Write a detailed RATIONALE (about 8–12 sentences, or a comparable number of concise bullets) "
        "analyzing the current state and justifying the shown MOVE as the best next action. "
        "Discuss matching vs. not yet matching, layout constraints, pathing, subgoals, and plan forward."
    ]
    user_text = "\n".join(cur_parts).strip()
    return sys_text, user_text

# ----------------------------- LLM backends -----------------------------

def _llm_backend(cli_backend: Optional[str]) -> str:
    if cli_backend:
        return cli_backend.lower()
    env = os.getenv("LLM_BACKEND", "").strip().lower()
    return env if env in ("openai", "gemini") else "openai"

def _call_openai(system_text: str, user_text: str) -> str:
    if OpenAI is None:
        _die("openai python SDK not installed. Install it or choose --backend gemini.")
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        _die("OPENAI_API_KEY is missing. Put it in .env at repo root (no quotes).")
    client = OpenAI(api_key=key)
    # GPT-5; no temperature; can pass reasoning_effort
    resp = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "system", "content": system_text},
                  {"role": "user", "content": user_text}],
        reasoning_effort="low",
    )
    return (resp.choices[0].message.content or "").strip()

def _ensure_genai():
    global _GENAI
    if _GENAI is None:
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as e:
            _die("google-generativeai is not installed. Install with `uv add google-generativeai` or `pip install google-generativeai`.")
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            _die("GEMINI_API_KEY is missing. Put it in .env at repo root (no quotes).")
        genai.configure(api_key=key)
        _GENAI = genai
    return _GENAI

def _call_gemini(system_text: str, user_text: str) -> str:
    genai = _ensure_genai()
    # Use Gemini 2.5 Pro, no temperature
    model = genai.GenerativeModel(model_name="gemini-2.5-pro", system_instruction=system_text)
    # For text-only, a single string is fine
    resp = model.generate_content(user_text)
    # google-generativeai returns .text on success
    out = getattr(resp, "text", "") or ""
    if not out and getattr(resp, "candidates", None):
        # Fallback: attempt to join text parts
        try:
            parts = resp.candidates[0].content.parts  # type: ignore
            out = "".join(getattr(p, "text", "") for p in parts)
        except Exception:
            pass
    if not out:
        _die("Gemini returned empty text.")
    return out.strip()

def _call_llm(backend: str, system_text: str, user_text: str) -> str:
    if backend == "gemini":
        return _call_gemini(system_text, user_text)
    return _call_openai(system_text, user_text)

# ----------------------------- combo run (L1 × L2) -----------------------------

def _combo_out_dir(ts_root: Path, l1_name: str, l2_name: str) -> Path:
    d = ts_root / "annotated_manual_traces" / "combos" / f"{l1_name}__x__{l2_name}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _ts_root() -> Path:
    base = Path(os.getenv("TRANSCRIPTS_DIR", "transcripts")).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = base / stamp
    root.mkdir(parents=True, exist_ok=True)
    return root

def _run_one_combo(
    *,
    game_id: str,
    l1: Dict[str, Any],
    l2: Dict[str, Any],
    print_frame_json: bool,
    flat_sft: bool,
    ts_root: Path,
    backend: str,
) -> None:
    out_dir = _combo_out_dir(ts_root, l1["name"], l2["name"])
    arc = ARCClient(game_id=game_id, out_dir=out_dir, print_frame_json=print_frame_json)

    # Verify game availability
    games = arc.list_games()
    if game_id not in games:
        _die(f"Game '{game_id}' not in /api/games for this key. Available: {games}")

    arc.open_scorecard(tags=["agent", f"manual-text-interleaved-{backend}", f"{l1['name']}__x__{l2['name']}"])

    blocks: List[Block] = []
    try:
        # RESET → start of L1
        pre = arc.reset()
        level_counter = 1
        last_score = int(pre.get("score", 0))

        # L1 moves (assumed successful)
        for mv in l1["moves"]:
            post = arc.move(_to_action_name(_norm_move(mv)))
            pre_state = str(pre.get("state"))
            pre_score = int(pre.get("score", 0))
            post_state = str(post.get("state"))
            post_score = int(post.get("score", 0))

            level_note = None
            if post_score > last_score:
                level_note = f"LEVEL {level_counter} COMPLETED → starting LEVEL {level_counter+1}"
                level_counter += 1
                last_score = post_score

            blocks.append(Block(
                idx=len(blocks),
                move=_norm_move(mv),
                pre_state=pre_state,
                pre_score=pre_score,
                pre_grid=_pretty_grid_3d(pre.get("frame") or []) if INCLUDE_GRIDS else "",
                post_state=post_state,
                post_score=post_score,
                post_grid=_pretty_grid_3d(post.get("frame") or []) if INCLUDE_GRIDS else "",
                level_note=level_note,
            ))
            pre = post

        # L2 moves (may or may not succeed)
        for mv in l2["moves"]:
            post = arc.move(_to_action_name(_norm_move(mv)))
            pre_state = str(pre.get("state"))
            pre_score = int(pre.get("score", 0))
            post_state = str(post.get("state"))
            post_score = int(post.get("score", 0))

            level_note = None
            if post_score > last_score:
                level_note = f"LEVEL {level_counter} COMPLETED → starting LEVEL {level_counter+1}"
                level_counter += 1
                last_score = post_score

            blocks.append(Block(
                idx=len(blocks),
                move=_norm_move(mv),
                pre_state=pre_state,
                pre_score=pre_score,
                pre_grid=_pretty_grid_3d(pre.get("frame") or []) if INCLUDE_GRIDS else "",
                post_state=post_state,
                post_score=post_score,
                post_grid=_pretty_grid_3d(post.get("frame") or []) if INCLUDE_GRIDS else "",
                level_note=level_note,
            ))
            pre = post

        # Annotate each step (interleaved; no state repetition)
        annot_path = out_dir / "interleaved.jsonl"
        md_path = out_dir / "interleaved.md"
        flat_path = out_dir / "sft.jsonl" if flat_sft else None

        with open(annot_path, "a", encoding="utf-8") as annot, open(md_path, "a", encoding="utf-8") as md:
            # Markdown header with INITIAL STATE (printed once)
            if blocks:
                md.write(f"# LS20 Interleaved Transcript — {l1['name']}__x__{l2['name']} ({datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')})\n\n")
                b0 = blocks[0]
                md.write("## INITIAL STATE\n\n")
                md.write(f"GameState={b0.pre_state} | Score={b0.pre_score}\n\n")
                if INCLUDE_GRIDS:
                    md.write("```\n" + b0.pre_grid + "\n```\n\n")

            for idx, b in enumerate(blocks):
                sys_text, user_text = _build_messages_for_step(blocks[:idx], b)
                text = _call_llm(backend, sys_text, user_text)
                if not text:
                    _die(f"LLM returned empty rationale at step {idx}.")

                b.cot = text

                # live preview (first 500 chars)
                preview = text[:RATIONAL_PREVIEW_CHARS].replace("\n", " ")
                print(f"[LLM:{backend}] step {idx:02d} preview: {preview}{'…' if len(text) > RATIONAL_PREVIEW_CHARS else ''}")

                # JSONL row with explicit ARC states
                row = {
                    "combo": f"{l1['name']}__x__{l2['name']}",
                    "index": b.idx,
                    "move": b.move,
                    "pre_state": b.pre_state,
                    "pre_score": b.pre_score,
                    "post_state": b.post_state,
                    "post_score": b.post_score,
                    "level_note": b.level_note,
                    "cot": text,
                }
                annot.write(json.dumps(row, ensure_ascii=False) + "\n")

                # Append to markdown without repeating current state at the start of next step
                md.write(f"### Step {b.idx:02d}\n\n")
                md.write("**RATIONALE**\n\n")
                md.write(b.cot + "\n\n")
                md.write("**MOVE**\n\n")
                md.write(b.move + "\n\n")
                md.write("**STATE AFTER**\n\n")
                md.write(f"GameState={b.post_state} | Score={b.post_score}\n\n")
                if b.level_note:
                    md.write(f">>> {b.level_note}\n\n")
                if INCLUDE_GRIDS:
                    md.write("```\n" + b.post_grid + "\n```\n\n")

                if flat_path is not None:
                    with open(flat_path, "a", encoding="utf-8") as flat:
                        flat.write(json.dumps({"move": b.move, "cot": b.cot}, ensure_ascii=False) + "\n")

        # Terminal links to outputs (absolute paths)
        print("\n[links]")
        print(f"  arc frames:     {out_dir.resolve() / 'arc_frames.jsonl'}")
        print(f"  jsonl interlv:  {out_dir.resolve() / 'interleaved.jsonl'}")
        print(f"  markdown:       {out_dir.resolve() / 'interleaved.md'}")
        if flat_sft:
            print(f"  flat sft:       {out_dir.resolve() / 'sft.jsonl'}")
        print("")

    finally:
        arc.close_scorecard()


# ----------------------------- CLI & driver -----------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LS20 L1×L2 combinations → ARC states → interleaved textual rationales.")
    p.add_argument("--game", type=str, default="ls20-fa137e247ce6", help="ARC game_id (default: ls20-fa137e247ce6)")
    p.add_argument("--print-frame-json", action="store_true", help="Print full ARC frame JSON to console (verbose).")
    p.add_argument("--flat-sft", action="store_true", help="Also write sft.jsonl with flat {move,cot} rows.")
    p.add_argument("--backend", type=str, choices=["openai", "gemini"], default=None, help="LLM backend (default from LLM_BACKEND env, else openai).")
    return p.parse_args()

def main() -> None:
    args = _parse_args()

    if not os.getenv("ARC_API_KEY"):
        _die("ARC_API_KEY missing in env.")
    # Only require the key for the selected backend
    backend = _llm_backend(args.backend)
    if backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        _die("OPENAI_API_KEY missing in env (or choose --backend gemini).")
    if backend == "gemini" and not os.getenv("GEMINI_API_KEY"):
        _die("GEMINI_API_KEY missing in env (or choose --backend openai).")

    ts_root = Path(os.getenv("TRANSCRIPTS_DIR", "transcripts")).resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ts_root.mkdir(parents=True, exist_ok=True)

    # Default = combinations (Cartesian product). For 2×2 → 4 combos.
    for l1 in TRACES_L1:
        l1_moves = [_norm_move(m) for m in l1["moves"]]
        for l2 in TRACES_L2:
            l2_moves = [_norm_move(m) for m in l2["moves"]]
            _run_one_combo(
                game_id=args.game,
                l1={"name": l1["name"], "moves": l1_moves},
                l2={"name": l2["name"], "moves": l2_moves},
                print_frame_json=args.print_frame_json,
                flat_sft=args.flat_sft,
                ts_root=ts_root,
                backend=backend,
            )

if __name__ == "__main__":
    main()
