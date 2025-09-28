# agents/templates/manual_trace_processing.py
# Text or Visual LS20 L1×L2 combinations → ARC states → interleaved rationales (no state repetition).
# Run (text):   uv run python agents/templates/manual_trace_processing.py
# Run (visual): uv run python agents/templates/manual_trace_processing.py --visual --backend gemini
# Flags: --print-prompts   Print FULL extracted prompts (also saved to files)
#
# Folder layout:
#   transcripts/<UTC>/annotated_manual_traces/
#     └── combos/<L1_name>__x__<L2_name>/
#         ├── arc_frames.jsonl
#         ├── interleaved.jsonl
#         ├── interleaved.md
#         ├── sft.jsonl (optional)
#         ├── visual_system_prompt.txt
#         └── text_system_prompt.txt

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import ast
import importlib.util
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image  # for visual mode

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

# ----------------------------- robust prompt extractors -----------------------------

LLM_AGENTS_PY = Path(__file__).resolve().parent / "llm_agents.py"

def _read_src() -> str:
    if not LLM_AGENTS_PY.exists():
        _die(f"Missing {LLM_AGENTS_PY}. Cannot extract prompts.")
    return LLM_AGENTS_PY.read_text(encoding="utf-8")

def _eval_stringish(node: ast.AST, resolvers: List[Dict[str, str]]) -> Optional[str]:
    """Evaluate string-like AST nodes with limited, safe semantics."""
    def _resolve_name(name: str) -> Optional[str]:
        for r in resolvers:
            if name in r:
                return r[name]
        return None

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    if isinstance(node, ast.JoinedStr):
        parts: List[str] = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                inner = _eval_stringish(v.value, resolvers)
                if inner is None:
                    return None
                parts.append(inner)
            else:
                return None
        return "".join(parts)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        l = _eval_stringish(node.left, resolvers)
        r = _eval_stringish(node.right, resolvers)
        if l is None or r is None:
            return None
        return l + r

    if isinstance(node, ast.Name):
        return _resolve_name(node.id)

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "dedent" and len(node.args) == 1:
            inner = _eval_stringish(node.args[0], resolvers)
            return textwrap.dedent(inner) if inner is not None else None

        if isinstance(node.func, ast.Attribute) and node.func.attr == "dedent" and len(node.args) == 1:
            inner = _eval_stringish(node.args[0], resolvers)
            return textwrap.dedent(inner) if inner is not None else None

        if isinstance(node.func, ast.Attribute) and node.func.attr in ("strip", "lstrip", "rstrip"):
            base = _eval_stringish(node.func.value, resolvers)
            if base is None:
                return None
            try:
                return getattr(base, node.func.attr)()
            except Exception:
                return None

        if isinstance(node.func, ast.Attribute) and node.func.attr == "format":
            base = _eval_stringish(node.func.value, resolvers)
            if base is None:
                return None
            if not node.args and not node.keywords:
                return base
            pos: List[str] = []
            for a in node.args:
                val = _eval_stringish(a, resolvers)
                if val is None:
                    return None
                pos.append(val)
            kw: Dict[str, str] = {}
            for kwarg in node.keywords or []:
                if kwarg.arg is None:
                    return None
                val = _eval_stringish(kwarg.value, resolvers)
                if val is None:
                    return None
                kw[kwarg.arg] = val
            try:
                return base.format(*pos, **kw)
            except Exception:
                return None
        return None

    return None

def _gather_const_assigns(nodes: List[ast.stmt], inherited: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env: Dict[str, str] = {} if inherited is None else dict(inherited)
    for n in nodes:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            v = _eval_stringish(n.value, [env])
            if isinstance(v, str):
                env[n.targets[0].id] = v
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
            v = _eval_stringish(n.value, [env])
            if isinstance(v, str):
                env[n.target.id] = v
    return env

def _extract_prompt_via_ast(class_name: str, method_name: str) -> str:
    src = _read_src()
    mod = ast.parse(src)

    module_consts = _gather_const_assigns(mod.body)

    cls = None
    for n in mod.body:
        if isinstance(n, ast.ClassDef) and n.name == class_name:
            cls = n
            break
    if cls is None:
        _die(f"{class_name} not found in llm_agents.py.")

    class_consts = _gather_const_assigns(cls.body, inherited=module_consts)

    fn = None
    for n in cls.body:
        if isinstance(n, ast.FunctionDef) and n.name == method_name:
            fn = n
            break
    if fn is None:
        _die(f"{method_name} not found in {class_name}.")

    locals_map: Dict[str, str] = {}
    ret: Optional[str] = None

    for stmt in fn.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            v = _eval_stringish(stmt.value, [locals_map, class_consts, module_consts])
            if isinstance(v, str):
                locals_map[stmt.targets[0].id] = v
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            v = _eval_stringish(stmt.value, [locals_map, class_consts, module_consts])
            if isinstance(v, str):
                locals_map[stmt.target.id] = v
        elif isinstance(stmt, ast.Return):
            ret = _eval_stringish(stmt.value, [locals_map, class_consts, module_consts]) if stmt.value is not None else ""
            break

    if not isinstance(ret, str) or not ret.strip():
        _die(f"Cannot statically evaluate {class_name}.{method_name}().")
    return ret.strip()

def _extract_prompt_via_runtime(class_name: str, method_name: str) -> str:
    spec = importlib.util.spec_from_file_location("llm_agents_runtime", LLM_AGENTS_PY)
    if spec is None or spec.loader is None:
        _die("Failed to import llm_agents.py at runtime.")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore
    cls = getattr(m, class_name, None)
    if cls is None:
        _die(f"{class_name} not found during runtime import.")
    meth = getattr(cls, method_name, None)
    if meth is None:
        _die(f"{method_name} not found on {class_name} during runtime import.")
    # Try static/class, then instance (no-arg)
    try:
        s = meth()  # type: ignore
        if isinstance(s, str) and s.strip():
            return s.strip()
    except TypeError:
        pass
    try:
        inst = cls()  # type: ignore
        bound = getattr(inst, method_name)
        s = bound()  # type: ignore
        if isinstance(s, str) and s.strip():
            return s.strip()
    except Exception as e:
        _die(f"Runtime probe failed for {class_name}.{method_name}(): {e}")
    _die(f"Runtime probe produced empty text for {class_name}.{method_name}().")

def _extract_or_probe_prompt(class_name: str, method_name: str) -> str:
    try:
        return _extract_prompt_via_ast(class_name, method_name)
    except Exception as e_ast:
        print(f"[PROMPT] AST extraction failed for {class_name}.{method_name}: {e_ast}\n[PROMPT] Falling back to runtime import probe...")
        return _extract_prompt_via_runtime(class_name, method_name)

# Pull both prompts (robust)
TEXT_CONTEXT = _extract_or_probe_prompt("GuidedLLM", "build_user_prompt")
VISUAL_CONTEXT = _extract_or_probe_prompt("VisualGuidedLLM", "build_game_context_prompt")

def _assert_prompt_sane(kind: str, s: str) -> None:
    if len(s) < 80:
        _die(f"{kind} prompt is suspiciously short ({len(s)} chars).")

# ----------------------------- training format utilities -----------------------------

INCLUDE_GRIDS = True         # include numeric matrices (disabled in --visual mode)
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
    level_note: Optional[str] = None
    cot: Optional[str] = None
    pre_img_path: Optional[Path] = None
    post_img_path: Optional[Path] = None
    pre_img_data_url: Optional[str] = None
    post_img_data_url: Optional[str] = None

# ----------------------------- example traces -----------------------------

TRACES_L1: List[Dict[str, Any]] = [
    {"name": "l1_success_a", "moves": ["Up", "Up", "Left", "Up", "Up"]},
    {"name": "l1_success_b", "moves": ["Up", "Up", "Left", "Up", "Up"]},
]
TRACES_L2: List[Dict[str, Any]] = [
    {"name": "l2_variant_a", "moves": ["Right", "Right", "Up", "Left", "Down"]},
    {"name": "l2_variant_b", "moves": ["Down", "Right", "Up", "Right", "Left"]},
]

# ----------------------------- context builders -----------------------------

def _build_full_timeline(blocks: List[Block], focus_idx: int) -> str:
    """
    Build a compact transcript with ALL steps (past & future).
    Include existing rationales for past steps if available; future ones have none.
    Avoid redundant grids here; rely on single image attachment in visual mode.
    """
    out: List[str] = []
    if blocks:
        b0 = blocks[0]
        out.append("## INITIAL STATE")
        out.append(f"GameState={b0.pre_state} | Score={b0.pre_score}")
        if INCLUDE_GRIDS and b0.pre_grid:
            out.append("```\n" + b0.pre_grid + "\n```")
        out.append("")
    for b in blocks:
        out.append(f"### Step {b.idx:02d}")
        if b.cot:
            out.append("RATIONALE (past):")
            out.append(b.cot)
            out.append("")
        out.append(f"MOVE: {b.move}")
        out.append(f"STATE AFTER: GameState={b.post_state} | Score={b.post_score}")
        if b.level_note:
            out.append(f">>> {b.level_note}")
        if INCLUDE_GRIDS and b.post_grid:
            out.append("```\n" + b.post_grid + "\n```")
        out.append("")
    out.append(f"### Focus: Step {focus_idx:02d} (write rationale for THIS step only)")
    out.append("Do NOT quote or cite future steps, even though you can see them below and above; use them only to infer plausible intent.")
    return "\n".join(out).strip()

def _build_messages_for_step(blocks: List[Block], focus_idx: int) -> Tuple[str, str]:
    """
    Text (non-visual) message builder with full (past+future) timeline context.
    """
    b = blocks[focus_idx]
    timeline = _build_full_timeline(blocks, focus_idx)

    sys_text = (
        TEXT_CONTEXT
        + "\n\nYou are writing natural reasoning notes for an LS20 gameplay transcript."
          " Your role is to write, in natural prose (or concise bullets), the rationale for a single move,"
          " as if you were the player at that exact moment. Do not output JSON or code."
          "\n\nGoal of the rationale:"
          " Given the current state and the move that was taken, explain clearly and convincingly WHY that move made sense"
          " at the time. The move does not need to be optimal nor guaranteed to lead to a win; your task is to supply"
          " reasonable, game-aware justification for taking it then and there."
          "\n\nAnalytic checklist (weigh all that apply):"
          "\n  • Key ↔ Exit match: Carefully assess whether the bottom-left key indicator visually matches the exit’s target pattern."
          "    If it matches (or if later success implies it must have matched), say so explicitly and prioritize heading to the exit."
          "    Avoid the key generator once matched; route to the exit via legal corridors."
          "\n  • Game dynamics & rules: Consider walls/legality of movement, corridor topology, proximity to generators/exit,"
          "    remaining moves/energy, lives, and HUD cues."
          "\n  • Subgoals & routing: Explain intermediate positioning (e.g., reaching corridors, avoiding generators when matched,"
          "    or seeking generators when not matched), and why the chosen direction advances that plan."
          "\n  • Consistency with outcomes: If a level completes soon after, acknowledge that the key must already have matched"
          "    and show how this move aligns with beelining to the exit; if no progress followed, explain the hypothesis tested."
          "\n\nSTRICT TRAINING RULE:"
          " Do NOT reveal or mention the move in the body of your rationale. Defer stating the move name until the very end."
          " Spend the beginning analyzing and thinking first."
          "\n\nVoice and perspective:"
          " Write in first person (e.g., “Looking at the state, I see… Therefore I…”). Ground claims in visible state and rules."
          " Example of acceptable phrasing: “Ah, looking at the state, the key matches the exit, so I should focus on exiting."
          " I appear to have enough moves, so I’ll head toward the exit without touching the generator…”"
          "\n\nDeliverable:"
          " A thoughtful paragraph (or a few concise bullets) integrating state, rules, and win conditions into a coherent"
          " justification for why THIS MOVE was chosen now. **you may not state future knowledge, in this paragraph, the action which was take is deducible from the current and past states!** On the FINAL lines only, append: “Move: <Up/Down/Left/Right>”."
          "Some examples of what not to say 'My key must already match the exit lock, as I'm on a direct path to victory.' or 'The level completes in the very next step, which tells me that my key must already match the exit's lock pattern. '"
          " this is crazy, you are writing as the person playing the game who shouldn't have access to the future (although I give it to you to help play this role),"
          "this means you must just look at the game and see 'Looking at the key and the exit lock it seems we are matching so I can proceed...'. Understand the game and your role in providing reasoning "
    )

    user_text = "\n".join([
        timeline,
        "",
        f"# Current step = {b.idx:02d}",
        f"(For your final line only) MOVE: {b.move}",
        "STATE BEFORE is the immediately preceding state in the transcript; use it to reason but do not repeat images here.",
    ]).strip()

    return sys_text, user_text

def _build_visual_messages_for_step(blocks: List[Block], focus_idx: int) -> Tuple[str, str]:
    """
    Visual message builder with full (past+future) timeline context.
    We attach ONLY the single 'state BEFORE' image for the focus step to avoid duplicate pre/post images.
    """
    b = blocks[focus_idx]
    timeline = _build_full_timeline(blocks, focus_idx)

    system = (
        VISUAL_CONTEXT
        + "\n\nYou are annotating a past move (not choosing a new action)."
          "\n\nGoal of the rationale:"
          " Given the displayed CURRENT state (image) and the move that was taken, explain WHY that move made sense at the time."
          " It need not be optimal—provide reasonable, game-aware justification."
          "\n\nAnalytic checklist (weigh all that apply):"
          "\n  • Key ↔ Exit match: visually check the indicator vs the exit; if matched (or implied by later success),"
          "    prioritize routing to exit and avoid the generator."
          "\n  • Legality & topology: walls, corridors, reachable paths; remaining moves/energy; lives; proximity to exit/generator."
          "\n  • Plan & routing: why this direction advances the plan now."
          "\n  • Outcomes consistency: use knowledge that the future sequence succeeds/doesn’t to refine your reconstruction,"
          "    but DO NOT reference future steps explicitly in your written rationale."
          "\n\nSTRICT TRAINING RULE:"
          " Do NOT reveal or mention the move in the body. Leave the move name for the final lines only."
          " Spend the opening on analysis and thinking."
          "\n\nVoice and perspective:"
          " First person, grounded in visible shapes/colors and known rules."
          "\n\nDeliverable:"
          " A coherent paragraph (or concise bullets) justifying THIS MOVE now. On the final lines only, state the move:"
          " “Move: <Up/Down/Left/Right>”."
    )

    user_text = "\n".join([
        "# FULL TRANSCRIPT SUMMARY (past + future available; do not cite future explicitly)",
        timeline,
        "",
        f"# Current step = {b.idx:02d}",
        f"(For your final line only) MOVE: {b.move}",
        "An image of the CURRENT state (before this move) is attached below.",
    ]).strip()

    return system, user_text

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
        except Exception:
            _die("google-generativeai is not installed. Install with `uv add google-generativeai` or `pip install google-generativeai`.")
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            _die("GEMINI_API_KEY is missing. Put it in .env at repo root (no quotes).")
        genai.configure(api_key=key)
        _GENAI = genai
    return _GENAI

def _call_gemini(system_text: str, user_text: str) -> str:
    genai = _ensure_genai()
    model = genai.GenerativeModel(model_name="gemini-2.5-pro", system_instruction=system_text)
    resp = model.generate_content(user_text)
    out = getattr(resp, "text", "") or ""
    if not out and getattr(resp, "candidates", None):
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

# --- visual LLM calls ---

def _call_openai_visual(system_text: str, user_text: str, images_data_urls: List[str]) -> str:
    if OpenAI is None:
        _die("openai python SDK not installed. Install it or choose --backend gemini.")
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        _die("OPENAI_API_KEY is missing. Put it in .env at repo root (no quotes).")
    client = OpenAI(api_key=key)
    content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    for url in images_data_urls:
        if not url:
            continue
        content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
    resp = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "system", "content": system_text},
                  {"role": "user", "content": content}],
        reasoning_effort="low",
    )
    return (resp.choices[0].message.content or "").strip()

def _call_gemini_visual(system_text: str, user_text: str, image_paths: List[Optional[Path]]) -> str:
    genai = _ensure_genai()
    model = genai.GenerativeModel(model_name="gemini-2.5-pro", system_instruction=system_text)
    parts: List[Any] = [user_text]
    for p in image_paths:
        if not p:
            continue
        try:
            img = Image.open(str(p))
            parts.append(img)
        except Exception:
            continue
    resp = model.generate_content(parts)
    out = getattr(resp, "text", "") or ""
    if not out and getattr(resp, "candidates", None):
        try:
            parts2 = resp.candidates[0].content.parts  # type: ignore
            out = "".join(getattr(pt, "text", "") for pt in parts2)
        except Exception:
            pass
    if not out:
        _die("Gemini returned empty text in visual mode.")
    return out.strip()

def _call_llm_visual(backend: str, system_text: str, user_text: str, *, data_urls: List[str], paths: List[Optional[Path]]) -> str:
    if backend == "gemini":
        return _call_gemini_visual(system_text, user_text, paths)
    return _call_openai_visual(system_text, user_text, data_urls)

# ----------------------------- image helpers (visual mode) -----------------------------

def _last_grid(arr3d: List[List[List[int]]]) -> List[List[int]]:
    if not arr3d:
        return []
    g = arr3d[-1] or arr3d[0]
    if not g or not isinstance(g, list) or not g or not isinstance(g[0], list):
        return []
    return g

def _grid_to_png_bytes(grid: List[List[int]]) -> bytes:
    KEY_COLORS = {
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
            return (136, 136, 136)
        return (int(hex_str[1:3], 16), int(hex_str[3:5], 16), int(hex_str[5:7], 16))
    h = len(grid)
    w = len(grid[0]) if h else 0
    im = Image.new("RGB", (w, h), (0, 0, 0))
    px = im.load()
    for y, row in enumerate(grid):
        for x, val in enumerate(row):
            rgb = _hex_to_rgb(KEY_COLORS.get((val & 15), "#888888"))
            px[x, y] = rgb
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()

def _b64_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")

# ----------------------------- combo run (L1 × L2) -----------------------------

def _combo_out_dir(ts_root: Path, l1_name: str, l2_name: str) -> Path:
    d = ts_root / "annotated_manual_traces" / "combos" / f"{l1_name}__x__{l2_name}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "images").mkdir(exist_ok=True)
    return d

def _ts_root() -> Path:
    base = Path(os.getenv("TRANSCRIPTS_DIR", "transcripts")).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = base / stamp
    root.mkdir(parents=True, exist_ok=True)
    return root

def _write_prompt_files(out_dir: Path, *, text_prompt: str, visual_prompt: str, print_full: bool) -> None:
    tp = out_dir / "text_system_prompt.txt"
    vp = out_dir / "visual_system_prompt.txt"
    tp.write_text(text_prompt, encoding="utf-8")
    vp.write_text(visual_prompt, encoding="utf-8")
    def _preview(label: str, s: str) -> None:
        print(f"\n[DEBUG] {label} system prompt ({len(s)} chars):")
        if print_full:
            print(s)
        else:
            print(s[:800] + ("…" if len(s) > 800 else ""))
    _preview("TEXT", text_prompt)
    _preview("VISUAL", visual_prompt)

def _run_one_combo(
    *,
    game_id: str,
    l1: Dict[str, Any],
    l2: Dict[str, Any],
    print_frame_json: bool,
    flat_sft: bool,
    ts_root: Path,
    backend: str,
    visual: bool,
    print_prompts: bool,
) -> None:
    out_dir = _combo_out_dir(ts_root, l1["name"], l2["name"])
    arc = ARCClient(game_id=game_id, out_dir=out_dir, print_frame_json=print_frame_json)

    # Verify game availability
    games = arc.list_games()
    if game_id not in games:
        _die(f"Game '{game_id}' not in /api/games for this key. Available: {games}")

    tag = "manual-visual-interleaved" if visual else "manual-text-interleaved"
    arc.open_scorecard(tags=["agent", f"{tag}-{backend}", f"{l1['name']}__x__{l2['name']}"])

    # Prompts: assert + persist + preview
    _assert_prompt_sane("text", TEXT_CONTEXT)
    _assert_prompt_sane("visual", VISUAL_CONTEXT)
    _write_prompt_files(out_dir, text_prompt=TEXT_CONTEXT, visual_prompt=VISUAL_CONTEXT, print_full=print_prompts)

    blocks: List[Block] = []
    try:
        # RESET → start of L1
        pre = arc.reset()
        level_counter = 1
        last_score = int(pre.get("score", 0))

        # Helper to render & save per-state PNGs (visual mode)
        def _save_image_for(frame_obj: Dict[str, Any], prefix: str) -> Tuple[Optional[Path], Optional[str]]:
            try:
                grid = _last_grid(frame_obj.get("frame") or [])
                if not grid or not grid[0]:
                    return (None, None)
                png = _grid_to_png_bytes(grid)
                path = out_dir / "images" / f"{prefix}.png"
                with open(path, "wb") as f:
                    f.write(png)
                return (path, _b64_data_url(png))
            except Exception:
                return (None, None)

        # L1 moves
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

            pre_path = post_path = None
            pre_url = post_url = None
            if visual:
                pre_path, pre_url = _save_image_for(pre, f"{len(blocks):02d}-pre")
                post_path, post_url = _save_image_for(post, f"{len(blocks):02d}-post")

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
                pre_img_path=pre_path, post_img_path=post_path,
                pre_img_data_url=pre_url, post_img_data_url=post_url,
            ))
            pre = post

        # L2 moves
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

            pre_path = post_path = None
            pre_url = post_url = None
            if visual:
                pre_path, pre_url = _save_image_for(pre, f"{len(blocks):02d}-pre")
                post_path, post_url = _save_image_for(post, f"{len(blocks):02d}-post")

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
                pre_img_path=pre_path, post_img_path=post_path,
                pre_img_data_url=pre_url, post_img_data_url=post_url,
            ))
            pre = post

        # Annotate each step
        annot_path = out_dir / "interleaved.jsonl"
        md_path = out_dir / "interleaved.md"
        flat_path = out_dir / "sft.jsonl" if flat_sft else None

        with open(annot_path, "a", encoding="utf-8") as annot, open(md_path, "a", encoding="utf-8") as md:
            if blocks:
                md.write(f"# LS20 Interleaved Transcript — {l1['name']}__x__{l2['name']} ({datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')})\n\n")
                b0 = blocks[0]
                md.write("## INITIAL STATE\n\n")
                md.write(f"GameState={b0.pre_state} | Score={b0.pre_score}\n\n")
                if INCLUDE_GRIDS:
                    md.write("```\n" + b0.pre_grid + "\n```\n\n")

            for idx, b in enumerate(blocks):
                if visual:
                    sys_text, user_text = _build_visual_messages_for_step(blocks, idx)
                    # Attach ONLY the 'state BEFORE' image for the focus step to avoid duplicate pre/post images
                    text = _call_llm_visual(
                        backend,
                        sys_text,
                        user_text,
                        data_urls=[b.pre_img_data_url or ""],
                        paths=[b.pre_img_path],
                    )
                else:
                    sys_text, user_text = _build_messages_for_step(blocks, idx)
                    text = _call_llm(backend, sys_text, user_text)

                if not text:
                    _die(f"LLM returned empty rationale at step {idx}.")

                b.cot = text

                preview = text[:RATIONAL_PREVIEW_CHARS].replace("\n", " ")
                print(f"[LLM:{backend}] step {idx:02d} preview: {preview}{'…' if len(text) > RATIONAL_PREVIEW_CHARS else ''}")

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
                if visual and (b.pre_img_path or b.post_img_path):
                    md.write("**IMAGES (this step)**\n\n")
                    if b.pre_img_path:
                        md.write(f"Pre:\n\n![]({(b.pre_img_path).as_posix()})\n\n")
                    if b.post_img_path:
                        md.write(f"Post:\n\n![]({(b.post_img_path).as_posix()})\n\n")

                if flat_path is not None:
                    with open(flat_path, "a", encoding="utf-8") as flat:
                        flat.write(json.dumps({"move": b.move, "cot": b.cot}, ensure_ascii=False) + "\n")

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
    p = argparse.ArgumentParser(description="LS20 L1×L2 combinations → ARC states → interleaved rationales (text or visual).")
    p.add_argument("--game", type=str, default="ls20-fa137e247ce6", help="ARC game_id (default: ls20-fa137e247ce6)")
    p.add_argument("--print-frame-json", action="store_true", help="Print full ARC frame JSON to console (verbose).")
    p.add_argument("--flat-sft", action="store_true", help="Also write sft.jsonl with flat {move,cot} rows.")
    p.add_argument("--backend", type=str, choices=["openai", "gemini"], default=None, help="LLM backend (default from LLM_BACKEND env, else openai).")
    p.add_argument("--visual", action="store_true", help="Enable visual mode (images instead of numeric matrices).")
    p.add_argument("--print-prompts", action="store_true", help="Print FULL extracted prompts (also saved to files in each combo dir).")
    return p.parse_args()

def main() -> None:
    args = _parse_args()

    if not os.getenv("ARC_API_KEY"):
        _die("ARC_API_KEY missing in env.")
    backend = _llm_backend(args.backend)
    if backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        _die("OPENAI_API_KEY missing in env (or choose --backend gemini).")
    if backend == "gemini" and not os.getenv("GEMINI_API_KEY"):
        _die("GEMINI_API_KEY missing in env (or choose --backend openai).")

    ts_root = Path(os.getenv("TRANSCRIPTS_DIR", "transcripts")).resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ts_root.mkdir(parents=True, exist_ok=True)

    # In visual mode we suppress numeric matrices in prompts/markdown
    global INCLUDE_GRIDS
    if args.visual:
        INCLUDE_GRIDS = False

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
                visual=args.visual,
                print_prompts=args.print_prompts,
            )

if __name__ == "__main__":
    main()
