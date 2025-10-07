# agents/templates/as66/inspect_api.py
from __future__ import annotations
import argparse, base64, io, json, os, sys
from pathlib import Path
import requests
from PIL import Image

# Robust import: works as "python -m ..." or "python path/to/inspect_api.py"
try:
    from .downsample import downsample_4x4, to_block_matrix_str
except Exception:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from agents.templates.as66.downsample import downsample_4x4, to_block_matrix_str
    
KEY_COLORS = {
    0: "#FFFFFF", 1: "#CCCCCC", 2: "#999999",
    3: "#666666", 4: "#000000", 5: "#202020",
    6: "#1E93FF", 7: "#F93C31", 8: "#FF851B",
    9: "#921231", 10: "#88D8F1", 11: "#FFDC00",
    12: "#FF7BCC", 13: "#4FCC30", 14: "#2ECC71",
    15: "#7F3FBF",
}
def _rgb(h): return (int(h[1:3],16), int(h[3:5],16), int(h[5:7],16))

def _root_url()->str:
    s=os.getenv("SCHEME","https"); h=os.getenv("HOST","three.arcprize.org"); p=os.getenv("PORT","443")
    return f"{s}://{h}" if (s=="https" and p=="443") or (s=="http" and p=="80") else f"{s}://{h}:{p}"

def _hdrs()->dict[str,str]:
    k=os.getenv("ARC_API_KEY","").strip()
    if not k: raise SystemExit("ARC_API_KEY missing")
    return {"X-API-Key":k,"Accept":"application/json","Content-Type":"application/json"}

def _grid_png(grid:list[list[int]])->bytes:
    h=len(grid); w=len(grid[0]) if h else 0
    im=Image.new("RGB",(w,h),(0,0,0)); px=im.load()
    for y,row in enumerate(grid):
        for x,val in enumerate(row):
            px[x,y]=_rgb(KEY_COLORS.get(val&15,"#888888"))
    buf=io.BytesIO(); im.save(buf,"PNG",optimize=True); return buf.getvalue()

def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--game", default="as66-821a4dcad9c2",
                    help="prefix or full game_id (you can leave the default)")
    ap.add_argument("--moves", nargs="*", default=["Down","Left","Down"],
                    help="sequence, e.g. Down Left Down")
    args=ap.parse_args()

    root=_root_url(); hdr=_hdrs()

    # choose exact game_id by prefix
    games=requests.get(f"{root}/api/games",headers=hdr,timeout=20).json()
    gids=[g.get("game_id") for g in games if isinstance(g.get("game_id"),str)]
    gid=next((g for g in gids if g.startswith(args.game)), None)
    if not gid: raise SystemExit(f"No game_id starting with '{args.game}' for this key")
    print(f"[inspect] game_id={gid}")

    # open scorecard
    card=requests.post(f"{root}/api/scorecard/open", json={"tags":["as66-inspect"]}, headers=hdr, timeout=30).json()["card_id"]

    def cmd(name:str, payload:dict)->dict:
        r=requests.post(f"{root}/api/cmd/{name}", json=payload, headers=hdr, timeout=30)
        data=r.json()
        print(f"\n== /api/cmd/{name} -> state={data.get('state')} score={data.get('score')}")
        frame=data.get("frame") or []
        if frame and frame[0]:
            print(f"   frame layers={len(frame)} size={len(frame[0])}x{len(frame[0][0])}")
        print((json.dumps(data, ensure_ascii=False)[:800]) + ("…" if len(json.dumps(data))>800 else ""))
        return data

    # RESET
    res=cmd("RESET", {"card_id":card, "game_id":gid})
    guid=res.get("guid")

    outdir = Path(os.getenv("TRANSCRIPTS_DIR","transcripts"))/"inspect"
    outdir.mkdir(parents=True, exist_ok=True)

    for m in args.moves:
        # 16×16 BEFORE move
        f=res.get("frame") or []
        ds16=downsample_4x4(f, take_last_grid=True, round_to_int=True)
        print("\n-- 16x16 (4x4-avg) BEFORE move --")
        print(to_block_matrix_str(ds16))

        last=f[-1] if f else []
        if last and last[0]:
            png=_grid_png(last)
            fn=outdir/f"{res.get('score',0):02d}-{m.lower()}.png"
            fn.write_bytes(png)
            print(f"saved image: {fn}")

        name={"up":"ACTION1","down":"ACTION2","left":"ACTION3","right":"ACTION4"}.get(m.lower())
        if not name: raise SystemExit(f"bad move {m}")
        res=cmd(name, {"game_id":gid, "guid":guid})
        guid=res.get("guid")

    requests.post(f"{root}/api/scorecard/close", json={"card_id":card}, headers=hdr, timeout=30)

if __name__=="__main__":
    main()
