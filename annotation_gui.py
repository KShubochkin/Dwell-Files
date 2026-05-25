#a5 - added slow delete annotation function, 
# fixed known tags (a5 not displaying), 
# made tag dropdown menu, 
# added tag label to annotation table, 
# fixed messages/status not showing, 
# added annotation count to status and display, 
# plotted head spine_0 point on trajectories, 
# added event duration to annotation table

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

NAME = "Katya" # put your own name here!! This will be saved as the annotator for any new annotations you add, so that we can keep track of who added what. You can also use this to distinguish between different annotation versions (e.g. "Katya v1", "Katya v2", etc) if you want to
FILE_IDS    = [1] # 0 = GA1, 1 = GA2, 2 = GA3, 3 = EA, 4 = H2O
IDS = slice(None) # can be int, list of ints, or slice (e.g. 0, [0,2], slice(1,4)) - indicates which IDs to include in the dropdown selector (after filtering by FILE_IDS)
ANN_MASTER  = "C:/Users/corna/honours/fresh1/hp_2/data_intermediate/annotation/annotation.csv"
SESSION_DIR = "C:/Users/corna/honours/fresh1/hp_2/data_intermediate/annotation"

FPS              = 6.0
TIME_MARKER_STEP = 2.5
MAIN_BEHAVIORS   = ["dwelling", "crawling", "turning", "nondwelling","dwelling_old","nondwelling_old", "other"]
INCLUSIVE_BOUNDS = True
SPINE_HEAD_X, SPINE_HEAD_Y = "xspine_0",  "yspine_0"
SPINE_TAIL_X, SPINE_TAIL_Y = "xspine_10", "yspine_10"

BEH_COLORS = {
    "dwelling":    "#4ade80",
    "nondwelling": "#c084fc",
    "crawling":    "#60a5fa",
    "turning":     "#fb923c",
    "dwelling_old": "#beedcf",
    "nondwelling_old": "#ddc4f7",
    "other":       "#94a3b8",
}
show_annotations = True

BEH_RGBA_PLOT = {
    "dwelling":    (0.18, 0.73, 0.38, 0.30),
    "crawling":    (0.22, 0.55, 0.95, 0.26),
    "turning":     (0.95, 0.45, 0.10, 0.30),
    "nondwelling": (0.65, 0.42, 0.95, 0.24),
    "dwelling_old": (0.13, 0.78, 0.47, 0.30),
    "nondwelling_old": (0.65, 0.42, 0.95, 0.24),
    "other":       (0.50, 0.55, 0.65, 0.20),
}

from pathlib import Path
OUT_DIR = Path("C:/Users/corna/honours/fresh1/hp_2/data_intermediate/annotation")
LOG_DIR = OUT_DIR / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
#  BOOTSTRAP
# ══════════════════════════════════════════════════════════════════
import importlib, pipeline
importlib.reload(pipeline)
_ctx = pipeline.get_context(file_ids=FILE_IDS, ann_master_csv=ANN_MASTER, session_dir=SESSION_DIR)
per_file, long_df, ann, annotated = _ctx.per_file, _ctx.long_df, _ctx.ann, _ctx.annotated

try:
    get_ipython  # type: ignore
    try:
        import ipympl  # noqa: F401
        get_ipython().run_line_magic("matplotlib", "ipympl")
    except Exception:
        pass
except Exception:
    pass

import math, json, uuid
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import ipywidgets as W
from IPython.display import display, clear_output, HTML as IHTML

USING_IPYMPL = ("ipympl" in matplotlib.get_backend().lower()
                or "widget" in matplotlib.get_backend().lower())

# ══════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════
def html_escape(s):
    return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def norm_behavior(b):
    if b is None: return None
    b = str(b).strip().lower()
    syn = {"dwell":"dwelling","turn":"turning","run":"crawling","crawl":"crawling"}
    if b == "Dwell": print("WEird")
    b = syn.get(b, b)
    if b == "Dwell": print("WEird")
    return b if b in MAIN_BEHAVIORS else "other"

def norm_tags(tag_str):
    if not tag_str: return ""
    parts = [t.strip().lower() for t in str(tag_str).split(";") if t.strip()]
    out, seen = [], set()
    for t in parts:
        if t not in seen: out.append(t); seen.add(t)
    return ";".join(out)

def norm_annotator(a):
    if not a: return "Katya (old)"
    return str(a).strip()

def nearest_idx(et, t):
    return int(np.argmin(np.abs(et - t)))

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def interval_overlap(a0, a1, b0, b1, inclusive=True):
    return not ((a1 < b0 or b1 < a0) if inclusive else (a1 <= b0 or b1 <= a0))

def compute_body_length_mm(df_one):
    need = [SPINE_HEAD_X, SPINE_HEAD_Y, SPINE_TAIL_X, SPINE_TAIL_Y]
    if not all(c in df_one.columns for c in need): return None
    d = np.hypot(df_one[SPINE_HEAD_X]-df_one[SPINE_TAIL_X],
                 df_one[SPINE_HEAD_Y]-df_one[SPINE_TAIL_Y]).to_numpy()
    d = d[np.isfinite(d)]
    return float(np.median(d)) if d.size else None

def compute_an_iteration_value(df, id_value, id_col="ID", an_col="AN",
                               group_cols=None, missing_fill=-1):
    group_cols = [] if group_cols is None else (
        [group_cols] if isinstance(group_cols, str) else group_cols)
    base = df.loc[:, group_cols+[id_col,an_col]].dropna(subset=[id_col,an_col])
    base[id_col] = base[id_col].astype(int)
    base = (base.sort_values(group_cols+[id_col])
                .drop_duplicates(subset=group_cols+[id_col], keep="first")
                .reset_index(drop=True))
    def _calc(g):
        an = g[an_col].to_numpy()
        return 1 + np.cumsum(np.r_[False, an[1:] < an[:-1]])
    if group_cols:
        per = [pd.DataFrame({id_col: g[id_col].to_numpy(), "AN_iter": _calc(g)})
               for _, g in base.groupby(group_cols, sort=False, dropna=False, observed=False)]
        mapper = pd.concat(per, ignore_index=True).set_index(id_col)["AN_iter"]
    else:
        mapper = pd.DataFrame({id_col: base[id_col].to_numpy(),
                               "AN_iter": _calc(base)}).set_index(id_col)["AN_iter"]
    val = mapper.get(id_value, np.nan)
    return int(val) if pd.notna(val) else int(missing_fill)

# ══════════════════════════════════════════════════════════════════
#  DATA SETUP
# ══════════════════════════════════════════════════════════════════
def normalize_master_ann(ann_df):
    if ann_df is None or ann_df.empty:
        return pd.DataFrame(columns=["source","AN","ID","t0","t1","Behavior","notes","tags"])
    df = ann_df.copy()
    ren = {}
    if "t0" not in df and "t0_sec" in df:         ren["t0_sec"]      = "t0"
    if "t1" not in df and "t1_sec" in df:         ren["t1_sec"]      = "t1"
    if "Behavior" not in df and "behavior" in df: ren["behavior"]    = "Behavior"
    if "source" not in df:
        if "File" in df:          ren["File"]        = "source"
        elif "source_name" in df: ren["source_name"] = "source"
    if ren: df.rename(columns=ren, inplace=True)
    for c in ("AN","ID"):
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ("t0","t1"):
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Behavior" in df: df["Behavior"] = df["Behavior"].astype(str).str.strip().str.lower()
    if "source"   in df: df["source"]   = df["source"].astype(str).str.strip()
    keep = [c for c in ("source","AN","ID","t0","t1","Behavior","annotator","notes","tags") if c in df.columns]
    df = df[keep].dropna(subset=["ID","t0","t1","Behavior"])
    df = df.loc[df["t1"] > df["t0"]].copy()
    if "notes"     not in df: df["notes"]     = ""
    if "tags"      not in df: df["tags"]      = ""
    if "annotator" not in df: df["annotator"] = "Katya (old)"
    df["Behavior"]  = df["Behavior"].apply(norm_behavior)
    df["tags"]      = df["tags"].apply(norm_tags)
    df["annotator"] = df["annotator"].apply(norm_annotator)
    return df

MASTER_ANN = normalize_master_ann(ann)

_loaded = long_df.copy()[IDS]
if "source" not in _loaded.columns and "File" in _loaded.columns:
    _loaded.rename(columns={"File":"source"}, inplace=True)
if "source" not in _loaded.columns:
    _loaded["source"] = "in_memory"
_loaded["source"] = _loaded["source"].astype(str).str.strip()
for c in ("AN","ID","frame"):
    _loaded[c] = _loaded[c].astype(int)

required = {"source","AN","ID","et","x","y","frame"}
missing  = required - set(_loaded.columns)
if missing: raise ValueError(f"long_df missing: {sorted(missing)}")

def calculate_global_scale(df, percentile=75):
    ranges = df.groupby(["source","AN","ID"]).agg(
        x=("x", lambda v: np.nanmax(v)-np.nanmin(v)),
        y=("y", lambda v: np.nanmax(v)-np.nanmin(v)),
    )
    return float(np.nanpercentile(ranges.max(axis=1), percentile) * 1.25)

UNIFORM_VIEW_SIZE = calculate_global_scale(_loaded, percentile=75)

def intervals_for(src, an, idv):
    if MASTER_ANN is None or MASTER_ANN.empty:
        print("ruh roh")
        return MASTER_ANN.iloc[0:0].copy()
    df = MASTER_ANN
    if "source" in df:
        out = df[(df["source"]==str(src)) & (df["ID"]==idv)]
    elif "AN" in df and not df["AN"].isna().all():
        out = df[(df["AN"].fillna(an)==an) & (df["ID"]==idv)]
    else:
        out = df[df["ID"]==idv]
    return out.sort_values(["t0","t1"]).reset_index(drop=True)

def subset_id_df(src, AN_val, ID_val):
    cols = ["source","AN","ID","et","x","y","frame",
            SPINE_HEAD_X,SPINE_HEAD_Y,SPINE_TAIL_X,SPINE_TAIL_Y]
    have = [c for c in cols if c in _loaded.columns]
    df = _loaded.loc[
        (_loaded["source"]==src)&(_loaded["AN"]==AN_val)&(_loaded["ID"]==ID_val), have
    ].copy()
    return df.sort_values("et").reset_index(drop=True)

def get_intervals_and_gaps(src, an, idv, df_one):
    if df_one.empty: return [], []
    et = df_one["et"].to_numpy()
    t_min, t_max = float(et.min()), float(et.max())
    m_df   = intervals_for(src, an, idv)
    m_ints = [(float(r["t0"]),float(r["t1"]),str(r["Behavior"])) for _,r in m_df.iterrows()]
    s_df   = annotations_df[
        (annotations_df["AN"]==an)&(annotations_df["ID"]==idv)&(annotations_df["source"]==src)]
    s_ints = [(float(r["t0_sec"]),float(r["t1_sec"]),str(r["behavior"])) for _,r in s_df.iterrows()]
    all_ints = sorted(m_ints+s_ints, key=lambda x: x[0])
    gaps, curr, frame_dur = [], t_min, 1.1/FPS
    for t0, t1, _ in all_ints:
        if t0 > curr+frame_dur: gaps.append((curr, t0))
        curr = max(curr, t1)
    if curr < t_max-frame_dur: gaps.append((curr, t_max))
    return all_ints, gaps

# ── ID maps ──────────────────────────────────────────────────────
_id_counts = _loaded.groupby(["ID"], observed=True)[["source","AN"]].nunique().reset_index()
_dupe_ids  = set(_id_counts.loc[(_id_counts["source"]>1)|(_id_counts["AN"]>1),"ID"].astype(int))

GLOBAL_ID_MAP: Dict[str, tuple] = {}
for (src, an, idv), _ in _loaded.groupby(["source","AN","ID"], sort=True, observed=True):
    key = (f"{int(idv)} [src={src}] (AN={int(an)})")
    GLOBAL_ID_MAP[key] = (str(src), int(an), int(idv))

def _sort_key(s):
    try: return int(s.split(" ")[0])
    except: return s

GLOBAL_ID_KEYS = sorted(GLOBAL_ID_MAP.keys(), key=_sort_key)

def _ann_counts_by_triplet():
    if MASTER_ANN is None or MASTER_ANN.empty: return {}
    g = MASTER_ANN.groupby(["source","AN","ID"], observed=True).size()
    return {(str(k[0]),int(k[1]),int(k[2])): int(v) for k,v in g.items()}

ANN_COUNT_BY_TRIPLET = _ann_counts_by_triplet()

def weight_for_key(key):
    src, an, idv = GLOBAL_ID_MAP[key]
    a = ANN_COUNT_BY_TRIPLET.get((src, an, idv), 0)
    return 5.0 / max(a**0.5, 1e-9)

source_file_guess = (_loaded["source"].mode().iloc[0]
                     if _loaded["source"].notna().any() else "in_memory")

# ══════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════
behavior_tag_memory: Dict[str, set] = {b: set() for b in MAIN_BEHAVIORS}

# ── Seed tag memory from ALL master annotations at load time ─────
# This means any tag ever used for a behavior in the master CSV will
# show up in the known-tags chip area, regardless of which ID you're on.
if MASTER_ANN is not None and not MASTER_ANN.empty and "tags" in MASTER_ANN.columns:
    for _, _mrow in MASTER_ANN.iterrows():
        _mbeh = norm_behavior(str(_mrow.get("Behavior", "")))
        _mtags = str(_mrow.get("tags", ""))
        if _mbeh and _mbeh in MAIN_BEHAVIORS and _mtags not in ("", "nan"):
            for _mt in _mtags.split(";"):
                _mt = _mt.strip()
                if _mt:
                    behavior_tag_memory[_mbeh].add(_mt)

ANNOT_COLS = [
    "annotation_id","timestamp_iso","source","AN","ID",
    "t0_sec","t1_sec","t0_frame","t1_frame",
    "behavior","tags","notes","inclusive_bounds","fps",
    "et_min_id","et_max_id","behavior_version","an_iteration","annotator",
]
annotations_df = pd.DataFrame(columns=ANNOT_COLS)

_state = {
    "key":      GLOBAL_ID_KEYS[0] if GLOBAL_ID_KEYS else "",
    "t0":       0.0,
    "t1":       0.0,
    "behavior": "dwelling",
    "filter_behavior": "None",
    "tags":     "",
    "notes":    "",
}

# ══════════════════════════════════════════════════════════════════
#  MATPLOTLIB DARK SETUP
# ══════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#0d1117",
    "axes.edgecolor":   "#2d333b",
    "axes.labelcolor":  "#768390",
    "axes.titlecolor":  "#cdd9e5",
    "xtick.color":      "#636e7b",
    "ytick.color":      "#636e7b",
    "grid.color":       "#1c2128",
    "text.color":       "#cdd9e5",
    "figure.dpi":       110,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

_old_interactive = plt.isinteractive()
plt.ioff()
fig, ax = plt.subplots(figsize=(5.8, 5.8))
if _old_interactive: plt.ion()
fig.patch.set_facecolor("#0d1117")
try:
    fig.canvas.toolbar_visible  = True
    fig.canvas.header_visible   = False
    fig.canvas.toolbar_position = "bottom"
except Exception:
    pass

out_plot = W.Output(layout=W.Layout(width="640px", min_width="640px"))

# ══════════════════════════════════════════════════════════════════
#  PLOT DRAW
# ══════════════════════════════════════════════════════════════════
def draw_plot(df_one, title):
    ax.clear()
    for a in list(ax.artists): a.remove()

    x, y, t = df_one["x"].to_numpy(), df_one["y"].to_numpy(), df_one["et"].to_numpy()
    
    try:
        src = str(df_one["source"].iat[0])
        an  = int(df_one["AN"].iat[0])
        idv = int(df_one["ID"].iat[0])
        ann_id = intervals_for(src, an, idv)
        if show_annotations and not ann_id.empty and len(t) > 1:
            for beh, part in ann_id.groupby("Behavior", sort=False):
                col  = BEH_RGBA_PLOT.get(beh, (0.4,0.4,0.4,0.2))
                segs = []
                for _, r in part.iterrows():
                    i0 = nearest_idx(t, float(r["t0"]))
                    i1 = nearest_idx(t, float(r["t1"]))
                    if i1 <= i0: continue
                    pts2 = np.column_stack([x[i0:i1+1],y[i0:i1+1]])
                    if len(pts2) < 2: continue
                    segs.append(np.stack([pts2[:-1],pts2[1:]], axis=1))
                if segs:
                    s = np.concatenate(segs, axis=0)
                    ax.add_collection(LineCollection(s, colors=[col]*len(s),
                                                     linewidths=6, zorder=2, alpha=0.4))
    except Exception:
        pass
    
    # ── Spine head (spine_0) as disconnected dots ─────────────────
    # Orange dots mark head position at each frame — no lines connecting them.
    if SPINE_HEAD_X in df_one.columns and SPINE_HEAD_Y in df_one.columns:
        hx = df_one[SPINE_HEAD_X].to_numpy()
        hy = df_one[SPINE_HEAD_Y].to_numpy()
        valid = np.isfinite(hx) & np.isfinite(hy)
        if valid.any():
            ax.scatter(hx[valid], hy[valid], s=3, c="#f97316",
                       alpha=0.40, zorder=3, linewidths=0)
            
            #connect head points to corresponding body points
            head_pts = np.column_stack([hx[valid], hy[valid]])
            base_pts = np.column_stack([x[valid], y[valid]])
            connector_segs = np.stack([head_pts, base_pts], axis=1)
            connectors = LineCollection(connector_segs, colors="#f97316", alpha=0.2, linewidths=0.5, zorder=2)
            ax.add_collection(connectors)

    if len(x) > 1:
        pts  = np.column_stack([x,y]).reshape(-1,1,2)
        segs = np.concatenate([pts[:-1],pts[1:]], axis=1)
        tn   = (t-t.min()) / max(1e-9, t.max()-t.min())
        lc   = LineCollection(segs, cmap="viridis", linewidths=1.8, alpha=0.92, zorder=2)
        lc.set_array(tn[:-1])
        ax.add_collection(lc)
        ax.plot(x[0],  y[0],  "o", ms=5, color="#4ade80", zorder=5, label="start")
        ax.plot(x[-1], y[-1], "s", ms=5, color="#fb923c", zorder=5, label="end")
    else:
        ax.plot(x, y, "o", ms=3, color="#60a5fa")

    

    fx, fy = x[np.isfinite(x)], y[np.isfinite(y)]
    if fx.size and fy.size:
        x_range = fx.max() - fx.min()
        y_range = fy.max() - fy.min()
        pad_x = (x_range * 0.05) if x_range > 0 else 1.0
        pad_y = (y_range * 0.05) if y_range > 0 else 1.0
        ax.set_xlim(fx.min() - pad_x, fx.max() + pad_x)
        ax.set_ylim(fy.min() - pad_y, fy.max() + pad_y + 4)

    ax.add_artist(AnchoredSizeBar(ax.transData, 2.0, "2 mm", loc="lower left",
        pad=1.0, color="#636e7b", frameon=False, size_vertical=0.08,
        fontproperties=fm.FontProperties(size=8, family="monospace")))
    bl = compute_body_length_mm(df_one)
    if bl and math.isfinite(bl):
        ax.add_artist(AnchoredSizeBar(ax.transData, bl, f"~{bl:.1f} mm", loc="lower left",
            pad=2.5, color="#444c56", frameon=False, size_vertical=0.05,
            fontproperties=fm.FontProperties(size=7.5, family="monospace")))

    if TIME_MARKER_STEP > 0 and np.isfinite(t).any():
        et_min, et_max = float(np.nanmin(t)), float(np.nanmax(t))
        k0 = math.ceil(et_min/TIME_MARKER_STEP)
        k1 = math.floor(et_max/TIME_MARKER_STEP)
        if k1 >= k0:
            et_grid = np.array([k*TIME_MARKER_STEP for k in range(k0,k1+1)])
            idxs    = [nearest_idx(t, tt) for tt in et_grid]
            xl, xr  = ax.get_xlim(); yl, yr = ax.get_ylim()
            for tx,ty,tt in zip(x[idxs], y[idxs], et_grid):
                if xl<=tx<=xr and yl<=ty<=yr:
                    ax.plot(tx,ty,"o",ms=3,mec="#636e7b",mfc="none",zorder=4,lw=0.8)
                    ax.text(tx,ty,f" {tt:.1f}s",fontsize=7,ha="left",va="bottom",
                            color="#ffffff",fontfamily="monospace")

    

    an_iter = compute_an_iteration_value(_loaded, int(df_one["ID"].iat[0]))
    info_txt = (f"AN={int(df_one['AN'].iat[0])}  ID={int(df_one['ID'].iat[0])}"
                f"  Iter={an_iter}  n={len(df_one)}"
                f"  [{t.min():.1f}→{t.max():.1f}s]")
    ax.text(0.01, 0.99, info_txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=8, fontfamily="monospace", color="#768390",
            bbox=dict(boxstyle="round,pad=0.3", fc="#0d1117", ec="#2d333b", alpha=0.9))

    ax.set_title(title, fontsize=10, fontfamily="monospace", pad=7)
    ax.set_xlabel("x (mm)", fontsize=8.5)
    ax.set_ylabel("y (mm)", fontsize=8.5)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=8)
    ax.grid(True, linewidth=0.35, alpha=0.6)
    ax.figure.canvas.draw_idle()

# ══════════════════════════════════════════════════════════════════
#  HTML PANEL BUILDER
# ══════════════════════════════════════════════════════════════════
ui_panel = W.HTML(value="<p style='color:#768390;font-family:monospace;'>Loading…</p>")

# Separate status widget — updates via direct Python assignment,
# avoiding the script-tag-injection approach that JupyterLab strips.
status_widget = W.HTML(
    value="<span style='font-family:monospace;font-size:11px;color:#444c56;'>"
          "🐛 ready — go annotate something!</span>",
    layout=W.Layout(padding="3px 0 0 4px"),
)

def _btn(accent=""):
    base = ("border-radius:6px;cursor:pointer;font-family:monospace;font-size:11px;"
            "font-weight:600;border:1px solid;transition:opacity .1s;padding:0 12px;"
            "letter-spacing:.02em;")
    if accent:
        return base + (f"background:{accent}20;border-color:{accent}60;color:{accent};")
    return base + ("background:#1c2128;border-color:#2d333b;color:#adbac7;")

def _inp():
    return ("background:#0d1117;border:1px solid #2d333b;color:#cdd9e5;"
            "border-radius:5px;padding:5px 8px;font-size:12px;font-family:monospace;"
            "outline:none;width:100%;box-sizing:border-box;")

def _lbl():
    return ("font-size:10px;color:#636e7b;font-family:monospace;"
            "display:block;margin-bottom:3px;text-transform:uppercase;"
            "letter-spacing:.06em;")

def _section_hdr(text):
    return (f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:.1em;color:#444c56;margin-bottom:6px;'>{text}</div>")

def _del_confirm_btn(cmd_str):
    """Two-click confirm delete button for session annotations.
    First click: turns orange and says ✓?, auto-resets after 2 s.
    Second click (within 2 s): fires the pyCmd.
    """
    return (
        f"<button "
        f"onclick=\"event.stopPropagation();var b=this;"
        f"if(b.dataset.c==='1'){{"
        f"pyCmd('{cmd_str}');"
        f"}}else{{"
        f"b.dataset.c='1';"
        f"b.textContent='✓?';"
        f"b.style.color='#fb923c';"
        f"setTimeout(function(){{b.dataset.c='';b.textContent='✕';"
        f"b.style.color='#f85149';}},2000);"
        f"}}\" "
        f"style='background:none;border:none;color:#f85149;cursor:pointer;"
        f"font-size:13px;padding:0 4px;line-height:1;transition:color .15s;'>✕</button>"
    )

def _tag_cell(tags_str):
    """Compact tag chips for annotation table cells."""
    if not tags_str or str(tags_str) in ("", "nan"):
        return "<span style='color:#444c56;font-size:9px;'>—</span>"
    chips = []
    for t in str(tags_str).split(";"):
        t = t.strip()
        if t:
            chips.append(
                f"<span style='display:inline-block;background:#21262d;"
                f"border:1px solid #2d333b;border-radius:2px;padding:0 3px;"
                f"font-size:9px;color:#768390;font-family:monospace;"
                f"margin:1px;white-space:nowrap;'>{html_escape(t)}</span>"
            )
    return "".join(chips) if chips else "<span style='color:#444c56;font-size:9px;'>—</span>"

def _annotation_stats_html():
    """Stats bar: total count + per-behavior colored chips.
    Hover a chip to see tag breakdown tooltip.
    Counts master (M) and session (S) annotations together.
    """
    beh_counts: Dict[str, int] = {b: 0 for b in MAIN_BEHAVIORS}
    beh_tag_counts: Dict[str, Dict[str, int]] = {b: {} for b in MAIN_BEHAVIORS}

    def _tally(beh_raw, tags_raw):
        b = norm_behavior(str(beh_raw)) or "other"
        beh_counts[b] = beh_counts.get(b, 0) + 1
        ts = str(tags_raw) if tags_raw else ""
        tags = ([t.strip() for t in ts.split(";") if t.strip()]
                if ts not in ("", "nan") else ["(no tag)"])
        for tag in tags:
            beh_tag_counts[b][tag] = beh_tag_counts[b].get(tag, 0) + 1

    master_n = 0
    if MASTER_ANN is not None and not MASTER_ANN.empty:
        master_n = len(MASTER_ANN)
        for _, r in MASTER_ANN.iterrows():
            _tally(r.get("Behavior", "other"), r.get("tags", ""))

    session_n = len(annotations_df)
    for _, r in annotations_df.iterrows():
        _tally(r.get("behavior", "other"), r.get("tags", ""))

    total = master_n + session_n

    chips = []
    for b in MAIN_BEHAVIORS:
        n = beh_counts.get(b, 0)
        if n == 0:
            continue
        col = BEH_COLORS.get(b, "#94a3b8")

        # Build tag breakdown rows for the hover tooltip
        tt_rows = "".join(
            f"<div style='display:flex;justify-content:space-between;"
            f"gap:12px;padding:1px 0;'>"
            f"<span style='color:#768390;'>{html_escape(tag)}</span>"
            f"<span style='color:#adbac7;font-weight:600;'>{cnt}</span></div>"
            for tag, cnt in sorted(beh_tag_counts[b].items(), key=lambda x: -x[1])
        ) or "<span style='color:#444c56;font-size:9px;'>no tags recorded</span>"

        tooltip = (
            f"<div class='ann-tt' style='display:none;position:absolute;z-index:1000;"
            f"top:calc(100% + 3px);left:0;background:#161b22;"
            f"border:1px solid #2d333b;border-radius:6px;padding:8px 10px;"
            f"min-width:150px;font-family:monospace;font-size:10px;"
            f"white-space:nowrap;box-shadow:0 4px 16px #000a;'>"
            f"<div style='color:#636e7b;font-size:9px;text-transform:uppercase;"
            f"letter-spacing:.06em;margin-bottom:5px;border-bottom:1px solid #2d333b;"
            f"padding-bottom:3px;'>{html_escape(b)} · by tag</div>"
            f"{tt_rows}</div>"
        )

        chips.append(
            f"<div style='position:relative;display:inline-block;margin:1px 2px;' "
            f"onmouseover=\"this.querySelector('.ann-tt').style.display='block'\" "
            f"onmouseout=\"this.querySelector('.ann-tt').style.display='none'\">"
            f"<span style='background:{col}22;color:{col};border:1px solid {col}44;"
            f"border-radius:3px;padding:2px 7px;font-size:10px;font-family:monospace;"
            f"cursor:default;white-space:nowrap;'>{html_escape(b)}: {n}</span>"
            f"{tooltip}</div>"
        )

    header = (
        f"<span style='font-family:monospace;font-size:10px;color:#636e7b;"
        f"white-space:nowrap;margin-right:6px;'>🐛 "
        f"<b style='color:#adbac7;'>{total}</b> intervals "
        f"<span style='color:#444c56;'>({master_n}M + {session_n}S)</span></span>"
    )

    return (
        f"<div style='display:flex;align-items:center;flex-wrap:wrap;gap:2px;"
        f"margin:6px 0 4px;padding:5px 8px;background:#161b22;"
        f"border:1px solid #2d333b;border-radius:6px;'>"
        f"{header}{''.join(chips)}</div>"
    )

def _timeline_html(src, an, idv, df_one):
    if df_one.empty: return ""
    et = df_one["et"].to_numpy()
    tmin, tmax = float(et.min()), float(et.max())
    dur = tmax - tmin
    if dur <= 0: return ""

    all_ints, gaps = get_intervals_and_gaps(src, an, idv, df_one)
    W_PX, H_PX = 440, 20

    def tx(t):
        return round((t - tmin) / dur * W_PX, 1)

    parts = []
    for g0, g1 in gaps:
        parts.append(
            f'<rect x="{tx(g0)}" y="0" width="{max(1,tx(g1)-tx(g0))}" height="{H_PX}" '
            f'fill="#f85149" opacity="0.3" rx="2"/>')
    for t0, t1, beh in all_ints:
        col = BEH_COLORS.get(beh, "#94a3b8")
        parts.append(
            f'<rect x="{tx(t0)}" y="2" width="{max(1,tx(t1)-tx(t0))}" height="{H_PX-4}" '
            f'fill="{col}" opacity="0.82" rx="2"/>'
        )

    if TIME_MARKER_STEP > 0:
        k0 = math.ceil(tmin/TIME_MARKER_STEP)
        k1 = math.floor(tmax/TIME_MARKER_STEP)
        for k in range(k0, k1+1):
            px = tx(k*TIME_MARKER_STEP)
            parts.append(f'<line x1="{px}" y1="0" x2="{px}" y2="{H_PX}" '
                         f'stroke="#2d333b" stroke-width="1"/>')

    n_gaps  = len(gaps)
    gap_dur = sum(g1-g0 for g0,g1 in gaps)
    gap_col = "#4ade80" if n_gaps == 0 else "#f85149"
    gap_lbl = ("✓ fully labeled" if n_gaps == 0
               else f"{n_gaps} gap{'s' if n_gaps>1 else ''} · {gap_dur:.1f}s unlabeled")

    return (
        f"<div style='margin:8px 0 4px;'>"
        f"<div style='display:flex;justify-content:space-between;margin-bottom:3px;'>"
        f"<span style='font-size:10px;color:#636e7b;font-family:monospace;"
        f"text-transform:uppercase;letter-spacing:.07em;'>Timeline</span>"
        f"<span style='font-size:10px;color:{gap_col};font-family:monospace;'>"
        f"{gap_lbl}</span></div>"
        f"<svg width='{W_PX}' height='{H_PX}' style='display:block;border-radius:4px;"
        f"background:#1c2128;border:1px solid #2d333b;'>"
        f"{''.join(parts)}</svg></div>"
    )

def _interval_table_html(src, an, idv):
    m_df = intervals_for(src, an, idv)
    s_df = annotations_df[
        (annotations_df["AN"]==an) &
        (annotations_df["ID"]==idv) &
        (annotations_df["source"]==src)
    ].reset_index(drop=True)

    if m_df.empty and s_df.empty:
        return ("<p style='color:#636e7b;font-size:11px;font-family:monospace;"
                "margin:4px 0;'>No intervals yet.</p>")

    hdr = ("font-size:9px;color:#444c56;font-weight:600;text-align:left;"
           "padding:2px 6px;text-transform:uppercase;letter-spacing:.07em;"
           "white-space:nowrap;")
    # 7 columns: Src | Behavior | Interval | Dur | By | Tags | (del)
    rows = [
        f"<thead><tr style='border-bottom:1px solid #2d333b;'>"
        f"<th style='{hdr}'>Src</th>"
        f"<th style='{hdr}'>Behavior</th>"
        f"<th style='{hdr}'>Interval</th>"
        f"<th style='{hdr}'>Dur</th>"
        f"<th style='{hdr}'>By</th>"
        f"<th style='{hdr}'>Tags</th>"
        f"<th style='{hdr}'></th>"
        f"</tr></thead><tbody>"
    ]

    td = "padding:3px 6px;font-size:10px;white-space:nowrap;"

    # Master CSV rows — no delete button
    for _, r in m_df.iterrows():
        beh      = str(r.get("Behavior",""))
        annotator = str(r.get("annotator",""))
        tags_str  = str(r.get("tags",""))
        col = BEH_COLORS.get(beh,"#94a3b8")
        t0, t1 = float(r["t0"]), float(r["t1"])
        chip = (f"<span style='background:{col}22;color:{col};"
                f"border:1px solid {col}55;border-radius:3px;"
                f"padding:1px 5px;font-size:10px;'>{html_escape(beh)}</span>")
        rows.append(
            f"<tr>"
            f"<td style='{td}color:#636e7b;font-size:9px;'>CSV</td>"
            f"<td style='{td}'>{chip}</td>"
            f"<td style='{td}font-family:monospace;color:#adbac7;'>{t0:.2f}→{t1:.2f}s</td>"
            f"<td style='{td}color:#636e7b;'>Δ{t1-t0:.2f}s</td>"
            f"<td style='{td}color:#636e7b;font-size:9px;'>{html_escape(annotator)}</td>"
            f"<td style='{td}'>{_tag_cell(tags_str)}</td>"
            f"<td></td>"
            f"</tr>"
        )

    # Session rows — two-click confirm delete button
    for i, r in s_df.iterrows():
        beh     = str(r.get("behavior",""))
        tags_str = str(r.get("tags",""))
        col     = BEH_COLORS.get(beh,"#94a3b8")
        t0, t1  = float(r["t0_sec"]), float(r["t1_sec"])
        ann_id  = str(r.get("annotation_id",""))
        chip    = (f"<span style='background:{col}22;color:{col};"
                   f"border:1px solid {col}55;border-radius:3px;"
                   f"padding:1px 5px;font-size:10px;'>{html_escape(beh)}</span>")
        load_js = f"pyCmd('load_interval:{t0}:{t1}:{html_escape(beh)}')"
        rows.append(
            f"<tr style='cursor:pointer;' onclick=\"{load_js}\" title='Click to load'>"
            f"<td style='{td}color:#c084fc;font-size:9px;'>SES</td>"
            f"<td style='{td}'>{chip}</td>"
            f"<td style='{td}font-family:monospace;color:#adbac7;'>{t0:.2f}→{t1:.2f}s</td>"
            f"<td style='{td}color:#636e7b;'>Δ{t1-t0:.2f}s</td>"
            f"<td style='{td}color:#636e7b;font-size:9px;'>{html_escape(NAME)}</td>"
            f"<td style='{td}'>{_tag_cell(tags_str)}</td>"
            f"<td>{_del_confirm_btn(f'del_interval:{ann_id}')}</td>"
            f"</tr>"
        )

    rows.append("</tbody>")
    # overflow-x:auto so Tags column is reachable on narrow panels
    return (
        f"<div style='overflow-x:auto;'>"
        f"<table style='min-width:max-content;width:100%;border-collapse:collapse;'>"
        f"{''.join(rows)}</table></div>"
    )

def _gap_rows_html(src, an, idv, df_one):
    if df_one.empty:
        return "<p style='color:#636e7b;font-size:11px;font-family:monospace;'>No data.</p>"
    _, gaps = get_intervals_and_gaps(src, an, idv, df_one)
    if not gaps:
        return ("<p style='color:#4ade80;font-size:11px;font-family:monospace;'>"
                "✓ No gaps — fully labeled!</p>")
    items = []
    for g0, g1 in gaps:
        js = f"pyCmd('load_gap:{g0}:{g1}')"
        items.append(
            f"<div onclick=\"{js}\" "
            f"style='cursor:pointer;display:flex;justify-content:space-between;"
            f"align-items:center;padding:5px 8px;border-radius:4px;margin-bottom:3px;"
            f"background:#1c2128;border:1px solid #f8514940;' "
            f"onmouseover=\"this.style.background='#2a1a1a'\" "
            f"onmouseout=\"this.style.background='#1c2128'\">"
            f"<span style='font-family:monospace;font-size:11px;color:#f87171;'>"
            f"{g0:.2f}s → {g1:.2f}s</span>"
            f"<span style='font-family:monospace;font-size:10px;color:#636e7b;'>"
            f"Δ{g1-g0:.2f}s</span></div>"
        )
    return "".join(items)

def rebuild_panel():
    key = _state["key"]
    if key not in GLOBAL_ID_MAP:
        ui_panel.value = "<p style='color:#f85149'>Invalid ID</p>"
        return

    src, an, idv = GLOBAL_ID_MAP[key]
    df_one       = subset_id_df(src, an, idv)
    an_iter      = compute_an_iteration_value(_loaded, idv)
    cur_idx      = GLOBAL_ID_KEYS.index(key)
    total_ids    = len(GLOBAL_ID_KEYS)

    id_options = "".join(
        f'<option value="{html_escape(k)}" {"selected" if k==key else ""}>'
        f'{html_escape(k)}</option>'
        for k in GLOBAL_ID_KEYS
    )
    
    f_beh_cur = _state.get("filter_behavior", "None")
    filter_opts_html = "".join(
        f'<option value="{b}" {"selected" if b==f_beh_cur else ""}>{b}</option>'
        for b in ["None"] + MAIN_BEHAVIORS
    )

    beh_cur   = _state["behavior"]
    beh_tiles = []
    for i, b in enumerate(MAIN_BEHAVIORS):
        col    = BEH_COLORS.get(b, "#94a3b8")
        if i == 4 or i == 5: continue
        active = f"border-color:{col};background:{col}20;" if b == beh_cur else ""
        tile   = (
            f"<div onclick=\"pyCmd('set_beh:{b}')\" "
            f"style='cursor:pointer;flex:1;text-align:center;padding:8px 2px;"
            f"border-radius:6px;border:2px solid #2d333b;{active}transition:all .12s;' "
            f"onmouseover=\"this.style.borderColor='{col}'\" "
            f"onmouseout=\"this.style.borderColor='"
            f"{col if b==beh_cur else '#2d333b'}';\">"
            f"<div style='width:8px;height:8px;border-radius:50%;background:{col};"
            f"margin:0 auto 4px;'></div>"
            f"<div style='font-size:10px;color:{'#cdd9e5' if b==beh_cur else '#768390'};"
            f"font-family:monospace;line-height:1.3;'>{b}</div>"
            f"<div style='font-size:9px;color:#444c56;margin-top:1px;'>[{i+1}]</div>"
            f"</div>"
        )
        beh_tiles.append(tile)

    timeline   = _timeline_html(src, an, idv, df_one)
    stats_bar  = _annotation_stats_html()
    int_table  = _interval_table_html(src, an, idv)
    gap_rows   = _gap_rows_html(src, an, idv, df_one)

    known_tags = sorted(behavior_tag_memory.get(beh_cur, set()))
    tag_chips  = "".join(
        f"<span onclick=\"appendTag('{html_escape(t)}')\" "
        f"style='cursor:pointer;background:#21262d;border:1px solid #2d333b;"
        f"border-radius:3px;padding:2px 7px;font-size:10px;color:#768390;"
        f"font-family:monospace;margin:2px;display:inline-block;' "
        f"title='Add tag'>{html_escape(t)}</span>"
        for t in known_tags
    ) or "<span style='color:#444c56;font-size:10px;font-family:monospace;'>none yet</span>"

    t0_val  = f"{_state['t0']:.4f}"
    t1_val  = f"{_state['t1']:.4f}"
    tags_v  = html_escape(_state["tags"])
    notes_v = html_escape(_state["notes"])

    beh_tiles_html = "".join(beh_tiles)

    def _leg_item(b):
        col = BEH_COLORS.get(b, "#94a3b8")
        return (f"<span style='display:inline-flex;align-items:center;gap:3px;"
                f"margin-right:8px;font-family:monospace;font-size:10px;color:#768390;'>"
                f"<span style='width:8px;height:8px;border-radius:2px;background:{col};"
                f"display:inline-block;'></span>{b}</span>")
    legend_html = "".join(_leg_item(b) for b in MAIN_BEHAVIORS)

    html = f"""<div style="font-family:system-ui,-apple-system,sans-serif;background:#0d1117;
border:1px solid #2d333b;border-radius:10px;color:#cdd9e5;overflow:hidden;">

<!-- HEADER -->
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:10px 16px;border-bottom:1px solid #2d333b;background:#161b22;">
  <span style="font-size:13px;font-weight:700;letter-spacing:-.01em;color:#cdd9e5;">
    LARVAL ANNOTATOR
    <span style="font-size:10px;color:#444c56;font-weight:400;font-family:monospace;margin-left:8px;">v4.0</span>
  </span>
  <span style="font-family:monospace;font-size:11px;color:#636e7b;">
    AN={an} · ID={idv} · Iter={an_iter} · {cur_idx+1}/{total_ids}
  </span>
</div>

<!-- BODY -->
<div style="display:grid;grid-template-columns:1fr 300px;min-height:500px;">

  <!-- LEFT -->
  <div style="padding:14px 12px 14px 16px;border-right:1px solid #2d333b;">

    <!-- ID nav -->
    <div style="display:flex;align-items:center;gap:5px;margin-bottom:10px;">
      <button onclick="pyCmd('prev')" style="{_btn()}width:30px;height:30px;font-size:15px;flex-shrink:0;">‹</button>
      <select onchange="pyCmd('set_id:'+this.value)"
        style="flex:1;background:#161b22;color:#cdd9e5;border:1px solid #2d333b;
               border-radius:6px;padding:5px 8px;font-size:11px;font-family:monospace;
               height:30px;cursor:pointer;min-width:0;">
        {id_options}
      </select>
      <button onclick="pyCmd('next')" style="{_btn()}width:30px;height:30px;font-size:15px;flex-shrink:0;">›</button>
      <button onclick="pyCmd('random')" style="{_btn()}height:30px;padding:0 9px;font-size:11px;flex-shrink:0;" title="Weighted random">⚄</button>
      <button onclick="pyCmd('reset_view')" style="{_btn()}height:30px;padding:0 9px;font-size:13px;flex-shrink:0;" title="Reset zoom">⌂</button>
      
      <div style="width:1px;height:20px;background:#2d333b;margin:0 2px;"></div>
      <select onchange="pyCmd('set_filter:'+this.value)" title="Filter navigation by behavior"
        style="background:#161b22;color:#cdd9e5;border:1px solid #2d333b;border-radius:6px;padding:0 4px;font-size:10px;font-family:monospace;height:30px;cursor:pointer;width:50px;">
        {filter_opts_html}
      </select>
    </div>

    <!-- Timeline -->
    {timeline}

    <!-- Annotation stats bar (hover chips for tag breakdown) -->
    {stats_bar}

    <!-- Behavior legend -->
    <div style="margin:2px 0 10px;">{legend_html}</div>

    <!-- EDITOR CARD -->
    <div style="background:#161b22;border:1px solid #2d333b;border-radius:8px;padding:12px;">
      {_section_hdr("Annotation Editor")}

      <!-- Behavior tiles -->
      <div style="display:flex;gap:5px;margin-bottom:12px;">
        {beh_tiles_html}
      </div>

      <!-- t0 / t1 -->
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">
        <div>
          <label style="{_lbl()}">t&#8320; (s)</label>
          <input id="t0-input" type="number" step="0.1" value="{t0_val}"
            onchange="pyCmd('set_t0:'+this.value)"
            style="{_inp()}">
        </div>
        <div>
          <label style="{_lbl()}">t&#8321; (s)</label>
          <input id="t1-input" type="number" step="0.1" value="{t1_val}"
            onchange="pyCmd('set_t1:'+this.value)"
            style="{_inp()}">
        </div>
      </div>

      <!-- Tags + Notes -->
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">
        <div>
          <label style="{_lbl()}">Tags (semicolon-sep)</label>
          <input id="tags-input" type="text" value="{tags_v}" placeholder="tag1;tag2"
            onchange="pyCmd('set_tags:'+this.value)"
            style="{_inp()}">
        </div>
        <div>
          <label style="{_lbl()}">Notes</label>
          <input id="notes-input" type="text" value="{notes_v}" placeholder="optional"
            onchange="pyCmd('set_notes:'+this.value)"
            style="{_inp()}">
        </div>
      </div>

      <!-- Known tag chips -->
      <div style="margin-bottom:10px;">
        <span style="font-size:9px;color:#444c56;font-family:monospace;
                     text-transform:uppercase;letter-spacing:.07em;">
          Known tags for {html_escape(beh_cur)}:</span>
        <div style="margin-top:4px;">{tag_chips}</div>
      </div>

      <!-- Hotkey hint -->
      <div style="font-size:10px;color:#444c56;font-family:monospace;margin-bottom:10px;">
        Keys: <b style="color:#636e7b;">1–5</b> behavior &nbsp;
        <b style="color:#636e7b;">←/→</b> prev/next ID &nbsp;
        <b style="color:#636e7b;">Enter</b> add &nbsp;
        <b style="color:#636e7b;">Ctrl+Z</b> undo
      </div>

      <!-- Action buttons -->
      <div style="display:flex;gap:6px;">
        <button onclick="pyCmd('add')"
          style="{_btn('#4ade80')}flex:2;height:34px;font-size:12px;">
          ＋ Add Interval</button>
        <button onclick="pyCmd('undo')"
          style="{_btn()}flex:1;height:34px;font-size:12px;">
          ↺ Undo</button>
        <button onclick="pyCmd('save')"
          style="{_btn('#60a5fa')}flex:1;height:34px;font-size:12px;">
          💾 Save</button>
      </div>
    </div>
  </div>

  <!-- RIGHT -->
  <div style="padding:14px 16px 14px 12px;overflow-y:auto;max-height:700px;
              display:flex;flex-direction:column;gap:12px;">

    <!-- Intervals -->
    <div>
      {_section_hdr("All Intervals")}
      <div style="background:#161b22;border:1px solid #2d333b;border-radius:6px;
                  padding:6px;max-height:260px;overflow-y:auto;overflow-x:auto;">
        {int_table}
      </div>
    </div>

    <!-- Gaps -->
    <div>
      {_section_hdr("Unlabeled Gaps")}
      <div style="margin-bottom:8px;">{gap_rows}</div>
      <div style="display:flex;flex-direction:column;gap:5px;">
        <button onclick="pyCmd('jump_gap')"
          style="{_btn('#c084fc')}height:30px;font-size:11px;width:100%;">
          → Jump to Next Gap (any ID)</button>
        <button onclick="pyCmd('fill_gaps')"
          style="{_btn('#fb923c')}height:30px;font-size:11px;width:100%;">
          Fill Gaps → nondwelling</button>
        <button onclick="pyCmd('set_entire')"
          style="{_btn('#f85149')}height:30px;font-size:11px;width:100%;">
          Set Entire → nondwelling</button>
      </div>
    </div>

  </div>
</div>

<!-- JS -->
<script>
(function() {{
  // Keyboard shortcuts event hub listener tracking routing
  var BEHS = {json.dumps(MAIN_BEHAVIORS)};
  document.removeEventListener('keydown', window._annKeydownHandler);
  
  window._annKeydownHandler = function(e) {{
    var tag = document.activeElement ? document.activeElement.tagName : '';
    if (tag==='INPUT'||tag==='TEXTAREA'||tag==='SELECT') return;
    var n = parseInt(e.key);
    if (n >= 1 && n <= BEHS.length) {{ pyCmd('set_beh:'+BEHS[n-1]); return; }}
    if (e.key==='ArrowLeft')      {{ pyCmd('prev'); return; }}
    if (e.key==='ArrowRight')     {{ pyCmd('next'); return; }}
    if (e.key==='Enter')          {{ pyCmd('add');  return; }}
    if (e.key==='z' && (e.ctrlKey||e.metaKey)) {{ pyCmd('undo'); return; }}
  }};
  
  document.addEventListener('keydown', window._annKeydownHandler);
}})();
</script>
</div>"""

    ui_panel.value = html


# ══════════════════════════════════════════════════════════════════
#  STATUS  — direct Python widget update (no script-tag injection)
# ══════════════════════════════════════════════════════════════════
def _set_status(msg, kind="info"):
    colors = {"ok": "#4ade80", "warn": "#fb923c", "err": "#f85149", "info": "#60a5fa"}
    col = colors.get(kind, "#768390")
    status_widget.value = (
        f"<span style='font-family:monospace;font-size:11px;color:{col};'>{msg}</span>"
    )

# ══════════════════════════════════════════════════════════════════
#  PLOT REDRAW
# ══════════════════════════════════════════════════════════════════
def redraw_plot():
    key = _state["key"]
    if key not in GLOBAL_ID_MAP: return
    src, an, idv = GLOBAL_ID_MAP[key]
    df_one  = subset_id_df(src, an, idv)
    an_iter = compute_an_iteration_value(_loaded, idv)
    with out_plot:
        clear_output(wait=True)
        if df_one.empty:
            print(f"[no data] AN={an} ID={idv}")
        else:
            draw_plot(df_one, title=f"AN={an} · ID={idv} · Iter={an_iter}")
            display(fig.canvas if USING_IPYMPL else fig)
            try:
                tb = getattr(fig.canvas, "toolbar", None)
                if tb and hasattr(tb, "push_current"): tb.push_current()
            except Exception:
                pass

def full_refresh():
    rebuild_panel()
    redraw_plot()

# ══════════════════════════════════════════════════════════════════
#  COMMAND DISPATCH
# ══════════════════════════════════════════════════════════════════
def _dispatch(change):
    global annotations_df
    raw = change["new"]
    if not raw: return
    cmd = raw.split("|")[0]

    if cmd == "prev":       _nav_delta(-1);    return
    if cmd == "next":       _nav_delta(+1);    return
    if cmd == "random":     _do_random();      return
    if cmd == "reset_view": _do_reset_view();  return
    if cmd == "add":        _do_add();         return
    if cmd == "undo":       _do_undo();        return
    if cmd == "save":       _do_save();        return
    if cmd == "jump_gap":   _do_jump_gap();    return
    if cmd == "fill_gaps":  _do_fill_gaps();   return
    if cmd == "set_entire": _do_set_entire();  return

    if cmd.startswith("set_id:"):
        key = cmd[7:]
        if key in GLOBAL_ID_MAP:
            _state["key"] = key
            full_refresh()
        return
    
    if cmd.startswith("set_filter:"):
        _state["filter_behavior"] = cmd[11:]
        rebuild_panel()
        return

    if cmd.startswith("set_beh:"):
        _state["behavior"] = cmd[8:]
        rebuild_panel(); return

    if cmd.startswith("set_t0:"):
        try: _state["t0"] = float(cmd[7:])
        except: pass
        return

    if cmd.startswith("set_t1:"):
        try: _state["t1"] = float(cmd[7:])
        except: pass
        return

    if cmd.startswith("set_tags:"):
        _state["tags"] = cmd[9:]; return

    if cmd.startswith("set_notes:"):
        _state["notes"] = cmd[10:]; return

    if cmd.startswith("load_interval:"):
        parts = cmd.split(":")
        try:
            _state["t0"] = float(parts[1])
            _state["t1"] = float(parts[2])
            beh = parts[3] if len(parts) > 3 else "dwelling"
            _state["behavior"] = beh if beh in MAIN_BEHAVIORS else "other"
        except: pass
        rebuild_panel(); return

    if cmd.startswith("load_gap:"):
        parts = cmd.split(":")
        try:
            _state["t0"] = float(parts[1])
            _state["t1"] = float(parts[2])
        except: pass
        rebuild_panel()
        _set_status(
            f"🕳️ gap loaded [{_state['t0']:.2f}→{_state['t1']:.2f}s] — label it!",
            "info"
        )
        return

    if cmd.startswith("del_interval:"):
        ann_id = cmd[13:]
        annotations_df = annotations_df[
            annotations_df["annotation_id"] != ann_id].copy()
        rebuild_panel(); redraw_plot()
        _set_status("🗑️ interval deleted — poof, gone, never existed", "warn")
        return

# ── action implementations ────────────────────────────────────────
# ── action implementations ────────────────────────────────────────
def _has_behavior(src, an, idv, beh):
    if beh == "None": return True
    
    # Check master annotations
    m_df = intervals_for(src, an, idv)
    if not m_df.empty and (m_df["Behavior"] == beh).any():
        return True
        
    # Check current session annotations
    s_df = annotations_df[
        (annotations_df["AN"]==an)&
        (annotations_df["ID"]==idv)&
        (annotations_df["source"]==src)
    ]
    if not s_df.empty and (s_df["behavior"] == beh).any():
        return True
        
    return False

def _nav_delta(delta):
    keys = GLOBAL_ID_KEYS
    if not keys: return
    try:    idx = keys.index(_state["key"])
    except: idx = 0
    
    f_beh = _state.get("filter_behavior", "None")
    
    # Scan forward or backward until we hit an ID with the behavior
    for step in range(1, len(keys) + 1):
        test_idx = (idx + delta * step) % len(keys)
        test_key = keys[test_idx]
        src, an, idv = GLOBAL_ID_MAP[test_key]
        
        if _has_behavior(src, an, idv, f_beh):
            _state["key"] = test_key
            full_refresh()
            return
            
    _set_status(f"🚫 No IDs found containing behavior: {f_beh}", "warn")

def _do_random():
    if not GLOBAL_ID_KEYS: return
    
    f_beh = _state.get("filter_behavior", "None")
    valid_keys = [k for k in GLOBAL_ID_KEYS if _has_behavior(*GLOBAL_ID_MAP[k], f_beh)]
    
    if not valid_keys:
        _set_status(f"🚫 No IDs found containing behavior: {f_beh}", "warn")
        return

    w = np.array([weight_for_key(k) for k in valid_keys], float)
    s = float(w.sum())
    
    if not np.isfinite(s) or s <= 0:
        _state["key"] = valid_keys[int(np.random.choice(len(valid_keys)))]
    else:
        _state["key"] = valid_keys[int(np.random.choice(len(valid_keys), p=w/s))]
        
    full_refresh()
    _set_status(f"🎲 mystery worm selected: {_state['key']}", "info")

def _do_reset_view():
    key = _state["key"]
    if key not in GLOBAL_ID_MAP: return
    src, an, idv = GLOBAL_ID_MAP[key]
    df_one = subset_id_df(src, an, idv)
    if df_one.empty: return
    x, y = df_one["x"].to_numpy(), df_one["y"].to_numpy()
    fx, fy = x[np.isfinite(x)], y[np.isfinite(y)]
    if fx.size and fy.size:
        xm, ym = (fx.max()+fx.min())/2, (fy.max()+fy.min())/2
        h = UNIFORM_VIEW_SIZE/2
        ax.set_xlim(xm-h, xm+h); ax.set_ylim(ym-h, ym+h)
        ax.figure.canvas.draw_idle()

def _do_add():
    global annotations_df
    key = _state["key"]
    if key not in GLOBAL_ID_MAP:
        _set_status("💀 no valid ID selected — are you even trying", "err"); return
    src, an, idv = GLOBAL_ID_MAP[key]
    df_one = subset_id_df(src, an, idv)
    if df_one.empty:
        _set_status("🪦 no trajectory data for this ID", "err"); return
    et = df_one["et"].to_numpy()
    et_min, et_max = float(et.min()), float(et.max())
    t0, t1 = _state["t0"], _state["t1"]
    if not (math.isfinite(t0) and math.isfinite(t1)):
        _set_status("🤯 t0/t1 must be actual numbers, not philosophical concepts", "err"); return
    if t1 < t0:
        _set_status("⏳ t1 < t0 — impossible", "err"); return
    t0 = clamp(t0, et_min, et_max); t1 = clamp(t1, et_min, et_max)
    i0, i1 = nearest_idx(et,t0), nearest_idx(et,t1)
    f0, f1 = int(df_one["frame"].iat[i0]), int(df_one["frame"].iat[i1])
    t0s, t1s = float(df_one["et"].iat[i0]), float(df_one["et"].iat[i1])
    if abs(t1s-t0s) < 1.0/FPS:
        _set_status(
            f"🔬 too short to matter (Δ={abs(t1s-t0s):.4f}s) — need ≥1 frame", "err"
        ); return
    beh  = norm_behavior(_state["behavior"]) or "other"
    tags = norm_tags(_state["tags"])
    if tags:
        behavior_tag_memory.setdefault(beh, set()).update(tags.split(";"))
    same = annotations_df[
        (annotations_df["AN"]==an)&(annotations_df["ID"]==idv)&
        (annotations_df["source"]==src)]
    n_ov = sum(interval_overlap(t0s,t1s,r["t0_sec"],r["t1_sec"],INCLUSIVE_BOUNDS)
               for _,r in same.iterrows())
    new_row = {
        "annotation_id":    str(uuid.uuid4()),
        "timestamp_iso":    datetime.now(timezone.utc).isoformat(),
        "source":           source_file_guess,
        "AN": an, "ID": idv,
        "t0_sec": min(t0s,t1s), "t1_sec": max(t0s,t1s),
        "t0_frame": min(f0,f1), "t1_frame": max(f0,f1),
        "behavior": beh, "tags": tags, "notes": _state["notes"].strip(),
        "inclusive_bounds": INCLUSIVE_BOUNDS, "fps": FPS,
        "et_min_id": et_min, "et_max_id": et_max,
        "behavior_version": "v1",
        "an_iteration": compute_an_iteration_value(_loaded, idv),
        "annotator": NAME,
    }
    annotations_df = pd.concat([annotations_df, pd.DataFrame([new_row])], ignore_index=True)
    msg = f"🐛 wiggle logged! {beh}  [{t0s:.2f}→{t1s:.2f}s]"
    if n_ov:
        msg += f"  · ⚠️ {n_ov} overlap(s), brave choice"
    rebuild_panel(); redraw_plot()
    _set_status(msg, "ok")

def _do_undo():
    global annotations_df
    if annotations_df.empty:
        _set_status("🤷 nothing to undo — the slate is already clean", "warn"); return
    annotations_df = annotations_df.iloc[:-1].copy()
    rebuild_panel(); redraw_plot()
    _set_status("↩️ last wriggle unwriggled — it never happened", "warn")

def _do_save():
    if annotations_df.empty:
        _set_status("😴 nothing to save yet — annotate something first", "warn"); return
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"annotations_{ts}.csv"
    log_path = LOG_DIR / f"annotations_log_{ts}.csv"
    annotations_df.to_csv(out_path, index=False)
    pd.DataFrame([{
        "log_id":        str(uuid.uuid4()),
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "action":        "save_annotations",
        "payload_json":  json.dumps({"n_rows": len(annotations_df), "out_file": str(out_path)}),
        "result": "ok", "message": "",
    }]).to_csv(log_path, index=False)
    _set_status(
        f"💾 {len(annotations_df)} wriggles cocooned → {out_path.name}", "ok"
    )

def _do_jump_gap():
    keys = GLOBAL_ID_KEYS
    try:    start = keys.index(_state["key"]) + 1
    except: start = 0
    search = keys[start:] + keys[:start]
    for key in search:
        src, an, idv = GLOBAL_ID_MAP[key]
        df_one = subset_id_df(src, an, idv)
        if df_one.empty: continue
        _, gaps = get_intervals_and_gaps(src, an, idv, df_one)
        if gaps:
            _state["key"] = key
            _state["t0"]  = round(float(gaps[0][0]), 4)
            _state["t1"]  = round(float(gaps[0][1]), 4)
            full_refresh()
            _set_status(
                f"🕳️ gap spotted in {key} [{gaps[0][0]:.2f}→{gaps[0][1]:.2f}s] — dive in!",
                "info"
            )
            return
    _set_status("🎉 all worms fully labeled! you are a hero 🏆", "ok")

def _do_fill_gaps():
    global annotations_df
    key = _state["key"]
    if key not in GLOBAL_ID_MAP: return
    src, an, idv = GLOBAL_ID_MAP[key]
    df_one = subset_id_df(src, an, idv)
    if df_one.empty: return
    _, gaps = get_intervals_and_gaps(src, an, idv, df_one)
    if not gaps:
        _set_status("✨ no gaps to fill, this maggot is complete", "info"); return
    et = df_one["et"].to_numpy()
    new_rows = []
    for g0, g1 in gaps:
        i0, i1 = nearest_idx(et,g0), nearest_idx(et,g1)
        new_rows.append({
            "annotation_id":    str(uuid.uuid4()),
            "timestamp_iso":    datetime.now(timezone.utc).isoformat(),
            "source":           source_file_guess,
            "AN": an, "ID": idv,
            "t0_sec": float(df_one["et"].iat[i0]),
            "t1_sec": float(df_one["et"].iat[i1]),
            "t0_frame": int(df_one["frame"].iat[i0]),
            "t1_frame": int(df_one["frame"].iat[i1]),
            "behavior": "nondwelling", "tags": "auto_fill_gap", "notes": "",
            "inclusive_bounds": INCLUSIVE_BOUNDS, "fps": FPS,
            "et_min_id": float(et.min()), "et_max_id": float(et.max()),
            "behavior_version": "v1",
            "an_iteration": compute_an_iteration_value(_loaded, idv),
            "annotator": NAME,
        })
    annotations_df = pd.concat([annotations_df, pd.DataFrame(new_rows)], ignore_index=True)
    rebuild_panel(); redraw_plot()
    _set_status(f"🪱 {len(new_rows)} gap(s) filled with nondwelling!", "ok")

def _do_set_entire():
    global annotations_df
    key = _state["key"]
    if key not in GLOBAL_ID_MAP: return
    src, an, idv = GLOBAL_ID_MAP[key]
    annotations_df = annotations_df[
        ~((annotations_df["AN"]==an)&(annotations_df["ID"]==idv)&
          (annotations_df["source"]==src))].copy()
    df_one = subset_id_df(src, an, idv)
    if df_one.empty: return
    et = df_one["et"].to_numpy()
    _state["t0"] = float(et.min()); _state["t1"] = float(et.max())
    _state["behavior"] = "nondwelling"
    _do_add()
    # _do_add sets its own status; override with a more specific one
    _set_status("🌊 entire trajectory set to nondwelling — maggot fully tamed", "ok")

# ══════════════════════════════════════════════════════════════════
#  TOOLBAR HOME OVERRIDE
# ══════════════════════════════════════════════════════════════════
def _toolbar_home_override(*_): _do_reset_view()
try:
    if hasattr(fig.canvas, "toolbar") and fig.canvas.toolbar is not None:
        fig.canvas.toolbar.home = _toolbar_home_override
except Exception:
    pass

# ══════════════════════════════════════════════════════════════════
#  WIDGET BRIDGE (BULLETPROOF VERSION)
# ══════════════════════════════════════════════════════════════════
# Use a hidden Text input instead of an HTML div.
# JS natively triggers ipywidgets by dispatching standard browser events,
# completely bypassing Jupyter's fragile internal API and preventing race conditions.

_cmd_widget = W.Text(value="")
_cmd_widget.layout = W.Layout(width="0px", height="0px", visibility="hidden", margin="0", padding="0")
_cmd_widget.add_class("py-bridge-hub-input")

def _bridge_dispatch(change):
    raw = change["new"]
    if not raw: return
    # We NO LONGER clear _cmd_widget.value here! 
    # The JS adds a unique timestamp (Date.now()) to every payload. 
    # This makes every click unique, so ipywidgets automatically fires this callback 
    # every time without us needing to clear it (which destroyed the DOM bridge previously).
    _dispatch({"new": raw})

_cmd_widget.observe(_bridge_dispatch, names="value")

_bridge_js = IHTML("""<script>
(function() {
  window.pyCmd = function(cmd) {
    var payload = cmd + '|' + Date.now();
    
    // Find all instances of our hidden bridge inputs (in case the cell was run multiple times)
    var inputs = document.querySelectorAll('.py-bridge-hub-input input[type="text"]');
    
    if (inputs.length > 0) {
      // Always use the most recently rendered one to prevent ghost-cell conflicts
      var input = inputs[inputs.length - 1]; 
      
      // Emulate a real user typing and hitting enter
      input.value = payload;
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new Event('change', { bubbles: true }));
    } else {
      console.error('Larval Annotator: Bridge input not found in DOM.');
    }
  };

  window.appendTag = function(t) {
    var inp = document.getElementById('tags-input');
    if (!inp) return;
    inp.value = inp.value.trim() ? inp.value.trim() + ';' + t : t;
    pyCmd('set_tags:' + inp.value);
  };
})();
</script>""")

# ══════════════════════════════════════════════════════════════════
#  ASSEMBLE & DISPLAY
# ══════════════════════════════════════════════════════════════════
rebuild_panel()

# Give your left panel a solid width constraint so the plot doesn't step on it
ui_panel.layout = W.Layout(width="780px", min_width="780px", flex="0 0 auto")

bridge_wrapper = W.Box([_cmd_widget])

root = W.VBox([
    bridge_wrapper,        # Hidden communication layer hub node
    W.HBox([
        ui_panel,
        out_plot,
    ], layout=W.Layout(align_items="flex-start", gap="10px")), 
    status_widget,         
], layout=W.Layout(max_width="2000px"))

display(_bridge_js)
display(root)
redraw_plot()
