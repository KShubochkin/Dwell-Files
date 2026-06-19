# feature_registry.py
#
# The single source of truth for all feature definitions.
#
# Design
# ──────
# Every feature is registered with:
#   - a unique name  (the column suffix, e.g. "tortuosity")
#   - which windows it applies to  (e.g. [11, 30, 50, 75], or [] for base)
#   - a version integer  — bump this when the formula changes; the cache
#     manifest will automatically flag the column as stale
#   - a compute function  — takes (df, feat, w, fps, g_inst, g_win, groups_ser)
#     and returns a pd.Series aligned to df.index
#   - an optional list of "prereqs"  — other feature names it reads from feat
#     (the orchestrator uses this to order computations within a window pass)
#
# Adding a feature
# ────────────────
# 1. Write a compute function at the bottom of the relevant section.
# 2. Call register() with the appropriate metadata.
#
# Versioning
# ──────────
# Each entry has an integer `version`.  When you change a formula, bump the
# version.  feature_store.update_cache() will detect the version mismatch via
# the manifest and recompute only that column, for all sources.
#
# CONFIG
# ──────
# Global pipeline config is here too — no more reaching into the logic file
# by string key.  feature_store.py imports CONFIG directly.

from __future__ import annotations

import gc
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import numba
from scipy.signal import spectrogram
from scipy.spatial import ConvexHull
import time


# ─────────────────────────────────────────────────────────────────────────────
# Global pipeline config
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "dwelling_tags":                  ["wonderful"],
    "nondwelling_ratio_to_dwelling":  2.0,
    "nondwelling_tag_ratios": {
        "crawl":       2,
        "long":        7,
        "arc":         2,
        "backtrack":   2,
        "sharp_turn":  2,
        "wide_turn":   2,
        "double_turn": 2,
        "paused":      2,
        "triple_turn": 2,
    },
    "pause_threshold":  0.15,
    "min_coverage":     0.1,
    "fps":              6.0,
    "turn_amp_quantile":  0.75,   # percentile of combined turn/head-cast signal used as event threshold
    "turn_min_sep_sec":   0.5,    # refractory period between detected turn events

}



# ─────────────────────────────────────────────────────────────────────────────
# Registry internals
# ─────────────────────────────────────────────────────────────────────────────

class _FeatureDef:
    """Everything the orchestrator needs to know about one feature."""
    __slots__ = ("name", "version", "fn", "prereqs", "is_base")

    def __init__(
        self,
        name: str,
        version: int,
        fn: Callable,
        prereqs: List[str],
        is_base: bool,
    ):
        self.name    = name
        self.version = version
        self.fn      = fn
        self.prereqs = prereqs   # feature names (no w{N}_ prefix) that must be computed first
        self.is_base = is_base   # True → column name is exactly `name`, no window prefix


_REGISTRY: Dict[str, _FeatureDef] = {}


def register(
    name: str,
    *,
    version: int,
    fn: Callable,
    prereqs: Optional[List[str]] = None,
    is_base: bool = False,
):
    """
    Register a feature definition.

    Parameters
    ----------
    name     : column suffix.  For windowed features the column will be
               f"w{w}_{name}" where w comes from FeatureSetConfig; for base
               features it will just be `name`.
    version  : integer; bump to invalidate cached values after a formula change.
               The manifest tracks this per-column via registry_versions().
    fn       : callable — see signature note below.
    prereqs  : list of feature *names* (not column names) that must be computed
               into the scratch frame before this one runs, within the same
               window pass.
    is_base  : if True, the column name is exactly `name` (no window prefix).

    Which windows a feature is computed at is NOT declared here.
    That is the sole responsibility of FeatureSetConfig (for model runs) or
    the explicit column list passed to calculate_columns() / update_cache().

    Compute function signature
    ──────────────────────────
    For windowed features:
        def my_feat(df, feat, w, fps, g_inst, g_win, groups_ser) -> pd.Series:
            ...

    For base features (is_base=True):
        def my_feat(df, feat, fps, g_inst) -> pd.Series:
            ...

    The returned Series must be aligned to df.index.
    """
    if name in _REGISTRY:
        raise ValueError(f"Feature '{name}' is already registered.")
    _REGISTRY[name] = _FeatureDef(
        name=name,
        version=version,
        fn=fn,
        prereqs=prereqs or [],
        is_base=is_base,
    )


def get_registry() -> Dict[str, _FeatureDef]:
    return dict(_REGISTRY)


def registry_versions() -> Dict[str, int]:
    """
    Flat map of {feature_name: version} for every registered feature.
    Fed to the cache manifest for staleness detection.
    Only base-feature names appear here (no w{N}_ prefix) — the manifest
    tracks windowed columns by their full column name, so version lookups
    strip the prefix before checking this map.
    """
    return {fdef.name: fdef.version for fdef in _REGISTRY.values()}


# ─────────────────────────────────────────────────────────────────────────────
# Low-level primitives  (imported by feature functions below)
# ─────────────────────────────────────────────────────────────────────────────

def _roll(g_win, col: str, stat: str) -> pd.Series:
    """Rolling aggregation returning a flat Series aligned to df.index."""
    res = getattr(g_win[col], stat)()
    if isinstance(res.index, pd.MultiIndex):
        return res.reset_index(level=[0, 1], drop=True)
    return res

def _roll_fresh(df, col: str, w_frames: int, stat: str) -> pd.Series:
    """Like _roll(), but builds its own GroupBy+Rolling off the live df instead
    of the orchestrator's pre-built g_win — required for any column you add
    to df during this window's pass, since g_win snapshots its column set at
    construction time and raises KeyError for anything added afterward."""
    res = getattr(
        df.groupby(["source", "ID"])[col].rolling(window=w_frames, min_periods=1, center=True),
        stat,
    )()
    if isinstance(res.index, pd.MultiIndex):
        return res.reset_index(level=[0, 1], drop=True)
    return res

def calc_angle_vec(p1x, p1y, p2x, p2y, p3x, p3y):
    """Vectorised bending angle (degrees)."""
    v1x, v1y = p1x - p2x, p1y - p2y
    v2x, v2y = p3x - p2x, p3y - p2y
    dot = v1x * v2x + v1y * v2y
    mag = np.sqrt(v1x**2 + v1y**2) * np.sqrt(v2x**2 + v2y**2)
    return np.degrees(np.arccos(np.clip(dot / (mag + 1e-6), -1.0, 1.0)))

def _angle_between_vectors(v1x, v1y, v2x, v2y):
    """Angle (degrees) between two free vectors (no shared vertex required)."""
    dot = v1x * v2x + v1y * v2y
    mag = np.sqrt(v1x**2 + v1y**2) * np.sqrt(v2x**2 + v2y**2)
    return np.degrees(np.arccos(np.clip(dot / (mag + 1e-6), -1.0, 1.0)))

@numba.njit
def _revisit_numba(x, y, w):
    n = len(x)
    out = np.zeros(n, dtype=np.float32)
    for i in range(w, n):
        min_dist = 1e9
        for j in range(1, w):
            d = np.sqrt((x[i] - x[i - j]) ** 2 + (y[i] - y[i - j]) ** 2)
            if d < min_dist:
                min_dist = d
        out[i] = min_dist
    return out

def _revisit_stats_shared(df, w, g_inst):
    cache = df.attrs.setdefault("_revisit_cache", {})
    if w not in cache:
        x_arr = df["x"].to_numpy(dtype=np.float64)
        y_arr = df["y"].to_numpy(dtype=np.float64)
        out = np.zeros(len(df), dtype=np.float32)
        for _, idx in g_inst.indices.items():
            order = np.sort(idx)
            out[order] = _revisit_numba(x_arr[order], y_arr[order], w)
        cache[w] = out
    return cache[w]

@numba.njit(fastmath=True)
def _rolling_pca_stats(x, y, w_frames):
    """
    Computes PCA ratio (Feature 11) and PCA Area (Feature 8 proxy) efficiently.
    Returns a 2D array: [pca_ratio, pca_area]
    """
    n = len(x)
    out = np.zeros((n, 2), dtype=np.float32)
    half_w = w_frames // 2
    
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        
        if end - start < 3:
            continue
            
        wx = x[start:end]
        wy = y[start:end]
        mx = np.mean(wx)
        my = np.mean(wy)
        
        cxx = np.sum((wx - mx)**2)
        cyy = np.sum((wy - my)**2)
        cxy = np.sum((wx - mx)*(wy - my))
        
        trace = cxx + cyy
        det = cxx * cyy - cxy**2
        discriminant = max(0.0, trace**2 - 4*det)
        
        l1 = (trace + np.sqrt(discriminant)) / 2.0
        l2 = (trace - np.sqrt(discriminant)) / 2.0
        
        if l1 > 1e-6:
            out[i, 0] = l2 / l1  # PCA Ratio
            out[i, 1] = np.pi * np.sqrt(l1 * l2)  # PCA Area
            
    return out

@numba.njit(fastmath=True)
def _rolling_path_efficiency(x, y, w_frames):
    """
    Computes Path Efficiency Index (Feature 13).
    Max distance from start / total path length.
    """
    n = len(x)
    out = np.zeros(n, dtype=np.float32)
    half_w = w_frames // 2
    
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        
        if end - start < 2:
            out[i] = 1.0
            continue
            
        x0, y0 = x[start], y[start]
        max_d = 0.0
        path_len = 0.0
        
        for j in range(start + 1, end):
            d = np.sqrt((x[j] - x0)**2 + (y[j] - y0)**2)
            if d > max_d: 
                max_d = d
            path_len += np.sqrt((x[j] - x[j-1])**2 + (y[j] - y[j-1])**2)
            
        if path_len > 1e-6:
            out[i] = max_d / path_len
            
    return out

@numba.njit(fastmath=True)
def _turn_event_kernel(x, y, amp, sign, bl, w_frames, thresh, min_sep):
    """
    Detects discrete turn/head-cast events (local maxima of `amp` above
    `thresh`, separated by >= min_sep frames), then for each frame computes
    centered-window stats over [i - w_frames//2, i + w_frames//2]:
        col0: event count
        col1: mean inter-event centroid distance, normalized by body length
        col2: max  inter-event centroid distance, normalized by body length
        col3: mean event amplitude
        col4: coefficient of variation of inter-event TIME gaps (regularity)
        col5: alternation score (fraction of consecutive event pairs with
              opposite turn-direction sign)
    """
    n = len(x)
    out = np.zeros((n, 6), dtype=np.float32)
    half_w = w_frames // 2

    events = np.zeros(n, dtype=np.uint8)
    last_event = -min_sep - 1
    for i in range(1, n - 1):
        if amp[i] >= thresh and amp[i] >= amp[i - 1] and amp[i] >= amp[i + 1]:
            if i - last_event >= min_sep:
                events[i] = 1
                last_event = i

    idxs = np.empty(w_frames + 2, dtype=np.int64)
    for i in range(n):
        start = max(0, i - half_w)
        end   = min(n, i + half_w + 1)

        cnt = 0
        for j in range(start, end):
            if events[j] == 1:
                idxs[cnt] = j
                cnt += 1

        out[i, 0] = cnt
        if cnt == 1:
            out[i, 3] = amp[idxs[0]]
        elif cnt >= 2:
            gap_sum = 0.0
            gap_max = 0.0
            amp_sum = amp[idxs[0]]
            alt_count = 0
            tg_sum = 0.0
            tg_sq_sum = 0.0
            for k in range(1, cnt):
                amp_sum += amp[idxs[k]]
                dx = x[idxs[k]] - x[idxs[k - 1]]
                dy = y[idxs[k]] - y[idxs[k - 1]]
                d = np.sqrt(dx * dx + dy * dy)
                gap_sum += d
                if d > gap_max:
                    gap_max = d
                tg = np.float64(idxs[k] - idxs[k - 1])
                tg_sum += tg
                tg_sq_sum += tg * tg
                if sign[idxs[k]] * sign[idxs[k - 1]] < 0:
                    alt_count += 1
            n_gaps = cnt - 1
            out[i, 1] = np.float32((gap_sum / n_gaps) / (bl + 1e-6))
            out[i, 2] = np.float32(gap_max / (bl + 1e-6))
            out[i, 3] = np.float32(amp_sum / cnt)
            tg_mean = tg_sum / n_gaps
            tg_var  = max(0.0, tg_sq_sum / n_gaps - tg_mean * tg_mean)
            out[i, 4] = np.float32(np.sqrt(tg_var) / (tg_mean + 1e-6))
            out[i, 5] = np.float32(alt_count / n_gaps)
    return out


def _turn_event_signals(df):
    """Combined turn/head-cast amplitude + signed laterality, unit-matched."""
    amp = np.maximum(
        df["omega_heading"].to_numpy(dtype=np.float32),
        np.radians(df["omega_relative"].to_numpy(dtype=np.float32)),  # deg/s -> rad/s
    )
    x0, y0   = df["xspine_0"].to_numpy(dtype=np.float32),  df["yspine_0"].to_numpy(dtype=np.float32)
    x5, y5   = df["xspine_5"].to_numpy(dtype=np.float32),  df["yspine_5"].to_numpy(dtype=np.float32)
    x10, y10 = df["xspine_10"].to_numpy(dtype=np.float32), df["yspine_10"].to_numpy(dtype=np.float32)
    v1x, v1y = x0 - x5, y0 - y5
    v2x, v2y = x10 - x5, y10 - y5
    sign = np.sign(v1x * v2y - v1y * v2x).astype(np.float32)  # CW/CCW body-bend direction
    return amp, sign


def _turn_event_wrapper(df, w, fps, g_inst, groups_ser, stat_idx):
    cache = df.attrs.setdefault("_turn_evt_cache", {})
    if w not in cache:
        w_frames = int(w * fps)
        amp, sign = _turn_event_signals(df)
        thresh  = np.float32(np.quantile(amp, CONFIG.get("turn_amp_quantile", 0.75)))
        min_sep = max(1, int(CONFIG.get("turn_min_sep_sec", 0.5) * fps))

        x_arr  = df["x"].to_numpy(dtype=np.float32)
        y_arr  = df["y"].to_numpy(dtype=np.float32)
        bl_arr = df["group_medians"].to_numpy(dtype=np.float32)

        out = np.zeros((len(df), 6), dtype=np.float32)
        for _, idx in g_inst.indices.items():
            order = np.sort(idx)
            out[order] = _turn_event_kernel(
                x_arr[order], y_arr[order], amp[order], sign[order],
                np.float32(bl_arr[order[0]]),
                w_frames, thresh, min_sep,
            )
        cache[w] = out

    return pd.Series(cache[w][:, stat_idx], index=df.index, dtype="float32")

def _turn_event_count(df, feat, w, fps, g_inst, g_win, groups_ser):
    return _turn_event_wrapper(df, w, fps,g_inst, groups_ser, 0)
register("turn_event_count", version=1, fn=_turn_event_count)

def _turn_event_gap_mean_bl(df, feat, w, fps, g_inst, g_win, groups_ser):
    return _turn_event_wrapper(df, w, fps,g_inst, groups_ser, 1)
register("turn_event_gap_mean_bl", version=1, fn=_turn_event_gap_mean_bl)

def _turn_event_gap_max_bl(df, feat, w, fps, g_inst, g_win, groups_ser):
    return _turn_event_wrapper(df, w, fps,g_inst, groups_ser, 2)
register("turn_event_gap_max_bl", version=1, fn=_turn_event_gap_max_bl)

def _turn_event_amp_mean(df, feat, w, fps, g_inst, g_win, groups_ser):
    return _turn_event_wrapper(df, w, fps,g_inst, groups_ser, 3)
register("turn_event_amp_mean", version=1, fn=_turn_event_amp_mean)

def _turn_event_interval_cv(df, feat, w, fps, g_inst, g_win, groups_ser):
    return _turn_event_wrapper(df, w, fps,g_inst, groups_ser, 4)
register("turn_event_interval_cv", version=1, fn=_turn_event_interval_cv)

def _turn_event_alternation(df, feat, w, fps, g_inst, g_win, groups_ser):
    return _turn_event_wrapper(df, w, fps, g_inst,groups_ser, 5)
register("turn_event_alternation", version=1, fn=_turn_event_alternation)


def _revisit_series(group_df: pd.DataFrame, w: int) -> pd.Series:
    arr = _revisit_numba(group_df["x"].values, group_df["y"].values, w)
    return pd.Series(arr, index=group_df.index, dtype="float32")


def get_windowed_freq(signal: np.ndarray, fps: float, window_seconds: float) -> np.ndarray:
    """Dominant frequency at each frame via STFT."""
    nperseg = int(window_seconds * fps)
    if len(signal) < nperseg:
        return np.zeros(len(signal))
    f, _t, Sxx = spectrogram(signal, fs=fps, nperseg=nperseg,
                              noverlap=nperseg - 1, mode="magnitude")
    dom_freq = f[np.argmax(Sxx, axis=0)]
    pad = len(signal) - len(dom_freq)
    return np.pad(dom_freq, (pad // 2, (pad + 1) // 2), mode="edge")


def has_target_tag(tag_string, target_tags):
    if pd.isna(tag_string):
        return False
    tags = [t.strip() for t in str(tag_string).split(";")]
    return any(t in tags for t in target_tags)


# ─────────────────────────────────────────────────────────────────────────────
# Base feature  (non-windowed, computed once)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_larva_body_length(df, feat, fps, g_inst):
    return df["group_medians"].astype("float32")

register(
    "larva_body_length", version=1, is_base=True,
    fn=_compute_larva_body_length,
)


# ─────────────────────────────────────────────────────────────────────────────
# Windowed features — angular / rotation
# ─────────────────────────────────────────────────────────────────────────────

def _omega_body_mean(df, feat, w, fps, g_inst, g_win, groups_ser):
    return _roll(g_win, "omega_body", "mean")

register("omega_body_mean", version=1, fn=_omega_body_mean)


def _omega_head_std(df, feat, w, fps, g_inst, g_win, groups_ser):
    return _roll(g_win, "omega_head", "std").fillna(0)

register("omega_head_std", version=1, fn=_omega_head_std)


def _omega_relative_mean(df, feat, w, fps, g_inst, g_win, groups_ser):
    return _roll(g_win, "omega_relative", "mean")

register("omega_relative_mean", version=1, fn=_omega_relative_mean)


def _omega_heading_mean(df, feat, w, fps, g_inst, g_win, groups_ser):
    return _roll(g_win, "omega_heading", "mean")

register("omega_heading_mean", version=1, fn=_omega_heading_mean)


# ─────────────────────────────────────────────────────────────────────────────
# Windowed features — spatial / displacement
# ─────────────────────────────────────────────────────────────────────────────

def _rog(df, feat, w, fps, g_inst, g_win, groups_ser):
    rog_x = _roll(g_win, "x", "var").fillna(0)
    rog_y = _roll(g_win, "y", "var").fillna(0)
    return (np.sqrt(rog_x + rog_y) / (df["group_medians"] + 1e-6)).astype("float32")

register("rog", version=1, fn=_rog)


def _tortuosity(df, feat, w, fps, g_inst, g_win, groups_ser):
    half_w  = int(w * fps) // 2
    first_x = g_inst["x"].shift( half_w).fillna(df["x"])
    last_x  = g_inst["x"].shift(-half_w).fillna(df["x"])
    first_y = g_inst["y"].shift( half_w).fillna(df["y"])
    last_y  = g_inst["y"].shift(-half_w).fillna(df["y"])
    disp    = np.sqrt((last_x - first_x)**2 + (last_y - first_y)**2)
    path_len = _roll(g_win, "v_com", "sum") / fps
    epsilon  = df["group_medians"] * 0.1
    return (path_len / (disp + epsilon)).astype("float32")

register("tortuosity", version=2, fn=_tortuosity)


def _msd(df, feat, w, fps, g_inst, g_win, groups_ser):
    half_w  = int(w * fps) // 2
    first_x = g_inst["x"].shift( half_w).fillna(df["x"])
    last_x  = g_inst["x"].shift(-half_w).fillna(df["x"])
    first_y = g_inst["y"].shift( half_w).fillna(df["y"])
    last_y  = g_inst["y"].shift(-half_w).fillna(df["y"])
    disp    = np.sqrt((last_x - first_x)**2 + (last_y - first_y)**2)
    return (disp**2 / w).astype("float32")

register("msd", version=2, fn=_msd)


def _msd_norm(df, feat, w, fps, g_inst, g_win, groups_ser):
    half_w  = int(w * fps) // 2
    first_x = g_inst["x"].shift( half_w).fillna(df["x"])
    last_x  = g_inst["x"].shift(-half_w).fillna(df["x"])
    first_y = g_inst["y"].shift( half_w).fillna(df["y"])
    last_y  = g_inst["y"].shift(-half_w).fillna(df["y"])
    disp    = np.sqrt((last_x - first_x)**2 + (last_y - first_y)**2)
    return (disp**2 / (w * (df["group_medians"]**2) + 1e-6)).astype("float32")

register("msd_norm", version=2, fn=_msd_norm)


def _angular_tortuosity(df, feat, w, fps, g_inst, g_win, groups_ser):
    half_w       = w // 2
    total_ang    = _roll(g_win, "omega_body", "sum") / fps
    first_ang    = g_inst["angle_body_unwrapped"].shift( half_w).fillna(df["angle_body_unwrapped"])
    last_ang     = g_inst["angle_body_unwrapped"].shift(-half_w).fillna(df["angle_body_unwrapped"])
    return (total_ang / (np.abs(last_ang - first_ang) + 0.01)).astype("float32")

register("angular_tortuosity", version=1, fn=_angular_tortuosity)


# ─────────────────────────────────────────────────────────────────────────────
# Windowed features — velocity
# ─────────────────────────────────────────────────────────────────────────────

def _vel_mean(df, feat, w, fps, g_inst, g_win, groups_ser):
    return _roll(g_win, "v_com", "mean")

register("vel_mean", version=1, fn=_vel_mean)


def _vel_std(df, feat, w, fps, g_inst, g_win, groups_ser):
    return _roll(g_win, "v_com", "std").fillna(0)

register("vel_std", version=1, fn=_vel_std)


def _vel_norm_mean(df, feat, w, fps, g_inst, g_win, groups_ser):
    return _roll(g_win, "v_mid_norm", "mean")

register("vel_norm_mean", version=1, fn=_vel_norm_mean)


def _head_vel_mean(df, feat, w, fps, g_inst, g_win, groups_ser):
    return _roll(g_win, "v_head", "mean")

register("head_vel_mean", version=1, fn=_head_vel_mean)


def _head_vel_std(df, feat, w, fps, g_inst, g_win, groups_ser):
    return _roll(g_win, "v_head", "std").fillna(0)

register("head_vel_std", version=1, fn=_head_vel_std)


def _vel_lag(df, feat, w, fps, g_inst, g_win, groups_ser):
    # depends on vel_mean at the same window
    shift_len = max(1, int(w / 6 * fps))
    col = f"w{w}_vel_mean"
    return feat.groupby(groups_ser)[col].shift(shift_len).fillna(0).astype("float32")

register("vel_lag", version=1, fn=_vel_lag, prereqs=["vel_mean"])


def _vel_lead(df, feat, w, fps, g_inst, g_win, groups_ser):
    shift_len = max(1, int(w / 6 * fps))
    col = f"w{w}_vel_mean"
    return feat.groupby(groups_ser)[col].shift(-shift_len).fillna(0).astype("float32")

register("vel_lead", version=1, fn=_vel_lead, prereqs=["vel_mean"])


# ─────────────────────────────────────────────────────────────────────────────
# Windowed features — bending / shape
# ─────────────────────────────────────────────────────────────────────────────

def _bending_std(df, feat, w, fps, g_inst, g_win, groups_ser):
    return _roll(g_win, "bending", "std").fillna(0)

register("bending_std", version=1, fn=_bending_std)


def _hc_ratio_mean(df, feat, w, fps, g_inst, g_win, groups_ser):
    return _roll(g_win, "hc_ratio", "mean")

register("hc_ratio_mean", version=1, fn=_hc_ratio_mean)


def _ht_ratio_mean(df, feat, w, fps, g_inst, g_win, groups_ser):
    return _roll(g_win, "ht_ratio", "mean")

register("ht_ratio_mean", version=1, fn=_ht_ratio_mean)


def _bend_peaks_rate(df, feat, w, fps, g_inst, g_win, groups_ser):
    valid_cnt = _roll(g_win, "v_com", "count")
    return (_roll(g_win, "is_peak", "sum") / (valid_cnt + 1e-6)).astype("float32")

register("bend_peaks_rate", version=1, fn=_bend_peaks_rate)


def _bend_freq_rolling(df, feat, w, fps, g_inst, g_win, groups_ser):
    return (
        g_inst["bending"]
          .transform(lambda x: get_windowed_freq(x.values, fps, w))
          .astype("float32")
    )

register("bend_freq_rolling", version=1, fn=_bend_freq_rolling)


# ─────────────────────────────────────────────────────────────────────────────
# Windowed features — behavioral states
# ─────────────────────────────────────────────────────────────────────────────

def _coverage(df, feat, w, fps, g_inst, g_win, groups_ser):
    valid_cnt = _roll(g_win, "v_com", "count")
    return (valid_cnt / w).astype("float32")

register("coverage", version=1, fn=_coverage)


def _pause_run_frac(df, feat, w, fps, g_inst, g_win, groups_ser):
    return (
        _roll(g_win, "has_neighbor", "mean")
        / (_roll(g_win, "is_paused", "mean") + 1e-6)
    ).astype("float32")

register("pause_run_frac", version=1, fn=_pause_run_frac)


def _reversal_rate(df, feat, w, fps, g_inst, g_win, groups_ser):
    return _roll(g_win, "high_bend_activity", "mean")

register("reversal_rate", version=1, fn=_reversal_rate)


# ─────────────────────────────────────────────────────────────────────────────
# Windowed features — revisitation
# ─────────────────────────────────────────────────────────────────────────────

def _revisitation_mean(df, feat, w, fps, g_inst, g_win, groups_ser):
    s = pd.Series(_revisit_stats_shared(df, w, g_inst), index=df.index)
    return s.groupby(groups_ser).rolling(window=w, min_periods=1).mean() \
            .reset_index(level=0, drop=True).astype("float32")

def _revisitation_mean_norm(df, feat, w, fps, g_inst, g_win, groups_ser):
    s = pd.Series(_revisit_stats_shared(df, w, g_inst), index=df.index)
    rolled = s.groupby(groups_ser).rolling(window=w, min_periods=1).mean() \
              .reset_index(level=0, drop=True)
    return (rolled / (df["group_medians"] + 1e-6)).astype("float32")

register("revisitation_mean", version=2, fn=_revisitation_mean)
register("revisitation_mean_norm", version=2, fn=_revisitation_mean_norm)

# windowed - other:
def _path_length_norm(df, feat, w, fps, g_inst, g_win, groups_ser):
    path_len = _roll(g_win, "v_com", "sum") / fps
    return (path_len / (df["group_medians"] + 1e-6)).astype("float32")
register("path_length_norm", version=1, fn=_path_length_norm)


def _head_bend_max(df, feat, w, fps, g_inst, g_win, groups_ser):
    if "head_bend" not in df.columns:
        v1x, v1y = df["xspine_2"] - df["xspine_0"], df["yspine_2"] - df["yspine_0"]
        v2x, v2y = df["xspine_10"] - df["xspine_5"], df["yspine_10"] - df["yspine_5"]
        df["head_bend"] = _angle_between_vectors(v1x, v1y, v2x, v2y).astype("float32")
    return _roll_fresh(df, "head_bend", int(w * fps),"max")
register("head_bend_max", version=1, fn=_head_bend_max)


def _curvature_index_mean(df, feat, w, fps, g_inst, g_win, groups_ser):
    if "curvature_index" not in df.columns:
        chord = np.sqrt((df["xspine_10"] - df["xspine_0"])**2 + (df["yspine_10"] - df["yspine_0"])**2)
        df["curvature_index"] = (df["body_len"] / (chord + 1e-6)).astype("float32")
    return _roll_fresh(df, "curvature_index", int(w * fps),"mean")
register("curvature_index_mean", version=1, fn=_curvature_index_mean)


def _omega_heading_max(df, feat, w, fps, g_inst, g_win, groups_ser):
    return _roll(g_win, "omega_heading", "max")
register("omega_heading_max", version=1, fn=_omega_heading_max)


def _pause_fraction(df, feat, w, fps, g_inst, g_win, groups_ser):
    return _roll(g_win, "is_paused", "mean").fillna(0).astype("float32")
register("pause_fraction", version=1, fn=_pause_fraction)

def _posture_asymmetry_max(df, feat, w, fps, g_inst, g_win, groups_ser):
    if "asym_norm" not in df.columns:
        x0, y0 = df["xspine_0"], df["yspine_0"]
        x5, y5 = df["xspine_5"], df["yspine_5"]
        x10, y10 = df["xspine_10"], df["yspine_10"]
        num = np.abs((x10 - x0)*(y0 - y5) - (x0 - x5)*(y10 - y0))
        den = np.sqrt((x10 - x0)**2 + (y10 - y0)**2) + 1e-6
        df["asym_norm"] = (num / den / (df["group_medians"] + 1e-6)).astype("float32")
    return _roll_fresh(df, "asym_norm",int(w * fps), "max",)

register("posture_asymmetry_max", version=1, fn=_posture_asymmetry_max)

def _turn_fraction(df, feat, w, fps, g_inst, g_win, groups_ser):
    if "is_turning" not in df.columns:
        df["is_turning"] = (df["omega_heading"] > 0.35).astype("float32")
    return _roll_fresh(df, "is_turning",int(w * fps), "mean")
register("turn_fraction", version=1, fn=_turn_fraction)

def _net_displacement_norm(df, feat, w, fps, g_inst, g_win, groups_ser):
    half_w  = int(w * fps) // 2
    first_x = g_inst["x"].shift(half_w).fillna(df["x"])
    last_x  = g_inst["x"].shift(-half_w).fillna(df["x"])
    first_y = g_inst["y"].shift(half_w).fillna(df["y"])
    last_y  = g_inst["y"].shift(-half_w).fillna(df["y"])
    
    disp = np.sqrt((last_x - first_x)**2 + (last_y - first_y)**2)
    return (disp / (df["group_medians"] + 1e-6)).astype("float32")

register("net_displacement_norm", version=2, fn=_net_displacement_norm)

def _pca_stats_cached(df, w, fps, stat_idx):
    cache = df.attrs.setdefault("_pca_cache", {})
    if w not in cache:
        w_frames = int(w * fps)
        def _apply_pca(g):
            arr = _rolling_pca_stats(g["x"].to_numpy(dtype=np.float32),
                                      g["y"].to_numpy(dtype=np.float32), w_frames)
            return pd.DataFrame(arr, index=g.index)
        full = df.groupby(["source", "ID"], group_keys=False).apply(_apply_pca)
        cache[w] = full.reindex(df.index).to_numpy(dtype=np.float32)
    return pd.Series(cache[w][:, stat_idx], index=df.index, dtype="float32")

def _pca_stats_wrapper(df, feat, w, fps, g_inst, g_win, groups_ser, stat_idx):
    cache = df.attrs.setdefault("_pca_stats_cache", {})
    if w not in cache:
        w_frames = int(w * fps)
        x_arr = df["x"].to_numpy(dtype=np.float32)
        y_arr = df["y"].to_numpy(dtype=np.float32)
        out = np.zeros((len(df), 2), dtype=np.float32)
        for _, idx in g_inst.indices.items():
            order = np.sort(idx)
            out[order] = _rolling_pca_stats(x_arr[order], y_arr[order], w_frames)
        cache[w] = out
    return pd.Series(cache[w][:, stat_idx], index=df.index, dtype="float32")

def _pca_ratio(df, feat, w, fps, g_inst, g_win, groups_ser):
    return _pca_stats_wrapper(df, feat, w, fps, g_inst, g_win, groups_ser, stat_idx=0)

def _pca_area_norm(df, feat, w, fps, g_inst, g_win, groups_ser):
    area = _pca_stats_wrapper(df, feat, w, fps, g_inst, g_win, groups_ser, stat_idx=1)
    return (area / (df["group_medians"]**2 + 1e-6)).astype("float32")

register("pca_ratio", version=2, fn=_pca_ratio)
register("pca_area_norm", version=2, fn=_pca_area_norm)

def _vel_autocorr(df, feat, w, fps, g_inst, g_win, groups_ser):
    if "autocorr_1s" not in df.columns:
        lag = int(fps)
        dx, dy = g_inst["x"].diff(), g_inst["y"].diff()
        dx_lag, dy_lag = g_inst["x"].diff().shift(lag), g_inst["y"].diff().shift(lag)
        dot = (dx * dx_lag) + (dy * dy_lag)
        mag = (dx**2 + dy**2) * (dx_lag**2 + dy_lag**2)
        df["autocorr_1s"] = (dot / (np.sqrt(mag) + 1e-6)).fillna(0).astype("float32")
    return _roll_fresh(df, "autocorr_1s",int(w * fps), "mean")

register("vel_autocorr", version=1, fn=_vel_autocorr)

def _path_efficiency(df, feat, w, fps, g_inst, g_win, groups_ser):
    cache = df.attrs.setdefault("_path_eff_cache", {})
    if w not in cache:
        w_frames = int(w * fps)
        x_arr = df["x"].to_numpy(dtype=np.float32)
        y_arr = df["y"].to_numpy(dtype=np.float32)
        out = np.zeros(len(df), dtype=np.float32)
        for _, idx in g_inst.indices.items():
            order = np.sort(idx)
            out[order] = _rolling_path_efficiency(x_arr[order], y_arr[order], w_frames)
        cache[w] = out
    return pd.Series(cache[w], index=df.index, dtype="float32")
register("path_efficiency", version=2, fn=_path_efficiency)

# ─────────────────────────────────────────────────────────────────────────────
# Windowed features — slope / trend  (depend on other windowed features)
# ─────────────────────────────────────────────────────────────────────────────

def _make_slope_fn(src_feat: str):
    """Factory: finite-difference slope of another windowed feature."""
    def _slope(df, feat, w, fps, g_inst, g_win, groups_ser):
        shift_len = max(1, int(w / 6 * fps))
        col = f"w{w}_{src_feat}"
        tg  = feat.groupby(groups_ser)
        return (
            tg[col].shift(-shift_len).ffill()
            - tg[col].shift( shift_len).bfill()
        ).astype("float32")
    _slope.__name__ = f"_{src_feat}_slope_smooth"
    return _slope


register(
    "omega_body_mean_slope_smooth", # w11 uses it; w50+ added in original
    version=1,
    fn=_make_slope_fn("omega_body_mean"),
    prereqs=["omega_body_mean"],
)

register(
    "revis_slope_smooth", version=1,
    fn=_make_slope_fn("revisitation_mean"),
    prereqs=["revisitation_mean"],
)

register(
    "rog_slope_smooth", version=1,
    fn=_make_slope_fn("rog"),
    prereqs=["rog"],
)

register(
    "tort_slope_smooth", version=1,
    fn=_make_slope_fn("tortuosity"),
    prereqs=["tortuosity"],
)


def _build_base_signals(df: pd.DataFrame, fps: float, pause_threshold: float) -> None:
    """
    Compute all instantaneous (non-windowed) columns on df in-place.
    These are intermediate signals consumed by windowed feature functions.
    """
    g_inst = df.groupby(["source", "ID"])

    x_spines = np.array([df[f"xspine_{i}"].values for i in range(11)])
    y_spines = np.array([df[f"yspine_{i}"].values for i in range(11)])
    dx = np.diff(x_spines, axis=0)
    dy = np.diff(y_spines, axis=0)

    df["body_len"] = np.sum(np.sqrt(dx**2 + dy**2), axis=0)
    df["group_medians"] = g_inst["body_len"].transform("median")
    gm = df["group_medians"]

    df["bending"] = calc_angle_vec(
        df["xspine_0"], df["yspine_0"],
        df["xspine_5"], df["yspine_5"],
        df["xspine_10"], df["yspine_10"],
    )
    df["bending_vel"] = g_inst["bending"].diff() * fps

    df["v_head"]      = np.sqrt(g_inst["xspine_0"].diff()**2  + g_inst["yspine_0"].diff()**2)  * fps
    df["v_head_norm"] = df["v_head"] / (gm + 1e-6)
    df["v_mid"]       = np.sqrt(g_inst["xspine_5"].diff()**2  + g_inst["yspine_5"].diff()**2)  * fps
    df["v_mid_norm"]  = df["v_mid"]  / (gm + 1e-6)
    df["v_tail"]      = np.sqrt(g_inst["xspine_10"].diff()**2 + g_inst["yspine_10"].diff()**2) * fps
    df["v_tail_norm"] = df["v_tail"] / (gm + 1e-6)
    df["v_com"]       = np.sqrt(g_inst["x"].diff()**2 + g_inst["y"].diff()**2) * fps
    df["v_com_norm"]  = df["v_com"]  / (gm + 1e-6)

    df["is_paused"]   = (df["v_com"] < pause_threshold).astype(int)
    df["has_neighbor"]= (df["is_paused"] == 1) & (
        (g_inst["is_paused"].shift(1) == 1) | (g_inst["is_paused"].shift(-1) == 1)
    )

    df["angle_body"]    = np.arctan2(df["yspine_0"] - df["yspine_10"],
                                     df["xspine_0"] - df["xspine_10"]).fillna(0)
    df["angle_head"]    = np.arctan2(df["yspine_0"] - df["yspine_5"],
                                     df["xspine_0"] - df["xspine_5"]).fillna(0)
    dx_com = g_inst["x"].diff(); dy_com = g_inst["y"].diff()
    df["angle_heading"] = np.arctan2(dy_com, dx_com)

    def _get_omega(angles, fps):
        clean = np.nan_to_num(angles.to_numpy(), nan=0.0)
        return np.gradient(np.unwrap(clean)) * fps

    df["omega_body"]     = g_inst["angle_body"].transform(lambda x: _get_omega(x, fps)).abs()
    df["omega_head"]     = g_inst["angle_head"].transform(lambda x: _get_omega(x, fps)).abs()
    df["omega_heading"]  = g_inst["angle_heading"].transform(lambda x: _get_omega(x, fps)).abs()
    df["omega_relative"] = g_inst["bending"].diff().abs() * fps

    fill_cols = ["omega_body", "omega_head", "omega_heading", "omega_relative"]
    df[fill_cols] = g_inst[fill_cols].ffill().bfill().fillna(0)

    df["angle_body_unwrapped"] = g_inst["angle_body"].transform(
        lambda x: np.unwrap(np.nan_to_num(x.to_numpy(), nan=0.0))
    )

    df["ht_ratio"] = (df["v_head"] + 1e-3) / (df["v_tail"] + 1e-3)
    df["hc_ratio"] = (df["v_head"] + 1e-3) / (df["v_mid"]  + 1e-3)

    df["bending_diff"] = g_inst["bending"].diff().abs() * fps
    larva_bend_median  = g_inst["bending_diff"].transform("median")
    df["high_bend_activity"] = (df["bending_diff"] > larva_bend_median).astype(int)

    df["is_peak"] = (
        (df["bending"] > g_inst["bending"].shift( 1)) &
        (df["bending"] > g_inst["bending"].shift(-1))
    ).astype(int)


def _topo_sort_features(fdefs: List[_FeatureDef]) -> List[_FeatureDef]:
    """
    Order feature defs so that each feature is computed after all its prereqs.
    Raises if a cycle is detected.
    """
    name_map = {f.name: f for f in fdefs}
    order, visited, visiting = [], set(), set()

    def visit(f):
        if f.name in visited:
            return
        if f.name in visiting:
            raise RuntimeError(f"Circular prereq detected for feature '{f.name}'")
        visiting.add(f.name)
        for p in f.prereqs:
            if p in name_map:
                visit(name_map[p])
        visiting.discard(f.name)
        visited.add(f.name)
        order.append(f)

    for f in fdefs:
        visit(f)
    return order


def calculate(df: pd.DataFrame, fps: float, pause_threshold: float, windows: List[int] = None):
    """
    Full-sweep compute — runs every registered feature at every window in `windows`.

    Used by build_full_cache() where you genuinely want everything.
    `windows` must be provided; there is no longer a registry-derived default
    because the registry no longer stores per-feature window membership.

    Returns a float32 DataFrame indexed like df, with no label columns.
    """
    if not windows:
        raise ValueError(
            "calculate() requires an explicit `windows` list.  "
            "Pass the full set of windows you want, e.g. windows=[11, 30, 50, 75]."
        )
    # Build the full column list: every non-base feature at every window,
    # plus all base features.
    all_cols = [fdef.name for fdef in _REGISTRY.values() if fdef.is_base]
    for w in sorted(windows):
        for fdef in _REGISTRY.values():
            if not fdef.is_base:
                all_cols.append(f"w{w}_{fdef.name}")
    return calculate_columns(df, fps, pause_threshold, all_cols)


def calculate_columns(
    df: pd.DataFrame,
    fps: float,
    pause_threshold: float,
    columns: List[str],
) -> pd.DataFrame:
    """
    Targeted compute — only calculates the specific columns requested.

    This is the right function to call from update_cache() when you only need
    a few new columns (e.g. adding window 20 for one feature).  It:
      - runs _build_base_signals() once (cheap, all instantaneous)
      - groups windows by which features are actually needed at each window
      - within each window, also runs any prereq features those columns depend on
        (but does NOT emit them in the output — they're scratch)
      - skips every other window and every other feature entirely

    Parameters
    ----------
    columns : explicit list of column names, e.g. ["w20_rog", "w20_tortuosity"]

    Returns
    -------
    DataFrame with exactly the requested columns (float32), indexed like df.
    Missing/unrecognised column names are silently skipped with a warning.
    """
    # ── Parse requested columns into (window, feature_name) pairs ────────────
    wanted: Dict[int, set] = {}   # window -> set of feature names needed as output
    unrecognised = []

    for col in columns:
        # Base features have no w{N}_ prefix
        if not (col.startswith("w") and "_" in col):
            if col in _REGISTRY and _REGISTRY[col].is_base:
                wanted.setdefault(None, set()).add(col)
            else:
                unrecognised.append(col)
            continue
        try:
            w_str, feat_name = col.split("_", 1)
            w = int(w_str[1:])
        except (ValueError, IndexError):
            unrecognised.append(col)
            continue
        fdef = _REGISTRY.get(feat_name)
        if fdef is None or fdef.is_base:
            unrecognised.append(col)
            continue
        wanted.setdefault(w, set()).add(feat_name)

    if unrecognised:
        print(f"[calculate_columns] WARNING: unrecognised columns skipped: {unrecognised}")

    if not wanted:
        return pd.DataFrame(index=df.index)

    _build_base_signals(df, fps, pause_threshold)
    groups_ser = df["source"] + "_" + df["ID"].astype(str)
    feat_out   = pd.DataFrame(index=df.index)

    if None in wanted:
        g_inst = df.groupby(["source", "ID"])
        for feat_name in wanted[None]:
            fdef = _REGISTRY[feat_name]
            feat_out[feat_name] = fdef.fn(df, feat_out, fps, g_inst)

    collected_frames = [feat_out]

    for w, feat_names_wanted in wanted.items():
        if w is None:
            continue
        print(f"  [calculate_columns] Window {w} — computing: {sorted(feat_names_wanted)}")
        g_inst = df.groupby(["source", "ID"])
        g_win  = g_inst.rolling(window=int(w * fps), min_periods=1, center=True)

        def _collect_with_prereqs(name: str, seen: set) -> List[str]:
            if name in seen:
                return []
            seen.add(name)
            fdef = _REGISTRY.get(name)
            if fdef is None:
                return []
            order = []
            for p in fdef.prereqs:
                order.extend(_collect_with_prereqs(p, seen))
            order.append(name)
            return order

        run_order = []
        seen: set = set()
        for name in feat_names_wanted:
            run_order.extend(_collect_with_prereqs(name, seen))

        applicable = [_REGISTRY[n] for n in run_order if n in _REGISTRY and not _REGISTRY[n].is_base]
        applicable = _topo_sort_features(applicable)

        feat_win = pd.DataFrame(index=df.index)   
        for fdef in applicable:
            col = f"w{w}_{fdef.name}"
            t0 = time.perf_counter()
            feat_win[col] = fdef.fn(df, feat_win, w, fps, g_inst, g_win, groups_ser)
            dt = time.perf_counter() - t0
            if dt > 5.0:
                print(f"    SLOW: {col} took {dt:.1f}s")
        
        cols_to_keep = [f"w{w}_{feat_name}" for feat_name in feat_names_wanted]
        cols_to_keep = [c for c in cols_to_keep if c in feat_win.columns]
        
        if cols_to_keep:
            chunk = feat_win[cols_to_keep].reset_index(drop=True)
            collected_frames.append(chunk)

        del g_inst, g_win, feat_win
        gc.collect()

    # Strip the index from the base features frame too
    collected_frames[0].reset_index(drop=True, inplace=True)
    
    final_out = pd.concat(collected_frames, axis=1)
    
    final_out.index = df.index

    return final_out.astype("float32").copy()
