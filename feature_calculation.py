from typing import List
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from scipy.spatial import ConvexHull
import numba
import gc

CONFIG = {
    "dwelling_tags": ["wonderful"],
    "nondwelling_ratio_to_dwelling": 1.0,
    "nondwelling_tag_ratios": {
        "crawl": 1,
        "long": 1,
        "arc": 1,
    },
    "windows": [11, 30, 50, 75],
    "max_window_size": 75,
    "pause_threshold": 0.3,
    "min_coverage": 0.1,
    "fps": 6.0,
}

# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers (defined once so numba compiles them once)
# ──────────────────────────────────────────────────────────────────────────────

def calc_angle_vec(p1x, p1y, p2x, p2y, p3x, p3y):
    """Vectorised bending angle (degrees)."""
    v1x, v1y = p1x - p2x, p1y - p2y
    v2x, v2y = p3x - p2x, p3y - p2y
    dot = v1x * v2x + v1y * v2y
    mag = np.sqrt(v1x**2 + v1y**2) * np.sqrt(v2x**2 + v2y**2)
    return np.degrees(np.arccos(np.clip(dot / (mag + 1e-6), -1.0, 1.0)))


@numba.njit
def _revisit_numba(x, y, w):
    """Per-frame minimum distance to any previous position within w frames."""
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


def _revisit_series(group_df: pd.DataFrame, w: int) -> pd.Series:
    """Wrapper that returns a Series with the group's original index."""
    arr = _revisit_numba(group_df["x"].values, group_df["y"].values, w)
    return pd.Series(arr, index=group_df.index, dtype="float32")


def get_hull_area(pts):
    if len(pts) < 3:
        return 0.0
    try:
        return ConvexHull(pts).area
    except Exception:
        return 0.0


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


# ──────────────────────────────────────────────────────────────────────────────
# Rolling helper
# ──────────────────────────────────────────────────────────────────────────────

def _roll(g_win, col: str, stat: str) -> pd.Series:
    """Apply a rolling aggregation and return a flat Series aligned to df.index."""
    return getattr(g_win[col], stat)().reset_index(level=[0, 1], drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Single-window feature builder
# ──────────────────────────────────────────────────────────────────────────────

def _compute_window_features(
    df: pd.DataFrame,
    s: int,
    fps: float,
    group_medians: pd.Series,
    g_inst,          # DataFrameGroupBy on ['source', 'ID']
) -> pd.DataFrame:
    """
    Compute all windowed features for window size `s` seconds.
    Supported sizes: 11, 30, 50, 75.

    `df` may be mutated temporarily (temp column added/dropped for revisitation)
    but is restored before the function returns.
    """
    if s not in (11, 30, 50, 75):
        raise ValueError(f"Unsupported window size {s}; expected one of (11, 30, 50, 75).")

    w = int(s * fps)
    half_w = w // 2
    shift_len = max(1, int(s / 6 * fps))
    p = f"w{s}_"

    g_win = g_inst.rolling(window=w, min_periods=1, center=True)
    groups_ser = df["source"] + "_" + df["ID"].astype(str)
    feat = pd.DataFrame(index=df.index)

    # ── Features present at every window size ─────────────────────────────────
    feat[f"{p}omega_body_mean"]     = _roll(g_win, "omega_body",     "mean")
    feat[f"{p}omega_head_std"]      = _roll(g_win, "omega_head",     "std").fillna(0)
    feat[f"{p}omega_relative_mean"] = _roll(g_win, "omega_relative", "mean")

    rog_x = _roll(g_win, "x", "var").fillna(0)
    rog_y = _roll(g_win, "y", "var").fillna(0)
    feat[f"{p}rog"] = np.sqrt(rog_x + rog_y).astype("float32")

    first_x = g_inst["x"].shift( half_w).fillna(df["x"])
    last_x  = g_inst["x"].shift(-half_w).fillna(df["x"])
    first_y = g_inst["y"].shift( half_w).fillna(df["y"])
    last_y  = g_inst["y"].shift(-half_w).fillna(df["y"])
    disp     = np.sqrt((last_x - first_x) ** 2 + (last_y - first_y) ** 2)
    path_len = _roll(g_win, "v_com", "sum") / fps
    epsilon  = group_medians * 0.1                    # aligned to df.index via transform
    feat[f"{p}tortuosity"] = (path_len / (disp + epsilon)).astype("float32")
    feat[f"{p}msd"]        = (disp ** 2 / s).astype("float32")

    # ── s == 11 only ──────────────────────────────────────────────────────────
    if s == 11:
        feat[f"{p}bending_std"]  = _roll(g_win, "bending", "std").fillna(0)
        feat[f"{p}hc_ratio_mean"]= _roll(g_win, "hc_ratio", "mean")
        feat[f"{p}ht_ratio_mean"]= _roll(g_win, "ht_ratio", "mean")
        tg = feat.groupby(groups_ser)
        feat[f"{p}omega_body_mean_slope_smooth"] = (
            tg[f"{p}omega_body_mean"].shift(-shift_len).ffill()
            - tg[f"{p}omega_body_mean"].shift( shift_len).bfill()
        ).astype("float32")

    # ── s >= 30 ───────────────────────────────────────────────────────────────
    if s >= 30:
        feat[f"{p}vel_mean"]           = _roll(g_win, "v_com",        "mean")
        feat[f"{p}vel_std"]            = _roll(g_win, "v_com",        "std").fillna(0)
        feat[f"{p}vel_norm_mean"]      = _roll(g_win, "v_mid_norm",   "mean")
        feat[f"{p}head_vel_mean"]      = _roll(g_win, "v_head",       "mean")
        feat[f"{p}head_vel_std"]       = _roll(g_win, "v_head",       "std").fillna(0)
        feat[f"{p}omega_heading_mean"] = _roll(g_win, "omega_heading","mean")

        valid_cnt = _roll(g_win, "v_com", "count")
        feat[f"{p}coverage"]        = (valid_cnt / w).astype("float32")
        feat[f"{p}bend_peaks_rate"] = (
            _roll(g_win, "is_peak", "sum") / (valid_cnt + 1e-6)
        ).astype("float32")
        feat[f"{p}pause_run_frac"] = (
            _roll(g_win, "has_neighbor", "mean")
            / (_roll(g_win, "is_paused", "mean") + 1e-6)
        ).astype("float32")
        feat[f"{p}reversal_rate"] = _roll(g_win, "high_bend_activity", "mean")

        # Angular tortuosity
        total_ang_path = _roll(g_win, "omega_body", "sum") / fps
        first_ang = g_inst["angle_body_unwrapped"].shift( half_w).fillna(df["angle_body_unwrapped"])
        last_ang  = g_inst["angle_body_unwrapped"].shift(-half_w).fillna(df["angle_body_unwrapped"])
        feat[f"{p}angular_tortuosity"] = (
            total_ang_path / (np.abs(last_ang - first_ang) + 0.01)
        ).astype("float32")

        # Revisitation — uses a temporary column on df, always cleaned up
        _tmp = f"__rev_{s}"
        df[_tmp] = (
            df.groupby(["source", "ID"], group_keys=False)
              .apply(lambda g: _revisit_series(g, w))
        )
        # g_inst holds a reference to df, so the new column is visible here
        feat[f"{p}revisitation_mean"] = (
            g_inst[_tmp]
              .rolling(window=w, min_periods=1)   # causal: revisit is already a lookback metric
              .mean()
              .reset_index(level=[0, 1], drop=True)
              .astype("float32")
        )
        df.drop(columns=[_tmp], inplace=True)

        # Slope features (finite-difference over a 1/6-window offset)
        tg = feat.groupby(groups_ser)
        _slope_pairs = [
            (f"{p}revisitation_mean", f"{p}revis_slope_smooth"),
            (f"{p}rog",               f"{p}rog_slope_smooth"),
            (f"{p}tortuosity",        f"{p}tort_slope_smooth"),
        ]
        # omega slope only for s >= 50 (matches original feature set)
        if s >= 50:
            _slope_pairs.append((f"{p}omega_body_mean", f"{p}omega_body_mean_slope_smooth"))

        for src_col, dst_col in _slope_pairs:
            feat[dst_col] = (
                tg[src_col].shift(-shift_len).ffill()
                - tg[src_col].shift( shift_len).bfill()
            ).astype("float32")

    # ── s >= 50 extras ────────────────────────────────────────────────────────
    if s >= 50:
        feat[f"{p}bending_std"]  = _roll(g_win, "bending", "std").fillna(0)
        feat[f"{p}hc_ratio_mean"]= _roll(g_win, "hc_ratio", "mean")
        feat[f"{p}ht_ratio_mean"]= _roll(g_win, "ht_ratio", "mean")
        tg = feat.groupby(groups_ser)
        feat[f"{p}vel_lag"]  = tg[f"{p}vel_mean"].shift( shift_len).fillna(0).astype("float32")
        feat[f"{p}vel_lead"] = tg[f"{p}vel_mean"].shift(-shift_len).fillna(0).astype("float32")

    # ── s == 75 only ──────────────────────────────────────────────────────────
    if s == 75:
        # transform() returns a flat Series already aligned to df.index —
        # do NOT call reset_index() on it (only rolling aggs need that).
        feat[f"{p}bend_freq_rolling"] = (
            g_inst["bending"]
              .transform(lambda x: get_windowed_freq(x.values, fps, s))
              .astype("float32")
        )

    return feat.astype("float32")


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def prepare_ml_dataset(
    context,
    windows: List[int] = None,
    fps: float = 6.0,
    id_slice=slice(None),
    file_str: List[str] = None,
    pause_threshold: float = CONFIG["pause_threshold"],
    min_coverage: float = CONFIG["min_coverage"],
):
    if windows is None:
        windows = CONFIG["windows"]
    if file_str is None:
        file_str = []

    df = context.annotated.copy()
    raw_counts = df.groupby("source")["behavior"].value_counts().unstack(fill_value=0)
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")

    valid_behaviors = ["dwelling", "nondwelling"]
    df = df[df["behavior"].isin(valid_behaviors)].copy()

    # ── Source / ID filtering ─────────────────────────────────────────────────
    print("Filtering data...")
    source_mask = (
        df["source"].str.startswith(tuple(file_str))
        if file_str
        else pd.Series(True, index=df.index)
    )

    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")

    if isinstance(id_slice, dict):
        slice_mask = pd.Series(False, index=df.index)
        for src_prefix, s in id_slice.items():
            src_match = df["source"].str.startswith(src_prefix)
            if isinstance(s, slice):
                start = s.start if s.start is not None else df.loc[src_match, "ID"].min()
                stop  = s.stop  if s.stop  is not None else df.loc[src_match, "ID"].max()
                cur   = src_match & df["ID"].between(start, stop)
            else:
                cur   = src_match & (df["ID"] == s)
            slice_mask |= cur
    elif isinstance(id_slice, slice):
        start = id_slice.start if id_slice.start is not None else df["ID"].min()
        stop  = id_slice.stop  if id_slice.stop  is not None else df["ID"].max()
        slice_mask = df["ID"].between(start, stop)
    else:
        slice_mask = df["ID"] == id_slice

    df = df[source_mask & slice_mask].sort_values(["source", "ID", "et"]).reset_index(drop=True)

    # ── Event IDs ─────────────────────────────────────────────────────────────
    time_gaps = df.groupby(["source", "ID"])["et"].diff() > 0.5
    df["event_id"] = (
        (df["behavior"] != df.groupby(["source", "ID"])["behavior"].shift())
        | (df["tags"]    != df.groupby(["source", "ID"])["tags"].shift())
        | time_gaps
    ).cumsum()

    # ── Dwelling / non-dwelling selection ─────────────────────────────────────
    print("Selecting Dwelling frames based on tags...")
    is_valid_dweller = df.apply(
        lambda row: row["behavior"] == "dwelling"
                    and has_target_tag(row["tags"], CONFIG["dwelling_tags"]),
        axis=1,
    )
    dwellers_df = df[is_valid_dweller].copy()
    total_dwelling_frames = len(dwellers_df)

    print("Sampling Non-Dwelling events based on tags and ratios...")
    target_nd_frames = int(total_dwelling_frames * CONFIG["nondwelling_ratio_to_dwelling"])
    ratio_dict = CONFIG["nondwelling_tag_ratios"]
    total_weight = sum(ratio_dict.values())
    nd_frame_targets = {
        tag: int(target_nd_frames * (w / total_weight)) for tag, w in ratio_dict.items()
    }

    nd_df = df[df["behavior"] == "nondwelling"].copy()
    rng = np.random.default_rng(seed=42)
    selected_nd_indices = []

    for tag, target_frames in nd_frame_targets.items():
        tag_mask = nd_df["tags"].apply(lambda x: has_target_tag(x, [tag]))
        available_events = nd_df[tag_mask]["event_id"].unique()
        rng.shuffle(available_events)
        accumulated = 0
        for event in available_events:
            if accumulated >= target_frames:
                break
            idx = nd_df[nd_df["event_id"] == event].index
            selected_nd_indices.extend(idx)
            accumulated += len(idx)
        print(f"  Tag '{tag}': collected {accumulated}/{target_frames} target frames.")

    selected_nd_indices = list(set(selected_nd_indices))
    df_sampled = pd.concat([dwellers_df, nd_df.loc[selected_nd_indices]])

    # ── Padding ───────────────────────────────────────────────────────────────
    print("Padding selected events with unannotated context frames...")
    pad_seconds = CONFIG["max_window_size"] / 2.0
    raw_df = context.long_df.copy()
    padded_chunks = []

    for event_id, event_data in df_sampled.groupby("event_id"):
        src     = event_data["source"].iloc[0]
        trk_id  = event_data["ID"].iloc[0]
        start_et = event_data["et"].min() - pad_seconds
        end_et   = event_data["et"].max() + pad_seconds

        chunk = raw_df[
            (raw_df["source"] == src)
            & (raw_df["ID"]   == trk_id)
            & (raw_df["et"]   >= start_et)
            & (raw_df["et"]   <= end_et)
        ].copy().sort_values("et")

        ann_safe = event_data[["et", "behavior"]].copy().sort_values("et")
        ann_safe["is_target_temp"] = True
        chunk    = chunk.astype({"et": "float32"})
        ann_safe = ann_safe.astype({"et": "float32"})

        chunk = pd.merge_asof(
            chunk, ann_safe, on="et",
            direction="nearest", tolerance=1.0 / fps * 0.6,
        )
        chunk["is_target_annotation"] = chunk["is_target_temp"].fillna(False)
        chunk.drop(columns=["is_target_temp"], inplace=True)
        padded_chunks.append(chunk)

    df = pd.concat(padded_chunks)
    df = df.sort_values(
        ["source", "ID", "et", "is_target_annotation"],
        ascending=[True, True, True, False],
    )
    df = df.drop_duplicates(subset=["source", "ID", "et"], keep="first").reset_index(drop=True)

    # ── Instantaneous base features ───────────────────────────────────────────
    print("Calculating base metrics...")
    g_inst = df.groupby(["source", "ID"])

    df["bending"] = calc_angle_vec(
        df["xspine_0"], df["yspine_0"],
        df["xspine_5"], df["yspine_5"],
        df["xspine_10"], df["yspine_10"],
    )
    df["bending_vel"] = g_inst["bending"].diff() * fps

    df["v_head"] = np.sqrt(g_inst["xspine_0"].diff() ** 2 + g_inst["yspine_0"].diff()  ** 2) * fps
    df["v_mid"]  = np.sqrt(g_inst["xspine_5"].diff() ** 2 + g_inst["yspine_5"].diff()  ** 2) * fps
    df["v_tail"] = np.sqrt(g_inst["xspine_10"].diff()** 2 + g_inst["yspine_10"].diff() ** 2) * fps
    df["v_com"]  = np.sqrt(g_inst["x"].diff() ** 2 + g_inst["y"].diff() ** 2) * fps

    df["is_paused"] = (df["v_com"] < pause_threshold).astype(int)
    df["has_neighbor"] = (df["is_paused"] == 1) & (
        (g_inst["is_paused"].shift( 1) == 1) |
        (g_inst["is_paused"].shift(-1) == 1)
    )

    df["angle_body"]    = np.arctan2(df["yspine_0"] - df["yspine_10"],
                                     df["xspine_0"] - df["xspine_10"]).fillna(0)
    df["angle_head"]    = np.arctan2(df["yspine_0"] - df["yspine_5"],
                                     df["xspine_0"] - df["xspine_5"]).fillna(0)
    dx_com = g_inst["x"].diff()
    dy_com = g_inst["y"].diff()
    df["angle_heading"] = np.arctan2(dy_com, dx_com)

    def get_omega(angles: pd.Series, fps: float) -> np.ndarray:
        clean    = np.nan_to_num(angles.to_numpy(), nan=0.0)
        unwrapped = np.unwrap(clean)
        return np.gradient(unwrapped) * fps

    df["omega_body"]    = g_inst["angle_body"].transform(lambda x: get_omega(x, fps)).abs()
    df["omega_head"]    = g_inst["angle_head"].transform(lambda x: get_omega(x, fps)).abs()
    df["omega_heading"] = g_inst["angle_heading"].transform(lambda x: get_omega(x, fps)).abs()
    df["omega_relative"]= g_inst["bending"].diff().abs() * fps

    # ── NaN fill for all angular/velocity derivatives ─────────────────────────
    # omega_relative is a diff so first frame per larva is NaN — include it here.
    fill_cols = ["omega_body", "omega_head", "omega_heading", "omega_relative"]
    df[fill_cols] = g_inst[fill_cols].ffill().bfill().fillna(0)

    def unwrap_group(x):
        return np.unwrap(np.nan_to_num(x.to_numpy(), nan=0.0))

    df["angle_body_unwrapped"] = g_inst["angle_body"].transform(unwrap_group)

    df["body_len"] = np.sqrt(
        (df["xspine_0"] - df["xspine_10"]) ** 2
        + (df["yspine_0"] - df["yspine_10"]) ** 2
    )
    group_medians = g_inst["body_len"].transform("median")
    df["v_mid_norm"] = df["v_mid"] / (group_medians + 1e-6)

    df["ht_ratio"] = (df["v_head"] + 1e-3) / (df["v_tail"] + 1e-3)
    df["hc_ratio"] = (df["v_head"] + 1e-3) / (df["v_mid"]  + 1e-3)

    df["bending_diff"] = g_inst["bending"].diff().abs() * fps

    # FIX 1: use per-larva median to avoid global data leakage across CV folds
    larva_bend_median = g_inst["bending_diff"].transform("median")
    df["high_bend_activity"] = (df["bending_diff"] > larva_bend_median).astype(int)

    df["is_peak"] = (
        (df["bending"] > g_inst["bending"].shift( 1)) &
        (df["bending"] > g_inst["bending"].shift(-1))
    ).astype(int)

    # ── Windowed features ─────────────────────────────────────────────────────
    print("Calculating windowed features...")
    X_list = []
    for s in windows:
        print(f"  Window {s}s ...")
        # Recreate g_inst each iteration so it reflects any df columns added
        # by the previous iteration (e.g. the temp revisit column if cleanup failed).
        g_inst_w = df.groupby(["source", "ID"])
        win_feat = _compute_window_features(df, s, fps, group_medians, g_inst_w)
        X_list.append(win_feat)
        del g_inst_w
        gc.collect()

    del win_feat
    gc.collect()

    print("Combining features...")
    X = pd.concat(X_list, axis=1)

    y      = (df["behavior"] == "dwelling").astype(int)
    groups = df["source"] + "_" + df["ID"].astype(str)
    X      = X.groupby(groups).transform(lambda x: x.ffill().bfill().fillna(0))

    cov_windows  = [s for s in windows if s >= 25]
    coverage_cols = [f"w{s}_coverage" for s in cov_windows]
    coverage_ok  = (X[coverage_cols] >= min_coverage).all(axis=1)
    valid_mask   = coverage_ok & df["is_target_annotation"]

    print("\n--- PIPELINE DIAGNOSTIC ---")
    print(f"Total rows in padded dataframe  : {len(df)}")
    print(f"Rows flagged as target annotations: {df['is_target_annotation'].sum()}")
    print(f"Rows passing window coverage    : {coverage_ok.sum()}")
    print(f"Rows passing BOTH (fed to model): {valid_mask.sum()}")
    print("---------------------------\n")

    X_final      = X[valid_mask]
    y_final      = y[valid_mask]
    groups_final = groups[valid_mask]
    meta_final   = df.loc[valid_mask, ["source", "ID", "et"]]

    # ── Representation summary ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'SOURCE':<15} | {'RAW POS':<8} | {'RAW NEG':<8} | {'FED POS':<8} | {'FED NEG':<8}")
    print("-" * 65)
    for src in raw_counts.index:
        r_pos = raw_counts.loc[src, "dwelling"] if "dwelling" in raw_counts.columns else 0
        r_neg = raw_counts.loc[src].sum() - r_pos
        src_mask = meta_final["source"] == src
        f_pos = y_final[src_mask].sum()
        f_neg = src_mask.sum() - f_pos
        print(f"{src:<15} | {r_pos:<8} | {r_neg:<8} | {f_pos:<8} | {f_neg:<8}")

    total_raw = raw_counts.values.sum()
    total_fed = len(X_final)
    print("-" * 65)
    print(f"TOTAL FRAMES: Raw Annotated = {total_raw} | Fed to Model = {total_fed}")
    print(f"Retention Rate: {total_fed / total_raw:.1%}" if total_raw > 0 else "")
    print("=" * 65 + "\n")

    gc.collect()
    return X_final, y_final, groups_final, meta_final
