# tp_export.py
#
# Pipeline entry-points: train, infer, predict, assess_performance, plot helpers.
#
# Cache integration (new system)
# ──────────────────────────────
# All feature I/O now goes through feature_store.py.
# train() and infer() accept a `fsc` (FeatureSetConfig) that specifies exactly
# which features/windows the model sees.  The cache is annotation-free and
# split-ignorant; train() merges annotation keys AFTER loading from cache.
#
# The old per-(source, mode) parquet caching system is removed entirely.
# If you have old "features_{src}_train.parquet" files they are simply ignored.

import joblib
import numpy as np
import importlib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import psutil
import gc
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import os
import math
from sklearn.metrics import (
    mean_squared_error, roc_curve, roc_auc_score, precision_recall_curve,
    log_loss, balanced_accuracy_score, brier_score_loss, cohen_kappa_score,
    confusion_matrix, f1_score, fbeta_score, matthews_corrcoef,
    ConfusionMatrixDisplay, DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    average_precision_score
)
from sklearn.calibration import calibration_curve, CalibrationDisplay
from scikitplot.metrics import plot_cumulative_gain, plot_lift_curve

import feature_store as fs
from feature_store import FeatureSetConfig, _parquet_path, _KEY_COLS

from scipy.ndimage import binary_closing, binary_opening, median_filter
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"No model file found at {model_path}")
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


def _fsc_from_path_or_obj(fsc_or_path) -> FeatureSetConfig:
    """Accept a FeatureSetConfig object or a path to a saved .json."""
    if isinstance(fsc_or_path, FeatureSetConfig):
        return fsc_or_path
    return FeatureSetConfig.load(fsc_or_path)

def plot_gini(names, imp, output_dir, num=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    imp = np.asarray(imp)
    if imp.ndim == 1:
        mean = imp
        std = np.zeros_like(imp)
    elif imp.ndim == 2:
        mean = np.mean(imp, axis=0)
        std = np.std(imp, axis=0)
    else:
        raise ValueError(
            f"plot_gini expects a 1D or 2D importance array, got shape {imp.shape}"
        )

    if num is None or num > len(mean):
        num = len(mean)
    if len(names) != len(mean):
        raise ValueError(
            f"names length ({len(names)}) does not match importance length ({len(mean)})"
        )

    order = np.argsort(mean)[::-1][:num]
    fig, ax = plt.subplots(figsize=(10, len(order) * 0.35))
    y_pos = np.arange(len(order))
    ax.barh(
        y_pos,
        mean[order][::-1],
        xerr=std[order][::-1],
        color='steelblue',
        alpha=0.8,
        align='center',
        error_kw=dict(ecolor='black', capsize=3),
    )
    ax.set_yticks(y_pos)
    name = np.asarray(names)
    ax.set_yticklabels(name[order][::-1], fontsize=8)
    ax.set_xlabel("Mean Decrease in Impurity (std across trees)")
    ax.set_title("Feature Importances")
    plt.tight_layout()
    ax.grid(False)
    ppath = output_dir / "feature_importance.png"
    fig.savefig(ppath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Feature Importances plotted → {ppath}")

@dataclass
class PostProcessConfig:
    median_filter: float = 5.166667
    threshold: float = 0.5
    gap_fill_size: float = 5.5
    minimum_dwell_length: float = 4.5
    def predict(self, probs):
        smoothed = median_filter(probs, size=int(self.median_filter * 6))
        preds    = (smoothed > self.threshold).astype(int)
        close_sz = int(self.gap_fill_size * 6)
        open_sz  = int(self.minimum_dwell_length * 6)
        pad      = max(close_sz, open_sz)
        preds_p  = np.pad(preds, pad, mode="edge")
        preds_p  = binary_closing(preds_p, structure=np.ones(close_sz)).astype(int)
        preds_p  = binary_opening(preds_p, structure=np.ones(open_sz)).astype(int)
        return preds_p[pad:-pad]
    def get_IDstr(self):
        return f"MF{self.median_filter}_TH{self.threshold}_GC{self.gap_fill_size}_GO{self.minimum_dwell_length}"


# ─────────────────────────────────────────────────────────────────────────────
# train()
# ─────────────────────────────────────────────────────────────────────────────

def train(
    ctx,
    slices,
    prefixes,
    logic_file,
    model_path,
    feature_path,
    metadata_path,
    seed,
    cache_dir,
    fsc: FeatureSetConfig,
    train_keys=None,
    do_plot_gini=False,
    plot_path=None,
):
    """
    Train a RandomForest classifier.

    Features come exclusively from the annotation-free feature cache.
    Annotation labels are joined AFTER loading from cache using ctx.annotated.

    Parameters
    ----------
    fsc         : FeatureSetConfig declaring which columns to use as model input.
                  Must have been validated against the cache beforehand (or
                  auto-validation is done here).
    train_keys  : optional DataFrame with (source, ID) to restrict training rows
                  to a specific train split.  All other cache rows are dropped.
    """
    cache_dir = Path(cache_dir)

    # Ensure all required columns exist in cache
    missing = fsc.validate_against_cache(
        cache_dir, ctx=ctx, logic_file=logic_file, auto_fill=True
    )
    if missing:
        raise RuntimeError(
            f"Cannot train: {len(missing)} feature columns are missing from "
            f"cache even after auto-fill attempt: {missing}"
        )

    feature_cols = fsc.get_all_columns()
    print(f"\n[train] FeatureSetConfig '{fsc.name}': {len(feature_cols)} features")
    print(f"[train] Loading from cache for sources: {prefixes}")

    # ── Load raw features from cache ──────────────────────────────────────────
    X_list, meta_list = [], []
    for src in prefixes:
        dest = _parquet_path(cache_dir, src)
        if not dest.exists():
            print(f"  WARNING: no cache for '{src}' — skipping")
            continue
        df_src = pd.read_parquet(dest)
        for c in feature_cols:
            if c not in df_src.columns:
                df_src[c] = 0.0
        meta_list.append(df_src[_KEY_COLS])
        X_list.append(df_src[feature_cols].astype("float32"))

    if not X_list:
        raise RuntimeError("No cached features found for any training source.")

    meta = pd.concat(meta_list, ignore_index=True)
    X    = pd.concat(X_list,    ignore_index=True)

    # ── Join annotation labels ────────────────────────────────────────────────
    # restrict ctx.annotated to train-only larvae FIRST, before
    # join.  This means val/test annotations are structurally excluded — not
    # just post-filtered — regardless of what is in ctx.annotated at call time.
    ann = ctx.annotated[["source", "ID", "et", "behavior", "tags"]].copy()
    ann["source"] = ann["source"].astype(str)
    ann["ID"]     = ann["ID"].astype(str)
    ann["et"]     = ann["et"].astype("float32").round(4)

    if train_keys is not None:
        print("[train] Pre-filtering annotation table to train-split larvae …")
        train_keys_copy = train_keys.copy()
        train_keys_copy["source"] = train_keys_copy["source"].astype(str)
        train_keys_copy["ID"]     = train_keys_copy["ID"].astype(str)
        n_ann_before = len(ann)
        ann = ann.merge(train_keys_copy, on=["source", "ID"], how="inner")
        print(f"[train] Annotations restricted: {n_ann_before} → {len(ann)} rows "
              f"({ann[['source','ID']].drop_duplicates().shape[0]} larvae)")

    meta["source"] = meta["source"].astype(str)
    meta["ID"]     = meta["ID"].astype(str)
    meta["et"]     = meta["et"].astype("float32").round(4)

    combined = meta.join(X)
    combined = combined.merge(ann, on=["source", "ID", "et"], how="left")

    # Keep only annotated dwelling/nondwelling frames
    labeled_mask = combined["behavior"].isin(["dwelling", "nondwelling"])
    combined = combined[labeled_mask].reset_index(drop=True)

    if combined.empty:
        raise RuntimeError("No labeled rows remain after annotation join + split filter.")

    y      = (combined["behavior"] == "dwelling").astype(np.int32)
    groups = combined["ID"].values
    mod_meta = combined[["source", "ID", "et"]].copy()
    mod_meta["true_behavior"] = y.values
    X_train = combined[feature_cols].astype("float32")

    print(f"[train] Training on {len(X_train):,} rows "
          f"({y.sum()} dwelling / {(1-y).sum()} nondwelling)")

    # ── Fit model ─────────────────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        max_features=0.3,
        min_samples_leaf=100,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
        oob_score=True,
    )
    model.fit(X_train.values, y.values)
    
    if do_plot_gini:
        plot_dir  = Path(plot_path)
        plot_dir.mkdir(parents=True, exist_ok=True)
        fis = model.feature_importances_
        plot_gini(feature_cols,fis,plot_path)
    
    
    # ── Persist ───────────────────────────────────────────────────────────────
    joblib.dump(model,  model_path);   print(f"Model saved      → {model_path}")
    joblib.dump(fsc,    feature_path); print(f"FSC saved        → {feature_path}")
    mod_meta.to_pickle(metadata_path); print(f"Metadata saved   → {metadata_path}")

    # Save FSC as human-readable JSON next to the model
    fsc_json = Path(model_path).parent / "feature_set_config.json"
    fsc.save(fsc_json)
    print(f"FSC (JSON) saved → {fsc_json}")

    return model, feature_cols, mod_meta, [f"Trained on {len(X_train):,} rows"]


# ─────────────────────────────────────────────────────────────────────────────
# infer()
# ─────────────────────────────────────────────────────────────────────────────

def infer(
    model_path,
    ctx,
    files,
    feature_path,
    probabilities_path,
    metadata_test_path,
    cache_dir,
    logic_file,
    fsc: FeatureSetConfig = None,
):
    """
    Run model inference over `files`.

    Features are loaded from the annotation-free feature cache.
    Ground-truth labels from ctx.annotated are joined for later evaluation.

    Parameters
    ----------
    fsc : FeatureSetConfig — if None, loads the saved FSC from feature_path.
    """
    cache_dir = Path(cache_dir)
    model     = load_model(model_path)

    # Load the FSC that was used at training time
    if fsc is None:
        fsc = joblib.load(feature_path)
        if not isinstance(fsc, FeatureSetConfig):
            # Legacy: feature_path stored a plain list of column names
            cols = fsc
            fsc  = FeatureSetConfig(
                name="legacy",
                base_features=[c for c in cols if not c.startswith("w")],
                windowed_features={},
            )
            # reconstruct windowed_features from column names
            for c in cols:
                if c.startswith("w") and "_" in c:
                    try:
                        w    = int(c.split("_")[0][1:])
                        feat = "_".join(c.split("_")[1:])
                        fsc.windowed_features.setdefault(feat, [])
                        fsc.windowed_features[feat].append(w)
                    except ValueError:
                        pass

    feature_cols = fsc.get_all_columns()
    print(f"\n[infer] FeatureSetConfig '{fsc.name}': {len(feature_cols)} features")

    # Validate cache for all inference files
    missing = fsc.validate_against_cache(
        cache_dir, ctx=ctx, logic_file=logic_file, auto_fill=True
    )
    if missing:
        raise RuntimeError(
            f"Cannot infer: {len(missing)} feature columns missing from cache "
            "even after auto-fill attempt."
        )

    probabilities_path = Path(probabilities_path)
    if probabilities_path.exists():
        probabilities_path.unlink()
        print(f"Cleared old probabilities file at {probabilities_path}")

    meta_test_list = []
    log_messages   = []

    for file_id in files:
        print(f"\n=== Processing file_id={file_id} ===")
        dest = _parquet_path(cache_dir, file_id)
        if not dest.exists():
            print(f"  WARNING: no cache parquet for '{file_id}' — skipping.")
            continue

        print(f"  Loading features from cache …")
        df_src = pd.read_parquet(dest)
        for c in feature_cols:
            if c not in df_src.columns:
                df_src[c] = 0.0

        meta = df_src[_KEY_COLS].copy()
        X_inf = df_src[feature_cols].astype("float32")
        print(f"  Source: {file_id}  |  Rows: {len(X_inf):,}")
        print(f"  RAM free: {psutil.virtual_memory().available/1e9:.1f} GB")

        probs = model.predict_proba(X_inf.values)[:, 1]
        res   = meta.copy()
        res["prob"] = probs

        del X_inf; gc.collect()

        # ── Join ground truth from ctx.annotated ──────────────────────────────
        cols_to_extract = (
            ["source", "ID", "et", "behavior", "tags"]
            if "tags" in ctx.annotated.columns
            else ["source", "ID", "et", "behavior"]
        )
        ann_gt = ctx.annotated[ctx.annotated["source"] == file_id][cols_to_extract].copy()
        if "tags" not in ann_gt.columns:
            ann_gt["tags"] = np.nan

        if not ann_gt.empty:
            ann_gt = ann_gt[ann_gt["behavior"].isin(["dwelling", "nondwelling"])].copy()
            for col in ["ID"]:
                res[col]    = res[col].astype(str)
                ann_gt[col] = ann_gt[col].astype(str)
            res["et"]    = res["et"].astype("float64").round(4)
            ann_gt["et"] = ann_gt["et"].astype("float64").round(4)

            res = res.merge(
                ann_gt[["source", "ID", "et", "behavior", "tags"]],
                on=["source", "ID", "et"], how="left"
            )
            res["true_label"] = np.nan
            won_mask = (
                (res["behavior"] == "dwelling")
                & res["tags"].astype(str).str.contains("wonderful", na=False, case=False)
            )
            res.loc[won_mask, "true_label"] = 1
            res.loc[(res["behavior"] == "nondwelling"), "true_label"] = 0

            print(f"  Matched {res['behavior'].notna().sum()} annotated frames.")
        else:
            res["behavior"] = np.nan
            res["tags"]     = np.nan

        # Append to streaming CSV
        res.to_csv(
            probabilities_path, mode="a",
            header=not probabilities_path.exists(), index=False
        )
        print(f"  Probabilities appended → {probabilities_path}")
        meta_test_list.append(res.copy())

        del df_src, res, meta, probs
        gc.collect()
        print(f"  RAM free after cleanup: {psutil.virtual_memory().available/1e9:.1f} GB")

    meta_test = (
        pd.concat(meta_test_list, ignore_index=True) if meta_test_list else None
    )
    if meta_test is not None:
        meta_test.to_pickle(metadata_test_path)
        print(f"\nTest metadata saved → {metadata_test_path}")

    log_messages.append(f"Inferred over {len(files)} sources using FSC '{fsc.name}'")
    return meta_test, log_messages


# ─────────────────────────────────────────────────────────────────────────────
# predict()
# ─────────────────────────────────────────────────────────────────────────────

def predict(probabilities_path, ppc, predictions_dir, ctx, logic_file, plot=False):
    """Post-process raw probabilities into binary predictions (unchanged logic)."""
    probabilities_path = Path(probabilities_path)
    predictions_dir    = Path(predictions_dir)

    if not probabilities_path.exists():
        raise FileNotFoundError(f"No probabilities file found at {probabilities_path}")

    df_probs = pd.read_csv(probabilities_path)
    df_probs["ID"] = df_probs["ID"].astype(str)
    df_probs = df_probs.set_index(["source", "ID"]).sort_index()

    ppc_id      = ppc.get_IDstr()
    output_path = predictions_dir / f"{ppc_id}_predictions.csv"

    final_preds_list = []
    unique_tracks = df_probs.index.unique()
    sorted_tracks = sorted(
        unique_tracks,
        key=lambda x: (x[0], int(x[1]) if str(x[1]).isdigit() else x[1])
    )

    for source, track_id in sorted_tracks:
        track_id_str = str(track_id)
        if (source, track_id_str) not in df_probs.index:
            continue

        if int(track_id_str) % 1000 == 0:
            print(f"Predicting for {source} track {track_id_str}…")

        track_data  = df_probs.loc[(source, track_id_str)].sort_values("et")
        probs_array = track_data["prob"].values
        binary_preds = ppc.predict(probs_array)

        track_preds_df = pd.DataFrame({
            "source":     source,
            "ID":         track_id_str,
            "et":         track_data["et"].values,
            "prob":       probs_array,
            "prediction": binary_preds,
        })
        if "behavior" in track_data.columns:
            track_preds_df["behavior"] = track_data["behavior"].values
        if "tags" in track_data.columns:
            track_preds_df["tags"] = track_data["tags"].values
        final_preds_list.append(track_preds_df)

    if final_preds_list:
        output_df = pd.concat(final_preds_list, ignore_index=True)
        output_df.to_csv(output_path, index=False)
        print(f"Saved post-processed predictions → {output_path}")

        if plot:
            for src in output_df["source"].unique():
                plot_source_grid(output_df, src, predictions_dir, ppc, cols=10)
        return output_df
    else:
        print("⚠ WARNING: No predictions generated.")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# plot_source_grid()  — unchanged from original
# ─────────────────────────────────────────────────────────────────────────────

def plot_source_grid(results_path, src, out_dir, ppc, cols=10,rows=None):
    """One figure per source — all larvae as a grid of small prob traces."""
    ppc_id    = ppc.get_IDstr()
    pred_path = results_path / f"{ppc_id}_predictions.csv"
    results   = pd.read_csv(pred_path).copy()

    plt.style.use("default")
    grp_src = results[results["source"] == src]
    larvae  = sorted(grp_src["ID"].unique(), key=lambda x: int(x) if str(x).isdigit() else x)

    if len(larvae) == 0:
        print(f"  No tracks found for source {src}. Skipping plot.")
        return

    r = 0
    if rows == None:
        r = int(np.ceil(len(larvae) / cols))
    else: 
        r = rows
        
    fig, axes = plt.subplots(r, cols, figsize=(cols * 4, rows * 1.8), facecolor="#111")

    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = np.atleast_1d(axes).ravel()

    print(f"  Plotting {r*cols} tracks for source {src} in a {r}×{cols} grid…")
    #num = len(larvae)
    for i, lid in enumerate(larvae):
        if i >= r*cols: break
        #print(f"\rProgress: |{i+1}/{num}|", end="")
        ax  = axes[i]
        grp = grp_src[grp_src["ID"] == lid].sort_values("et")
        et, prob, pred = grp["et"].values, grp["prob"].values, grp["prediction"].values

        ax.fill_between(et, pred, alpha=0.45, color="#00FFFF", step="post")
        ax.plot(et, prob, color="#BBF1FF", linewidth=0.8)

        if np.any(pred):
            diffs  = np.diff(np.concatenate(([0], pred, [0])))
            starts = np.where(diffs ==  1)[0]
            ends   = np.where(diffs == -1)[0] - 1
            dark_outline = [pe.withStroke(linewidth=1, foreground="#111111")]

            for s_idx, e_idx in zip(starts, ends):
                if s_idx < len(et) and e_idx < len(et):
                    s_time = et[s_idx]; e_time = et[e_idx]; dur = e_time - s_time
                    if dur > 0:
                        ax.text(s_time, 0.75, f"{s_time:.1f}", color="#cffafe",
                                fontsize=5, ha="right", va="center", path_effects=dark_outline)
                        ax.text(e_time, 0.35, f"{e_time:.1f}", color="#cffafe",
                                fontsize=5, ha="right", va="center", path_effects=dark_outline)
                        ax.text(s_time + dur/2, 0.05, f"{dur:.1f}s", color="#082a2f",
                                fontsize=5, ha="center", va="bottom")

        if "behavior" in grp.columns:
            won_mask    = ((grp["behavior"] == "dwelling") &
                           grp["tags"].astype(str).str.contains("wonderful", na=False, case=False)).astype(int)
            unsure_mask = ((grp["behavior"] == "dwelling") &
                           grp["tags"].astype(str).str.contains("unsure",   na=False, case=False)).astype(int)
            alright_mask= ((grp["behavior"] == "dwelling") &
                           grp["tags"].astype(str).str.contains("alright",  na=False, case=False)).astype(int)
            nd_mask     = (grp["behavior"] == "nondwelling").astype(int)

            if won_mask.any():
                ax.fill_between(et, won_mask    * -0.25, 0, alpha=0.6, color="#22c55e", step="post")
            if unsure_mask.any():
                ax.fill_between(et, unsure_mask * -0.25, 0, alpha=0.6, color="#ef4444", step="post")
            if alright_mask.any():
                ax.fill_between(et, alright_mask* -0.25, 0, alpha=0.6, color="#eab308", step="post")
            if nd_mask.any():
                ax.fill_between(et, nd_mask     * -0.10, 0, alpha=0.4, color="#9ca3af", step="post")

        ax.set_ylim(-0.3, 1.05)
        ax.set_title(f"ID {lid}", fontsize=7, color="#aaa", pad=2)
        if len(et) > 0:
            ax.text(0.01, 0.05, f"{et[0]:.1f}s",  transform=ax.transAxes, fontsize=5, color="#666")
            ax.text(0.99, 0.05, f"{et[-1]:.1f}s", transform=ax.transAxes, fontsize=5, color="#666",
                    ha="right")
        ax.set_facecolor("#111")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes[len(larvae):]:
        ax.set_visible(False)

    fig.suptitle(src, color="#ccc", fontsize=11)
    fig.tight_layout(pad=0.3)
    path = Path(out_dir) / f"grid_{src}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close(fig)
    print(f"\n  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# assess_performance()  — logic unchanged, signature unchanged
# ─────────────────────────────────────────────────────────────────────────────

def assess_performance(
    preds_dir, val_dir, ppc, prefixes, sources, report_path, plots,
    dwell_tags=("wonderful",), pred_assess=True, prob_assess=True,
    rf_assess=False, descriptive=True,
    val_keys=None,       # DataFrame (source, ID) — validation larvae only
    train_keys=None,     # DataFrame (source, ID) — used for contamination check
):
    """
    Assess model performance on validation predictions.

    Parameters
    ----------
    val_keys   : (source, ID) DataFrame of validation larvae.  When provided,
                 predictions are filtered to ONLY these larvae before any metric
                 is computed, so stray rows from other splits cannot pollute the
                 evaluation.
    train_keys : (source, ID) DataFrame of training larvae.  When provided,
                 a contamination check is run: any prediction row whose larva
                 appears in train_keys raises a loud warning (or error).
    """
    ppc_id   = ppc.get_IDstr()
    preds_dir = Path(preds_dir)
    val_dir   = Path(val_dir)
    plot_dir  = Path(plots) / ppc_id
    plot_dir.mkdir(parents=True, exist_ok=True)

    pred_path = preds_dir / f"{ppc_id}_predictions.csv"
    results   = pd.read_csv(pred_path).copy()
    results["source"] = results["source"].astype(str)
    results["ID"]     = results["ID"].astype(str)

    # ── Restrict to val_keys if provided ─────────────────────────────────────
    if val_keys is not None:
        vk = val_keys.copy()
        vk["source"] = vk["source"].astype(str)
        vk["ID"]     = vk["ID"].astype(str)
        n_before = len(results)
        results = results.merge(vk, on=["source", "ID"], how="inner")
        n_larvae_kept = results[["source", "ID"]].drop_duplicates().shape[0]
        print(
            f"[assess_performance] Val-key filter: {n_before} → {len(results)} rows "
            f"({n_larvae_kept} validation larvae)"
        )
        if results.empty:
            raise RuntimeError(
                "[assess_performance] No rows remain after filtering to val_keys. "
                "Check that infer() was run on the correct sources."
            )
    else:
        print(
            "[assess_performance] WARNING: val_keys not provided — evaluating on "
            "ALL rows in predictions file. Pass val_keys= to enforce val-only evaluation."
        )
        
    # ── Contamination guard ───────────────────────────────────────────────────
    # Check for train larvae appearing in the predictions before any filtering.
    if train_keys is not None:
        tk = train_keys.copy()
        tk["source"] = tk["source"].astype(str)
        tk["ID"]     = tk["ID"].astype(str)
        tk["_is_train"] = True
        check = results[["source", "ID"]].drop_duplicates().merge(
            tk, on=["source", "ID"], how="left"
        )
        n_leaked = check["_is_train"].sum()
        if n_leaked > 0:
            leaked = check[check["_is_train"] == True][["source", "ID"]].values.tolist()
            raise RuntimeError(
                f"[assess_performance] DATA LEAK DETECTED: {n_leaked} larvae "
                f"from the TRAINING set appear in the EVALUATION set.\n"
                f"Leaking larvae: {leaked}\n"
                "This means your val_keys explicitly contain training data, or you evaluated on everything without passing val_keys."
            )
        print(f"[assess_performance] ✓ Contamination check passed — no train larvae in evaluation subset.")
        
    et     = results["et"].values
    probs  = results["prob"].values
    preds  = results["prediction"].values
    beh    = results["behavior"].values
    tags   = results["tags"].values

    wonderful  = ((results["behavior"] == "dwelling") &
                   results["tags"].astype(str).str.contains("wonderful", na=False, case=False)).astype(int)
    nondwelling= (results["behavior"] == "nondwelling").astype(int)

    def get_confusion_data(preds, nd, beh, tags, dtags):
        dwell     = (beh == "dwelling").astype(int)
        tag_match = np.isin(tags, dtags)
        dwell     = np.where(tag_match, dwell, 0)
        tps = (dwell == 1) & (preds == 1)
        fns = (dwell == 1) & (preds == 0)
        fps = (nd    == 1) & (preds == 1)
        tns = (nd    == 1) & (preds == 0)
        return tps.sum(), tns.sum(), fps.sum(), fns.sum(), dwell.sum(), nd.sum()
    
    def event_confusion_matrix(y_true, y_pred, meta_df, overlap_threshold=0.5):
        """
        Computes event-based TP, FP, FN by aggregating overlapping predictions
        per GT interval to avoid falsely punishing fragmented predictions as FPs.
        """
        from scipy.ndimage import label
        import numpy as np
        
        meta_df = meta_df.reset_index(drop=True)
        groups = (meta_df['source'] + "_" + meta_df['ID'].astype(str)).values
        
        tp = 0
        fn = 0
        fp = 0
        
        for g in np.unique(groups):
            mask = groups == g
            g_et = meta_df.loc[mask, 'et'].values
            gap_locs = np.where(np.diff(g_et) > 0.5)[0] + 1
            
            y_true_seg_list = np.split(y_true[mask], gap_locs)
            y_pred_seg_list = np.split(y_pred[mask], gap_locs)
            
            for yt_seg, yp_seg in zip(y_true_seg_list, y_pred_seg_list):
                if len(yt_seg) == 0:
                    continue
                    
                gt_labels, num_gt = label(yt_seg)
                pred_labels, num_pred = label(yp_seg)
                
                # Keep track of which predicted labels are 'whitelisted' as useful
                whitelisted_preds = set()
                # Keep track of predicted labels that touched failed GT intervals
                tainted_preds = set()
                
                # Evaluate Ground Truth Events first
                for gt_idx in range(1, num_gt + 1):
                    gt_mask = (gt_labels == gt_idx)
                    gt_len = np.sum(gt_mask)
                    
                    # Identify all predicted events overlapping this specific GT interval
                    overlapping_preds = np.unique(pred_labels[gt_mask])
                    overlapping_preds = overlapping_preds[overlapping_preds != 0]
                    
                    # Calculate aggregated overlap frames from ALL intersecting predictions
                    total_overlap_frames = 0
                    for p_idx in overlapping_preds:
                        pred_mask = (pred_labels == p_idx)
                        total_overlap_frames += np.sum(gt_mask & pred_mask)
                    
                    aggregate_overlap_pct = total_overlap_frames / gt_len
                    
                    if aggregate_overlap_pct >= overlap_threshold:
                        tp += 1
                        # Whitelist EVERY prediction interval that contributed to this success
                        for p_idx in overlapping_preds:
                            whitelisted_preds.add(p_idx)
                    else:
                        fn += 1
                        # Mark these predictions as having touched a failed GT event
                
                # Evaluate Predicted Events for False Positives
                for p_idx in range(1, num_pred + 1):
                    # An interval is a False Positive if it wasn't part of any successful TP
                    if p_idx not in whitelisted_preds:
                        fp += 1
        return tp, fp, fn

    def expected_calibration_error(y_true, y_prob, n_bins=10):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins,
                                                  strategy="uniform")
        bin_edges      = np.linspace(0, 1, n_bins + 1)
        bin_assignments= np.clip(np.digitize(y_prob, bin_edges) - 1, 0, n_bins - 1)
        ece = 0.0
        n   = len(y_true)
        for i in range(n_bins):
            mask = bin_assignments == i
            sz   = mask.sum()
            if sz > 0:
                ece += (sz / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
        return ece

    dwell_mask = (beh == "dwelling") & np.isin(tags, list(dwell_tags))
    nd_mask    = (beh == "nondwelling")
    gt         = np.zeros_like(beh, dtype=int)
    gt[dwell_mask] = 1
    eval_mask  = dwell_mask | nd_mask

    gt_eval    = gt[eval_mask]
    probs_eval = probs[eval_mask]
    preds_eval = preds[eval_mask]

    report_lines = [
        f"\n{'='*50}\nModel Assessment Report: {ppc_id}\n{'='*50}"
    ]

    if pred_assess:
        tp, tn, fp, fn, p, n = get_confusion_data(preds, nondwelling, beh, tags, list(dwell_tags))
        p = max(p, 1); n = max(n, 1)
        tpr = tp / p; tnr = tn / n; fnr = fn / p; fpr = fp / n
        informedness = tpr + tnr - 1
        plr = tpr / fpr if fpr > 0 else np.nan
        nlr = fnr / tnr if tnr > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        acc = (tp + tn) / (p + n)
        ba  = (tpr + tnr) / 2
        f1  = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
        fm  = math.sqrt(ppv * tpr)
        mcc_denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        mcc = ((tp*tn) - (fp*fn)) / mcc_denom if mcc_denom > 0 else 0
        dor = plr / nlr if (nlr and nlr > 0 and not np.isnan(plr)) else np.nan
        csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
        ck  = cohen_kappa_score(gt, preds)
        fbeta = fbeta_score(gt, preds, beta=1.0)
        markedness = ppv + npv - 1
        prev = p / (p + n)
        fdr = 1 - ppv

        report_lines.extend([
            "--- PREDICTION METRICS ---",
            f"Accuracy:              {acc:.4f}          | the percentage of predictions a model gets right across all classifications", 
            f"Balanced Accuracy (BA):{ba:.4f}           | the average of recall (accuracy) obtained from each individual class, primarily used to evaluate", 
             "                                          | classification models on imbalanced datasets where one class vastly outnumbers the others",
            f"Sensitivity (TPR):     {tpr:.4f}          | measures a model’s ability to correctly identify all actual positive cases. Also known as Recall",             
             "                                          | or the True Positive Rate, it answers the question: “Out of all the truly positive instances in the ",
             "                                          | data, how many did the model manage to find?",
            f"Specificity (TNR):     {tnr:.4f}          | (or True Negative Rate) measures a model's ability to correctly identify actual negative cases. It shows ",
             "                                          | the proportion of negatives the model avoids flagging as positive (false alarms)"
            f"Precision (PPV):       {ppv:.4f}          | When the model predicts a positive class, how often is it actually correct?",
            f"Negative Pred Val (NPV):{npv:.4f}         | the probability that a prediction of 'Negative' is actually correct",
            f"F1 Score:              {f1:.4f}           | the harmonic mean of precision and recall, stays closer to the lower of the two numbers, so must be good at both",
            f"F-Beta Score (b=1):    {fbeta:.4f}        | ",
            f"Matthews Corrcoef:     {mcc:.4f}          | the Pearson correlation coefficient between your model’s predictions and the actual true labels, widely ",
             "                                          | considered one of the most reliable and truthful single-number metrics in machine learning",
            f"Cohen's Kappa:         {ck:.4f}           | how closely your classifier’s predictions match the actual ground truth, discounting the accuracy you would ",
             "                                          | expect by pure chance"
            f"Informedness:          {informedness:.4f} | -1 to 1; the probability that a model makes a reliable, informed decision rather than simply guessing based on chance (0)",
            f"Markedness:            {markedness:.4f}   | -1 to 1; When the model predicts a specific class, how often is it actually right? ",
            f"Diagnostic Odds Ratio: {dor:.4f}          | compares the model's ability to identify true signals against its susceptibility to false alarms",
            f"Critical Success Index:{csi:.4f}          | evaluates a machine learning model's predictive accuracy for rare events; calculates the proportion of ",
             "                                          | correctly predicted positive events out of all total actual events and incorrectly predicted events, ",
             "                                          | intentionally ignoring the massive number of True Negatives to avoid skewed results"
            f"Prevalence:            {prev:.4f}         | the proportion of dwelling within the total dataset",
            f"FPR: {fpr:.4f}",
            f"FNR:  {fnr:.4f}"
            f"FM Index: {fm:.4f}"
            f"FDR: {fdr:4f}",
            "",
        ])

    if prob_assess:
        brier = brier_score_loss(gt_eval, probs_eval)
        auroc = roc_auc_score(gt_eval, probs_eval)
        ap    = average_precision_score(gt_eval, probs_eval)
        ece   = expected_calibration_error(gt_eval, probs_eval)

        report_lines.extend([
            "--- PROBABILITY METRICS ---",
            f"AUROC:                  {auroc:.4f}",
            f"Average Precision (AP): {ap:.4f}",
            f"Brier Score:            {brier:.4f}",
            f"Expected Calib Error:   {ece:.4f}",
            "",
        ])

        plt.style.use("seaborn-v0_8-whitegrid")

        fpr_c, tpr_c, _ = roc_curve(gt_eval, probs_eval)
        fig, ax = plt.subplots(figsize=(6, 6))
        RocCurveDisplay(fpr=fpr_c, tpr=tpr_c, roc_auc=auroc).plot(ax=ax)
        ax.set_title(f"ROC Curve - {ppc_id}")
        fig.savefig(plot_dir / "ROC_Curve.png", bbox_inches="tight", dpi=300); plt.close(fig)

        prec, rec, _ = precision_recall_curve(gt_eval, probs_eval)
        fig, ax = plt.subplots(figsize=(6, 6))
        PrecisionRecallDisplay(precision=prec, recall=rec, average_precision=ap).plot(ax=ax)
        ax.set_title(f"Precision-Recall Curve - {ppc_id}")
        fig.savefig(plot_dir / "PR_Curve.png", bbox_inches="tight", dpi=300); plt.close(fig)

        prob_true_c, prob_pred_c = calibration_curve(gt_eval, probs_eval)
        fig, ax = plt.subplots(figsize=(6, 6))
        CalibrationDisplay(prob_true_c, prob_pred_c, probs_eval).plot(ax=ax)
        ax.set_title(f"Calibration Curve - {ppc_id}")
        fig.savefig(plot_dir / "Calibration.png", bbox_inches="tight", dpi=300); plt.close(fig)

        cm = confusion_matrix(gt_eval, preds_eval)
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues")
        ax.grid(False)
        ax.set_title(f"Confusion Matrix - {ppc_id}")
        plt.text(0.8, -0.12,
            f"TPR/Sensitivity = {tpr:.3f}\n"
            f"TNR/Specificity = {tnr:.3f}\n"
            f"FPR = {fpr:.3f}",
            transform=plt.gca().transAxes, ha="left", va="bottom", color = "#000000", fontsize=7,
        )
        fig.savefig(plot_dir / "Confusion_Matrix.png", bbox_inches="tight", dpi=300); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 6))
        DetCurveDisplay.from_predictions(gt_eval, probs_eval, ax=ax)
        ax.set_title(f"DET Curve - {ppc_id}")
        fig.savefig(plot_dir / "DET_Curve.png", bbox_inches="tight", dpi=300); plt.close(fig)

        probs_2d = np.vstack([1 - probs_eval, probs_eval]).T
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_cumulative_gain(gt_eval, probs_2d, ax=ax)
        fig.savefig(plot_dir / "Cumulative_Gain.png", bbox_inches="tight", dpi=300); plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        plot_lift_curve(gt_eval, probs_2d, ax=ax)
        fig.savefig(plot_dir / "Lift_Curve.png", bbox_inches="tight", dpi=300); plt.close(fig)
        
        fig, ax = plt.subplots(figsize = (7,6))
        meta_eval = results.loc[eval_mask, ['source', 'ID', 'et']].reset_index(drop=True)
        tp_ev, fp_ev, fn_ev = event_confusion_matrix(gt_eval, preds_eval, meta_eval, overlap_threshold=0.3)
        cm_events = np.array([[0, fp_ev], 
                            [fn_ev, tp_ev]])
        disp_e = ConfusionMatrixDisplay(cm_events,display_labels=["No Event", "Event"])
        ax.grid(False)
        disp_e.plot(ax=ax, colorbar=True, cmap="Purples")
        ax.set_title(f"Event Confusion Matrix - {ppc_id}")
        fig.savefig(plot_dir / f"Event_Confusion_Matrix.png",bbox_inches="tight",dpi=300)

    with open(report_path, "a") as f:
        f.write("\n".join(report_lines) + "\n")

    return {"f1": f1,
            "precision": ppv,
            "recall/sensitivity": tpr}
