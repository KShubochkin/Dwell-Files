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
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.inspection import permutation_importance
from dataclasses import dataclass

import pyarrow.parquet as pq

import numpy as np
from scipy.ndimage import label, binary_dilation

import shap

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
    ranked_features = list(np.asarray(names)[order])
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
    return ranked_features

def plot_shap(model, X_df, output_dir, subsample_size=10000):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[SHAP] Calculating SHAP values (subsampling {subsample_size} rows for speed)...")
    
    if len(X_df) > subsample_size:
        X_sample = X_df.sample(n=subsample_size, random_state=42)
    else:
        X_sample = X_df

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Scikit-Learn RF outputs a list of shap arrays [Negative Class, Positive Class]
    if isinstance(shap_values, list):
        shap_vals_dwelling = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_vals_dwelling = shap_values[:, :, 1]
    else:
        shap_vals_dwelling = shap_values

    # 1. Beeswarm Summary Plot (Feature impact distribution)
    fig, ax = plt.subplots(figsize=(70, 6))
    shap.summary_plot(shap_vals_dwelling, X_sample, show=False,max_display = 70)
    plt.title("SHAP Summary (Impact on 'Dwelling')")
    plt.tight_layout()
    summary_path = output_dir / "shap_summary_beeswarm.png"
    fig.savefig(summary_path, bbox_inches="tight", dpi=300, facecolor='white')
    plt.close(fig)

    # 2. Global Bar Plot (The SHAP equivalent of your Gini plot)
    fig, ax = plt.subplots(figsize=(70, 6))
    shap.summary_plot(shap_vals_dwelling, X_sample, plot_type="bar", show=False,max_display = 70)
    plt.title("SHAP Feature Importance (Mean |SHAP|)")
    plt.tight_layout()
    bar_path = output_dir / "shap_importance_bar.png"
    fig.savefig(bar_path, bbox_inches="tight", dpi=300, facecolor='white')
    plt.close(fig)

    print(f"[SHAP] Beeswarm plotted → {summary_path}")
    print(f"[SHAP] Bar plot plotted  → {bar_path}")

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
    do_plot_gini=False,do_plot_shap=False,shap_subsample=10000,
    do_spearman_clustering=False, spearman_corr_threshold=0.8, enact_clustering = False,
    plot_path=None,
    n_est=300,max_dep=16,max_feat=0.3,min_samples=100,min_split=2,
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
        try:
            # Let feature_store handle fragmented reading and stitching
            X_src, meta_src = fs._load_features_for_sources(cache_dir, [src], fsc)
            X_list.append(X_src)
            meta_list.append(meta_src)
        except FileNotFoundError:
            print(f"  WARNING: no cache parquet for '{src}' — skipping.")
            continue

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
    
    
    if do_spearman_clustering:
        print(f"\n[train] Executing Spearman hierarchical clustering (Cutoff |ρ| ≥ {spearman_corr_threshold})...")
        
        # 1. Compute absolute Spearman distance matrix: d(x,y) = 1 - |ρ|
        corr_matrix = X_train.corr(method='spearman').fillna(0).values
        dist_matrix = 1.0 - np.abs(corr_matrix)
        
        # Guard against float64 floating-point errors (e.g. 1e-16 instead of 0.0) crashing SciPy
        dist_matrix = (dist_matrix + dist_matrix.T) / 2.0 
        np.fill_diagonal(dist_matrix, 0.0)
        
        condensed_dist = squareform(dist_matrix, checks=False)
        linkage_matrix = hierarchy.linkage(condensed_dist, method='average')
        
        dist_threshold = 1.0 - spearman_corr_threshold
        cluster_labels = hierarchy.fcluster(linkage_matrix, dist_threshold, criterion='distance')
        
        # 2. Pick the representative feature per cluster most strongly correlated to target `y`
        corrs_with_y = np.abs(X_train.apply(lambda col: spearmanr(col, y)[0]).fillna(0).values)
        
        kept_features = []
        pruning_log = []
        
        for cid in np.unique(cluster_labels):
            c_indices = np.where(cluster_labels == cid)[0]
            if len(c_indices) == 1:
                kept_features.append(feature_cols[c_indices[0]])
            else:
                best_sub_idx = c_indices[np.argmax(corrs_with_y[c_indices])]
                best_feat = feature_cols[best_sub_idx]
                kept_features.append(best_feat)
                
                dropped = [feature_cols[i] for i in c_indices if i != best_sub_idx]
                pruning_log.append((best_feat, dropped))
                
        print(f"[train] Pruned {len(feature_cols) - len(kept_features)} collinear features. {len(kept_features)} survive.")
        
        # 3. Create Dendrogram Plot
        p_dir = Path(plot_path) if plot_path else Path(model_path).parent
        p_dir.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        hierarchy.dendrogram(
            linkage_matrix, labels=feature_cols, ax=ax, leaf_rotation=90, 
            leaf_font_size=8, color_threshold=dist_threshold
        )
        ax.axhline(y=dist_threshold, color='r', linestyle='--', label=f'Cut Threshold (Dist ≤ {dist_threshold:.2f})')
        ax.set_title(f"Spearman Collinearity Dendrogram (Kept {len(kept_features)} of {len(feature_cols)} features)")
        ax.set_ylabel("Distance: 1.0 - |Spearman ρ|")
        ax.legend()
        plt.tight_layout()
        dendro_path = p_dir / "spearman_clustering_dendrogram.png"
        fig.savefig(dendro_path, dpi=300, facecolor='white')
        plt.close(fig)
        
        # 4. Write exhaustive log to the report text file
        report_file = p_dir / "spearman_feature_selection_report.txt"
        with open(report_file, "w") as rf:
            rf.write(f"=== SPEARMAN RANK-ORDER CLUSTERING SELECTION REPORT ===\n")
            rf.write(f"Correlation Cutoff Threshold : |rho| >= {spearman_corr_threshold}  (Distance <= {dist_threshold:.2f})\n")
            rf.write(f"Original Feature Count       : {len(feature_cols)}\n")
            rf.write(f"Surviving Feature Count      : {len(kept_features)}\n")
            rf.write(f"Eliminated Feature Count     : {len(feature_cols) - len(kept_features)}\n\n")
            rf.write("SURVIVING REPRESENTATIVES (Passed to Model):\n")
            rf.write(", ".join(kept_features) + "\n\n")
            if pruning_log:
                rf.write("PRUNING REPLACEMENT MAP (Surviving Representative  <===  [Eliminated Collinear Features]):\n")
                for rep, drops in pruning_log:
                    rf.write(f" • [KEPT] {rep:<25} <=== replaced: {', '.join(drops)}\n")
        
        print(f"[train] Dendrogram saved → {dendro_path}")
        print(f"[train] Text report saved → {report_file}")
        
        if enact_clustering:
            feature_cols = kept_features
            X_train = X_train[feature_cols]
            
            surv_base = [c for c in feature_cols if not c.startswith("w")]
            surv_win = {}
            for c in feature_cols:
                if c.startswith("w") and "_" in c:
                    try:
                        w_val = int(c.split("_")[0][1:])
                        feat_name = "_".join(c.split("_")[1:])
                        surv_win.setdefault(feat_name, []).append(w_val)
                    except ValueError: pass
                    
            fsc = FeatureSetConfig(name=f"{fsc.name}_spearman_pruned", base_features=surv_base, windowed_features=surv_win)

    # ── Fit model ─────────────────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=max_dep,
        max_features=max_feat,
        min_samples_split=min_split,
        min_samples_leaf=min_samples,
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
        ranked_features = plot_gini(feature_cols,fis,plot_path)
        
        with open(plot_path / "feature_ranking.txt", "w") as f:
            for feat in ranked_features:
                f.write(feat + "\n")
                
    if do_plot_shap:
        if plot_path is None:
            print("⚠ WARNING: do_plot_shap is True, but plot_path is None. Skipping SHAP.")
        else:
            plot_shap(model, X_train, plot_path, subsample_size=shap_subsample)
    
    
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
    infer_keys=None,
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
        try:
            # Let feature_store handle the fragmented reading and stitching
            X_inf, meta = fs._load_features_for_sources(cache_dir, [file_id], fsc)
        except FileNotFoundError:
            print(f"  WARNING: no cache parquet for '{file_id}' — skipping.")
            continue
        
        if infer_keys is not None:
            ik = infer_keys.copy()
            ik["source"] = ik["source"].astype(str)
            ik["ID"]     = ik["ID"].astype(str)
            
            meta["source"] = meta["source"].astype(str)
            meta["ID"]     = meta["ID"].astype(str)
            
            # Store the original row index so we know exactly which rows survive
            meta["_row_idx"] = np.arange(len(meta))
            
            # Perform the inner merge on the metadata
            meta = meta.merge(ik, on=["source", "ID"], how="inner")
            
            if meta.empty:
                print(f"  No validation larvae in {file_id}. Skipping.")
                continue
                
            # Slice X_inf using the specific indices that survived the merge
            X_inf = X_inf.iloc[meta["_row_idx"]].reset_index(drop=True)
            
            # Clean up the temporary tracking column
            meta = meta.drop(columns=["_row_idx"])
        
        
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

        del res, meta, probs
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

def event_confusion_matrix(y_true, y_pred, meta_df, overlap_threshold=0.5, safe_zone_size=30):
        """
        Computes event-based TP, FP, FN using a safe-zone (tolerance margin) approach.
        safe_zone_size: Number of frames (y time points) to extend before and after a GT event.
        """
        meta_df = meta_df.reset_index(drop=True)
        groups = (meta_df['source'] + "_" + meta_df['ID'].astype(str)).values
        
        tp = 0
        fn = 0
        fp = 0
        
        tp_mask = np.zeros(len(y_pred), dtype=bool)
        fp_mask = np.zeros(len(y_pred), dtype=bool)
        fn_mask = np.zeros(len(y_pred), dtype=bool)
        safe_frames_global = np.zeros(len(y_pred), dtype=bool)
        
        global_indices = np.arange(len(y_pred))
            
        for g in np.unique(groups):
            mask = groups == g
            g_et = meta_df.loc[mask, 'et'].values
            gap_locs = np.where(np.diff(g_et) > 0.5)[0] + 1
            
            y_true_seg_list = np.split(y_true[mask], gap_locs)
            y_pred_seg_list = np.split(y_pred[mask], gap_locs)
            idx_seg_list = np.split(global_indices[mask], gap_locs)
            
            for yt_seg, yp_seg,idx_seg in zip(y_true_seg_list, y_pred_seg_list,idx_seg_list):
                if len(yt_seg) == 0:
                    continue
                    
                gt_labels, num_gt = label(yt_seg)
                yp_bool = (yp_seg > 0)  # Convert predictions to a simple boolean mask
                
                # ---------------------------------------------------------
                # RULE 1: True Positives & False Negatives
                # "for each GT event: if > %frames are predicted dwelling, +1 TP. else, +1 FN."
                # ---------------------------------------------------------
                for gt_idx in range(1, num_gt + 1):
                    gt_mask = (gt_labels == gt_idx)
                    gt_len = np.sum(gt_mask)
                    
                    # Count how many frames in this GT event were positively predicted
                    predicted_frames_in_gt = np.sum(yp_bool & gt_mask)
                    
                    if (predicted_frames_in_gt / gt_len) > overlap_threshold:
                        tp += 1
                        tp_mask[idx_seg[gt_mask]] = True
                    else:
                        fn += 1
                        fn_mask[idx_seg[gt_mask]] = True
                
                # ---------------------------------------------------------
                # RULE 2: The Safe Zone
                # "for each dwelling interval, there is a safe zone of y time points around it, 
                # also including the gt dwell event itself."
                # ---------------------------------------------------------
                if safe_zone_size > 0:
                    # Dilation expands the 1s in yt_seg by 'safe_zone_size' in both directions
                    structure = np.ones(2 * safe_zone_size + 1, dtype=bool)
                    safe_zone_mask = binary_dilation(yt_seg > 0, structure=structure)
                else:
                    safe_zone_mask = (yt_seg > 0)
                    
                # ---------------------------------------------------------
                # RULE 3: False Positives
                # "for each NONsafe zone area, count the number of INTERVALS of 
                # positively predicted frames. for each, false positive +1."
                # ---------------------------------------------------------
                # Isolate predictions that fall completely outside the safe zones
                nonsafe_preds = yp_bool & (~safe_zone_mask)
                labels_fp, num_fp_intervals = label(nonsafe_preds)
                fp += num_fp_intervals
                fp_mask[idx_seg[nonsafe_preds]] = True
            
                safe_only = safe_zone_mask & (~(yt_seg > 0))
                safe_frames_global[idx_seg[safe_only]] = True
                
        return tp, fp, fn, tp_mask,fp_mask,fn_mask,safe_frames_global

def plot_source_grid(results_path, src, out_dir, ppc, val_keys,cols=10,rows=None,dwell_tags=["wonderful"]):
    """One figure per source — all larvae as a grid of small prob traces."""
    ppc_id    = ppc.get_IDstr()
    pred_path = results_path / f"{ppc_id}_predictions.csv"
    results   = pd.read_csv(pred_path).copy()
    
    results["source"] = results["source"].astype(str)
    results["ID"]     = results["ID"].astype(str)
    
    # preds  = results["prediction"].values
    # beh    = results["behavior"].values
    # tags   = results["tags"].values
    # dwell_mask = (beh == "dwelling") & np.isin(tags, list(dwell_tags))
    # nd_mask    = (beh == "nondwelling")
    # gt         = np.zeros_like(beh, dtype=int)
    # gt[dwell_mask] = 1
    # eval_mask  = dwell_mask | nd_mask
    # gt_eval    = gt[eval_mask]
    # preds_eval = preds[eval_mask]
    # meta_eval = results.loc[eval_mask, ['source', 'ID', 'et']].reset_index(drop=True)
    
    #_,_,_,tp_mask,fn_mask,fp_mask,safe_frames = event_confusion_matrix(gt_eval,preds_eval,meta_eval)

    plt.style.use("default")
    grp_src = results[results["source"] == src]
    larvae  = sorted(grp_src["ID"].unique(), key=lambda x: int(x) if str(x).isdigit() else x)

    if len(larvae) == 0:
        print(f"  No tracks found for source {src}. Skipping plot.")
        return

    r = int(np.ceil(len(larvae) / cols)) if rows is None else rows
        
    fig, axes = plt.subplots(r, cols, figsize=(cols * 4, r * 1.8), facecolor="#111")
    axes = [axes] if isinstance(axes, plt.Axes) else np.atleast_1d(axes).ravel()

    print(f"  Plotting {r*cols} tracks for source {src} in a {r}×{cols} grid…")
    
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
            is_val = False
            if val_keys is not None:
                is_val = ((val_keys["source"].astype(str) == str(src)) & (val_keys["ID"].astype(str) == str(lid))).any()

            # Local Evaluation logic for the grid visual
            safe_zone_size = 30
            if safe_zone_size > 0:
                structure = np.ones(2 * safe_zone_size + 1, dtype=bool)
                safe_zone_mask = binary_dilation(won_mask > 0, structure=structure)
            else:
                safe_zone_mask = (won_mask > 0)
            
            safe_only_mask = safe_zone_mask & (won_mask == 0)

            # 1. Safe Frames (Light blue diagonal hatch bottom 0.3)
            safe_labels, num_safe = label(safe_only_mask)
            for s_idx in range(1, num_safe + 1):
                s_mask = (safe_labels == s_idx)
                s_start, s_end = np.where(s_mask)[0][0], np.where(s_mask)[0][-1]
                if et[s_end] > et[s_start]:
                    ax.add_patch(plt.Rectangle((et[s_start], 0), et[s_end] - et[s_start], 0.3, 
                                               facecolor='none', edgecolor='lightblue', hatch='///', linewidth=0))

            # 2. TP & FN Logic
            gt_labels, num_gt = label(won_mask)
            for gt_idx in range(1, num_gt + 1):
                g_mask = (gt_labels == gt_idx)
                g_start, g_end = np.where(g_mask)[0][0], np.where(g_mask)[0][-1]
                s_time, e_time = et[g_start], et[g_end]
                mid_time = (s_time + e_time) / 2
                
                # Check overlap (TP vs FN)
                overlap = np.sum(pred[g_mask]) / np.sum(g_mask)
                
                if overlap > 0.5: # True Positive
                    ax.text(mid_time, -0.125, "TP", color="darkgreen", fontsize=7, ha="center", va="center", fontweight="bold")
                    ax.add_patch(plt.Rectangle((s_time, 0), e_time - s_time, 1.0, 
                                               fill=False, edgecolor="darkgreen", linestyle="--", linewidth=1))
                else: # False Negative
                    ax.text(mid_time, -0.125, "FN", color="yellow", fontsize=10, ha="center", va="center", fontweight="bold")
                    ax.add_patch(plt.Rectangle((s_time, 0), e_time - s_time, 1.0, 
                                               fill=False, edgecolor="yellow", linestyle="--", linewidth=1))

            # 3. FP Logic
            nonsafe_preds = (pred > 0) & (~safe_zone_mask) & (nd_mask > 0)
            fp_labels, num_fp = label(nonsafe_preds)
            for fp_idx in range(1, num_fp + 1):
                f_mask = (fp_labels == fp_idx)
                f_start, f_end = np.where(f_mask)[0][0], np.where(f_mask)[0][-1]
                s_time, e_time = et[f_start], et[f_end]
                mid_time = (s_time + e_time) / 2
                
                ax.text(mid_time, 0.5, "FP", color="red", fontsize=8, ha="center", va="center", fontweight="bold")
                ax.add_patch(plt.Rectangle((s_time, 0), e_time - s_time, 1.0, 
                                               fill=False, edgecolor="red", linestyle="--", linewidth=1))

            # 4. VAL KEYS Unseen Model Star
            if is_val:
                annotated_mask = grp["behavior"].isin(["dwelling", "nondwelling"])
                ann_labels, num_ann = label(annotated_mask)
                for a_idx in range(1, num_ann + 1):
                    a_mask = (ann_labels == a_idx)
                    start_i = np.where(a_mask)[0][0]
                    # Put a white star in the top left corner of the rectangle
                    ax.plot(et[start_i], -0.025, marker='*', color='white', markersize=5, zorder=10)

        # Axis cleanup
        ax.set_ylim(-0.3, 1.05)
        ax.set_title(f"ID {lid}", fontsize=7, color="#aaa", pad=2)
        if len(et) > 0:
            ax.text(0.01, 0.05, f"{et[0]:.1f}s",  transform=ax.transAxes, fontsize=5, color="#666")
            ax.text(0.99, 0.05, f"{et[-1]:.1f}s", transform=ax.transAxes, fontsize=5, color="#666", ha="right")
        
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

def get_event_info(preds_dir, val_dir, ppc,report_path,dwell_tags=("wonderful",),val_keys=None,train_keys=None,):
    ppc_id   = ppc.get_IDstr()
    preds_dir = Path(preds_dir)
    val_dir   = Path(val_dir)

    pred_path = preds_dir / f"{ppc_id}_predictions.csv"
    results   = pd.read_csv(pred_path).copy()
    results["source"] = results["source"].astype(str)
    results["ID"]     = results["ID"].astype(str)
    
    probs  = results["prob"].values
    preds  = results["prediction"].values
    beh    = results["behavior"].values
    tags   = results["tags"].values
    
    dwell_mask = (beh == "dwelling") & np.isin(tags, list(dwell_tags))
    nd_mask    = (beh == "nondwelling")
    gt         = np.zeros_like(beh, dtype=int)
    gt[dwell_mask] = 1
    eval_mask  = dwell_mask | nd_mask
    
    gt_eval    = gt[eval_mask]
    probs_eval = probs[eval_mask]
    preds_eval = preds[eval_mask]

    report_lines = [("\nEVENT PERFORMANCE INTERVALS - ******** = validation data (unseen!)")]
    
    meta_eval = results.loc[eval_mask, ['source', 'ID', 'et']].reset_index(drop=True)
    tp_ev, fp_ev, fn_ev, tp_mask, fp_mask, fn_mask, safe_frames = event_confusion_matrix(
        gt_eval, preds_eval, meta_eval, overlap_threshold=0.5
    )
    
    report_lines.extend([
                f"TP_ev = {tp_ev}",
                f"FN_ev = {fn_ev}",
                f"FP_ev = {fp_ev}",
                "FALSE POSITIVE EVENTS:",
            ])
    fp_labels, num_fp = label(fp_mask)
    for fp in range(1, num_fp + 1):
        f_mask_bool = (fp_labels == fp)
        fst, fe = np.where(f_mask_bool)[0][0], np.where(f_mask_bool)[0][-1]
        
        # Extract details safely from meta_eval
        src_val = meta_eval.loc[fst, 'source']
        id_val  = meta_eval.loc[fst, 'ID']
        st      = meta_eval.loc[fst, 'et']
        et_val  = meta_eval.loc[fe, 'et']
        
        is_val = False
        if val_keys is not None:
            is_val = ((val_keys["source"].astype(str) == str(src_val)) & (val_keys["ID"].astype(str) == str(id_val))).any()
        
        if is_val:
            report_lines.extend([
                f"{src_val} {id_val}: {st:.1f} - {et_val:.1f}s ********"
            ])
        else:
            report_lines.extend([
                f"{src_val} {id_val}: {st:.1f} - {et_val:.1f}s"
            ])

    report_lines.extend([
        "FALSE NEGATIVE EVENTS:",
    ])
    
    # Label the MASK, not the integer count
    fn_labels, num_fn = label(fn_mask)
    for fn in range(1, num_fn + 1):
        f_mask_bool = (fn_labels == fn)
        fst, fe = np.where(f_mask_bool)[0][0], np.where(f_mask_bool)[0][-1]
        
        # Extract details safely from meta_eval
        src_val = meta_eval.loc[fst, 'source']
        id_val  = meta_eval.loc[fst, 'ID']
        st      = meta_eval.loc[fst, 'et']
        et_val  = meta_eval.loc[fe, 'et']
        
        is_val = False
        if val_keys is not None:
            is_val = ((val_keys["source"].astype(str) == str(src_val)) & (val_keys["ID"].astype(str) == str(id_val))).any()
        if is_val:
            report_lines.extend([
                f"{src_val} {id_val}: {st:.1f} - {et_val:.1f}s ********"
            ])
        else:
            report_lines.extend([
                f"{src_val} {id_val}: {st:.1f} - {et_val:.1f}s"
            ])
            
    with open(report_path, "a") as f:
        f.write("\n".join(report_lines) + "\n")
        
def evaluate_validation_permutation_importance(
    model_path,
    ctx,
    val_prefixes,
    logic_file,
    feature_path,
    cache_dir,
    val_keys,
    report_path,
    plots_dir,
    n_repeats=10,
    seed=42,
):
    """
    Evaluates unbiased Scikit-Learn Permutation Importance strictly over the validation split.
    Outputs weights to report and isolates pure noise (Importance_Mean <= 0).
    """
    cache_dir, plots_dir, report_path = Path(cache_dir), Path(plots_dir), Path(report_path)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    model = load_model(model_path)
    
    # Re-hydrate the FSC saved at training time
    fsc = joblib.load(feature_path)
    if not isinstance(fsc, FeatureSetConfig):
        cols = fsc
        fsc = FeatureSetConfig(name="legacy", base_features=[c for c in cols if not c.startswith("w")], windowed_features={})
        for c in cols:
            if c.startswith("w") and "_" in c:
                try: fsc.windowed_features.setdefault("_".join(c.split("_")[1:]), []).append(int(c.split("_")[0][1:]))
                except ValueError: pass

    feature_cols = fsc.get_all_columns()
    print(f"\n[Permutation Importance] Reconstructing validation set for {len(feature_cols)} features...")
    
    # 1. Gather cached feature data for validation prefixes
    X_list, meta_list = [], []
    for src in val_prefixes:
        try:
            X_src, meta_src = fs._load_features_for_sources(cache_dir, [src], fsc)
            X_list.append(X_src)
            meta_list.append(meta_src)
        except FileNotFoundError: continue
            
    if not X_list:
        raise RuntimeError("No cached features found for requested validation sources.")

    meta = pd.concat(meta_list, ignore_index=True)
    X_raw = pd.concat(X_list, ignore_index=True)

    # 2. Slice strictly to validation larvae
    vk = val_keys.copy()
    vk["source"], vk["ID"] = vk["source"].astype(str), vk["ID"].astype(str)
    meta["source"], meta["ID"] = meta["source"].astype(str), meta["ID"].astype(str)
    meta["et"] = meta["et"].astype("float32").round(4)
    
    meta["_row_idx"] = np.arange(len(meta))
    meta_filtered = meta.merge(vk, on=["source", "ID"], how="inner")
    
    if meta_filtered.empty:
        raise RuntimeError("No validation larvae matched the feature rows.")

    X_val = X_raw.iloc[meta_filtered["_row_idx"]].reset_index(drop=True)
    del X_raw; gc.collect()
    
    meta_filtered = meta_filtered.drop(columns=["_row_idx"]).reset_index(drop=True)

    # 3. Join ground truth targets
    ann = ctx.annotated[["source", "ID", "et", "behavior"]].copy()
    ann["source"], ann["ID"] = ann["source"].astype(str), ann["ID"].astype(str)
    ann["et"] = ann["et"].astype("float32").round(4)

    meta_with_features = meta_filtered.join(X_val)
    combined = meta_with_features.merge(ann, on=["source", "ID", "et"], how="inner")
    combined = combined[combined["behavior"].isin(["dwelling", "nondwelling"])].reset_index(drop=True)
    
    y_val = (combined["behavior"] == "dwelling").astype(np.int32)
    X_val_final = combined[feature_cols].astype("float32")

    print(f"[Permutation Importance] Shuffling {len(X_val_final):,} validation rows ({n_repeats} passes per feature)...")
    
    # 4. Execute Scikit-Learn Permutation
    pi = permutation_importance(
        model, X_val_final.values, y_val.values,
        n_repeats=n_repeats, random_state=seed, n_jobs=-1, scoring='balanced_accuracy'
    )

    df_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std
    }).sort_values(by="importance_mean", ascending=False)

    noise_df = df_imp[df_imp["importance_mean"] <= 0.0]

    # 5. Append findings to the main assessment report text file
    with open(report_path, "a") as f:
        f.write(f"\n\n{'='*50}\nUNBIASED VALIDATION PERMUTATION IMPORTANCE\n{'='*50}\n")
        f.write(f"Evaluated on {len(X_val_final):,} validation rows. Scorer: Balanced Accuracy.\n\n")
        f.write("TOP 15 HIGHEST IMPACT VALIDATION FEATURES:\n")
        for _, r in df_imp.head(15).iterrows():
            f.write(f" • {r['feature']:<28} mean: {r['importance_mean']:+.4f}  (±{r['importance_std']:.4f})\n")
            
        f.write(f"\n### ZERO OR NEGATIVE IMPORTANCE FEATURES (PURE NOISE) [{len(noise_df)} total] ###\n")
        f.write("A <= 0 indicates scrambling the feature caused validation accuracy to stay flat or IMPROVE.\n")
        if noise_df.empty:
            f.write(" ✓ Zero noise features detected. All model inputs contributed positively.\n")
        else:
            for _, r in noise_df.iterrows():
                f.write(f" X [NOISE] {r['feature']:<26} mean: {r['importance_mean']:+.5f}  (±{r['importance_std']:.5f})\n")

    # 6. Generate Plot (Color-coded: Blue = Signal, Red = Noise)
    top_pos = df_imp[df_imp["importance_mean"] > 0].head(25)
    plot_df = pd.concat([top_pos, noise_df]).drop_duplicates().sort_values(by="importance_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.3)))
    colors = ['#ef4444' if val <= 0 else '#3b82f6' for val in plot_df["importance_mean"]]
    
    ax.barh(plot_df["feature"], plot_df["importance_mean"], xerr=plot_df["importance_std"], color=colors, alpha=0.85, capsize=3)
    ax.axvline(0, color='black', linewidth=1.2, linestyle='--')
    ax.set_xlabel("Mean Decrease in Validation Balanced Accuracy")
    ax.set_title("Validation Permutation Importance (Red bars = Pure Noise)")
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    
    plot_file = plots_dir / "validation_permutation_importance.png"
    fig.savefig(plot_file, bbox_inches="tight", dpi=300, facecolor='white')
    plt.close(fig)

    print(f"[Permutation Importance] Saved plot   → {plot_file}")
    print(f"[Permutation Importance] Logged to    → {report_path}")
    return df_imp, noise_df

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
    if plots is not None:
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

        meta_eval = results.loc[eval_mask, ['source', 'ID', 'et']].reset_index(drop=True)
        tp_ev, fp_ev, fn_ev, tp_mask, fp_mask, fn_mask, safe_frames = event_confusion_matrix(
            gt_eval, preds_eval, meta_eval, overlap_threshold=0.5
        )
                
        if plots is not None:
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
            ax.text(0.8, -0.12,
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
            
            
            cm_events = np.array([[0, fp_ev], 
                                [fn_ev, tp_ev]])
            disp_e = ConfusionMatrixDisplay(cm_events,display_labels=["No Event", "Event"])
            ax.grid(False)
            ax.text(0.8, -0.12,
                f"TPR/Sensitivity = {tpr:.3f}\n"
                f"TNR/Specificity = {tnr:.3f}\n"
                f"FPR = {fpr:.3f}",
                transform=plt.gca().transAxes, ha="left", va="bottom", color = "#000000", fontsize=7, 
            )
            disp_e.plot(ax=ax, colorbar=True, cmap="Purples")
            ax.set_title(f"Event Confusion Matrix - {ppc_id}")
            fig.savefig(plot_dir / f"Event_Confusion_Matrix.png",bbox_inches="tight",dpi=300)
            plt.close(fig)
            
            
                
                
             

    with open(report_path, "a") as f:
        f.write("\n".join(report_lines) + "\n")
    out_metrics = {}
    
    if pred_assess:
        out_metrics.update({
            "f1": f1,
            "precision": ppv,
            "recall": tpr,
            "mcc": mcc,
            "csi": csi,
            "accuracy": acc,
            "balanced_accuracy": ba,
            "specificity": tnr,
            "npv": npv,
            "fbeta": fbeta,
            "cohen_kappa": ck,
            "informedness": informedness,
            "markedness": markedness,
            "dor": dor,
            "prevalence": prev,
            "fpr": fpr,
            "fnr": fnr,
            "fm_index": fm,
            "fdr": fdr,
            "tp_ev": tp_ev, 
            "fp_ev":fp_ev, 
            "fn_ev": fn_ev
        })
        
    if prob_assess:
        out_metrics.update({
            "auroc": auroc,
            "average_precision": ap,
            "brier_score": brier,
            "ece": ece
        })

    return out_metrics
