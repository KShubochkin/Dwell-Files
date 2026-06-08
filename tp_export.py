#tp_export.py

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
import joblib
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

def _resolve_cache_dir(cache_path):
    cache_path = Path(cache_path)
    if cache_path.exists() and cache_path.is_file():
        return cache_path.parent
    if cache_path.name.startswith("features_") or cache_path.suffix == ".parquet":
        return cache_path.parent
    return cache_path


def _feature_cache_file(cache_path, src, mode="train"):
    cache_dir = _resolve_cache_dir(cache_path)
    return cache_dir / f"features_{src}_{mode}.parquet"


def _feature_cache_files_exist(cache_path, prefixes, mode="train"):
    return all(_feature_cache_file(cache_path, src, mode).exists() for src in prefixes)

def load_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"No model file found at {model_path}")
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model

def train(ctx, slices, prefixes, logic_file,model_path,feature_path,metadata_path,seed,cache_path,train_keys=None):
    
    feature_calc = importlib.import_module(logic_file)
    importlib.reload(feature_calc)
    
    cache_path = Path(cache_path)
    cache_dir = _resolve_cache_dir(cache_path)
    
    if _feature_cache_files_exist(cache_path, prefixes,mode="train"):
        print("Loading features from cache...")
        X_list, y_list, groups_list, meta_list, log_messages = [], [], [], [], []
        for src in prefixes:
            feat_cache = _feature_cache_file(cache_path, src, mode="train")
            df_cache = pd.read_parquet(feat_cache)
            y_list.append(df_cache['true_behavior'])
            groups_list.append(df_cache['ID'])
            meta_list.append(df_cache.drop(columns=[col for col in df_cache.columns if col not in ['source', 'ID', 'et', 'true_behavior']]))
            X_list.append(df_cache.drop(columns=['source', 'ID', 'et', 'true_behavior']))
        X = pd.concat(X_list, ignore_index=True)
        y = pd.concat(y_list, ignore_index=True)
        groups = pd.concat(groups_list, ignore_index=True)
        meta = pd.concat(meta_list, ignore_index=True)
        mod_meta = meta.copy()
        mod_meta['true_behavior'] = y.values
        log_messages.append(f"Loaded features from cache for sources: {', '.join(prefixes)}")
    else:

        print("Extracting features from training data...")
        X, y, groups, meta, log_messages,*_ = feature_calc.prepare_ml_dataset(ctx, fps=6, id_slice=slices, file_str=prefixes)

        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
        for src in prefixes:
            src_mask = meta['source'] == src
            if not src_mask.any():
                continue
            save_path = _feature_cache_file(cache_path, src, mode="train")
            mod_meta = meta.copy()
            mod_meta['true_behavior'] = y.values
            pd.concat(
                [mod_meta.loc[src_mask, ['source', 'ID', 'et', 'true_behavior']].reset_index(drop=True),
                 X.loc[src_mask].reset_index(drop=True)],
                axis=1,
            ).to_parquet(save_path, index=False)
            log_messages.append(f"Saved feature cache for source: {src}")

    if train_keys is not None:
        print("Filtering Universal Cache to match current Train split...")
        
        feature_cols = list(X.columns)
        
        # Combine everything so we can filter rows safely
        full_data = mod_meta.copy()
        full_data = pd.concat([full_data, X], axis=1)

        # Ensure string matching to prevent datatype errors (e.g., 1 vs "1")
        full_data['source'] = full_data['source'].astype(str)
        full_data['ID'] = full_data['ID'].astype(str)
        train_keys_copy = train_keys.copy()
        train_keys_copy['source'] = train_keys_copy['source'].astype(str)
        train_keys_copy['ID'] = train_keys_copy['ID'].astype(str)

        # Inner merge keeps ONLY the larvae that are in the current training split
        filtered_data = full_data.merge(train_keys_copy, on=['source', 'ID'], how='inner')

        # Re-split into X, y, meta for the model
        y = filtered_data['true_behavior']
        groups = filtered_data['ID']
        meta = filtered_data[['source', 'ID', 'et']]
        mod_meta = filtered_data[['source', 'ID', 'et', 'true_behavior']]
        X = filtered_data[feature_cols]
        print(f"Filtered features: {len(full_data)} -> {len(X)} rows.")

    feature_names = list(X.columns)
    X_values = X.values.astype(np.float32)
    y_values = y.values.astype(np.int32)
    groups_arr = np.array(groups)
    
    
    print("Training model...")
    
    X_values = X_values.astype(np.float32)
    y_values = y_values.astype(np.int32)
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        max_features=0.3,
        min_samples_leaf=100,
        random_state=seed,
        n_jobs=-1,
        class_weight='balanced_subsample',
        oob_score=True,
    )
    
    model.fit(X_values, y_values)
    
    package = {
        'model': model,
        'features': feature_names,
        'meta': mod_meta,
        'log_messages': log_messages
    }
    
    joblib.dump(model, model_path)
    print(f"Model saved → {model_path}")

    joblib.dump(list(X.columns), feature_path)
    print(f"Feature names saved → {feature_path}")
    
    mod_meta.to_pickle(metadata_path)
    print(f"Metadata saved → {metadata_path}")

    return model, list(X.columns), mod_meta, log_messages

def infer(model_path,ctx,files,feature_path,probabilities_path,metadata_test_path,cache_path,logic_file):
    feature_calc = importlib.import_module(logic_file)
    importlib.reload(feature_calc)
    
    model = load_model(model_path)
    features = joblib.load(feature_path)

    if probabilities_path.exists():
        os.remove(probabilities_path)
    
    cache_path = Path(cache_path)
    cache_dir = _resolve_cache_dir(cache_path)
    probabilities_path = Path(probabilities_path)
    
    if probabilities_path.exists():
        probabilities_path.unlink()
        print(f"Cleared old probabilities file at {probabilities_path}")
    
    print("Extracting features...")
    
    meta_test_list = []
    log_messages = []
    
    for file_id in files:
        print(f"\n=== Processing file_id={file_id} ===")

        sub_df = ctx.long_df[ctx.long_df['source'] == file_id].copy()
        if sub_df.empty:
            print(f"  WARNING: No data found in context for {file_id}. Skipping.")
            continue

        print(f"  Source: {file_id}  |  Frames: {len(sub_df):,}")
        print(f"  RAM free: {psutil.virtual_memory().available/1e9:.1f} GB")

        cached = False
        feat_cache_file = _feature_cache_file(cache_path, file_id, mode="infer")
        
        if cache_dir.exists() and feat_cache_file.exists():
            print(f"  Loading features for {file_id} from cache...")
            df_cache = pd.read_parquet(feat_cache_file)
            X_inf = df_cache.drop(columns=[col for col in df_cache.columns if col not in ['source', 'ID', 'et'] + features])
            meta = df_cache[['source', 'ID', 'et']].copy()
            log_messages.append(f"Loaded features from cache for source: {file_id}")
            cached = True
        else:
            print(f"  No cache found for {file_id}. Calculating features...")
            X_inf = feature_calc.calculate(df=sub_df, fps=6.0, pause_threshold=feature_calc.CONFIG['pause_threshold'], windows=feature_calc.CONFIG['windows'])
            meta = sub_df[['source', 'ID', 'et']].copy()

        # Align features to training feature list
        for col in features:
            if col not in X_inf.columns:
                X_inf[col] = 0.0
        extra = set(X_inf.columns) - set(features)
        if extra:
            X_inf = X_inf.drop(columns=list(extra))
        X_inf = X_inf[features].astype(np.float32)

        # Cache features if not cached
        if not cached:
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True, exist_ok=True)
            pd.concat([meta.reset_index(drop=True), X_inf.reset_index(drop=True)], axis=1).to_parquet(feat_cache_file, index=False)

        probs = model.predict_proba(X_inf)[:, 1]
        res = meta.copy()
        res['prob'] = probs
        
        del X_inf; gc.collect()

        # Attach ground truth if available
        cols_to_extract = ['source', 'ID', 'et', 'behavior', 'tags'] if 'tags' in ctx.annotated.columns else ['source', 'ID', 'et', 'behavior']
        ann_gt = ctx.annotated[ctx.annotated['source'] == file_id][cols_to_extract].copy()

        if 'tags' not in ann_gt.columns:
            ann_gt['tags'] = np.nan
            print("ruh roh, no tags column in annotations! Filling with NaN and proceeding without tag-based filtering.")
                
        if not ann_gt.empty:
            ann_gt = ann_gt[ann_gt['behavior'].isin(['dwelling', 'nondwelling'])].copy()
            res['ID'] = res['ID'].astype(str)
            ann_gt['ID'] = ann_gt['ID'].astype(str)

            res['et'] = res['et'].astype(np.float64).round(4)
            ann_gt['et'] = ann_gt['et'].astype(np.float64).round(4)

            res = res.merge(ann_gt[['source', 'ID', 'et', 'behavior', 'tags']], on=['source', 'ID', 'et'], how='left')

            res['true_label'] = np.nan
            
            # Set 1 strictly for wonderful dwelling
            won_mask = (res['behavior'] == 'dwelling') & res['tags'].astype(str).str.contains('wonderful', na=False, case=False)
            res.loc[won_mask, 'true_label'] = 1
            
            # Set 0 strictly for annotated nondwelling
            nd_mask = (res['behavior'] == 'nondwelling')
            res.loc[nd_mask, 'true_label'] = 0
                        
            print(f"  Matched {res['behavior'].notna().sum()} annotated frames.")
            
        else:
            res['behavior'] = np.nan
            res['tags'] = np.nan

        #res['ID'] = pd.to_numeric(res['ID'], errors='coerce').fillna(0).astype(int).astype(str)

        # to CSV
        res.to_csv(probabilities_path, mode='a', header=not probabilities_path.exists(), index=False)
        print(f"Predicted probabilities saved → {probabilities_path}")
        
        meta_test_list.append(res.copy())

        del sub_df, res, meta, probs
        gc.collect()
        print(f"  RAM free after cleanup: {psutil.virtual_memory().available/1e9:.1f} GB")    
    
    meta_test = pd.concat(meta_test_list, ignore_index=True) if meta_test_list else None
    
    if meta_test is not None:
        meta_test.to_pickle(metadata_test_path)
        print(f"Test metadata saved → {metadata_test_path}")
    
    return meta_test, log_messages

# In tp_export.py -> predict()
def predict(probabilities_path, ppc, predictions_dir,ctx,logic_file,plot=False): 
    probabilities_path = Path(probabilities_path)
    predictions_dir = Path(predictions_dir)
    
    if not probabilities_path.exists():
        raise FileNotFoundError(f"No probabilities file found at {probabilities_path}")
        
    # Read saved streaming probabilities and create MultiIndex for rapid slicing
    df_probs = pd.read_csv(probabilities_path)
    df_probs['ID'] = df_probs['ID'].astype(str)
    df_probs = df_probs.set_index(['source', 'ID']).sort_index()
    
    ppc_id = ppc.get_IDstr() 
    output_path = predictions_dir / f"{ppc_id}_predictions.csv"
    
    final_preds_list = []
    
    unique_tracks = df_probs.index.unique()
    sorted_tracks = sorted(unique_tracks, key=lambda x: (x[0], int(x[1]) if str(x[1]).isdigit() else x[1]))
    
    for source, track_id in sorted_tracks:
        track_id_str = str(track_id)
        if (source, track_id_str) not in df_probs.index:
            continue
        
        if int(track_id_str) % 1000 == 0: 
            print(f"Predicting for {source} track {track_id_str}...")
        
        track_data = df_probs.loc[(source, track_id_str)].sort_values('et')
        probs_array = track_data['prob'].values
        
        binary_preds = ppc.predict(probs_array) 
        
        track_preds_df = pd.DataFrame({
            'source': source,
            'ID': track_id_str,
            'et': track_data['et'].values,
            'prob': probs_array,
            'prediction': binary_preds
        })
        if 'behavior' in track_data.columns:
            track_preds_df['behavior'] = track_data['behavior'].values
        if 'tags' in track_data.columns:
            track_preds_df['tags'] = track_data['tags'].values
        final_preds_list.append(track_preds_df)
        
    if final_preds_list:
        output_df = pd.concat(final_preds_list, ignore_index=True)
        
        output_df.to_csv(output_path, index=False)
        print(f"Saved complete post-processed predictions -> {output_path}")

        if plot:
            for src in output_df['source'].unique():
                plot_source_grid(output_df, src, predictions_dir, ctx, logic_file, cols=10)

        return output_df
    else:
        print("⚠ WARNING: No matched predictions generated. Check if test IDs match inference sources.")
        return pd.DataFrame()

def plot_source_grid(results_path, src, out_dir,ppc, cols=10):
    """One figure per source — all larvae as a grid of small prob traces."""
    ppc_id = ppc.get_IDstr() 
    pred_path = results_path / f"{ppc_id}_predictions.csv"
    results = pd.read_csv(pred_path)
    
    plt.style.use('default')

    results = results.copy()
    
    grp_src = results[results['source'] == src]
    
    larvae = sorted(grp_src['ID'].unique(), key=lambda x: int(x) if str(x).isdigit() else x)
    
    if len(larvae) == 0:
        print(f"  No tracks found for source {src}. Skipping plot.")
        return

    rows = int(np.ceil(len(larvae) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 1.8), facecolor='#111')
    
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = np.atleast_1d(axes).ravel()

    print(f"  Plotting {len(larvae)} tracks for source {src} in a {rows}x{cols} grid...")
    #progress bar
    print()
    num = len(larvae)
    for i, lid in enumerate(larvae):
        print(f"\rProgress: |{i+1}/{num}|", end="")

        ax  = axes[i]
        grp = grp_src[grp_src['ID'] == lid].sort_values('et')
        et, prob, pred = grp['et'].values, grp['prob'].values, grp['prediction'].values

        # A "predictiony" vibrant cyan
        ax.fill_between(et, pred, alpha=0.45, color='#00FFFF', step='post') 
        ax.plot(et, prob, color="#BBF1FF", linewidth=0.8) 
        
        if np.any(pred):
            diffs = np.diff(np.concatenate(([0], pred, [0])))
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0] - 1 
            dark_outline = [pe.withStroke(linewidth=1, foreground='#111111')]
            
            for s_idx, e_idx in zip(starts, ends):
                if s_idx < len(et) and e_idx < len(et):
                    s_time = et[s_idx]
                    e_time = et[e_idx]
                    dur = e_time - s_time
                    
                    if dur > 0: 
                        ax.text(s_time, 0.75, f"{s_time:.1f}", color='#cffafe', fontsize=5, 
                                ha='right', va='center', path_effects=dark_outline)
                        
                        ax.text(e_time, 0.35, f"{e_time:.1f}", color='#cffafe', fontsize=5, 
                                ha='right', va='center', path_effects=dark_outline)
                        
                        mid_time = s_time + (dur / 2)
                        ax.text(mid_time, 0.05, f"{dur:.1f}s", color="#082a2f", fontsize=5, 
                                ha='center', va='bottom',)
        
        if 'behavior' in grp.columns:
            
            # 1. Wonderful Dwelling (Bright Green)
            won_mask = ((grp['behavior'] == 'dwelling') & grp['tags'].astype(str).str.contains('wonderful', na=False, case=False)).astype(int)
            if won_mask.any():
                ax.fill_between(et, won_mask * -0.25, 0, alpha=0.6, color='#22c55e', step='post')

            # 2. Unsure Dwelling (Red)
            unsure_mask = ((grp['behavior'] == 'dwelling') & grp['tags'].astype(str).str.contains('unsure', na=False, case=False)).astype(int)
            if unsure_mask.any():
                ax.fill_between(et, unsure_mask * -0.25, 0, alpha=0.6, color='#ef4444', step='post')

            # 3. Alright Dwelling (Yellow)
            alright_mask = ((grp['behavior'] == 'dwelling') & grp['tags'].astype(str).str.contains('alright', na=False, case=False)).astype(int)
            if alright_mask.any():
                ax.fill_between(et, alright_mask * -0.25, 0, alpha=0.6, color='#eab308', step='post')

            # 4. Strictly Nondwelling (Grey)
            nd_mask = (grp['behavior'] == 'nondwelling').astype(int)
            if nd_mask.any():
                ax.fill_between(et, nd_mask * -0.1, 0, alpha=0.4, color='#9ca3af', step='post')

        ax.set_ylim(-0.3, 1.05)
        ax.set_title(f"ID {lid}", fontsize=7, color='#aaa', pad=2)
        
        if len(et) > 0:
            min_et, max_et = et[0], et[-1]
            ax.text(0.01, 0.05, f"{min_et:.1f}s", transform=ax.transAxes, fontsize=5, color='#666666', va='bottom', ha='left')
            ax.text(0.99, 0.05, f"{max_et:.1f}s", transform=ax.transAxes, fontsize=5, color='#666666', va='bottom', ha='right')
            
        ax.set_facecolor('#111')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    i = 0
    for ax in axes[len(larvae):]:
        print(f"\rProgress: |{i}/{num}|", end="")
        i += 1
        ax.set_visible(False)

    fig.suptitle(src, color='#ccc', fontsize=11)
    fig.tight_layout(pad=0.3)
    path = Path(out_dir) / f"grid_{src}.png"
    fig.savefig(path, dpi=120, bbox_inches='tight', facecolor='#111')
    plt.close(fig)
    print(f"  Saved: {path}")

def assess_performance(preds_dir,val_dir, ppc, prefixes, sources, report_path, plots, 
                       dwell_tags=["wonderful"], pred_assess=True, prob_assess=True, 
                       rf_assess=False, descriptive=True):
    
    # 1. Setup & Data Ingestion
    ppc_id = ppc.get_IDstr() 
    preds_dir = Path(preds_dir)
    val_dir = Path(val_dir)
    plot_dir = Path(plots)
    plot_dir = plot_dir / ppc_id
    plot_dir.mkdir(parents=True, exist_ok=True) # Ensure plot directory exists
    
    pred_path = preds_dir / f"{ppc_id}_predictions.csv"
    results = pd.read_csv(pred_path).copy()
    
    larvae = sorted(results['ID'].unique(), key=lambda x: int(x) if str(x).isdigit() else x)
    et, probs, preds, beh, tags = results['et'].values, results['prob'].values, results['prediction'].values, results['behavior'].values, results['tags'].values
    
    wonderful = ((results['behavior'] == 'dwelling') & results['tags'].astype(str).str.contains('wonderful', na=False, case=False)).astype(int)
    alright = ((results['behavior'] == 'dwelling') & results['tags'].astype(str).str.contains('alright', na=False, case=False)).astype(int)
    nondwelling = (results['behavior'] == 'nondwelling').astype(int)

    def get_confusion_data(preds, nd, beh, tags, dtags):
        dwell = (beh == "dwelling").astype(int)
        tag_match = np.isin(tags, dtags)
        dwell = np.where(tag_match, dwell, 0)
        
        # (nd is already the Negative Class Mask passed into the function)
        
        # 2. Compare directly against the independent masks!
        tps = (dwell == 1) & (preds == 1) # True Positives
        fns = (dwell == 1) & (preds == 0) # False Negatives
        
        fps = (nd == 1) & (preds == 1)    # False Positives
        tns = (nd == 1) & (preds == 0)    # True Negatives
        
        return tps.sum(), tns.sum(), fps.sum(), fns.sum(), dwell.sum(), nd.sum()
        
    def expected_calibration_error(y_true, y_prob, n_bins=10):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_assignments = np.digitize(y_prob, bin_edges) - 1
        bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)
        
        ece = 0.0
        n_samples = len(y_true)
        
        for i in range(n_bins):
            bin_mask = bin_assignments == i
            bin_size = np.sum(bin_mask)
            if bin_size > 0:
                actual_true = np.mean(y_true[bin_mask])
                actual_pred = np.mean(y_prob[bin_mask])
                ece += (bin_size / n_samples) * np.abs(actual_true - actual_pred)
                
        return ece

    def get_train_test_idx(preds_dir, ppc, train_sources, test_sources):
        """
        Extracts the row indices for training and testing data from the predictions CSV.
        
        Args:
            preds_dir (str or Path): Directory containing the predictions CSV.
            ppc: The post-processing classifier object (used to get the ID string).
            train_sources (list): List of source IDs/prefixes used for training.
            test_sources (list): List of source IDs/prefixes used for testing/inference.
            
        Returns:
            train_idx (np.ndarray): Integer array of training indices.
            test_idx (np.ndarray): Integer array of testing indices.
        """
        preds_dir = Path(preds_dir)
        ppc_id = ppc.get_IDstr()
        pred_path = preds_dir / f"{ppc_id}_predictions.csv"
        
        if not pred_path.exists():
            raise FileNotFoundError(f"Predictions file not found at {pred_path}")
            
        df = pd.read_csv(pred_path, usecols=['source']) # Only load 'source' to save RAM
        
        # Extract integer indices where the source matches your splits
        train_idx = df.index[df['source'].isin(train_sources)].to_numpy()
        test_idx = df.index[df['source'].isin(test_sources)].to_numpy()
        
        # Quick sanity check
        if len(train_idx) == 0:
            print("⚠ WARNING: train_idx is empty. Check if train_sources match the CSV.")
        if len(test_idx) == 0:
            print("⚠ WARNING: test_idx is empty. Check if test_sources match the CSV.")
            
        return train_idx, test_idx

    #train_idx, test_idx = get_train_test_idx(preds_dir,ppc,prefixes,sources)
    dwell_mask = (beh == "dwelling") & np.isin(tags, dwell_tags)
    nd_mask = (beh == "nondwelling")
    
    gt = np.zeros_like(beh, dtype=int)
    gt[dwell_mask] = 1

    eval_mask = dwell_mask | nd_mask
    
    gt_eval = gt[eval_mask]
    probs_eval = probs[eval_mask]
    preds_eval = preds[eval_mask]
    
    report_lines = [f"\n{'='*50}\nModel Assessment Report: {ppc_id}\n{'='*50}"]
    
    if pred_assess:
        tp, tn, fp, fn, p, n = get_confusion_data(preds, nondwelling, beh, tags, dwell_tags)
        
        # Guard against division by zero
        p = max(p, 1)
        n = max(n, 1)
        
        tpr = tp / p      
        tnr = tn / n      
        fnr = fn / p      
        fpr = fp / n
        
        informedness = tpr + tnr - 1
        plr = tpr / fpr if fpr > 0 else np.nan
        nlr = fnr / tnr if tnr > 0 else np.nan
        
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        fo_r = 1 - npv
        fdr = 1 - ppv
        acc = (tp + tn) / (p + n)
        prev = p / (p + n)
        ba = (tpr + tnr) / 2
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        fm = math.sqrt(ppv * tpr) # Usually calculated with precision and recall
        
        # Safe MCC calculation
        mcc_denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        mcc = ((tp*tn) - (fp*fn)) / mcc_denom if mcc_denom > 0 else 0
        
        dor = plr / nlr if (nlr > 0 and not np.isnan(plr)) else np.nan
        markedness = ppv + npv - 1
        csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
        
        ck = cohen_kappa_score(gt, preds)
        fbeta = fbeta_score(gt, preds, beta=1.0) # Fixed: added beta parameter
        
        # Append nicely formatted text to report
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
            f"FDR: {fdr:4f}"
            ""
        ])

    # 3. Probability Assessment (Continuous Metrics & Visuals)
    if prob_assess:
        pointwise_log_loss = -(gt_eval * np.log(np.clip(probs_eval, 1e-15, 1-1e-15)) + (1 - gt_eval) * np.log(1 - np.clip(probs_eval, 1e-15, 1-1e-15)))
        
        brier = brier_score_loss(gt_eval, probs_eval)
        auroc = roc_auc_score(gt_eval, probs_eval)
        ap = average_precision_score(gt_eval, probs_eval)
        ece = expected_calibration_error(gt_eval, probs_eval)
        
        report_lines.extend([
            "--- PROBABILITY METRICS ---",
            f"AUROC:                 {auroc:.4f}",
            f"Average Precision (AP):{ap:.4f}",
            f"Brier Score:           {brier:.4f}",
            f"Expected Calib Error:  {ece:.4f}",
            ""
        ])

        plt.style.use('seaborn-v0_8-whitegrid') 
        
        # ROC Curve
        fpr, tpr_curve, _ = roc_curve(gt_eval, probs_eval)
        fig, ax = plt.subplots(figsize=(6, 6))
        RocCurveDisplay(fpr=fpr, tpr=tpr_curve, roc_auc=auroc).plot(ax=ax)
        ax.set_title(f"ROC Curve - {ppc_id}")
        fig.savefig(plot_dir / f"ROC_Curve.png", bbox_inches='tight', dpi=300)
        plt.close(fig)

        # Precision-Recall Curve
        prec, rec, _ = precision_recall_curve(gt_eval, probs_eval)
        fig, ax = plt.subplots(figsize=(6, 6))
        PrecisionRecallDisplay(precision=prec, recall=rec, average_precision=ap).plot(ax=ax)
        ax.set_title(f"Precision-Recall Curve - {ppc_id}")
        fig.savefig(plot_dir / f"PR_Curve.png", bbox_inches='tight', dpi=300)
        plt.close(fig)

        # Calibration Curve
        prob_true, prob_pred = calibration_curve(gt_eval, probs_eval)
        fig, ax = plt.subplots(figsize=(6, 6))
        CalibrationDisplay(prob_true, prob_pred, probs_eval).plot(ax=ax)
        ax.set_title(f"Calibration Curve - {ppc_id}")
        fig.savefig(plot_dir / f"Calibration.png", bbox_inches='tight', dpi=300)
        plt.close(fig)

        # Confusion Matrix
        cm = confusion_matrix(gt_eval, preds_eval)
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues')
        ax.set_title(f"Confusion Matrix - {ppc_id}")
        fig.savefig(plot_dir / f"Confusion_Matrix.png", bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        # DET Curve
        fig, ax = plt.subplots(figsize=(6, 6))
        DetCurveDisplay.from_predictions(gt_eval, probs_eval, ax=ax)
        ax.set_title(f"DET Curve - {ppc_id}")
        fig.savefig(plot_dir / f"DET_Curve.png", bbox_inches='tight', dpi=300)
        plt.close(fig)

        # Cumulative Gain & Lift Curves
        probs_2d = np.vstack([1 - probs_eval, probs_eval]).T
        
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_cumulative_gain(gt_eval, probs_2d, ax=ax)
        fig.savefig(plot_dir / f"Cumulative_Gain.png", bbox_inches='tight', dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        plot_lift_curve(gt_eval, probs_2d, ax=ax)
        fig.savefig(plot_dir / f"Lift_Curve.png", bbox_inches='tight', dpi=300)
        plt.close(fig)

    # Append to Output Text File
    with open(report_path, 'a') as f:
        f.write("\n".join(report_lines) + "\n")
        
    return True
