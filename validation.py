#mte.py

import joblib
import importlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score, log_loss
from scipy.ndimage import binary_opening, binary_closing, median_filter, label
from scipy.stats import wilcoxon
import sys
sys.path.append(r"C:\Users\corna\honours\fresh1\hp_2\notebooks&helpers")
from joblib import Parallel, delayed
import shap
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GroupKFold
import sys


def post_process(probs, fps=6, threshold=0.5, 
        smooth_s=1.5, min_bout_s=4.5, gap_fill_s=5.5):
  """Encapsulated smoothing logic."""
  smoothed_probs = median_filter(probs, size=int(smooth_s * fps))
  preds = (smoothed_probs > threshold).astype(int)
  
  close_size = int(gap_fill_s * fps)
  open_size = int(min_bout_s * fps)
  pad_len = max(close_size, open_size)
  preds_padded = np.pad(preds, pad_width=pad_len, mode='edge')
  
  # 3. Apply morphological operations on the padded array
  preds_padded = binary_closing(preds_padded, structure=np.ones(close_size)).astype(int)
  preds_padded = binary_opening(preds_padded, structure=np.ones(open_size)).astype(int)
  
  # 4. Strip the padding back off to return to original length
  preds = preds_padded[pad_len:-pad_len]
  
  return preds

def post_process_pooled(y_true_arr, probs_arr, meta_df):
    """Apply post_process per larva, respecting time gaps."""
    preds = np.zeros_like(probs_arr, dtype=int)
    meta_df = meta_df.reset_index(drop=True)
    groups = (meta_df['source'] + "_" + meta_df['ID'].astype(str)).values
    for g in np.unique(groups):
        mask = groups == g
        g_et = meta_df.loc[mask, 'et'].values
        g_probs = probs_arr[mask]
        # Split further on time gaps > 0.5s within the same larva
        gap_locs = np.where(np.diff(g_et) > 0.5)[0] + 1
        segments = np.split(np.where(mask)[0], gap_locs)
        seg_probs = np.split(g_probs, gap_locs)
        for seg_idx, seg_p in zip(segments, seg_probs):
            preds[seg_idx] = post_process(seg_p)
    return preds
  
def _post_process_fold(probs, test_groups, meta_df_subset):
    """Apply post_process per larva per contiguous segment (gap-aware)."""
    preds_flat = np.zeros(len(probs), dtype=int)
    meta_df_subset = meta_df_subset.reset_index(drop=True)
    groups = (meta_df_subset["source"] + "_" + meta_df_subset["ID"].astype(str)).values

    for g in np.unique(groups):
        mask = groups == g
        g_et   = meta_df_subset.loc[mask, "et"].values
        g_probs = probs[mask]
        gap_locs = np.where(np.diff(g_et) > 0.5)[0] + 1
        global_idx  = np.where(mask)[0]
        segments    = np.split(global_idx, gap_locs)
        seg_probs   = np.split(g_probs,    gap_locs)
        for seg_idx, seg_p in zip(segments, seg_probs):
            preds_flat[seg_idx] = post_process(seg_p)
    return preds_flat
  
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.ensemble import RandomForestClassifier

def _optimize_postprocess_per_larva(probs_by_larva, y_by_larva,
                                     fps=6,
                                     filter_candidates=(0, 5, 11, 21, 31, 41, 61),
                                     threshold_candidates=None):
    pass


def _apply_postprocess_per_larva(probs_by_larva, filter_len, threshold,
                                  gap_fill_frames=33, min_bout_frames=27):
    """
    Apply the optimized filter+threshold to each larva's probs independently.
    Returns a list of binary prediction arrays (one per larva).
    """
    preds_by_larva = []
    for probs in probs_by_larva:
        if filter_len > 1:
            sm = median_filter(probs, size=filter_len)
        else:
            sm = probs.copy()
        p = (sm > threshold).astype(int)
        p = binary_closing(p, structure=np.ones(gap_fill_frames)).astype(int)
        p = binary_opening(p, structure=np.ones(min_bout_frames)).astype(int)
        preds_by_larva.append(p)
    return preds_by_larva


def _shuffled_group_kfold(groups, n_splits, seed):
    """
    GroupKFold with shuffled group order so each seed gets different folds.
    Yields (train_idx, test_idx) arrays into the original groups array.
    """
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled_groups = rng.permutation(unique_groups)

    # Assign each shuffled group to a fold (round-robin)
    group_to_fold = {g: i % n_splits for i, g in enumerate(shuffled_groups)}

    fold_assignments = np.array([group_to_fold[g] for g in groups])

    for fold in range(n_splits):
        test_mask = fold_assignments == fold
        train_mask = ~test_mask
        yield np.where(train_mask)[0], np.where(test_mask)[0]


def _one_seed(seed, seed_idx, X_values, y_values, groups, n_splits,
              feature_names,meta_df, store_raw=False, optimize_postprocess=False):
    """
    Train one RF over GroupKFold (shuffled per seed).
    Post-processing pipeline: median filter sweep → threshold sweep,
    both applied per-larva to prevent bleed.
    """
    print(f"  Training seed {seed}...")
    X_values = X_values.astype(np.float32)
    y_values = y_values.astype(np.int32)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=16,
        max_features=0.3,
        min_samples_leaf=100,
        random_state=seed,
        n_jobs=-1,
        class_weight='balanced_subsample'
    )

    fold_f1s = np.zeros(n_splits)
    fold_log_losses = np.zeros(n_splits)
    fold_auprcs = np.zeros(n_splits)
    fold_preds = {} if seed_idx == 0 else None
    importance_sum = np.zeros(len(feature_names))
    roc_data = []
    raw_probs = []

    # Use seed-shuffled GroupKFold instead of vanilla GroupKFold
    splits = list(_shuffled_group_kfold(groups, n_splits, seed))

    for fold_num, (train_idx, test_idx) in enumerate(splits):
        print(f"    Seed {seed} | Fold {fold_num + 1}/{n_splits}")
        X_train, X_test = X_values[train_idx], X_values[test_idx]
        y_train, y_test = y_values[train_idx], y_values[test_idx]

        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds_flat = np.zeros_like(probs)
        probs_flat = probs

        # Extract the groups for just this test fold so we can separate larvae
        test_groups = groups[test_idx]
        unique_test_groups = np.unique(test_groups)

        test_meta = meta_df.iloc[test_idx].reset_index(drop=True)  # meta_df passed as new arg
        preds_flat = _post_process_fold(probs, test_groups, test_meta)
            
        fold_f1s[fold_num] = f1_score(y_test, preds_flat, zero_division=0)
        fold_log_losses[fold_num] = log_loss(y_test, probs, labels=[0, 1])
        fold_auprcs[fold_num] = average_precision_score(y_test, probs)

        if store_raw:
            fpr, tpr, thresholds = roc_curve(y_test, probs)
            roc_data.append({
                "fpr": fpr, "tpr": tpr,
                "auc": auc(fpr, tpr), "thresholds": thresholds
            })
            raw_probs.append((test_idx, probs))

        if seed_idx == 0:
            fold_preds[tuple(test_idx)] = preds_flat

        importance_sum += model.feature_importances_
        model.estimators_ = []

    del model
    return (
        np.mean(fold_f1s),
        np.mean(fold_log_losses),
        np.mean(fold_auprcs),
        fold_preds,
        importance_sum / n_splits,
        roc_data,
        raw_probs,
    )
    
def tune_rf_hyperparameters(X, y, groups, n_splits=4, n_iter=80, metric='auprc'):
  """
  Tunes RandomForest hyperparameters using threshold-agnostic metrics.
  No post-processing is applied here. Evaluates the full probability distribution.
  
  Parameters:
  - X, y, groups: Your extracted features, labels, and larva IDs.
  - n_splits: Number of folds for GroupKFold.
  - n_iter: Number of random parameter combinations to try.
  - metric: 'auprc' (Average Precision) or 'log_loss'.
  """
  print(f"--- Starting Hyperparameter Tuning optimizing for {metric.upper()} ---")
  
  # 1. Define the parameter space
  # (Adjust ranges based on your compute limits and domain knowledge)
  param_dist = {
    'n_estimators': [200],
    'max_depth': [8, 12, 16, 20, 25],
    'min_samples_leaf': [10, 25, 50, 100, 200],
    'class_weight': ['balanced', 'balanced_subsample'],
    'max_features': ['sqrt', 'log2', 0.3]
  }
  
  # 2. Select the correct threshold-agnostic scorer
  # 'average_precision' calculates the area under the PR curve (auPRC).
  # 'neg_log_loss' is used because scikit-learn optimization always tries to maximize the score.
  if metric.lower() == 'auprc':
    scoring = 'average_precision'
  elif metric.lower() == 'log_loss':
    scoring = 'neg_log_loss'
  else:
    raise ValueError("Metric must be 'auprc' or 'log_loss'")

  # 3. Setup the model and cross-validation strategy
  # Note: We set n_jobs=1 on the RF, and n_jobs=-1 on the Search. 
  # This parallelizes the search combinations rather than the individual trees, 
  # which is usually much faster and avoids CPU thread thrashing.
  rf = RandomForestClassifier(random_state=42, n_jobs=1)
  gkf = GroupKFold(n_splits=n_splits)
  
  # 4. Setup Randomized Search
  search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=n_iter,
    scoring=scoring,
    cv=gkf,
    n_jobs=1, 
    verbose=3,
    random_state=42
  )
  
  from sklearn.utils import resample
  import numpy as np

  # Stratified subsample preserving class and group structure
  tune_frac = 0.20
  idx = []
  for grp in np.unique(groups):
    grp_idx = np.where(groups == grp)[0]
    n = max(1, int(len(grp_idx) * tune_frac))
    idx.extend(resample(grp_idx, n_samples=n, random_state=42, replace=False))

  X_tune = X.iloc[idx]
  y_tune = y.iloc[idx]
  groups_tune = groups.iloc[idx]
  # 5. Execute Search 
  # Passing groups here is strictly enforced by GroupKFold to keep test larvae pure.
  search.fit(X_tune, y_tune, groups=groups_tune)
  
  # 6. Report
  best_score = search.best_score_ if metric.lower() == 'auprc' else -search.best_score_
  
  print("\n" + "="*40)
  print(" TUNING COMPLETE")
  print("="*40)
  print(f"Best {metric.upper()} Score: {best_score:.4f}")
  print("Best Parameters:")
  for param, value in search.best_params_.items():
    print(f" {param}: {value}")
    
  return search.best_params_, search.best_estimator_

def select_features_rfe(X, y, groups, step=5):
  """
  Finds the optimal feature subset using RFECV.
  Step=5 drops 5 features at a time to speed it up.
  """
  print(f"--- Starting RFE on {X.shape[1]} features ---")
  
  # Use a slightly 'faster' version of your RF for selection
  selector_rf = RandomForestClassifier(
    n_estimators=100, # Fewer trees for speed during selection
    max_depth=16,
    max_features=0.3,
    min_samples_leaf=100,
    n_jobs=-1, 
    class_weight='balanced_subsample',
    random_state=42
  )
  
  # Using GroupKFold to respect your larva-level groups
  cv = GroupKFold(n_splits=4)
  
  selector = RFECV(
    estimator=selector_rf,
    step=step, 
    cv=cv,
    scoring='f1', # Since you care about F1
    min_features_to_select=5,
    n_jobs=1,
    verbose=2
  )
  
  # RFECV needs the 'groups' passed here
  selector.fit(X, y, groups=groups)
  
  results = pd.DataFrame({
    'Feature': X.columns,
    'Rank': selector.ranking_,
    'Keep': selector.support_
  }).sort_values('Rank')
  
  print("\n--- Feature Survival Rankings ---")
  
  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', None)
  print(results)
  print("\n--- CLEAN LIST FOR COPY-PASTING ---")
  for feat in results['Feature'].tolist():
    print(feat)
  pd.reset_option('display.max_rows')
  pd.reset_option('display.max_columns')

  plt.figure()
  plt.xlabel("Number of features selected")
  plt.ylabel("Cross-validation score")
  plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1), 
      selector.cv_results_['mean_test_score'])
  plt.show()
  
  selected_features = X.columns[selector.support_].tolist()
  print(f"--- RFE Complete: Kept {len(selected_features)} features ---")
  return selected_features

def _cm_scalars(y_true, y_pred):
  """
  Returns a dict with:
   - balanced_accuracy : (sensitivity + specificity) / 2
   - informedness   : sensitivity + specificity - 1 (Youden's J)
   - delta_p      : precision - (1 - specificity) (Loevinger / p)

  These are all deterministic functions of the 22 CM  zero extra inference.
  """
  cm = confusion_matrix(y_true, y_pred)
  if cm.shape != (2, 2):
    return {}
  tn, fp, fn, tp = cm.ravel()
  sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
  specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
  precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
  balanced_acc = (sensitivity + specificity) / 2
  informedness = sensitivity + specificity - 1
  delta_p   = precision - (1 - specificity)  
  return {
    "balanced_accuracy": balanced_acc,
    "informedness":   informedness,
    "delta_p":      delta_p,
    "sensitivity":    sensitivity,
    "specificity":    specificity,
    "precision":     precision,
  }

def run_experiment(training_data, slice_val, files_prefix, logic_modules, 
         num_seeds=5,n_splits=5,use_shap=False,do_rfe=False,optimize_postprocess=False):
  all_logic_scores = {}
  all_logic_log_loss = {} 
  all_logic_auprc = {}
  representative_preds = {}  
  representative_metas = {}
  importance_records  = {}  
  roc_records      = {} 
  raw_prob_records   = {}
  shap_records     = {}

  for module_name in logic_modules:
    print(f"\n>>> Processing Logic: {module_name}")
    module = importlib.import_module(module_name)
    importlib.reload(module)
    
    print("Extracting features...")
    X, y, groups, meta, log_messages = module.prepare_ml_dataset(training_data, fps=6, id_slice=slice_val, file_str=files_prefix)
    
    if do_rfe:
      selected_cols = select_features_rfe(X, y, groups, step=5)
      X = X[selected_cols]

    mod_meta = meta.copy()
    mod_meta['true_behavior'] = y.values
    representative_metas[module_name] = mod_meta
    
    feature_names = list(X.columns)
    X_values = X.values.astype(np.float32)
    y_values = y.values.astype(np.int32)
    groups_arr = np.array(groups)
    
    
    print(f"Training {num_seeds} seeds...")
    
    # results = []
    # for i in range(num_seeds):
    #   is_representative = (i == 0) 
    #   res = _one_seed(42 + i, i, X_values, y_values, groups_arr, n_splits, feature_names, store_raw=is_representative)
    #   results.append(res)
    results = []
    for i in range(num_seeds):
      res = _one_seed(42 + i, i, X_values, y_values, groups_arr, n_splits, feature_names,meta, store_raw=(i==0),optimize_postprocess=optimize_postprocess)
      results.append(res)
    # results = Parallel(n_jobs=-1, prefer="threads")(
    #   delayed(_one_seed)(
    #     42 + i, i, X_values, y_values, groups_arr, n_splits, feature_names, store_raw=(i==0)
    #   ) for i in range(num_seeds)
    # )
    
      
    if use_shap:
      print(f"Generating SHAP summary for {module_name}...")
      
      final_model = RandomForestClassifier(n_estimators=150, max_depth=16,min_samples_leaf=100, max_features=0.3, n_jobs=-1, class_weight='balanced_subsample')
      final_model.fit(X, y) 
    
      explainer = shap.TreeExplainer(final_model)
      subsample = X.sample(min(1000, len(X)), random_state=42)
      shap_vals = explainer.shap_values(subsample)
      sv = shap_vals[1] if isinstance(shap_vals, list) else (shap_vals[:,:,1] if shap_vals.ndim==3 else shap_vals)
      shap_records[module_name] = {"names": feature_names, "mean_abs": np.abs(sv).mean(axis=0)}
      
      del final_model, explainer
      gc.collect()

    logic_f1_runs = [r[0] for r in results]
    all_logic_scores[module_name] = logic_f1_runs
    all_logic_log_loss[module_name] = [r[1] for r in results]
    all_logic_auprc[module_name]  = [r[2] for r in results]
    
    seed0_fold_preds = results[0][3]
    if seed0_fold_preds:
      pred_series = pd.Series(index=range(len(y)), dtype=int)
      for test_idx_tuple, preds in seed0_fold_preds.items():
        pred_series.iloc[list(test_idx_tuple)] = preds
      # Re-index to match meta's actual DataFrame index
      pred_series.index = meta.index
      representative_preds[module_name] = pred_series
      
    importance_records[module_name] = {
      "names": feature_names,
      "mean": np.mean([r[4] for r in results], axis=0),
      "std":  np.std( [r[4] for r in results], axis=0),
    }
    
    roc_records[module_name] = [rd for r in results for rd in r[5]]

    # Confusion matrix: pool all seed-0 fold probs (avoids double-counting)
    seed0_raw = results[0][6]
    all_probs = np.full(len(y), np.nan)
    for test_idx_tuple, probs in seed0_raw:
      all_probs[list(test_idx_tuple)] = probs
    raw_prob_records[module_name] = (y_values, all_probs, meta.reset_index(drop=True))
    
    del X, y, groups, X_values, y_values, groups_arr
    gc.collect()

  return representative_metas, representative_preds, all_logic_scores,all_logic_log_loss, all_logic_auprc,importance_records, roc_records, raw_prob_records, shap_records, log_messages

def _significance_report(scores_dict, log_loss_dict, auprc_dict, raw_prob_dict,log_file=None):
  """
  Print median F1, variance ratio, and Wilcoxon signed-rank test between
  every pair of logic modules.
  """
  def custom_print(msg):
    print(msg)
    if log_file:
      with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        
  modules = list(scores_dict.keys())
  print("\n" + "=" * 60)
  print("STATISTICAL SUMMARY")
  print("=" * 60)

  for m in modules:
    vals = np.array(scores_dict[m])
    ll_vals = np.array(log_loss_dict[m])
    pr_vals = np.array(auprc_dict[m])
    custom_print(f"\n {m}")
    custom_print(f"  Median F1 : {np.median(vals):.4f}")
    custom_print(f"  Mean F1  : {np.mean(vals):.4f}")
    custom_print(f"  Std    : {np.std(vals):.4f}")
    custom_print(f"  Min/Max  : {vals.min():.4f} / {vals.max():.4f}")
    custom_print(f"  Mean Log Loss : {np.mean(ll_vals):.4f}  {np.std(ll_vals):.4f}")
    custom_print(f"  Mean auPRC  : {np.mean(pr_vals):.4f}  {np.std(pr_vals):.4f}")
    
    if m in raw_prob_dict:
      y_true, probs, meta_df = raw_prob_dict[m]
      preds_pp = post_process_pooled(y_true, probs, meta_df)
      scalars = _cm_scalars(y_true, preds_pp)
      if scalars:
        custom_print(f"  --- Seed-0 pooled predictions ---")
        custom_print(f"  Balanced Accuracy : {scalars['balanced_accuracy']:.4f}")
        custom_print(f"  Informedness (J) : {scalars['informedness']:.4f} "
           f" (sensitivity={scalars['sensitivity']:.3f}, "
           f"specificity={scalars['specificity']:.3f})")
        custom_print(f"  DeltaP (p)    : {scalars['delta_p']:.4f} "
           f" (precision={scalars['precision']:.3f}, "
           f"FPR={1 - scalars['specificity']:.3f})")

  if len(modules) >= 2:
    custom_print("\n Pairwise Wilcoxon signed-rank tests (H0: equal medians):")
    for i in range(len(modules)):
      for j in range(i + 1, len(modules)):
        a = np.array(scores_dict[modules[i]])
        b = np.array(scores_dict[modules[j]])
        if np.all(a == b):
          custom_print(f"  {modules[i]} vs {modules[j]}: identical scores, test skipped")
          continue
        try:
          stat, p = wilcoxon(a, b)
          diff_vs_var = abs(np.mean(a) - np.mean(b)) / (np.std(a) + np.std(b) + 1e-9)
          custom_print(f"  {modules[i]} vs {modules[j]}:")
          custom_print(f"   W={stat:.2f}, p={p:.4f} {'*' if p < 0.05 else '(ns)'}")
          custom_print(f"   |mean| / (_A+_B) = {diff_vs_var:.3f} "
             f"{'(diff >> within-set variance)' if diff_vs_var > 1 else '(within-set variance dominates)'}")
        except Exception as e:
          custom_print(f"  {modules[i]} vs {modules[j]}: test failed ({e})")
  custom_print("=" * 60 + "\n")
  
def calculate_event_metrics(y_true, y_pred, meta_df, overlap_threshold=0.3):
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
                    for p_idx in overlapping_preds:
                        tainted_preds.add(p_idx)
            
            # Evaluate Predicted Events for False Positives
            for p_idx in range(1, num_pred + 1):
                # An interval is a False Positive if it wasn't part of any successful TP
                if p_idx not in whitelisted_preds:
                    fp += 1

    return tp, fp, fn
  
def plot_tagged_confusion_matrix(y_true, y_pred, meta_df, output_dir, module_name):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    meta_df = meta_df.reset_index(drop=True)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tag_col = next(
        (c for c in ['tag', 'tags', 'Tag', 'Tags'] if c in meta_df.columns), None
    )
    if tag_col is None:
        print(f"Warning: No tag column found in meta_df. Skipping tagged CM for {module_name}.")
        return

    df_analysis = pd.DataFrame({
        'GT':   y_true,
        'Pred': y_pred,
        'Tag':  meta_df[tag_col].fillna('None').astype(str).values
    })

    df_analysis['Tag'] = df_analysis['Tag'].str.split(';')
    df_analysis = df_analysis.explode('Tag')
    df_analysis['Tag'] = df_analysis['Tag'].str.strip().replace('', 'None')

    counts = (
        df_analysis
        .groupby(['GT', 'Tag'])['Pred']
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
        .rename(columns={0: 'Predicted: No Behavior (0)', 1: 'Predicted: Behavior (1)'})
    )

    # Row-normalised version (% of frames in that GT+Tag bucket)
    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    row_labels = [
        f"{'Behavior (1)' if gt == 1 else 'No Behav. (0)'} [{tag}]  (n={counts.loc[(gt,tag)].sum()})"
        for gt, tag in counts.index
    ]
    counts.index = row_labels
    pct.index    = row_labels

    fig, (ax_counts, ax_pct) = plt.subplots(
        1, 2,
        figsize=(18, max(6, len(counts) * 0.5))
    )

    sns.heatmap(
        counts, annot=True, fmt="d", cmap="Purples",
        cbar=True, linewidths=0.5, ax=ax_counts
    )
    ax_counts.set_title(f"Frame Counts\n{module_name}")
    ax_counts.set_ylabel("Ground Truth Condition [Tag]  (n=total frames)")
    ax_counts.set_xlabel("Model Predictions")

    sns.heatmap(
        pct, annot=True, fmt=".1f", cmap="Greens",
        cbar=True, linewidths=0.5, ax=ax_pct,
        vmin=0, vmax=100
    )
    ax_pct.set_title(f"Row-Normalised (% of GT+Tag frames)\n{module_name}")
    ax_pct.set_ylabel("")
    ax_pct.set_xlabel("Model Predictions")

    plt.tight_layout()
    safe_name = module_name.replace("/", "_").replace("\\", "_")
    plt.savefig(os.path.join(output_dir, f"tagged_cm_{safe_name}.png"), dpi=150)
    plt.close()
    print(f"Saved: tagged_cm_{safe_name}.png")
  
def plot_results(metas_dict, preds_dict, scores_dict, log_loss_dict, auprc_dict, 
        importance_dict, roc_dict, raw_prob_dict, shap_dict, output_dir,log_file):
  #plot feature importances
  def logs(msg):
    if log_file:
      with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
  _significance_report(scores_dict, log_loss_dict, auprc_dict, raw_prob_dict,log_file=log_file)
    
  # Box-plot of F1 scores with median annotated           #
  fig, ax = plt.subplots(figsize=(max(6, 3 * len(scores_dict)), 5))
  labels  = list(scores_dict.keys())
  score_data = [scores_dict[m] for m in labels]
  bp = ax.boxplot(score_data, labels=labels, patch_artist=True,
          medianprops=dict(color='red', linewidth=2))
  colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
  for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
  for i, vals in enumerate(score_data, 1):
    med = np.median(vals)
    ax.text(i, med + 0.005, f"{med:.3f}", ha='center', va='bottom',
        fontsize=9, color='red')
  ax.set_title("Model Performance (F1 over Seeds)")
  ax.set_ylabel("F1 Score")
  ax.set_ylim(0, 1.05)
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "f1_boxplot.png"), dpi=150)
  plt.close()
  print("Saved: f1_boxplot.png")
  
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
  
  # F1 Boxplot (ax1)
  labels = list(scores_dict.keys())
  ax1.boxplot([scores_dict[m] for m in labels], labels=labels, patch_artist=True)
  ax1.set_title("F1 Score (Higher is Better)")
  
  # Log Loss Boxplot (ax2)
  ax2.boxplot([log_loss_dict[m] for m in labels], labels=labels, patch_artist=True)
  ax2.set_title("Log Loss (Lower is Better)")
  
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "f1_logloss.png"), dpi=150)
  plt.close()

  # Feature importances                       
  for module_name, imp in importance_dict.items():
    names = np.array(imp["names"])
    mean = imp["mean"]
    std  = imp["std"]

    order = np.argsort(mean)[::-1][:74]  # top 74
    fig, ax = plt.subplots(figsize=(10, len(order) * 0.35))
    y_pos = np.arange(len(order))
    ax.barh(y_pos, mean[order][::-1], xerr=std[order][::-1],
        color='steelblue', alpha=0.8, align='center',
        error_kw=dict(ecolor='black', capsize=3))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[order][::-1], fontsize=8)
    ax.set_xlabel("Mean Decrease in Impurity ( std across seeds)")
    ax.set_title(f"Feature Importances  {module_name} (top 74)")
    plt.tight_layout()
    safe = module_name.replace("/", "_").replace("\\", "_")
    plt.savefig(os.path.join(output_dir, f"feature_importance_{safe}.png"), dpi=150)
    plt.close()
    
    logs(f"Feature importances for {module_name}:" + ", ".join(f"{n} ({m:.3f}{s:.3f})" for n, m, s in zip(names[order][:10], mean[order][:10], std[order][:10])))
    print(f"Saved: feature_importance_{safe}.png")
    
  for module_name, sv in shap_dict.items():
    names   = np.array(sv["names"])
    mean_abs = sv["mean_abs"]
    order   = np.argsort(mean_abs)[::-1][:25]
    fig, ax  = plt.subplots(figsize=(10, max(4, len(order) * 0.35)))
    y_pos   = np.arange(len(order))
    ax.barh(y_pos, mean_abs[order][::-1],
        color='darkorange', alpha=0.8, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[order][::-1], fontsize=8)
    ax.set_xlabel("Mean |SHAP value| (seed-0 pooled, top 25)")
    ax.set_title(f"SHAP Feature Importance  {module_name}")
    plt.tight_layout()
    safe = module_name.replace("/", "_").replace("\\", "_")
    plt.savefig(os.path.join(output_dir, f"shap_importance_{safe}.png"), dpi=150)
    plt.close()
    logs(f"SHAP feature importances for {module_name}:" + ", ".join(f"{n} ({m:.3f})" for n, m in zip(names[order][:10], mean_abs[order][:10])))
    print(f"Saved: shap_importance_{safe}.png")

  # 4. ROC curves (one panel per module, all folds as thin lines +
  #  mean curve interpolated)                     
  n_mods = len(roc_dict)
  fig, axes = plt.subplots(1, n_mods, figsize=(6 * n_mods, 5), squeeze=False)
  for ax, (module_name, roc_list) in zip(axes[0], roc_dict.items()):
    mean_fpr = np.linspace(0, 1, 200)
    interp_tprs = []
    best_thresholds = []
    for rd in roc_list:
      ax.plot(rd["fpr"], rd["tpr"], color='steelblue',
          alpha=0.15, linewidth=0.8)
      interp_tprs.append(np.interp(mean_fpr, rd["fpr"], rd["tpr"]))
      j_scores = rd["tpr"] - rd["fpr"]
      best_idx = np.argmax(j_scores)
      best_thresholds.append(rd["thresholds"][best_idx])
    mean_tpr = np.mean(interp_tprs, axis=0)
    std_tpr  = np.std( interp_tprs, axis=0)
    mean_auc = np.mean([rd["auc"] for rd in roc_list])
    
    std_auc  = np.std( [rd["auc"] for rd in roc_list])
    avg_best_threshold = np.mean(best_thresholds)
    std_threshold = np.std(best_thresholds)
    ax.plot(mean_fpr, mean_tpr, color='navy', linewidth=2,
        label=f"Mean AUC = {mean_auc:.3f}  {std_auc:.3f}\nOpt. Threshold = {avg_best_threshold:.3f}")
    logs(f"ROC AUC for {module_name}: {mean_auc:.3f}  {std_auc:.3f}, Optimal Threshold: {avg_best_threshold:.3f}  {std_threshold:.3f}")
    mean_j_scores = mean_tpr - mean_fpr
    opt_idx = np.argmax(mean_j_scores)
    ax.plot(mean_fpr[opt_idx], mean_tpr[opt_idx], 'ro', markersize=6, label='Best Operating Point')
    ax.fill_between(mean_fpr,
            np.clip(mean_tpr - std_tpr, 0, 1),
            np.clip(mean_tpr + std_tpr, 0, 1),
            alpha=0.2, color='navy')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_title(f"ROC  {module_name}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
  plt.close()
  print("Saved: roc_curves.png")
  
  n_mods = len(raw_prob_dict)
  fig, axes = plt.subplots(1, n_mods, figsize=(5 * n_mods, 5), squeeze=False)
  for ax, (module_name, (y_true, probs, meta_df)) in zip(axes[0], raw_prob_dict.items()):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)

    # F1 iso-curves for reference
    f1_grid = np.linspace(0.1, 0.9, 9)
    for f1_val in f1_grid:
      x = np.linspace(0.01, 1.0, 300)
      y_iso = f1_val * x / (2 * x - f1_val)
      mask = (y_iso >= 0) & (y_iso <= 1)
      ax.plot(x[mask], y_iso[mask], color='gray', alpha=0.25,
          linewidth=0.7, linestyle='--')
      if np.any(mask):
        idx = np.argmin(np.abs(x[mask] - 0.85))
        ax.text(x[mask][idx], y_iso[mask][idx],
            f"F1={f1_val:.1f}", fontsize=6, color='gray', alpha=0.6)

    ax.plot(recall, precision, color='darkorange', linewidth=2,
        label=f"AP = {ap:.3f}")

    preds_pp = post_process_pooled(y_true, probs, meta_df)
    op_f1  = f1_score(y_true, preds_pp)
    pos_mask = (y_true == 1)
    op_prec = preds_pp[pos_mask].sum() / preds_pp.sum() if preds_pp.sum() > 0 else 0
    op_rec  = preds_pp[pos_mask].sum() / pos_mask.sum() if pos_mask.sum() > 0 else 0
    ax.plot(op_rec, op_prec, 'ro', markersize=7,
        label=f"Post-proc. op. pt.\nF1={op_f1:.3f}")

    # Chance line 
    chance = pos_mask.mean()
    ax.axhline(chance, color='navy', linestyle='--', linewidth=0.8,
         label=f"Chance = {chance:.3f}")

    ax.set_title(f"Precision-Recall  {module_name}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "pr_curves.png"), dpi=150)
  plt.close()
  print("Saved: pr_curves.png")

  # Confusion matrices (seed-0 pooled predictions)         #
  n_mods = len(raw_prob_dict)
  fig, axes = plt.subplots(2, n_mods, figsize=(5 * n_mods, 8), squeeze=False)
  for col_idx, (module_name, (y_true, probs, meta_df)) in enumerate(raw_prob_dict.items()):
    preds_pp = post_process_pooled(y_true, probs, meta_df)
    
    # 1. Frame-Level CM (Row 0)
    cm_frames = confusion_matrix(y_true, preds_pp)
    disp_f = ConfusionMatrixDisplay(cm_frames, display_labels=["No Behav.", "Behavior"])
    disp_f.plot(ax=axes[0, col_idx], colorbar=False, cmap='Blues')
    axes[0, col_idx].set_title(f"Frame-Level CM\n{module_name}")
    
    # 2. Event-Level pseudo-CM (Row 1)
    # Using a 0.3 (30%) overlap threshold as a reasonable baseline
    tp_ev, fp_ev, fn_ev = calculate_event_metrics(y_true, preds_pp, meta_df, overlap_threshold=0.3)
    
    # Array structure mapping onto a 2x2 grid visually, setting TN to 0
    cm_events = np.array([[0, fp_ev], 
                          [fn_ev, tp_ev]])
    
    # Custom display handling the structural absence of TN
    disp_e = ConfusionMatrixDisplay(cm_events, display_labels=["No Event", "Event"])
    disp_e.plot(ax=axes[1, col_idx], colorbar=False, cmap='Oranges')
    axes[1, col_idx].set_title(f"Event-Based Counts\n{module_name} (TN=0)")
    
    # Log the event performance metrics explicitly since CM visuals omit TN depth
    ev_precision = tp_ev / (tp_ev + fp_ev) if (tp_ev + fp_ev) > 0 else 0
    ev_recall = tp_ev / (tp_ev + fn_ev) if (tp_ev + fn_ev) > 0 else 0
    ev_f1 = 2 * (ev_precision * ev_recall) / (ev_precision + ev_recall) if (ev_precision + ev_recall) > 0 else 0
    
    plot_tagged_confusion_matrix(y_true, preds_pp, meta_df, output_dir, module_name)
    
    logs(f"\nEvent-Based Metrics for {module_name}:")
    logs(f"  TP={tp_ev}, FP={fp_ev}, FN={fn_ev}")
    logs(f"  Event Precision: {ev_precision:.4f} | Event Recall: {ev_recall:.4f} | Event F1: {ev_f1:.4f}")

  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=150)
  plt.close()
  print("Saved: confusion_matrices.png")
  
  
  

  first_mod = list(metas_dict.keys())[0]
  unique_larvae = metas_dict[first_mod][['source', 'ID']].drop_duplicates()
  cmap = plt.cm.get_cmap('tab10', len(preds_dict))
  
  larva_path = os.path.join(output_dir, "larva")
  os.makedirs(larva_path, exist_ok=True)

  for _, row in unique_larvae.iterrows():
        src, larva_id = row['source'], row['ID']
        fig, ax = plt.subplots(figsize=(15, 4))

        for i, module_name in enumerate(preds_dict.keys()):
            df = metas_dict[module_name].copy()
            
            df['preds'] = preds_dict[module_name].values
            df['probs'] = raw_prob_dict[module_name][1]

            # Filter for the specific larva and sort by time
            mask = (df['source'] == src) & (df['ID'] == larva_id)
            larva_meta = df[mask].sort_values('et').copy()
            
            if larva_meta.empty:
                continue

            # Because they are all in the same DataFrame, they are guaranteed to be the exact same length!
            et_vals   = larva_meta['et'].values.astype(float)
            true_vals = larva_meta['true_behavior'].values.astype(float)
            pred_vals = larva_meta['preds'].values.astype(float)
            prob_vals = larva_meta['probs'].values.astype(float)

            # Insert gaps
            gap_locs = np.where(np.diff(et_vals) > 0.5)[0] + 1
            et_plot    = np.insert(et_vals,   gap_locs, np.nan)
            true_plot  = np.insert(true_vals, gap_locs, np.nan)
            pred_plot  = np.insert(pred_vals, gap_locs, np.nan)
            probs_plot = np.insert(prob_vals, gap_locs, np.nan)

            if i == 0:
                ax.plot(et_plot, true_plot,
                        label='Ground Truth', color='black',
                        alpha=0.2, linewidth=10, drawstyle='steps-post')

            ax.plot(et_plot, pred_plot,
                    label=f'Model: {module_name}',
                    color=cmap(i), linestyle='--',linewidth = 3, drawstyle='steps-post')
            
            ax.plot(et_plot, probs_plot, color=cmap(i), alpha=0.6, linewidth=1, label=f'Prob. {module_name}')

        src_tag = src.replace("/", "_").replace("\\", "_")
        ax.set_title(f"Comparison — Larva {larva_id} ({src_tag})")
        ax.set_ylim(0,1)
        ax.legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(os.path.join(larva_path, f"larva_{src_tag}_{larva_id}_comp.png"), dpi=120)
        plt.close()

  print("Saved: per-larva prediction plots")
    
# MAIN RUN BLOCK

# if __name__ == "__main__":
#   ANN_MASTER = "C:/Users/corna/honours/fresh1/hp_2/data_intermediate/annotation/annotation.csv"
#   SESSION_DIR = "C:/Users/corna/honours/fresh1/hp_2/data_intermediate/annotation"
#   OUT_DIR = r"C:\Users\corna\honours\fresh1\hp_2\data_intermediate\prediction_plots"
  
#   target_ids = [0,1,3] 
#   ctx = get_context(file_ids=target_ids, ann_master_csv=ANN_MASTER, session_dir=SESSION_DIR)
  
#   all_sources = ["GA1", "EA", "GA2"]
# # Define which logic files you want to compare
# slices = {
#   "GA1": slice(None),
#   "EA": slice(0, 586),
#   "GA2": slice(0,203)
# }
# prefixes = list(slices.keys())
# logic_files = ['larva_logic2', 'll4'] 

# gt, reps, scores, log_losses, auprcs,importances, rocs, raw_probs, shaps = run_experiment(
#   ctx, 
#   slices, 
#   prefixes, 
#   logic_files, 
#   num_seeds=5,
#   n_splits=5
# )

# plot_results(gt, reps, scores, importances, rocs, raw_probs, shaps,OUT_DIR)
# print("All experiments complete.")
