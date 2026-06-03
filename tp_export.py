#tp_export.py

import joblib
import numpy as np
import importlib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import psutil
import gc


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


def train(ctx, slices, prefixes, logic_file,model_path,feature_path,seed,cache_path):
    
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
        X, y, groups, meta, log_messages = feature_calc.prepare_ml_dataset(ctx, fps=6, id_slice=slices, file_str=prefixes)

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
        class_weight='balanced_subsample'
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

    return model, list(X.columns), mod_meta, log_messages

def infer(model,ctx,files,features,probabilities_path,cache_path,logic_file):
    feature_calc = importlib.import_module(logic_file)
    importlib.reload(feature_calc)
    
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
            target_tags = feature_calc.CONFIG['dwelling_tags']
            ann_gt['true_label'] = (
                (ann_gt['behavior'] == 'dwelling')
                & ann_gt['tags'].apply(lambda tag_string: feature_calc.has_target_tag(tag_string, target_tags))
            ).astype(int)

            res['ID'] = res['ID'].astype(str)
            if not ann_gt.empty:
                ann_gt['ID'] = ann_gt['ID'].astype(str)
    
            res['et'] = res['et'].astype(np.float64).round(4)
            ann_gt['et'] = ann_gt['et'].astype(np.float64).round(4)

            res = res.merge(ann_gt[['source', 'ID', 'et', 'true_label']], on=['source', 'ID', 'et'], how='left')
            print(f"  Matched {res['true_label'].notna().sum()} annotated frames.")
            
        else:
            res['true_label'] = np.nan

        #res['ID'] = pd.to_numeric(res['ID'], errors='coerce').fillna(0).astype(int).astype(str)

        # Stream to CSV
        res.to_csv(probabilities_path, mode='a', header=not probabilities_path.exists(), index=False)
        print(f"Predicted probabilities saved → {probabilities_path}")
        
        meta_test_list.append(res.copy())

        del sub_df, res, meta, probs
        gc.collect()
        print(f"  RAM free after cleanup: {psutil.virtual_memory().available/1e9:.1f} GB")    
    
    meta_test = pd.concat(meta_test_list, ignore_index=True) if meta_test_list else None
    return meta_test, log_messages

# In tp_export.py -> predict()
def predict(probabilities_path, ppc, metadata, predictions_dir,ctx,logic_file,plot=False): 
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
    
    for source, track_id in unique_tracks:
        track_id_str = str(track_id)
        if (source, track_id_str) not in df_probs.index:
            continue
        
        # Fixed print statement parsing 
        if int(track_id_str) % 250 == 0: 
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
        final_preds_list.append(track_preds_df)
        
    if final_preds_list:
        output_df = pd.concat(final_preds_list, ignore_index=True)
        
        output_df.to_csv(output_path, index=False)
        print(f"Saved complete post-processed predictions -> {output_path}")

        if plot:
            for src in metadata['source'].unique():
                plot_source_grid(output_df, src, predictions_dir, ctx, logic_file, cols=10)

        return output_df
    else:
        print("⚠ WARNING: No matched predictions generated. Check if test IDs match inference sources.")
        return pd.DataFrame()

import matplotlib.pyplot as plt 

def plot_source_grid(results, src, out_dir, ctx, logic_file, cols=4):
    """One figure per source — all larvae as a grid of small prob traces."""
    results = results.copy()
    
    feature_calc = importlib.import_module(logic_file)
    importlib.reload(feature_calc)

    if hasattr(ctx, 'annotated') and not ctx.annotated.empty:
        # 1. Clean and filter annotations
        ann_gt = ctx.annotated[['source', 'ID', 'et', 'behavior']].copy()
        ann_gt['tags'] = ctx.annotated['tags'] if 'tags' in ctx.annotated.columns else np.nan
        ann_gt = ann_gt[ann_gt['behavior'].isin(['dwelling', 'nondwelling'])].copy()

        target_tags = feature_calc.CONFIG['dwelling_tags']
        ann_gt['true_label'] = (
            (ann_gt['behavior'] == 'dwelling')
            & ann_gt['tags'].apply(lambda tag_string: feature_calc.has_target_tag(tag_string, target_tags))
        ).astype(int)

        # Force type alignment
        results['ID'] = results['ID'].astype(str)
        ann_gt['ID'] = ann_gt['ID'].astype(str)
        
        results['et'] = results['et'].astype(np.float32)
        ann_gt['et'] = ann_gt['et'].astype(np.float32)
        
        results = results.sort_values('et')
        ann_gt = ann_gt.sort_values('et')

        merged_pieces = []
        for (s_name, gid), grp in results.groupby(['source', 'ID']):
            ann_sub = ann_gt[(ann_gt['source'] == s_name) & (ann_gt['ID'] == gid)]
            if ann_sub.empty:
                grp['true_label'] = np.nan
                merged_pieces.append(grp)
                continue
                
            merged = pd.merge_asof(
                grp, 
                ann_sub[['et', 'true_label']], 
                on='et', 
                direction='nearest', 
                tolerance=0.05 
            )
            merged_pieces.append(merged)
            
        results = pd.concat(merged_pieces, ignore_index=True)
        print(f"  Matched {results['true_label'].notna().sum()} annotated frames.")
    else:
        results['true_label'] = np.nan

    grp_src = results[results['source'] == src]
    larvae = sorted(grp_src['ID'].unique())
    if len(larvae) == 0:
        print(f"  No tracks found for source {src}. Skipping plot.")
        return

    rows = int(np.ceil(len(larvae) / cols))

    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 4, rows * 1.8),
                              facecolor='#111')
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = np.atleast_1d(axes).ravel()

    for i, lid in enumerate(larvae):
        ax  = axes[i]
        grp = grp_src[grp_src['ID'] == lid].sort_values('et')
        et, prob, pred = grp['et'].values, grp['prob'].values, grp['prediction'].values

        ax.fill_between(et, pred, alpha=0.3, color='#4ade80', step='post')
        ax.plot(et, prob, color='#60a5fa', linewidth=0.6)
        if 'true_label' in grp.columns and grp['true_label'].notna().any():
            gt = grp['true_label'].fillna(0).values
            ax.fill_between(et, gt * -0.2, 0, alpha=0.5, color='#fb923c', step='post')
        ax.set_ylim(-0.25, 1.05)
        ax.set_title(f"ID {lid}", fontsize=7, color='#aaa', pad=2)
        ax.set_facecolor('#111')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes[len(larvae):]:
        ax.set_visible(False)

    fig.suptitle(src, color='#ccc', fontsize=11)
    fig.tight_layout(pad=0.3)
    path = Path(out_dir) / f"grid_{src}.png"
    fig.savefig(path, dpi=120, bbox_inches='tight', facecolor='#111')
    plt.close(fig)
    print(f"  Saved: {path}")
