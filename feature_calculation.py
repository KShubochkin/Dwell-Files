from typing import List
import numpy as np
import pandas as pd
from scipy.signal import spectrogram, welch
from scipy.spatial import ConvexHull
from numpy.lib.stride_tricks import sliding_window_view
import numba
import gc

#EDITS: ll39 with modified filtering for work with Karen

CONFIG = {
    "dwelling_tags": ["wonderful"],
    "nondwelling_ratio_to_dwelling": 4.0, 
    "nondwelling_tag_ratios": {
        "crawl": 1,
        "turn": 1,
        "arc": 3
    },
    "windows": [11, 30, 50, 75],
    "max_window_size": 75,
    "pause_threshold": 0.3,
    "min_coverage": 0.4,
    "fps": 6.0
}

def calc_angle_vec(p1x, p1y, p2x, p2y, p3x, p3y):
    """Vectorized calculation of bending angle."""
    v1x, v1y = p1x - p2x, p1y - p2y
    v2x, v2y = p3x - p2x, p3y - p2y
    dot = v1x * v2x + v1y * v2y
    mag = np.sqrt(v1x**2 + v1y**2) * np.sqrt(v2x**2 + v2y**2)
    return np.degrees(np.arccos(np.clip(dot / (mag + 1e-6), -1.0, 1.0)))

@numba.njit
def calc_revisitation_metric(x, y, w):
    n = len(x)
    out = np.zeros(n, dtype=np.float32)
    for i in range(w, n):
        min_dist = 1e9
        for j in range(1, w):
            dist = np.sqrt((x[i] - x[i-j])**2 + (y[i] - y[i-j])**2)
            if dist < min_dist:
                min_dist = dist
        out[i] = min_dist
    return out

def get_hull_area(pts):
    if len(pts) < 3: return 0.0
    try:
        return ConvexHull(pts).area # Use .area for 2D perimeter/surface
    except:
        return 0.0

def get_windowed_freq(signal, fps, window_seconds):
    """Uses STFT to get dominant frequency across the whole signal."""
    nperseg = int(window_seconds * fps)
    if len(signal) < nperseg:
        return np.zeros(len(signal))
    
    # noverlap = nperseg - 1 gives us a value for every single frame
    f, t, Sxx = spectrogram(signal, fs=fps, nperseg=nperseg, noverlap=nperseg-1, mode='magnitude')
    
    # Find max frequency at each time step
    max_freq_indices = np.argmax(Sxx, axis=0)
    dom_freq = f[max_freq_indices]
    
    # Spectrogram output is slightly shorter due to windowing, pad to match original
    pad_width = len(signal) - len(dom_freq)
    return np.pad(dom_freq, (pad_width // 2, (pad_width + 1) // 2), mode='edge')

def has_target_tag(tag_string, target_tags):
    if pd.isna(tag_string): return False
    tags = [t.strip() for t in str(tag_string).split(';')]
    return any(t in tags for t in target_tags)

def prepare_ml_dataset(context, windows=[11, 30, 50, 75],
                       fps: float=6, id_slice=slice(None), file_str=[], pause_threshold: float = 0.3,min_coverage=0.4):

    df = context.annotated.copy()
    float_cols = df.select_dtypes(include=['float64']).columns 
    df[float_cols] = df[float_cols].astype('float32')
    
    valid_behaviors = ['dwelling', 'nondwelling', 'crawling']
    df = df[df['behavior'].isin(valid_behaviors)].copy()
    
    print("Filtering data...")
    #make sure id is within provided slice
    if file_str:
        source_mask = df['source'].str.startswith(tuple(file_str))
    else:
        source_mask = pd.Series(True, index=df.index)

    df['ID'] = pd.to_numeric(df['ID'], errors='coerce')
    
    if isinstance(id_slice, dict):
        # Initialize a mask of all False
        slice_mask = pd.Series(False, index=df.index)
        for src_prefix, s in id_slice.items():
            # Find rows belonging to this specific source prefix
            src_match = df['source'].str.startswith(src_prefix)
            
            if isinstance(s, slice):
                start = s.start if s.start is not None else df.loc[src_match, 'ID'].min()
                stop = s.stop if s.stop is not None else df.loc[src_match, 'ID'].max()
                current_mask = src_match & df['ID'].between(start, stop)
            else:
                current_mask = src_match & (df['ID'] == s)
            
            # Combine with the master slice mask
            slice_mask |= current_mask
    elif isinstance(id_slice, slice):
        start = id_slice.start if id_slice.start is not None else df['ID'].min()
        stop = id_slice.stop if id_slice.stop is not None else df['ID'].max()
        slice_mask = df['ID'].between(start, stop)
    else:
        slice_mask = df['ID'] == id_slice


    mask = source_mask & slice_mask
    df = df.sort_values(['source', 'ID', 'et']).reset_index(drop=True)

    # Create a unique event ID for contiguous blocks of the same behavior AND same tag within the same track
    #df['event_id'] = (df['behavior'] != df.groupby(['source', 'ID'])['behavior'].shift()).cumsum()
    df['event_id'] = ((df['behavior'] != df.groupby(['source', 'ID'])['behavior'].shift()) |
                      (df['tags'] != df.groupby(['source', 'ID'])['tags'].shift())).cumsum()
    

    # 3. Filter Dwelling by Tag
    print("Selecting Dwelling frames based on tags...")
    is_valid_dweller = df.apply(lambda row: row['behavior'] == 'dwelling' and has_target_tag(row['tags'], CONFIG['dwelling_tags']), axis=1)
    dwellers_df = df[is_valid_dweller].copy()
    total_dwelling_frames = len(dwellers_df)
    
    # 4. Filter Non-Dwelling by Tag & Ratio
    print("Sampling Non-Dwelling events based on tags and ratios...")
    target_nd_frames = int(total_dwelling_frames * CONFIG['nondwelling_ratio_to_dwelling'])
    
    ratio_dict = CONFIG['nondwelling_tag_ratios']
    total_weight = sum(ratio_dict.values())
    nd_frame_targets = {tag: int(target_nd_frames * (weight / total_weight)) for tag, weight in ratio_dict.items()}
    
    nd_df = df[df['behavior'] == 'nondwelling'].copy()
    rng = np.random.default_rng(seed=42)
    selected_nd_indices = set()
    
    for tag, target_frames in nd_frame_targets.items():
        # Find all events that contain this specific non-dwelling tag
        tag_mask = nd_df['tags'].apply(lambda x: has_target_tag(x, [tag]))
        available_events = nd_df[tag_mask]['event_id'].unique()
        rng.shuffle(available_events)
        
        accumulated_frames = 0
        for event in available_events:
            if accumulated_frames >= target_frames:
                break
            
            event_indices = nd_df[nd_df['event_id'] == event].index
            selected_nd_indices.update(event_indices)
            accumulated_frames += len(event_indices)
            
        print(f"Tag '{tag}': Collected {accumulated_frames}/{target_frames} target frames.")

    # Combine chosen labels
    df_sampled = pd.concat([dwellers_df, nd_df.loc[selected_nd_indices]])
    
    # 
    print("Padding selected events with unannotated frames from raw data...")
    pad_seconds = CONFIG['max_window_size'] / 2.0
    
    # Group our selected frames by their continuous events
    selected_events = df_sampled.groupby('event_id')
    padded_chunks = []
    
    raw_df = context.long_df.copy()
    
    for event_id, event_data in selected_events:
        src = event_data['source'].iloc[0]
        trk_id = event_data['ID'].iloc[0]
        start_et = event_data['et'].min() - pad_seconds
        end_et = event_data['et'].max() + pad_seconds
        
        # Pull the padded chunk from the RAW data
        chunk = raw_df[(raw_df['source'] == src) & 
                       (raw_df['ID'] == trk_id) & 
                       (raw_df['et'] >= start_et) & 
                       (raw_df['et'] <= end_et)].copy()
        
        # Tag the core frames so we know what to keep after the math
        chunk['is_target_annotation'] = chunk['et'].between(event_data['et'].min(), event_data['et'].max())
        # Merge back the behavior labels for the target frames
        chunk = chunk.merge(event_data[['et', 'behavior']], on='et', how='left')
        
        padded_chunks.append(chunk)
        
    df = pd.concat(padded_chunks).sort_values(['source', 'ID', 'et']).reset_index(drop=True)
    
    # 1. Instantaneous Base Metrics
    print("Calculating base metrics...")
    g_inst = df.groupby(['source', 'ID'])
    
    # df['_traj_len']   = g_inst['et'].transform('count')
    # df['_frame_rank'] = g_inst.cumcount()                        # 0-based position from start
    # df['_frame_rank_rev'] = df['_traj_len'] - df['_frame_rank'] - 1  # frames from end
    
    df['bending'] = calc_angle_vec(df['xspine_0'], df['yspine_0'], 
                                   df['xspine_5'], df['yspine_5'], 
                                   df['xspine_10'], df['yspine_10'])
    
    df['bending_vel'] = g_inst['bending'].diff() * fps
    
    
    df['v_head'] = np.sqrt(g_inst['xspine_0'].diff()**2 + g_inst['yspine_0'].diff()**2) * fps
    df['v_mid']  = np.sqrt(g_inst['xspine_5'].diff()**2 + g_inst['yspine_5'].diff()**2) * fps
    df['v_tail'] = np.sqrt(g_inst['xspine_10'].diff()**2 + g_inst['yspine_10'].diff()**2) * fps
    df['v_com'] = np.sqrt(g_inst['x'].diff()**2 + g_inst['y'].diff()**2) * fps
    
    df['is_paused'] = (df['v_com'] < pause_threshold).astype(int)
    df['has_neighbor'] = (df['is_paused'] == 1) & (
        (df.groupby(['source', 'ID'])['is_paused'].shift(1) == 1) | 
        (df.groupby(['source', 'ID'])['is_paused'].shift(-1) == 1)
    )
    
    # Body Orientation (Tail to Head)
    # Using atan2 to get the absolute angle relative to the background (x-axis)
    df['angle_body'] = np.arctan2(df['yspine_0'] - df['yspine_10'], 
                                  df['xspine_0'] - df['xspine_10']).fillna(0)
    
    # Head Orientation (Mid to Head)
    df['angle_head'] = np.arctan2(df['yspine_0'] - df['yspine_5'], 
                                  df['xspine_0'] - df['xspine_5']).fillna(0)
    
    # CoM Heading (Movement Direction)
    # Note: This is the direction the CoM is moving, not the direction it's facing
    dx_com = g_inst['x'].diff()
    dy_com = g_inst['y'].diff()
    df['angle_heading'] = np.arctan2(dy_com, dx_com)
    
    # Calculate Angular Velocity (omega)
    def get_omega(angles, fps):
        # Unwrap handles the -pi to pi jump
        angles_clean = np.nan_to_num(angles.to_numpy(), nan=0.0)
        unwrapped = np.unwrap(angles_clean)
        grad = np.gradient(unwrapped) * fps
        return grad

    # Apply within groups to avoid bleeding between different trajectories
    df['omega_body'] = g_inst['angle_body'].transform(lambda x: get_omega(x, fps)).abs()
    df['omega_head'] = g_inst['angle_head'].transform(lambda x: get_omega(x, fps)).abs()
    df['omega_heading'] = g_inst['angle_heading'].transform(lambda x: get_omega(x, fps)).abs()
    
    # Internal Angular Velocity (Change in bending angle over time)
    df['omega_relative'] = g_inst['bending'].diff().abs() * fps
    
    angular_cols = ['omega_body', 'omega_head', 'omega_heading']
    df[angular_cols] = df.groupby(['source', 'ID'])[angular_cols].ffill().bfill().fillna(0)

    def unwrap_group(x):
        return np.unwrap(np.nan_to_num(x.to_numpy(), nan=0.0))

    df['angle_body_unwrapped'] = g_inst['angle_body'].transform(unwrap_group)
    
    # Body length normalization
    df['body_len'] = np.sqrt((df['xspine_0']-df['xspine_10'])**2 + (df['yspine_0']-df['yspine_10'])**2)
    group_medians = g_inst['body_len'].transform('median')
    df['v_mid_norm'] = df['v_mid'] / (group_medians + 1e-6)
    
    df['ht_ratio'] = (df['v_head'] + 1e-3) / (df['v_tail'] + 1e-3)
    df['hc_ratio'] = (df['v_head'] + 1e-3) / (df['v_mid'] + 1e-3)
    df['bending_diff'] = g_inst['bending'].diff().abs() * fps
    df['high_bend_activity'] = (df['bending_diff'] > df['bending_diff'].median()).astype(int)
    
    df['is_peak'] = (
        (df['bending'] > g_inst['bending'].shift(1)) & 
        (df['bending'] > g_inst['bending'].shift(-1))
    ).astype(int)
    
    X_list = []
    
    print("Calculating windowed features...")
    for s in windows:
        print(f"Processing window size: {s}s")
        w = int(s * fps)
        p = f"w{s}_"
        half_w = w//2
        
        g_win = g_inst.rolling(window=w, min_periods=1, center=True)
        
        win_feat = pd.DataFrame(index=df.index)
                
        if s ==11:
            win_feat[f'{p}omega_relative_mean'] = g_win['omega_relative'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            win_feat[f'{p}omega_body_mean'] = g_win['omega_body'].mean().reset_index(level=[0,1], drop=True).astype('float32').astype('float32')            
            win_feat[f'{p}bending_std'] = g_win['bending'].std().reset_index(level=[0,1], drop=True).fillna(0).astype('float32')
            win_feat[f'{p}hc_ratio_mean'] = g_win['hc_ratio'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            win_feat[f'{p}ht_ratio_mean'] = g_win['ht_ratio'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            win_feat[f'{p}omega_head_std'] = g_win['omega_head'].std().reset_index(level=[0,1], drop=True).fillna(0).astype('float32')
            rog_x = g_win['x'].var().reset_index(level=[0,1], drop=True).fillna(0).astype('float32')
            rog_y = g_win['y'].var().reset_index(level=[0,1], drop=True).fillna(0).astype('float32')
            win_feat[f'{p}rog'] = np.sqrt(rog_x + rog_y)
            shift_len = int(s/6 * fps)
            groups_ser = df['source'] + "_" + df['ID'].astype(str)
            temp_g = win_feat.groupby(groups_ser)
            win_feat[f'{p}omega_body_mean_slope_smooth'] = (
                temp_g[f'{p}omega_body_mean'].shift(-shift_len).ffill() - 
            temp_g[f'{p}omega_body_mean'].shift(shift_len).bfill()).astype('float32')
            first_x = g_inst['x'].shift(half_w).fillna(df['x'])
            last_x  = g_inst['x'].shift(-half_w).fillna(df['x'])
            first_y = g_inst['y'].shift(half_w).fillna(df['y'])
            last_y  = g_inst['y'].shift(-half_w).fillna(df['y'])

            disp     = np.sqrt((last_x - first_x)**2 + (last_y - first_y)**2)
            path_len = g_win['v_com'].sum().reset_index(level=[0,1], drop=True) / fps
            epsilon  = group_medians * 0.1
            win_feat[f'{p}tortuosity'] = path_len / (disp + epsilon)
            win_feat[f'{p}msd']        = disp**2 / s

        
        elif s == 30:
            win_feat[f'{p}omega_heading_mean'] = g_win['omega_heading'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            win_feat[f'{p}omega_relative_mean'] = g_win['omega_relative'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            win_feat[f'{p}vel_mean'] = g_win['v_com'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            win_feat[f'{p}vel_norm_mean'] = g_win['v_mid_norm'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            win_feat[f'{p}omega_body_mean'] = g_win['omega_body'].mean().reset_index(level=[0,1], drop=True).astype('float32').astype('float32')
            win_feat[f'{p}omega_head_std'] = g_win['omega_head'].std().reset_index(level=[0,1], drop=True).fillna(0).astype('float32')
            rog_x = g_win['x'].var().reset_index(level=[0,1], drop=True).fillna(0).astype('float32')
            rog_y = g_win['y'].var().reset_index(level=[0,1], drop=True).fillna(0).astype('float32')
            win_feat[f'{p}rog'] = np.sqrt(rog_x + rog_y)
            win_feat[f'{p}head_vel_std'] = g_win['v_head'].std().reset_index(level=[0,1], drop=True).fillna(0).astype('float32')
            win_feat[f'{p}head_vel_mean'] = g_win['v_head'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            win_feat[f'{p}vel_std'] = g_win['v_com'].std().reset_index(level=[0,1], drop=True).fillna(0).astype('float32')
            first_x = g_inst['x'].shift(half_w).fillna(df['x'])
            last_x  = g_inst['x'].shift(-half_w).fillna(df['x'])
            first_y = g_inst['y'].shift(half_w).fillna(df['y'])
            last_y  = g_inst['y'].shift(-half_w).fillna(df['y'])

            disp     = np.sqrt((last_x - first_x)**2 + (last_y - first_y)**2)
            path_len = g_win['v_com'].sum().reset_index(level=[0,1], drop=True) / fps
            epsilon  = group_medians * 0.1
            win_feat[f'{p}tortuosity'] = path_len / (disp + epsilon)
            win_feat[f'{p}msd']        = disp**2 / s
            win_feat[f'{p}pause_run_frac'] = (
                g_win['has_neighbor'].mean() / (g_win['is_paused'].mean() + 1e-6)
            ).reset_index(level=[0, 1], drop=True)
            
            win_feat[f'{p}reversal_rate'] = g_win['high_bend_activity'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            
            total_angular_path = g_win['omega_body'].sum().reset_index(level=[0,1], drop=True) / fps
            first_ang = g_inst['angle_body_unwrapped'].shift(half_w).fillna(df['angle_body_unwrapped'])
            last_ang  = g_inst['angle_body_unwrapped'].shift(-half_w).fillna(df['angle_body_unwrapped'])
            net_angular_change = np.abs(last_ang - first_ang)
            ang_epsilon = 0.01 
            win_feat[f'{p}angular_tortuosity'] = total_angular_path / (net_angular_change + ang_epsilon)
            
            win_feat[f'{p}coverage'] = g_win['v_com'].count().reset_index(level=[0,1], drop=True) / w 
            raw_peak_sum = g_win['is_peak'].sum().reset_index(level=[0,1], drop=True)
            valid_frame_count = g_win['v_com'].count().reset_index(level=[0,1], drop=True)
            # Calculate Peak Density (Rate)
            win_feat[f'{p}bend_peaks_rate'] = raw_peak_sum / (valid_frame_count + 1e-6)

            # Calculate revisitation per frame within each trajectory group
            # We use the current window 'w' as the lookback N
            df[f'temp_rev_{s}'] = g_inst.apply(lambda g: pd.Series(calc_revisitation_metric(g['x'].values, g['y'].values, w), index=g.index)).reset_index(level=[0,1], drop=True)
            win_feat[f'{p}revisitation_mean'] = g_inst[f'temp_rev_{s}'].rolling(window=w, min_periods=1).mean().reset_index(level=[0,1], drop=True).astype('float32')
            df.drop(columns=[f'temp_rev_{s}'], inplace=True)
            
            shift_len = int(s/6 * fps)
            groups_ser = df['source'] + "_" + df['ID'].astype(str)
            temp_g = win_feat.groupby(groups_ser)
            win_feat[f'{p}revis_slope_smooth'] = (
                temp_g[f'{p}revisitation_mean'].shift(-shift_len).ffill() - temp_g[f'{p}revisitation_mean'].shift(shift_len).bfill()).astype('float32')
        
            shift_len = int(s/6 * fps)
            groups_ser = df['source'] + "_" + df['ID'].astype(str)
            temp_g = win_feat.groupby(groups_ser)
            win_feat[f'{p}rog_slope_smooth'] = (
                temp_g[f'{p}rog'].shift(-shift_len).ffill() - 
            temp_g[f'{p}rog'].shift(shift_len).bfill()).astype('float32')

            shift_len = int(s/6 * fps)
            groups_ser = df['source'] + "_" + df['ID'].astype(str)
            temp_g = win_feat.groupby(groups_ser)
            win_feat[f'{p}tort_slope_smooth'] = (
                temp_g[f'{p}tortuosity'].shift(-shift_len).ffill() - 
            temp_g[f'{p}tortuosity'].shift(shift_len).bfill()).astype('float32')        
        
        elif s == 50:
            win_feat[f'{p}vel_mean'] = g_win['v_com'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            win_feat[f'{p}vel_norm_mean'] = g_win['v_mid_norm'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            win_feat[f'{p}vel_std'] = g_win['v_com'].std().reset_index(level=[0,1], drop=True).fillna(0).astype('float32')
            win_feat[f'{p}head_vel_std'] = g_win['v_head'].std().reset_index(level=[0,1], drop=True).fillna(0).astype('float32')
            win_feat[f'{p}head_vel_mean'] = g_win['v_head'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            win_feat[f'{p}omega_body_mean'] = g_win['omega_body'].mean().reset_index(level=[0,1], drop=True).astype('float32').astype('float32')
            win_feat[f'{p}omega_head_std'] = g_win['omega_head'].std().reset_index(level=[0,1], drop=True).fillna(0).astype('float32')
            win_feat[f'{p}omega_heading_mean'] = g_win['omega_heading'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            win_feat[f'{p}omega_relative_mean'] = g_win['omega_relative'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            win_feat[f'{p}bending_std'] = g_win['bending'].std().reset_index(level=[0,1], drop=True).fillna(0).astype('float32')
            win_feat[f'{p}hc_ratio_mean'] = g_win['hc_ratio'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            win_feat[f'{p}ht_ratio_mean'] = g_win['ht_ratio'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            rog_x = g_win['x'].var().reset_index(level=[0,1], drop=True).fillna(0).astype('float32')
            rog_y = g_win['y'].var().reset_index(level=[0,1], drop=True).fillna(0).astype('float32')
            win_feat[f'{p}rog'] = np.sqrt(rog_x + rog_y)
            first_x = g_inst['x'].shift(half_w).fillna(df['x'])
            last_x  = g_inst['x'].shift(-half_w).fillna(df['x'])
            first_y = g_inst['y'].shift(half_w).fillna(df['y'])
            last_y  = g_inst['y'].shift(-half_w).fillna(df['y'])
            disp     = np.sqrt((last_x - first_x)**2 + (last_y - first_y)**2)
            path_len = g_win['v_com'].sum().reset_index(level=[0,1], drop=True) / fps
            epsilon  = group_medians * 0.1
            win_feat[f'{p}tortuosity'] = path_len / (disp + epsilon)
            win_feat[f'{p}msd']        = disp**2 / s
            win_feat[f'{p}coverage'] = g_win['v_com'].count().reset_index(level=[0,1], drop=True) / w 
            win_feat[f'{p}pause_run_frac'] = (
                            g_win['has_neighbor'].mean() / (g_win['is_paused'].mean() + 1e-6)
                        ).reset_index(level=[0, 1], drop=True)
            raw_peak_sum = g_win['is_peak'].sum().reset_index(level=[0,1], drop=True)
            valid_frame_count = g_win['v_com'].count().reset_index(level=[0,1], drop=True)
            # Calculate Peak Density (Rate)
            win_feat[f'{p}bend_peaks_rate'] = raw_peak_sum / (valid_frame_count + 1e-6)

            # Calculate revisitation per frame within each trajectory group
            # We use the current window 'w' as the lookback N
            df[f'temp_rev_{s}'] = g_inst.apply(lambda g: pd.Series(calc_revisitation_metric(g['x'].values, g['y'].values, w), index=g.index)).reset_index(level=[0,1], drop=True)
            win_feat[f'{p}revisitation_mean'] = g_inst[f'temp_rev_{s}'].rolling(window=w, min_periods=1).mean().reset_index(level=[0,1], drop=True).astype('float32')
            df.drop(columns=[f'temp_rev_{s}'], inplace=True)
            win_feat[f'{p}reversal_rate'] = g_win['high_bend_activity'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            
            total_angular_path = g_win['omega_body'].sum().reset_index(level=[0,1], drop=True) / fps
            first_ang = g_inst['angle_body_unwrapped'].shift(half_w).fillna(df['angle_body_unwrapped'])
            last_ang  = g_inst['angle_body_unwrapped'].shift(-half_w).fillna(df['angle_body_unwrapped'])
            net_angular_change = np.abs(last_ang - first_ang)
            ang_epsilon = 0.01 
            win_feat[f'{p}angular_tortuosity'] = total_angular_path / (net_angular_change + ang_epsilon)
            
            shift_len = int((s / 6) * fps)  # ~1/6 of window size
            groups_ser = df['source'] + "_" + df['ID'].astype(str)
            temp_g = win_feat.groupby(groups_ser)
            win_feat[f'{p}vel_lag'] = temp_g[f'{p}vel_mean'].shift(shift_len).fillna(0)
            win_feat[f'{p}vel_lead'] = temp_g[f'{p}vel_mean'].shift(-shift_len).fillna(0)
            
            shift_len = int(s/6 * fps)
            groups_ser = df['source'] + "_" + df['ID'].astype(str)
            temp_g = win_feat.groupby(groups_ser)
            win_feat[f'{p}revis_slope_smooth'] = (
                temp_g[f'{p}revisitation_mean'].shift(-shift_len).ffill() - temp_g[f'{p}revisitation_mean'].shift(shift_len).bfill()).astype('float32')
        
            shift_len = int(s/6 * fps)
            groups_ser = df['source'] + "_" + df['ID'].astype(str)
            temp_g = win_feat.groupby(groups_ser)
            win_feat[f'{p}rog_slope_smooth'] = (
                temp_g[f'{p}rog'].shift(-shift_len).ffill() - 
            temp_g[f'{p}rog'].shift(shift_len).bfill()).astype('float32')

            shift_len = int(s/6 * fps)
            groups_ser = df['source'] + "_" + df['ID'].astype(str)
            temp_g = win_feat.groupby(groups_ser)
            win_feat[f'{p}tort_slope_smooth'] = (
                temp_g[f'{p}tortuosity'].shift(-shift_len).ffill() - 
            temp_g[f'{p}tortuosity'].shift(shift_len).bfill()).astype('float32')        

            shift_len = int(s/6 * fps)
            groups_ser = df['source'] + "_" + df['ID'].astype(str)
            temp_g = win_feat.groupby(groups_ser)
            win_feat[f'{p}omega_body_mean_slope_smooth'] = (
                temp_g[f'{p}omega_body_mean'].shift(-shift_len).ffill() - 
            temp_g[f'{p}omega_body_mean'].shift(shift_len).bfill()).astype('float32')
        elif s == 75:        
            win_feat[f'{p}vel_mean'] = g_win['v_com'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            win_feat[f'{p}vel_norm_mean'] = g_win['v_mid_norm'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            win_feat[f'{p}vel_std'] = g_win['v_com'].std().reset_index(level=[0,1], drop=True).fillna(0).astype('float32')
            win_feat[f'{p}head_vel_std'] = g_win['v_head'].std().reset_index(level=[0,1], drop=True).fillna(0).astype('float32')
            win_feat[f'{p}head_vel_mean'] = g_win['v_head'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            win_feat[f'{p}omega_body_mean'] = g_win['omega_body'].mean().reset_index(level=[0,1], drop=True).astype('float32').astype('float32')
            win_feat[f'{p}omega_head_std'] = g_win['omega_head'].std().reset_index(level=[0,1], drop=True).fillna(0).astype('float32')
            win_feat[f'{p}omega_heading_mean'] = g_win['omega_heading'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            win_feat[f'{p}omega_relative_mean'] = g_win['omega_relative'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            win_feat[f'{p}bending_std'] = g_win['bending'].std().reset_index(level=[0,1], drop=True).fillna(0).astype('float32')
            win_feat[f'{p}hc_ratio_mean'] = g_win['hc_ratio'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            win_feat[f'{p}ht_ratio_mean'] = g_win['ht_ratio'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            rog_x = g_win['x'].var().reset_index(level=[0,1], drop=True).fillna(0).astype('float32')
            rog_y = g_win['y'].var().reset_index(level=[0,1], drop=True).fillna(0).astype('float32')
            win_feat[f'{p}rog'] = np.sqrt(rog_x + rog_y)
            first_x = g_inst['x'].shift(half_w).fillna(df['x'])
            last_x  = g_inst['x'].shift(-half_w).fillna(df['x'])
            first_y = g_inst['y'].shift(half_w).fillna(df['y'])
            last_y  = g_inst['y'].shift(-half_w).fillna(df['y'])
            disp     = np.sqrt((last_x - first_x)**2 + (last_y - first_y)**2)
            path_len = g_win['v_com'].sum().reset_index(level=[0,1], drop=True) / fps
            epsilon  = group_medians * 0.1
            win_feat[f'{p}tortuosity'] = path_len / (disp + epsilon)
            win_feat[f'{p}msd']        = disp**2 / s
            win_feat[f'{p}coverage'] = g_win['v_com'].count().reset_index(level=[0,1], drop=True) / w 
            win_feat[f'{p}pause_run_frac'] = (
                            g_win['has_neighbor'].mean() / (g_win['is_paused'].mean() + 1e-6)
                        ).reset_index(level=[0, 1], drop=True)
            raw_peak_sum = g_win['is_peak'].sum().reset_index(level=[0,1], drop=True)
            valid_frame_count = g_win['v_com'].count().reset_index(level=[0,1], drop=True)
            # Calculate Peak Density (Rate)
            win_feat[f'{p}bend_peaks_rate'] = raw_peak_sum / (valid_frame_count + 1e-6)

            # Calculate revisitation per frame within each trajectory group
            # We use the current window 'w' as the lookback N
            df[f'temp_rev_{s}'] = g_inst.apply(lambda g: pd.Series(calc_revisitation_metric(g['x'].values, g['y'].values, w), index=g.index)).reset_index(level=[0,1], drop=True)
            win_feat[f'{p}revisitation_mean'] = g_inst[f'temp_rev_{s}'].rolling(window=w, min_periods=1).mean().reset_index(level=[0,1], drop=True).astype('float32')
            df.drop(columns=[f'temp_rev_{s}'], inplace=True)
            win_feat[f'{p}reversal_rate'] = g_win['high_bend_activity'].mean().reset_index(level=[0,1], drop=True).astype('float32')
            
            total_angular_path = g_win['omega_body'].sum().reset_index(level=[0,1], drop=True) / fps
            first_ang = g_inst['angle_body_unwrapped'].shift(half_w).fillna(df['angle_body_unwrapped'])
            last_ang  = g_inst['angle_body_unwrapped'].shift(-half_w).fillna(df['angle_body_unwrapped'])
            net_angular_change = np.abs(last_ang - first_ang)
            ang_epsilon = 0.01 
            win_feat[f'{p}angular_tortuosity'] = total_angular_path / (net_angular_change + ang_epsilon)
            
            shift_len = int((s / 6) * fps)  # ~1/6 of window size
            groups_ser = df['source'] + "_" + df['ID'].astype(str)
            temp_g = win_feat.groupby(groups_ser)
            win_feat[f'{p}vel_lag'] = temp_g[f'{p}vel_mean'].shift(shift_len).fillna(0)
            win_feat[f'{p}vel_lead'] = temp_g[f'{p}vel_mean'].shift(-shift_len).fillna(0)
            
            shift_len = int(s/6 * fps)
            groups_ser = df['source'] + "_" + df['ID'].astype(str)
            temp_g = win_feat.groupby(groups_ser)
            win_feat[f'{p}revis_slope_smooth'] = (
                temp_g[f'{p}revisitation_mean'].shift(-shift_len).ffill() - temp_g[f'{p}revisitation_mean'].shift(shift_len).bfill()).astype('float32')
        
            shift_len = int(s/6 * fps)
            groups_ser = df['source'] + "_" + df['ID'].astype(str)
            temp_g = win_feat.groupby(groups_ser)
            win_feat[f'{p}rog_slope_smooth'] = (
                temp_g[f'{p}rog'].shift(-shift_len).ffill() - 
            temp_g[f'{p}rog'].shift(shift_len).bfill()).astype('float32')

            shift_len = int(s/6 * fps)
            groups_ser = df['source'] + "_" + df['ID'].astype(str)
            temp_g = win_feat.groupby(groups_ser)
            win_feat[f'{p}tort_slope_smooth'] = (
                temp_g[f'{p}tortuosity'].shift(-shift_len).ffill() - 
            temp_g[f'{p}tortuosity'].shift(shift_len).bfill()).astype('float32')        

            shift_len = int(s/6 * fps)
            groups_ser = df['source'] + "_" + df['ID'].astype(str)
            temp_g = win_feat.groupby(groups_ser)
            win_feat[f'{p}omega_body_mean_slope_smooth'] = (
                temp_g[f'{p}omega_body_mean'].shift(-shift_len).ffill() - 
            temp_g[f'{p}omega_body_mean'].shift(shift_len).bfill()).astype('float32')
            df[f'{p}bend_freq'] = g_inst['bending'].transform(lambda x: get_windowed_freq(x, fps, s))
            win_feat[f'{p}bend_freq_rolling'] = df[f'{p}bend_freq'].astype('float32')
        
        win_feat = win_feat.astype('float32') #5:27
        
        X_list.append(win_feat)
        
    del win_feat
    gc.collect()
    
    print("Combining features...")    
    
    X = pd.concat(X_list, axis=1)
    
    y = (df['behavior'] == 'dwelling').astype(int)
    groups = df['source'] + "_" + df['ID'].astype(str)
    X = X.groupby(groups).transform(lambda x: x.ffill().bfill().fillna(0))

    cov_windows = [s for s in windows if s >= 25]  # Only consider larger windows for coverage
    coverage_cols = [f"w{s}_coverage" for s in cov_windows]
    coverage_ok   = (X[coverage_cols] >= min_coverage).all(axis=1)

    valid_mask   = coverage_ok & df['is_target_annotation']

    X_final      = X[valid_mask]
    y_final      = y[valid_mask]
    groups_final = groups[valid_mask]
    meta_final   = df.loc[valid_mask, ['source', 'ID', 'et']]

    # --- Print Representation Summary ---
    print("\n" + "="*65)
    print(f"{'SOURCE':<15} | {'RAW POS':<8} | {'RAW NEG':<8} | {'FED POS':<8} | {'FED NEG':<8}")
    print("-" * 65)
    for src in raw_counts.index:
        # Raw counts (from annotation CSV)
        r_pos = raw_counts.loc[src, 'dwelling']
        r_neg = raw_counts.loc[src].sum() - r_pos
        
        # Fed counts (survived windowing and NaNs)
        src_mask = meta_final['source'] == src
        f_pos = y_final[src_mask].sum()
        f_neg = src_mask.sum() - f_pos
        
        print(f"{src:<15} | {r_pos:<8} | {r_neg:<8} | {f_pos:<8} | {f_neg:<8}")
    
    total_raw = raw_counts.values.sum()
    total_fed = len(X_final)
    print("-"*65)
    print(f"TOTAL FRAMES: Raw Annotated = {total_raw} | Fed to Model = {total_fed}")
    print(f"Retention Rate: {total_fed/total_raw:.1%}")
    print("="*65 + "\n")

    gc.collect()
    return X_final, y_final, groups_final, meta_final
