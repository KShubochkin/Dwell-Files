import pandas as pd
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def get_coords(data_path, required_cols, extra_cols, sources):
    path = str(data_path)
    lf = pl.scan_parquet(path)
    available = lf.collect_schema().names()

    keep_cols = list(required_cols)
    if extra_cols is not None:
        keep_cols += [c for c in extra_cols if c not in keep_cols]

    keep_cols = [c for c in keep_cols if c in available]
    missing = set(required_cols) - set(keep_cols)
    if missing:
        print(f"[parquet_pipeline] WARNING: parquet is missing columns {missing}")

    lf = lf.select(keep_cols)

    if sources is not None:
        lf = lf.filter(pl.col("source").is_in(list(sources)))

    df = lf.collect().to_pandas()
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    df = df.sort_values(["source", "ID", "et"]).reset_index(drop=True)
    return df

def merge(coords, preds):
    preds = preds.sort_values(["source", "ID", "et"]).reset_index(drop=True)
    coords = coords.sort_values(["source", "ID", "et"]).reset_index(drop=True)
    
    # Explicitly merging on the shared tracking keys is safer
    data = pd.merge(coords, preds, on=["source", "ID", "et"], how="left")
    return data

def get_intervals(df: pd.DataFrame):
    blocks = df['prediction'] != df['prediction'].shift()
    df['label'] = blocks.cumsum() - 1  # 0-indexed alignment
    
    dwell_first = False
    if df['prediction'].iloc[0] == 1:
        dwell_first = True

    n_labels = df['label'].nunique()
    times = np.zeros((n_labels, 2))
    
    for idx, label in enumerate(df['label'].unique()):
        l = df.loc[df['label'] == label]
        start = l['et'].min()
        end = l['et'].max()
        times[idx] = np.array([start, end])
    
    intervals = {}
    if not dwell_first:
        intervals["dwelling"] = times[1::2]
        intervals["nondwelling"] = times[::2]
    else:
        intervals["nondwelling"] = times[1::2]
        intervals["dwelling"] = times[::2]
    
    return intervals

def dwell_proportion_total(data: pd.DataFrame, plot_dir):
    path = Path(plot_dir)
    sources = data["source"].unique()
    
    proportions = {}
    for src in sources:
        df = data.loc[data['source'] == src]
        proportions[src] = df['prediction'].mean()
        
    fig, ax = plt.subplots(figsize=(6, 4)) 
    ax.bar(sources, [proportions[s] for s in sources])
    ax.set_title('Total Proportion of Recording Time Spent Dwelling')
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Condition")
    
    fig.savefig(path / "DwellPropBar.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

def dwell_proportion_violin(data: pd.DataFrame, plot_dir, reducer="mean"):
    path = Path(plot_dir)
    sources = data["source"].unique()
    
    to_plot = {}
    for src in sources:
        df_src = data.loc[data['source'] == src]
        ids = df_src["ID"].unique()
        proportions = []
        
        for id in ids:
            df_id = df_src.loc[df_src['ID'] == id]
            if reducer == "mean":
                proportions.append(df_id['prediction'].mean())
            elif reducer == "std":
                proportions.append(df_id['prediction'].std())
            elif reducer == "median":
                proportions.append(df_id['prediction'].median())
        to_plot[src] = proportions
    
    fig, ax = plt.subplots(figsize=(6, 4)) 
    dataset = [to_plot[s] for s in sources]
    
    ax.violinplot(dataset)
    ax.set_xticks(range(1, len(sources) + 1))
    ax.set_xticklabels(sources)
    
    ax.set_title(f'{reducer.capitalize()} Proportion of Time Spent Dwelling Per Larva')
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Condition")
    
    fig.savefig(path / "DwellPropViolin.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

def behavior_length_violin(data: pd.DataFrame, plot_dir):
    path = Path(plot_dir)
    sources = data["source"].unique()
    
    to_plot = {}
    for src in sources:
        df_src = data.loc[data['source'] == src]
        ids = df_src["ID"].unique()
        lengths = []
        
        for id in ids:
            df_id = df_src.loc[df_src['ID'] == id]
            intervals = get_intervals(df_id)
            for event in intervals["dwelling"]:
                duration = event[1] - event[0]
                lengths.append(duration)
        to_plot[src] = lengths
        
    fig, ax = plt.subplots(figsize=(6, 4))
    dataset = [to_plot[s] for s in sources]
    
    # Fixed: Corrected violin plotting steps
    ax.violinplot(dataset)
    ax.set_xticks(range(1, len(sources) + 1))
    ax.set_xticklabels(sources)
    
    ax.set_title("Average Length Per Dwell Event")
    ax.set_ylabel("Length (s)")
    ax.set_xlabel("Condition")
    
    fig.savefig(path / "DwellLengthViolin.png", bbox_inches="tight", dpi=300)
    plt.close(fig)  
            
def dwell_proportion_log_violin(data: pd.DataFrame, plot_dir):
    """
     Log-transformed Violin plot.
    Adds a tiny epsilon offset to manage true 0.0 values, effectively
    stretching out the micro-differences at the bottom of the axis.
    """
    path = Path(plot_dir)
    sources = data["source"].unique()
    epsilon = 0.001
    
    dataset = []
    for src in sources:
        df_src = data.loc[data['source'] == src]
        proportions = [df_src.loc[df_src['ID'] == i, 'prediction'].mean() for i in df_src["ID"].unique()]
        log_proportions = np.log(np.array(proportions) + epsilon)
        dataset.append(log_proportions)
        
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.violinplot(dataset)
    ax.set_xticks(range(1, len(sources) + 1))
    ax.set_xticklabels(sources)
    
    ax.set_title('Log-Scaled Mean Dwell Proportion Per Larva (with 0.001 offset)')
    ax.set_ylabel("$\log_{10}(\text{Proportion} + \epsilon)$")
    ax.set_xlabel("Condition")
    
    fig.savefig(path / "DwellPropViolin_Log10.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    
def dwell_proportion_log_box(data: pd.DataFrame, plot_dir):
    
    path = Path(plot_dir)
    sources = data["source"].unique()
    epsilon = 0.001
    
    dataset = []
    for src in sources:
        df_src = data.loc[data['source'] == src]
        proportions = [df_src.loc[df_src['ID'] == i, 'prediction'].mean() for i in df_src["ID"].unique()]
        log_proportions = np.log(np.array(proportions) + epsilon)
        dataset.append(log_proportions)
        
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(dataset)
    ax.set_xticks(range(1, len(sources) + 1))
    ax.set_xticklabels(sources)
    
    ax.set_title('Log-Scaled Mean Dwell Proportion Per Larva (with 0.001 offset)')
    ax.set_ylabel("$\log_{10}(\text{Proportion} + \epsilon)$")
    ax.set_xlabel("Condition")
    
    fig.savefig(path / "DwellPropBox_Log.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

def dwell_proportion_cube_rt_box(data: pd.DataFrame, plot_dir):
    
    path = Path(plot_dir)
    sources = data["source"].unique()
    
    dataset = []
    for src in sources:
        df_src = data.loc[data['source'] == src]
        proportions = [df_src.loc[df_src['ID'] == i, 'prediction'].mean() for i in df_src["ID"].unique()]
        log_proportions = np.cbrt(np.array(proportions))
        dataset.append(log_proportions)
        
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(dataset)
    ax.set_xticks(range(1, len(sources) + 1))
    ax.set_xticklabels(sources)
    
    ax.set_title('Cube Rooted Mean Dwell Proportion Per Larva')
    ax.set_ylabel("Cube Root(Proportion + \epsilon)$")
    ax.set_xlabel("Condition")
    
    fig.savefig(path / "DwellPropBox_CubeRt.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

def dwell_proportion_raincloud(data: pd.DataFrame, plot_dir):
    """
    Matplotlib Raincloud Plot.
    Combines asymmetric violins (clouds) with jittered raw data points (rain)
    and a boxplot underneath to expose the true individual counts near zero.
    """
    path = Path(plot_dir)
    sources = data["source"].unique()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for i, src in enumerate(sources):
        df_src = data.loc[data['source'] == src]
        proportions = np.array([df_src.loc[df_src['ID'] == i, 'prediction'].mean() for i in df_src["ID"].unique()])
        
        pos = i + 1
        vp = ax.violinplot([proportions], positions=[pos], showextrema=False, widths=0.4)
        for body in vp['bodies']:
            m = np.mean(body.get_paths()[0].vertices[:, 0])
            body.get_paths()[0].vertices[:, 0] = np.clip(body.get_paths()[0].vertices[:, 0], m, np.inf)
            body.set_facecolor('C0')
            body.set_alpha(0.4)
            
        ax.boxplot([proportions], positions=[pos], widths=0.1, showfliers=False, 
                   manage_ticks=False, boxprops=dict(mfc='none', color='black'))
        
        jitter = np.random.normal(pos - 0.2, 0.04, size=len(proportions))
        ax.scatter(jitter, proportions, alpha=0.5, s=15, color='C0', edgecolor='none')
        
    ax.set_xticks(range(1, len(sources) + 1))
    ax.set_xticklabels(sources)
    ax.set_title('Raincloud Plot: Individual Dwell Proportions')
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Condition")
    
    fig.savefig(path / "DwellProp_Raincloud.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

def dwell_two_part_analysis(data: pd.DataFrame, plot_dir):
    """
    Two-Part Analysis.
    Generates a 1x2 grid: Left subplot measures the percentage of larvae that 
    dwelled at all. Right subplot checks the average dwell rate ONLY among active larvae
    """
    path = Path(plot_dir)
    sources = data["source"].unique()
    
    pct_active = []
    active_proportions = []
    
    for src in sources:
        df_src = data.loc[data['source'] == src]
        proportions = np.array([df_src.loc[df_src['ID'] == i, 'prediction'].mean() for i in df_src["ID"].unique()])
        
        # Segment data
        active = proportions[proportions > 0]
        pct_active.append((len(active) / len(proportions)) * 100)
        active_proportions.append(active if len(active) > 0 else [0])
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(sources, pct_active, color='teal', alpha=0.7)
    ax1.set_title("% of Larvae That Dwelled At All")
    ax1.set_ylabel("Percent (%)")
    ax1.set_xlabel("Condition")
    
    ax2.violinplot(active_proportions)
    ax2.set_xticks(range(1, len(sources) + 1))
    ax2.set_xticklabels(sources)
    ax2.set_title("Mean Proportion (Excluding True Zeros)")
    ax2.set_ylabel("Proportion")
    ax2.set_xlabel("Condition")
    
    fig.tight_layout()
    fig.savefig(path / "Dwell_TwoPartAnalysis.png", dpi=300)
    plt.close(fig)

def dwell_proportion_zoomed_box(data: pd.DataFrame, plot_dir, y_limit=0.15):
    """
    Zoom View.
     boxplot
    and crops the y-axis to focus exclusively on your small-scale micro shifts.
    """
    path = Path(plot_dir)
    sources = data["source"].unique()
    
    dataset = []
    for src in sources:
        df_src = data.loc[data['source'] == src]
        proportions = [df_src.loc[df_src['ID'] == i, 'prediction'].mean() for i in df_src["ID"].unique()]
        dataset.append(proportions)
        
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(dataset, labels=sources, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='blue', alpha=0.6))
    
    ax.set_ylim(-0.005, y_limit) 
    
    ax.set_title(f'Zoomed Mean Dwell Proportion (Capped at {y_limit})')
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Condition")
    
    ax.text(0.5, y_limit * 0.93, "*Outliers extending up to 1.0 omitted for scale*", 
            color='red', fontsize=9, transform=ax.transData, style='italic')
    
    fig.savefig(path / "DwellProp_ZoomedBox.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
def dwell_frequency_heavy_jitter(data: pd.DataFrame, plot_dir):
    """
    dual-axis jitter 
    and overlays a clean median marker to show population density.
    """
    path = Path(plot_dir)
    sources = data["source"].unique()
    counts_data = []
    for src in sources:
        df_src = data.loc[data['source'] == src]
        for idx in df_src["ID"].unique():
            df_id = df_src.loc[df_src['ID'] == idx]
            intervals = get_intervals(df_id)
            num_events = len(intervals["dwelling"])
            counts_data.append({"Condition": src, "Event_Count": num_events})
            
    df_counts = pd.DataFrame(counts_data)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.boxplot(data=df_counts, x="Condition", y="Event_Count", ax=ax, 
                showfliers=False, color="white", width=0.4,
                boxprops=dict(edgecolor='gray'), whiskerprops=dict(color='gray'))
    
    df_counts['Jittered_Count'] = df_counts['Event_Count'] + np.random.uniform(-0.15, 0.15, size=len(df_counts))
    
    sns.stripplot(data=df_counts, x="Condition", y="Jittered_Count", ax=ax, 
                  alpha=0.25, size=4, jitter=0.3, palette="tab10")
    
    ax.set_title("Density Cloud of Dwell Events Per Larva")
    ax.set_ylabel("Number of Events (Jittered)")
    ax.set_yticks(range(8))
    ax.set_yticklabels(range(8)) 
    
    fig.savefig(path / "Dwell_Frequency_Jittered.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
def dwell_frequency_percentage_bar(data: pd.DataFrame, plot_dir):
    """
    Plots the distribution of dwell event counts as a normalized 
    percentage bar chart to completely eliminate overplotting.
    """
    path = Path(plot_dir)
    sources = data["source"].unique()
    
    counts_data = []
    for src in sources:
        df_src = data.loc[data['source'] == src]
        for idx in df_src["ID"].unique():
            df_id = df_src.loc[df_src['ID'] == idx]
            intervals = get_intervals(df_id)
            num_events = len(intervals["dwelling"])
            counts_data.append({"Condition": src, "Event_Count": num_events})
            
    df_counts = pd.DataFrame(counts_data)
    
    # Calculate percentages per condition
    # Cap counts at 4 o 5 if you want to clean up  outliers
    df_counts['Event_Count_Capped'] = df_counts['Event_Count'].apply(lambda x: str(x) if x < 4 else '4+')
    
    pivot_df = pd.crosstab(df_counts['Condition'], df_counts['Event_Count_Capped'], normalize='index') * 100
    pivot_df = pivot_df.reindex(sources)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot_df.plot(kind='bar', ax=ax, edgecolor='black', width=0.8)
    
    ax.set_title("Distribution of Dwell Event Counts Per Larva")
    ax.set_ylabel("Percentage of Larvae (%)")
    ax.set_xlabel("Condition")
    ax.set_xticklabels(sources, rotation=0)
    ax.legend(title="Number of Events")
    ax.grid(True, axis='y', ls="--", alpha=0.3)
    
    fig.savefig(path / "Dwell_Frequency_Percentages.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
def dwell_event_duration_analysis(data: pd.DataFrame, plot_dir):
    """
    Extracts every single individual dwell event duration across all larvae 
    to see if conditions change the physical length
    """
    path = Path(plot_dir)
    sources = data["source"].unique()
    
    event_data = []
    for src in sources:
        df_src = data.loc[data['source'] == src]
        for idx in df_src["ID"].unique():
            df_id = df_src.loc[df_src['ID'] == idx]
            intervals = get_intervals(df_id)
            for event in intervals["dwelling"]:
                duration = event[1] - event[0]
                event_data.append({"Condition": src, "Duration_Sec": duration})
                
    df_events = pd.DataFrame(event_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.boxplot(data=df_events, x="Condition", y="Duration_Sec", ax=ax1, palette="Set2")
    ax1.set_yscale('log')
    ax1.set_title("Distribution of Single Dwell Durations")
    ax1.set_ylabel("Duration (Seconds, Log Scale)")
    
    sns.pointplot(data=df_events, x="Condition", y="Duration_Sec", 
                  errorbar=("ci", 95), capsize=0.1, join=False, ax=ax2, color="black")
    ax2.set_title("Mean Event Duration with 95% CI")
    ax2.set_ylabel("Duration (Seconds, Linear Scale)")
    
    fig.tight_layout()
    fig.savefig(path / "Dwell_Event_Durations.png", dpi=300)
    plt.close(fig)
    
def dwell_frequency_per_larva(data: pd.DataFrame, plot_dir):
    """
    Counts the total number of distinct dwell events initiated by each larva.
    """
    path = Path(plot_dir)
    sources = data["source"].unique()
    
    counts_data = []
    for src in sources:
        df_src = data.loc[data['source'] == src]
        for idx in df_src["ID"].unique():
            df_id = df_src.loc[df_src['ID'] == idx]
            intervals = get_intervals(df_id)
            num_events = len(intervals["dwelling"])
            counts_data.append({"Condition": src, "Event_Count": num_events})
            
    df_counts = pd.DataFrame(counts_data)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df_counts, x="Condition", y="Event_Count", ax=ax, showfliers=False, color="white")
    sns.stripplot(data=df_counts, x="Condition", y="Event_Count", ax=ax, alpha=0.4, jitter=0.2, palette="tab10")
    
    ax.set_title("Number of Dwell Events Initiated Per Larva")
    ax.set_ylabel("Number of Events")
    
    fig.savefig(path / "Dwell_Frequency_Per_Larva.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
def time_to_first_dwell(data: pd.DataFrame, plot_dir):
    """
    Measures the relative latency to the first dwell event for each larva
    by calculating time elapsed since its individual tracking  began.
    """
    path = Path(plot_dir)
    sources = data["source"].unique()
    
    latency_data = []
    for src in sources:
        df_src = data.loc[data['source'] == src]
        for idx in df_src["ID"].unique():
            df_id = df_src.loc[df_src['ID'] == idx]
            
            tracking_start_time = df_id['et'].min()
            
            intervals = get_intervals(df_id)
            if len(intervals["dwelling"]) > 0:
                raw_first_start = intervals["dwelling"][0][0]
                
                relative_latency = raw_first_start - tracking_start_time
                
                latency_data.append({"Condition": src, "Latency_Sec": relative_latency})
                
    df_latency = pd.DataFrame(latency_data)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.pointplot(data=df_latency, x="Condition", y="Latency_Sec", 
                  errorbar=("ci", 95), capsize=0.1, join=False, ax=ax, color="purple")
    
    ax.set_title("Relative Latency to First Dwell Event (95% CI)")
    ax.set_ylabel("Seconds Elapsed Since Individual Tracking Started")
    ax.grid(True, ls="--", alpha=0.3)
    
    fig.savefig(path / "Latency_To_First_Dwell_Fixed.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
def dwell_magnitude_confidence_intervals(data: pd.DataFrame, plot_dir):
    """
    Plots the MEAN dwell proportion of active larvae with proper 
    95% Confidence Interval error bars on a linear scale.
    """
    path = Path(plot_dir)
    
    larva_data = []
    for src in data["source"].unique():
        df_src = data.loc[data['source'] == src]
        for idx in df_src["ID"].unique():
            prop = df_src.loc[df_src['ID'] == idx, 'prediction'].mean()
            if prop > 0: # Active larvae only
                larva_data.append({"Condition": src, "Dwell_Proportion": prop})
    
    df_active = pd.DataFrame(larva_data)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.pointplot(data=df_active, x="Condition", y="Dwell_Proportion", 
                  errorbar=("ci", 95), capsize=0.1, join=False, color="crimson", ax=ax)
    
    ax.set_title("Mean Dwell Proportion (Active Larvae Only) with 95% CI")
    ax.set_ylabel("Mean Dwell Proportion (Linear Scale 0-1)")
    ax.set_ylim(0.2, 0.6) 
    ax.grid(True, ls="--", alpha=0.3)
    
    fig.savefig(path / "Dwell_Magnitude_95CI.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
def dwell_proportion_ecdf_log(data: pd.DataFrame, plot_dir):
    """
    Empirical Cumulative Distribution Function (ECDF) with Pseudo-Log Scale.
    """
    path = Path(plot_dir)
    sources = data["source"].unique()
    epsilon = 1e-4 
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for src in sources:
        df_src = data.loc[data['source'] == src]
        # Calculate mean prediction per unique larva
        proportions = np.array([df_src.loc[df_src['ID'] == i, 'prediction'].mean() for i in df_src["ID"].unique()])
        
        pseudo_log_props = proportions + epsilon
        
        x = np.sort(pseudo_log_props)
        y = np.arange(1, len(x) + 1) / len(x)
        
        # Plot steps
        ax.step(x, y, label=src, where='post', linewidth=2)
        
    ax.set_xscale('log')
    ax.set_title('ECDF of Mean Dwell Proportion Per Larva (Log Scale)')
    ax.set_xlabel(f'Dwell Proportion (Log Scale; shifted by +{epsilon})')
    ax.set_ylabel('Proportion of Population (Percentile)')
    
    
    ax.set_xlim(epsilon * 0.9, 1.1)
    ax.legend(title="Condition")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    fig.savefig(path / "DwellProp_ECDF_Log.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def dwell_two_part_analysis(data: pd.DataFrame, plot_dir):
    """
    Two-Part (Hurdle) Analysis.
    Generates a 1x2 grid: 
    Left subplot: % of larvae that performed ANY dwelling behavior (Proportion > 0).
    Right subplot: Box/Violin distribution of rates ONLY among those active larvae.
    """
    path = Path(plot_dir)
    sources = data["source"].unique()
    
    pct_active = []
    active_proportions = []
    
    for src in sources:
        df_src = data.loc[data['source'] == src]
        proportions = np.array([df_src.loc[df_src['ID'] == i, 'prediction'].mean() for i in df_src["ID"].unique()])
        
        active = proportions[proportions > 0]
        
        pct_active.append((len(active) / len(proportions)) * 100)
        
        active_proportions.append(active if len(active) > 0 else np.array([0.0]))
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    bars = ax1.bar(sources, pct_active, color='teal', alpha=0.7, edgecolor='black', width=0.6)
    ax1.set_title("1. Larvae That Dwelled At All")
    ax1.set_ylabel("Percentage of Total Larvae (%)")
    ax1.set_xlabel("Condition")
    ax1.set_ylim(0, 105)
    
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax2.boxplot(active_proportions, labels=sources, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue', alpha=0.6))
    
    ax2.set_yscale('log')
    ax2.set_title("2. Dwell Magnitude (Active Larvae Only)")
    ax2.set_ylabel("Mean Dwell Proportion (Log Scale)")
    ax2.set_xlabel("Condition")
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(path / "Dwell_TwoPartAnalysis_Fixed.png", dpi=300)
    plt.close(fig)
    

def _calculate_heading_angle(df: pd.DataFrame) -> pd.Series:
    """
    Helper function
    """
    head_x_col = 'xspine_0' if 'xspine_0' in df.columns else 'spinex_0'
    head_y_col = 'yspine_0' if 'yspine_0' in df.columns else 'spiney_0'
    tail_x_col = 'xspine_10' if 'xspine_10' in df.columns else 'spinex_10'
    tail_y_col = 'yspine_10' if 'yspine_10' in df.columns else 'spiney_10'
    
    return np.arctan2(df[head_y_col] - df[tail_y_col], df[head_x_col] - df[tail_x_col])

def _wrap_angle(angles):
    """Wraps angles to the interval [-pi, pi]"""
    return (angles + np.pi) % (2 * np.pi) - np.pi

def prob_dwell_heatmap(data: pd.DataFrame, plot_dir):
    """
    Plots a 2D density heatmap of overall dwelling frames across the arena 
    for each condition. Shows where larvae spend the bulk of their dwelling time.
    """
    path = Path(plot_dir)
    sources = data["source"].unique()
    
    df_dwelling = data[data['prediction'] == 1]
    
    fig, axes = plt.subplots(1, len(sources), figsize=(4 * len(sources), 4), sharex=True, sharey=True)
    if len(sources) == 1: axes = [axes]
        
    for ax, src in zip(axes, sources):
        df_src = df_dwelling[df_dwelling['source'] == src]
        
        if len(df_src) > 10:
            counts, xedges, yedges, im = ax.hist2d(df_src['x'], df_src['y'], bins=30, cmap='viridis', cmin=1)
            ax.set_title(f"{src} (n={len(df_src)} frames)")
        else:
            ax.text(0.5, 0.5, "Insufficient Data", ha='center', va='center')
            ax.set_title(src)
            
        ax.set_xlabel("X coordinate")
    
    axes[0].set_ylabel("Y coordinate")
    fig.suptitle("Overall 2D Spatial Density of Dwelling Behavior", y=1.05, fontsize=14, weight='bold')
    fig.savefig(path / "Spatial_Dwell_Occupancy_Heatmap.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def prob_dwell_start_heatmap(data: pd.DataFrame, plot_dir):
    """
    plots the exact 2D coordinates where a dwelling event starts.
    Reveals if certain odor concentration zones trigger the decision to stop and cast.
    """
    path = Path(plot_dir)
    sources = data["source"].unique()
    
    start_coords = []
    for src in sources:
        df_src = data[data['source'] == src]
        for idx in df_src["ID"].unique():
            df_id = df_src[df_src['ID'] == idx].copy()
            
            is_start = (df_id['prediction'] == 1) & (df_id['prediction'].shift(1) == 0)
            df_starts = df_id[is_start]
            
            for _, row in df_starts.iterrows():
                start_coords.append({"Condition": src, "x": row['x'], "y": row['y']})
                
    df_starts_all = pd.DataFrame(start_coords)
    
    fig, axes = plt.subplots(1, len(sources), figsize=(4 * len(sources), 4), sharex=True, sharey=True)
    if len(sources) == 1: axes = [axes]
        
    for ax, src in zip(axes, sources):
        df_src = df_starts_all[df_starts_all['Condition'] == src]
        
        if len(df_src) > 5:
            ax.scatter(df_src['x'], df_src['y'], color='darkred', alpha=0.5, s=15, edgecolor='none')
            ax.set_title(f"{src} (n={len(df_src)} events)")
        else:
            ax.text(0.5, 0.5, "No Events Started", ha='center', va='center')
            ax.set_title(src)
            
        ax.set_xlabel("X coordinate")
        
    axes[0].set_ylabel("Y coordinate")
    fig.suptitle("2D Spatial Coordinates of Dwell Event Initiations", y=1.05, fontsize=14, weight='bold')
    fig.savefig(path / "Spatial_Dwell_Initiation_Heatmap.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

def prop_dwell_started_distance_histogram(data: pd.DataFrame, plot_dir, odor="right"):
    """
    Plots the frequency of dwell initiations along the chemical gradient (X-axis).
    Since odor is uniformly on the right, high X values represent high odor concentration.
    """
    path = Path(plot_dir)
    sources = data["source"].unique()
    
    start_x_coords = []
    for src in sources:
        df_src = data[data['source'] == src]
        for idx in df_src["ID"].unique():
            df_id = df_src[df_src['ID'] == idx]
            is_start = (df_id['prediction'] == 1) & (df_id['prediction'].shift(1) == 0)
            for x_val in df_id.loc[is_start, 'x']:
                start_x_coords.append({"Condition": src, "X_Position": x_val})
                
    df_x_starts = pd.DataFrame(start_x_coords)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=df_x_starts, x="X_Position", hue="Condition", common_norm=False, bw_adjust=0.7, linewidth=2.5, ax=ax)
    
    ax.set_title("Dwell Initiation Density Along the Olfactory Gradient")
    ax.set_xlabel("Position Along Arena Axis (Odor Source Located Continuously at Right Edge ->)")
    ax.set_ylabel("Relative Initiation Density")
    ax.grid(True, ls="--", alpha=0.3)
    
    fig.savefig(path / "Odor_Gradient_Dwell_Initiation.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def dirs_before_after_polar(data: pd.DataFrame, plot_dir):
    """
    polar coordinate rose-plot showing body alignment before vs after a dwelling event to check if the head-casting sequence redirects their bodies.
    """
    path = Path(plot_dir)
    sources = data["source"].unique()
    
    # Pre-calculate absolute body heading for the whole dataset
    data = data.copy()
    data['heading'] = _calculate_heading_angle(data)
    
    angles_before = {src: [] for src in sources}
    angles_after = {src: [] for src in sources}
    
    for src in sources:
        df_src = data[data['source'] == src]
        for idx in df_src["ID"].unique():
            df_id = df_src[df_src['ID'] == idx].reset_index(drop=True)
            if len(df_id) < 5: continue
            
            # Find index blocks of dwelling behavior
            blocks = df_id['prediction'] != df_id['prediction'].shift()
            df_id['block_labels'] = blocks.cumsum() - 1
            
            for block in df_id['block_labels'].unique():
                sub_block = df_id[df_id['block_labels'] == block]
                if sub_block['prediction'].iloc[0] == 1:  # It's a dwell block
                    start_idx = sub_block.index[0]
                    end_idx = sub_block.index[-1]
                    
                    if start_idx > 0:
                        angles_before[src].append(df_id.loc[start_idx - 1, 'heading'])
                    if end_idx < len(df_id) - 1:
                        angles_after[src].append(df_id.loc[end_idx + 1, 'heading'])
                        
    fig, axes = plt.subplots(2, len(sources), figsize=(3.5 * len(sources), 7), subplot_kw={'projection': 'polar'})
    if len(sources) == 1: axes = np.expand_dims(axes, axis=1)
        
    for i, src in enumerate(sources):
        ax_bef = axes[0, i]
        if len(angles_before[src]) > 0:
            ax_bef.hist(angles_before[src], bins=16, range=(-np.pi, np.pi), color='navy', alpha=0.6, edgecolor='black')
        ax_bef.set_title(f"{src}\nBefore Dwell", fontsize=10)
        ax_bef.set_xticklabels([]) # Keep polar clean
        
        ax_aft = axes[1, i]
        if len(angles_after[src]) > 0:
            ax_aft.hist(angles_after[src], bins=16, range=(-np.pi, np.pi), color='orange', alpha=0.6, edgecolor='black')
        ax_aft.set_title(f"After Dwell", fontsize=10)
        
    fig.suptitle("Anatomical Orientation Distribution (0 rad = Pointing Straight at Odor Source)", y=1.02, weight='bold')
    fig.tight_layout()
    fig.savefig(path / "Body_Direction_Before_After_Polar.png", dpi=300)
    plt.close(fig)


def _extract_turn_dynamics(data: pd.DataFrame, sources, odor="right"):
    """ helper to calculate angular steering values across all events."""
    data = data.copy()
    data['heading'] = _calculate_heading_angle(data)
    
    records = []
    for src in sources:
        df_src = data[data['source'] == src]
        for idx in df_src["ID"].unique():
            df_id = df_src[df_src['ID'] == idx].reset_index(drop=True)
            blocks = (df_id['prediction'] != df_id['prediction'].shift()).cumsum() - 1
            
            for b in blocks.unique():
                sub = df_id[blocks == b]
                if sub['prediction'].iloc[0] == 1:
                    s_idx, e_idx = sub.index[0], sub.index[-1]
                    if s_idx > 0 and e_idx < len(df_id) - 1:
                        theta_bef = df_id.loc[s_idx - 1, 'heading']
                        theta_aft = df_id.loc[e_idx + 1, 'heading']
                        x_pos = df_id.loc[s_idx, 'x']
                        
                        delta_theta = _wrap_angle(theta_aft - theta_bef)
                        
                        err_bef = np.abs(_wrap_angle(theta_bef - 0.0))
                        err_aft = np.abs(_wrap_angle(theta_aft - 0.0))
                        
                        alignment_improvement = err_bef - err_aft
                        
                        records.append({
                            "Condition": src,
                            "Abs_Turn_Magnitude": np.abs(delta_theta),
                            "Steering_Improvement": alignment_improvement,
                            "X_Position": x_pos
                        })
    return pd.DataFrame(records)


def delta_dir_violin(data: pd.DataFrame, plot_dir, odor="right"):
    """
    Plots the absolute angular change (|Delta Theta|) executed during a dwell state.
    Measures 'Reorientation Intensity'—how sharply they turned during the casting sequence.
    """
    path = Path(plot_dir)
    sources = data["source"].unique()
    df_turns = _extract_turn_dynamics(data, sources, odor)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(data=df_turns, x="Condition", y="Abs_Turn_Magnitude", palette="pastel", ax=ax, inner="quartile")
    
    ax.set_title("Total Reorientation Magnitude Executed During Dwell State")
    ax.set_ylabel("Absolute Turning Angle |$\Delta\\theta$| (Radians)")
    ax.set_xlabel("Condition")
    ax.set_yticklabels([f"{val:.1f} rad" for val in ax.get_yticks()])
    ax.grid(True, ls="--", alpha=0.3)
    
    fig.savefig(path / "Dwell_Turn_Magnitude_Violin.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def delta_dir_closer_violin(data: pd.DataFrame, plot_dir, odor="right"):
    """
    Measures Navigational Guidance Accuracy. Plots the change in angular error relative to the odor.
    Values > 0 indicate the larva successfully used the dwell to align closer to the target vector.
    """
    path = Path(plot_dir)
    sources = data["source"].unique()
    df_turns = _extract_turn_dynamics(data, sources, odor)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    sns.violinplot(data=df_turns, x="Condition", y="Steering_Improvement", palette="vlag", ax=ax)
    ax.axhline(0, color='black', ls='--', alpha=0.7, linewidth=1.5)
    
    ax.set_title("Odor Steering Vector Correction via Dwelling")
    ax.set_ylabel("Error Reduction Vector ($\Delta$ Alignment Error)\n[ > 0 Means Turned Closer to Odor ]")
    ax.set_xlabel("Condition")
    ax.grid(True, ls="--", alpha=0.3)
    
    fig.savefig(path / "Dwell_Odor_Steering_Accuracy.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def delta_dir_closer_distance_histogram(data: pd.DataFrame, plot_dir, odor="right"):
    """ steering correction performance across the spatial coordinates of the gradient.
    Uses spatial binning to replace individual line noise with clean, smooth trend lines.
    """
    path = Path(plot_dir)
    sources = data["source"].unique()
    df_turns = _extract_turn_dynamics(data, sources, odor)
    
    
    bin_size = 10
    df_turns['X_Bin'] = (df_turns['X_Position'] // bin_size) * bin_size + (bin_size / 2)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    sns.lineplot(data=df_turns, x="X_Bin", y="Steering_Improvement", hue="Condition", 
                 linewidth=3, marker="o", markersize=6, errorbar=("ci", 95), ax=ax)
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.6, linewidth=1.5)
    
    ax.set_title("Closed-Loop Steering Accuracy Along the Odor Gradient", weight='bold', fontsize=12)
    ax.set_xlabel("X Location of Dwell Action (Odor Source Located Continuously at Right ->)")
    ax.set_ylabel("Steering Correction Magnitude\n[ Higher Means More Accurate Turns Toward Odor ]")
    ax.set_xlim(-5, 205)
    ax.grid(True, ls="--", alpha=0.3)
    
    fig.savefig(path / "Closed_Loop_Steering_vs_Distance_Fixed.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
