#pp.py

from __future__ import annotations

import pandas as pd
import polars as pl
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

EXCLUDE_BEHAVIORS = {"dwelling_old", "nondwelling_old"}

# Context

@dataclass
class PipelineContext:
    """Drop-in replacement for the old PipelineContext.  ll40 only uses
    .long_df and .annotated, so that is all we expose here."""
    long_df:   pd.DataFrame   # every raw frame from the parquet
    annotated: pd.DataFrame   # frames inside annotation intervals (+behavior, +tags)
    ann:       pd.DataFrame   # raw merged annotation table (for inspection)

# Annotation loading

_BEHAVIOR_MAP: Dict[str, str] = {
    # canonical forms
    "dwelling":     "dwelling",
    "dwell":        "dwelling",
    "nondwelling":  "nondwelling",
    "non-dwelling": "nondwelling",
    "non_dwelling": "nondwelling",
    "crawling":     "crawling",
    "crawl":        "crawling",
    "turning":      "nondwelling",   
    "turn":         "nondwelling",
    "arc":          "nondwelling",
}

def _load_ann_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load a single annotation CSV and normalise column names."""
    df = pd.read_csv(path, dtype={"AN": "Int64", "ID": "Int64"})
    renames = {
        "File":        "source",
        "source_file": "source",
        "t0_sec":      "t0",
        "t1_sec":      "t1",
        "behavior":    "Behavior",
        "Tag":         "tags",
        "tag":         "tags",
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})
    if "source" in df.columns:
        df["source"] = df["source"].astype(str).str.strip()
    return df


def load_annotations(
    ann_csv: Union[str, Path],
    session_dir: Optional[Union[str, Path]] = None,
    sources: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    pieces = [_load_ann_csv(ann_csv)]

    if session_dir is not None:
        for p in sorted(Path(session_dir).glob("annotations_*.csv")):
            pieces.append(_load_ann_csv(p))

    ann = pd.concat(pieces, ignore_index=True)

    #minimum viable columns
    required = {"source", "ID", "t0", "t1", "Behavior"}
    missing = required - set(ann.columns)
    if missing:
        raise ValueError(
            f"Annotation table is missing columns: {missing}\n"
            f"  Available: {list(ann.columns)}"
        )

    # Ensure tags column exists
    if "tags" not in ann.columns:
        ann["tags"] = np.nan

    ann["ID"] = pd.to_numeric(ann["ID"], errors="coerce")
    ann["t0"] = pd.to_numeric(ann["t0"], errors="coerce")
    ann["t1"] = pd.to_numeric(ann["t1"], errors="coerce")
    ann = ann.dropna(subset=["ID", "t0", "t1", "Behavior"])
    ann = ann[ann["t1"] > ann["t0"]].reset_index(drop=True)

    ann["Behavior"] = (
        ann["Behavior"].astype(str).str.strip().str.lower()
        .map(lambda s: _BEHAVIOR_MAP.get(s, s))
    )

    if sources is not None:
        ann = ann[ann["source"].isin(sources)].reset_index(drop=True)
        
    ann = ann[~ann["Behavior"].isin(EXCLUDE_BEHAVIORS)].reset_index(drop=True)

    return ann

def annotate_behaviors(
    df: pd.DataFrame,
    ann: pd.DataFrame,
    *,
    id_col:     str = "ID",
    source_col: str = "source",
    et_col:     str = "et",
    t0_col:     str = "t0",
    t1_col:     str = "t1",
    behavior_col: str = "Behavior",
    tags_col:     str = "tags",
) -> pd.DataFrame:

    if ann is None or ann.empty:
        out = df.iloc[0:0].copy()
        out["behavior"] = pd.Series([], dtype="object")
        out["tags"]     = pd.Series([], dtype="object")
        return out

    pieces: List[pd.DataFrame] = []

    for (src, gid), ann_g in ann.groupby([source_col, id_col], sort=False):
        frame_mask = (df[source_col] == src) & (df[id_col] == gid)
        if not frame_mask.any():
            continue

        idx_g  = df.index[frame_mask]
        et_g   = df.loc[idx_g, et_col].to_numpy()

        for _, row in ann_g.iterrows():
            t0  = float(row[t0_col])
            t1  = float(row[t1_col])
            beh = row[behavior_col]
            tag = row[tags_col] if pd.notna(row.get(tags_col)) else np.nan

            sel = (et_g >= t0) & (et_g < t1)
            if not sel.any():
                continue

            chunk = df.loc[idx_g[sel]].copy()
            chunk["behavior"] = beh
            chunk["tags"]     = tag
            pieces.append(chunk)

    if not pieces:
        out = df.iloc[0:0].copy()
        out["behavior"] = pd.Series([], dtype="object")
        out["tags"]     = pd.Series([], dtype="object")
        return out

    out = pd.concat(pieces)

    # If a frame was covered by multiple annotations, last one wins
    
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_values([source_col, id_col, et_col], kind="mergesort")
    return out

# Parquet loading

# Minimum columns ll40 needs from the raw parquet.
# Everything else is ignored to keep memory low (important!)
_LL40_REQUIRED_COLS = [
    "source", "ID", "et", "x", "y",
    *(f"xspine_{i}" for i in range(11)),
    *(f"yspine_{i}" for i in range(11)),
]


def load_parquet(
    path: Union[str, Path],
    sources: Optional[Sequence[str]] = None,
    extra_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    
    path = str(path)
    lf = pl.scan_parquet(path)
    available = lf.collect_schema().names()

    keep_cols = list(_LL40_REQUIRED_COLS)
    if extra_cols:
        keep_cols += [c for c in extra_cols if c not in keep_cols]

    # Only select columns that actually exist
    keep_cols = [c for c in keep_cols if c in available]
    missing = set(_LL40_REQUIRED_COLS) - set(keep_cols)
    if missing:
        print(f"[parquet_pipeline] WARNING: parquet is missing columns {missing}")

    lf = lf.select(keep_cols)

    if sources is not None:
        lf = lf.filter(pl.col("source").is_in(list(sources)))

    df = lf.collect().to_pandas()
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    df = df.sort_values(["source", "ID", "et"]).reset_index(drop=True)
    return df


# entry point

def get_context(
    parquet_path:  Union[str, Path],
    ann_csv:       Union[str, Path],
    session_dir:   Optional[Union[str, Path]] = None,
    sources:       Optional[Sequence[str]] = None,
    extra_cols:    Optional[List[str]] = None,
) -> PipelineContext:
    print("Loading raw trajectories from parquet...")
    long_df = load_parquet(parquet_path, sources=sources, extra_cols=extra_cols)
    print(f"  → {len(long_df):,} frames, {long_df['ID'].nunique()} tracks, "
          f"sources: {sorted(long_df['source'].unique())}")

    print("Loading annotations...")
    ann = load_annotations(ann_csv, session_dir=session_dir, sources=sources)
    print(f"  → {len(ann)} annotation intervals")

    # Behavior breakdown
    if not ann.empty:
        counts = ann["Behavior"].value_counts()
        for beh, n in counts.items():
            print(f"      {beh}: {n}")

    print("Labelling frames...")
    annotated = annotate_behaviors(long_df, ann)
    print(f"  → {len(annotated):,} labeled frames")
    if not annotated.empty:
        beh_counts = annotated["behavior"].value_counts()
        for beh, n in beh_counts.items():
            print(f"      {beh}: {n:,} frames")

    return PipelineContext(long_df=long_df, annotated=annotated, ann=ann)

def validate_context(ctx: PipelineContext) -> None:
    print("\n=== Context Validation ===")

    issues = []

    # 1. Required columns in long_df
    for col in _LL40_REQUIRED_COLS:
        if col not in ctx.long_df.columns:
            issues.append(f"long_df missing column: {col}")

    # 2. Annotated has behavior + tags
    for col in ("behavior", "tags"):
        if col not in ctx.annotated.columns:
            issues.append(f"annotated missing column: {col}")

    # 3. Dwelling frames have tags
    if "behavior" in ctx.annotated.columns and "tags" in ctx.annotated.columns:
        dwell = ctx.annotated[ctx.annotated["behavior"] == "dwelling"]
        n_no_tag = dwell["tags"].isna().sum()
        if n_no_tag > 0:
            print(f"  WARNING: {n_no_tag} dwelling frames have no tag. "
                  f"they will be ignored by ll40's tag filter.")

        nd = ctx.annotated[ctx.annotated["behavior"] == "nondwelling"]
        print(f"  Dwelling frames:    {len(dwell):,}")
        print(f"  Nondwelling frames: {len(nd):,}")

        if "tags" in ctx.annotated.columns:
            print("  Tag distribution (nondwelling):")
            if not nd.empty:
                for tag, cnt in nd["tags"].value_counts().items():
                    print(f"    {tag}: {cnt:,}")

    # 4. source/ID overlap between long_df and annotated
    ann_ids = set(zip(ctx.annotated["source"], ctx.annotated["ID"]))
    raw_ids = set(zip(ctx.long_df["source"],   ctx.long_df["ID"]))
    orphans = ann_ids - raw_ids
    if orphans:
        issues.append(
            f"{len(orphans)} annotated (source, ID) pairs not found in long_df: "
            f"{list(orphans)[:5]}..."
        )

    if issues:
        msg = "\n".join(f"  ✗ {i}" for i in issues)
        raise ValueError(f"Context validation failed:\n{msg}")

    print("  ✓ All checks passed.\n")
