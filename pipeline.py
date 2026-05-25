# pipeline.py

#from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
from pathlib import Path
import larva_pipeline_minimal_refactor
importlib.reload(larva_pipeline_minimal_refactor)
from larva_pipeline_minimal_refactor import (
    load_files, compute_metrics, annotate_behaviors,
)

# ---------------------
# Public API
# ---------------------

@dataclass
class PipelineContext:
    per_file: List  # list of FileResult (whatever your loader returns)
    long_df: pd.DataFrame
    ann: pd.DataFrame
    annotated: pd.DataFrame

def get_context(
    file_ids: Sequence[int] | Sequence[str],
    ann_master_csv: str | Path,
    session_dir: str | Path | None = None,
) -> PipelineContext:
    """
    Build a fresh pipeline context from the given file IDs and annotations.

    Parameters
    ----------
    file_ids : sequence of IDs (ints/strs) that your loader understands
    ann_master_csv : path to the master annotation CSV
    session_dir : optional directory with 'annotations_*.csv' session files

    Returns
    -------
    PipelineContext(per_file, long_df, ann, annotated)
    """
    # Load raw
    per_file = load_files(list(file_ids))
    
    # Annotations: master + any session files (standardize column names)
    ann_master = pd.read_csv(ann_master_csv, dtype={"AN": "Int64", "ID": "Int64"})
    ann_master = ann_master.rename(
        columns={
            "File": "source",
            "source_file": "source", # Catch this just in case
            "t0_sec": "t0",
            "t1_sec": "t1",
            "behavior": "Behavior"
        }
    )


    def take(canon, alias, cast=None):
        has_canon = canon in df.columns
        has_alias = alias in df.columns
        if has_canon and has_alias:
            if cast:
                df[alias] = pd.to_numeric(df[alias], errors="coerce") if cast=="num" else df[alias].astype(str)
            df[canon] = df[canon].where(df[canon].notna(), df[alias])
            df.drop(columns=[alias], inplace=True)
        elif has_alias and not has_canon:
            df.rename(columns={alias: canon}, inplace=True)


    # for session files, look for incorrect column names and standardize
    ann_pieces: list[pd.DataFrame] = [ann_master]
    if session_dir is not None:
        session_dir = Path(session_dir)
        session_files = sorted(session_dir.glob("annotations_*.csv"))
        for p in session_files:
            df = pd.read_csv(p)
            df = df.rename(
                columns={"t0_sec": "t0", "t1_sec": "t1", "behavior": "Behavior"}
            )
            take("source", "source_file")
            take("source", "File")
            ann_pieces.append(df)

    # combine annotations, then filter to just loaded files
    ann = pd.concat(ann_pieces, ignore_index=True) if ann_pieces else ann_master
    DEFAULT_REGISTRY: Dict[int,str] = {
    0: ("GA1"),
    1: ("GA2"),
    2: ("GA3"),
    3: ("EA"),
    4: ("H2O"),}
    
    # 1. First, make sure the column is named correctly
    if "source" not in ann.columns:
        if "File" in ann.columns:
            ann = ann.rename(columns={"File": "source"})
        elif "source_file" in ann.columns:
            ann = ann.rename(columns={"source_file": "source"})

    # 2. Clean the strings (strip whitespace)
    if "source" in ann.columns:
        ann["source"] = ann["source"].astype(str).str.strip()

    # 3. NOW filter by the loaded IDs
    loaded_files = [DEFAULT_REGISTRY.get(item, item) for item in file_ids]
    ann = ann[ann['source'].isin(loaded_files)]


    # Long DF - compute metrics for all files, concatenate into one big DF
    long_df = pd.concat([r.ecdf for r in per_file], ignore_index=True)
    long_df = compute_metrics(long_df)

    # Tag & return
    annotated = annotate_behaviors(long_df, ann)
    return PipelineContext(
        per_file=per_file,
        long_df=long_df,
        ann=ann,
        annotated=annotated,
    )
    
    #from parquet_writer import write_feature_store

    #root = Path("C:/Users/corna/honours/fresh1/hp_2/data_intermediate/metrics_storage")
    #write_feature_store(long_df, root)


# ---------------------
# Legacy globals (optional)
# ---------------------
# If someone still does: `from pipeline import per_file, long_df, ann, annotated`
# we’ll populate **once** using either an env var or a safe default.
# Set PIPELINE_FILE_IDS="1,2,3" in your env to override.

per_file = None
long_df = None
ann = None
annotated = None

def _init_globals_once():
    global per_file, long_df, ann, annotated
    if per_file is not None:
        return  # already initialized

    file_ids_env = os.getenv("PIPELINE_FILE_IDS", "1")
    # supports comma-separated ints or strings
    file_ids = [int(x) if x.strip().isdigit() else x.strip()
                for x in file_ids_env.split(",") if x.strip()]

    ann_master_csv = "C:/Users/corna/honours/fresh1/hp_2/data_intermediate/annotation/annotation.csv"
    session_dir = "C:/Users/corna/honours/fresh1/hp_2/data_intermediate/annotation"

    ctx = get_context(file_ids=file_ids, ann_master_csv=ann_master_csv, session_dir=session_dir)
    per_file, long_df, ann, annotated = ctx.per_file, ctx.long_df, ctx.ann, ctx.annotated

# initialize the legacy globals at import time
#_init_globals_once()
