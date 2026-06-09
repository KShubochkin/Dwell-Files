# feature_store.py
#
# Master feature cache — annotation-free, split-ignorant

# 1. The cache stores EVERY frame for EVERY source, no label info.
#    Downstream code (train/infer) is responsible for joining annotations.
# 2. One parquet per source: features_{source}.parquet
#    Indexed by (source, ID, et)
# 3. A manifest (cache_manifest.json) records exactly which columns exist,
#    the logic-file content hash, the fps used, and a per-column version tag.
#    Any time you add a new feature bump its version; stale columns are
#    flagged automatically.
# 4. Adding new features never rewrites old ones — update_cache() appends
#    only the columns that are missing or version-bumped.
# 5. FeatureSetConfig declares what a model run wants.  It resolves against
#    the cache and triggers a targeted recalculation if anything is missing.
#
# Typical workflow
# ────────────────
# ONCE (or when raw data changes):
#     feature_store.build_full_cache(ctx, logic_file="exp_feature_calculation",
#                                    cache_dir=CACHE_DIR, fps=6.0)
#
# OPTIONAL (new feature added to logic file):
#     feature_store.update_cache(ctx, logic_file="exp_feature_calculation",
#                                cache_dir=CACHE_DIR, fps=6.0,
#                                new_columns=["w30_new_metric", ...])
#     # or just: feature_store.update_cache(...) — it auto-detects missing cols.
#
# PER RUN:
#     fsc = FeatureSetConfig(
#         base_features=["larva_body_length"],
#         windowed_features={"v_com": [30, 75], "omega_body": [11, 50]},
#     )
#     # fsc.validate_against_cache(cache_dir) will auto-fill missing cols
#
#     model, features, meta, log = tp.train(ctx, ..., fsc=fsc, cache_dir=CACHE_DIR)
#     meta_test, log_inf     = tp.infer(model_path, ctx, ..., fsc=fsc, cache_dir=CACHE_DIR)

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_MANIFEST_FILE = "cache_manifest.json"
_KEY_COLS = ["source", "ID", "et"]   # immutable index — never a feature


# ─────────────────────────────────────────────────────────────────────────────
# Manifest helpers
# ─────────────────────────────────────────────────────────────────────────────

def _logic_hash(logic_file: str) -> str:
    """
    MD5 of the logic module's source file — used as a coarse staleness signal.

    For registry-based logic files (those that export `registry_versions()`),
    per-column versioning is preferred; the file hash is only checked as a
    secondary guard.
    """
    mod = importlib.import_module(logic_file)
    src = Path(mod.__file__).read_bytes()
    return hashlib.md5(src).hexdigest()[:12]


def _registry_versions(logic_file: str) -> Optional[Dict[str, int]]:
    """
    Return the per-column version map from a registry-based logic file,
    or None if the logic file doesn't expose registry_versions().
    """
    mod = importlib.import_module(logic_file)
    importlib.reload(mod)
    fn = getattr(mod, "registry_versions", None)
    return fn() if callable(fn) else None


def _stale_columns(cache_dir: Path, logic_file: str) -> List[str]:
    """
    Compare the per-column versions recorded in the manifest against the
    current registry.  Returns a list of column names whose version has
    increased (i.e. the formula changed).

    registry_versions() is keyed by feature name (e.g. "rog"), while the
    manifest stores full column names (e.g. "w30_rog", "w50_rog").  We strip
    the w{N}_ prefix to look up the version.

    Returns [] if the logic file doesn't use the registry system.
    """
    import re as _re
    rv = _registry_versions(logic_file)
    if rv is None:
        return []
    manifest = _read_manifest(cache_dir)
    stale = []
    for col, meta in manifest.get("columns", {}).items():
        cached_ver = meta.get("version", 1)
        m = _re.match(r"^w\d+_(.+)$", col)
        feat_name = m.group(1) if m else col
        current_ver = rv.get(feat_name)
        if current_ver is not None and cached_ver < current_ver:
            stale.append(col)
    return stale


def _read_manifest(cache_dir: Path) -> dict:
    p = cache_dir / _MANIFEST_FILE
    if p.exists():
        return json.loads(p.read_text())
    return {
        "logic_file": None,
        "logic_hash": None,
        "fps": None,
        "columns": {},      # col_name -> {"version": int, "added_at": str}
        "sources": [],      # source names present in cache
        "created_at": None,
        "updated_at": None,
    }


def _write_manifest(cache_dir: Path, manifest: dict) -> None:
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (cache_dir / _MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, default=str)
    )


def _parquet_path(cache_dir: Path, source: str) -> Path:
    safe = source.replace("/", "_").replace("\\", "_")
    return cache_dir / f"features_{safe}.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# Public cache API
# ─────────────────────────────────────────────────────────────────────────────

def build_full_cache(
    ctx,
    logic_file: str,
    cache_dir: Path | str,
    fps: float = 6.0,
    windows: Optional[List[int]] = None,
    sources: Optional[List[str]] = None,
    force: bool = False,
) -> None:
    """
    One-time (or full-refresh) build of the feature cache.

    Runs `calculate()` from `logic_file` over ALL frames in ctx.long_df for
    every source.  No annotation data is touched — the cache is entirely
    label-free and split-ignorant.

    Parameters
    ----------
    ctx         : pipeline context with `.long_df`
    logic_file  : module name, e.g. "exp_feature_calculation"
    cache_dir   : directory to store parquets + manifest
    fps         : recording frame-rate
    sources     : limit to these source IDs (default = all in ctx.long_df)
    force       : overwrite existing source parquets even if present
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    feature_calc = importlib.import_module(logic_file)
    importlib.reload(feature_calc)

    manifest = _read_manifest(cache_dir)
    lhash = _logic_hash(logic_file)

    if manifest["logic_hash"] and manifest["logic_hash"] != lhash and not force:
        print(
            f"[feature_store] WARNING: logic_file hash changed "
            f"({manifest['logic_hash']} → {lhash}).  "
            "Pass force=True to rebuild, or call update_cache() for additive changes."
        )

    all_sources = sources or ctx.long_df["source"].unique().tolist()
    print(f"[feature_store] Building cache for {len(all_sources)} sources → {cache_dir}")

    all_columns: dict[str, dict] = {}

    for src in all_sources:
        dest = _parquet_path(cache_dir, src)
        if dest.exists() and not force:
            print(f"  [{src}] already cached — skipping (pass force=True to overwrite)")
            # still harvest column list from manifest
            continue

        print(f"  [{src}] calculating features …")
        sub = ctx.long_df[ctx.long_df["source"] == src].copy().reset_index(drop=True)
        if sub.empty:
            print(f"  [{src}] WARNING: no rows in long_df — skipping")
            continue

        if not windows:
            raise ValueError(
                "build_full_cache() requires an explicit `windows` list, "
                "e.g. windows=[11, 30, 50, 75].  "
                "There is no longer a registry-derived default."
            )
        X = feature_calc.calculate(
            df=sub,
            fps=fps,
            pause_threshold=feature_calc.CONFIG["pause_threshold"],
            windows=windows,
        )

        # Attach the key columns so the parquet is self-describing
        # (sub may have been mutated by calculate — re-align by position)
        result = sub[_KEY_COLS].reset_index(drop=True).join(
            X.reset_index(drop=True)
        )
        result.to_parquet(dest, index=False)
        print(f"  [{src}] → {dest.name}  ({len(result):,} rows, {len(X.columns)} features)")

        for col in X.columns:
            if col not in all_columns:
                all_columns[col] = {
                    "version": 1,
                    "added_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }

        del sub, X, result
        gc.collect()

    # Merge any previously recorded columns we didn't just compute
    for col, meta in manifest.get("columns", {}).items():
        if col not in all_columns:
            all_columns[col] = meta

    # Overlay per-column versions from the registry.
    # registry_versions() is keyed by feature name; column names are w{N}_feat.
    import re as _re
    rv = _registry_versions(logic_file)
    if rv:
        for col in all_columns:
            m = _re.match(r"^w\d+_(.+)$", col)
            feat_name = m.group(1) if m else col
            ver = rv.get(feat_name)
            if ver is not None:
                all_columns[col]["version"] = ver

    manifest.update({
        "logic_file": logic_file,
        "logic_hash": lhash,
        "fps": fps,
        "columns": all_columns,
        "sources": sorted(set(manifest.get("sources", [])) | set(all_sources)),
        "created_at": manifest.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    _write_manifest(cache_dir, manifest)
    print(f"[feature_store] Manifest written → {cache_dir / _MANIFEST_FILE}")
    print(f"[feature_store] Done.  {len(all_columns)} total feature columns cached.")


def update_cache(
    ctx,
    logic_file: str,
    cache_dir: Path | str,
    fps: float = 6.0,
    new_columns: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    version_bump: Optional[Dict[str, int]] = None,
) -> None:
    """
    Additive update — append new feature columns to existing parquets.

    Pass `new_columns` to limit which columns are added; omit it to let the
    function auto-detect any columns produced by `calculate()` that are not
    yet in the manifest.

    Pass `version_bump` = {"col_name": new_version} to force-recalculate
    columns whose formula changed (the old values are overwritten in-place).

    Parameters
    ----------
    new_columns   : explicit list of column names to add (or None = auto-detect)
    version_bump  : {col_name: new_version_int} to overwrite stale columns
    """
    cache_dir = Path(cache_dir)
    manifest = _read_manifest(cache_dir)

    if not manifest["logic_file"]:
        raise RuntimeError(
            "No manifest found.  Run build_full_cache() first."
        )

    feature_calc = importlib.import_module(logic_file)
    importlib.reload(feature_calc)

    all_sources = sources or ctx.long_df["source"].unique().tolist()
    known_cols = set(manifest["columns"].keys())
    version_bump = version_bump or {}

    print(f"[feature_store] Updating cache for {len(all_sources)} sources …")

    for src in all_sources:
        dest = _parquet_path(cache_dir, src)
        if not dest.exists():
            print(f"  [{src}] no parquet found — run build_full_cache() first, skipping.")
            continue

        existing = pd.read_parquet(dest)
        existing_cols = set(existing.columns) - set(_KEY_COLS)

        sub = ctx.long_df[ctx.long_df["source"] == src].copy().reset_index(drop=True)
        if sub.empty:
            continue

        # ── Determine which columns to (re)compute ────────────────────────────
        if new_columns is None:
            # Auto-detect mode: full calculate() to discover what's missing.
            # This is intentionally expensive — it's for "find everything new"
            # not for targeted updates.  Use explicit new_columns= to avoid it.
            X_calc = feature_calc.calculate(
                df=sub.copy(),
                fps=fps,
                pause_threshold=feature_calc.CONFIG["pause_threshold"],
            )
            cols_to_add  = [c for c in X_calc.columns if c not in existing_cols]
            cols_to_bump = [c for c in version_bump   if c in X_calc.columns]
            cols_needed  = list(set(cols_to_add) | set(cols_to_bump))

            if not cols_needed:
                print(f"  [{src}] already up-to-date — nothing to add.")
                del sub, X_calc
                gc.collect()
                continue

            new_data = X_calc[cols_needed].reset_index(drop=True)
            del X_calc
        else:
            # Explicit mode: only compute exactly the columns requested
            # (plus any version-bumped ones), skipping those already present.
            # Uses calculate_columns() — runs only the needed windows/features,
            # not the full feature matrix.
            cols_to_add  = [c for c in new_columns  if c not in existing_cols]
            cols_to_bump = [c for c in version_bump if c in existing_cols]
            cols_needed  = list(set(cols_to_add) | set(cols_to_bump))

            if not cols_needed:
                print(f"  [{src}] all requested columns already present — skipping.")
                del sub
                gc.collect()
                continue

            print(f"  [{src}] targeted compute for {len(cols_needed)} column(s): {cols_needed}")
            X_calc = feature_calc.calculate_columns(
                df=sub.copy(),
                fps=fps,
                pause_threshold=feature_calc.CONFIG["pause_threshold"],
                columns=cols_needed,
            )
            cols_needed = [c for c in cols_needed if c in X_calc.columns]
            new_data = X_calc[cols_needed].reset_index(drop=True)
            del X_calc

        # ── Merge into existing parquet ───────────────────────────────────────
        for col in cols_needed:
            if col in new_data.columns:
                existing[col] = new_data[col].values

        existing.to_parquet(dest, index=False)
        print(f"  [{src}] added/updated {len(cols_needed)} columns: {cols_needed}")

        # ── Update manifest column registry ───────────────────────────────────
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        for col in cols_needed:
            if col in version_bump:
                manifest["columns"][col] = {
                    "version": version_bump[col],
                    "added_at": now,
                }
            elif col not in manifest["columns"]:
                manifest["columns"][col] = {"version": 1, "added_at": now}

        del sub, existing, new_data
        gc.collect()

    _write_manifest(cache_dir, manifest)
    print("[feature_store] Update complete.")


def list_cached_features(cache_dir: Path | str) -> pd.DataFrame:
    """
    Return a DataFrame summarising every cached feature column.

    Columns: feature_name, version, added_at, windows (list of ints if windowed)
    """
    cache_dir = Path(cache_dir)
    manifest = _read_manifest(cache_dir)
    rows = []
    for col, meta in manifest.get("columns", {}).items():
        # Try to parse window size from name prefix w{N}_
        windows = None
        if col.startswith("w") and "_" in col:
            try:
                windows = int(col.split("_")[0][1:])
            except ValueError:
                pass
        rows.append({
            "feature_name": col,
            "version": meta.get("version", 1),
            "added_at": meta.get("added_at", "unknown"),
            "window_size": windows,
        })
    df = pd.DataFrame(rows).sort_values(["feature_name"])
    return df


def load_source_features(
    cache_dir: Path | str,
    source: str,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load cached features for one source, optionally selecting specific columns.

    Always returns key columns (source, ID, et) + requested feature columns.
    """
    cache_dir = Path(cache_dir)
    dest = _parquet_path(cache_dir, source)
    if not dest.exists():
        raise FileNotFoundError(
            f"No cached features for source '{source}' at {dest}.  "
            "Run build_full_cache() first."
        )
    if columns is not None:
        load_cols = list(set(_KEY_COLS) | set(columns))
        return pd.read_parquet(dest, columns=load_cols)
    return pd.read_parquet(dest)


def check_stale_columns(cache_dir: Path | str, logic_file: str) -> List[str]:
    """
    Return column names whose version in the manifest is lower than the
    current registry version.  Empty list = everything is up-to-date.

    Only meaningful when logic_file uses the registry system (i.e. exposes
    registry_versions()).  Returns [] for legacy logic files.

    Usage
    -----
    stale = check_stale_columns(CACHE_DIR, "feature_registry")
    if stale:
        update_cache(ctx, "feature_registry", CACHE_DIR, fps=FPS,
                     version_bump={col: NEW_VER for col in stale})
    """
    return _stale_columns(Path(cache_dir), logic_file)


def check_cache_health(cache_dir: Path | str, ctx=None, logic_file: str = None) -> None:
    """
    Print a summary of the cache state: sources, column count, missing sources.
    If `ctx` is given, cross-checks against ctx.long_df sources.
    If `logic_file` is given and uses the registry system, also reports stale columns.
    """
    cache_dir = Path(cache_dir)
    manifest = _read_manifest(cache_dir)

    print("=" * 60)
    print(f"Cache directory : {cache_dir}")
    print(f"Logic file      : {manifest.get('logic_file', '—')}")
    print(f"Logic hash      : {manifest.get('logic_hash', '—')}")
    print(f"FPS             : {manifest.get('fps', '—')}")
    print(f"Total columns   : {len(manifest.get('columns', {}))}")
    print(f"Cached sources  : {len(manifest.get('sources', []))}")
    print(f"Last updated    : {manifest.get('updated_at', '—')}")

    parquets = sorted(cache_dir.glob("features_*.parquet"))
    print(f"Parquet files   : {len(parquets)}")

    if ctx is not None:
        ctx_sources = set(ctx.long_df["source"].unique())
        cached = set(manifest.get("sources", []))
        missing = ctx_sources - cached
        extra = cached - ctx_sources
        if missing:
            print(f"\n⚠ Sources in ctx but NOT in cache ({len(missing)}): {sorted(missing)}")
        if extra:
            print(f"  Sources in cache but not in ctx ({len(extra)}): {sorted(extra)}")
        if not missing:
            print("✓ All ctx sources are cached.")

    if logic_file is not None:
        stale = _stale_columns(cache_dir, logic_file)
        if stale:
            print(f"\n⚠ Stale columns (version bumped in registry, {len(stale)} total):")
            for col in stale:
                print(f"    {col}")
            print("  Run update_cache(..., version_bump={col: new_ver for col in stale})"
                  " or rebuild to refresh.")
        else:
            print("✓ All column versions are current.")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# FeatureSetConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureSetConfig:
    """
    Declares which features a model run uses.

    Parameters
    ----------
    base_features : list of non-windowed column names, e.g. ["larva_body_length"]
    windowed_features : {base_name: [window_sizes]}
        e.g. {"v_com": [11, 30, 75], "omega_body": [50]}
        Column names are resolved as f"w{w}_{feat}"
    name : optional human-readable label for this config (used in logs/paths)
    description : optional free-text note about why this feature set was chosen

    Usage
    -----
    fsc = FeatureSetConfig(
        name="baseline_v1",
        base_features=["larva_body_length"],
        windowed_features={
            "omega_body": [11, 30, 50, 75],
            "v_com":      [30, 50, 75],
            "rog":        [30, 50, 75],
            "tortuosity": [11, 30, 50],
        },
    )
    fsc.validate_against_cache(cache_dir)   # auto-fills missing columns
    X = fsc.select_from(df_with_all_features)
    """

    base_features: List[str] = field(default_factory=list)
    windowed_features: Dict[str, List[int]] = field(default_factory=dict)
    name: str = "unnamed"
    description: str = ""

    # ── Derived helpers ────────────────────────────────────────────────────────

    def get_all_columns(self) -> List[str]:
        """Full flat list of expected feature column names, in declaration order."""
        cols: List[str] = list(self.base_features)
        for feat, windows in self.windowed_features.items():
            for w in sorted(windows):
                cols.append(f"w{w}_{feat}")
        # deduplicate while preserving order
        seen: set[str] = set()
        out: List[str] = []
        for c in cols:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def get_windows_for_feature(self, feat: str) -> List[int]:
        return sorted(self.windowed_features.get(feat, []))

    def summary(self) -> str:
        lines = [
            f"FeatureSetConfig '{self.name}'",
            f"  Description   : {self.description or '—'}",
            f"  Base features : {self.base_features}",
            f"  Windowed      :",
        ]
        for feat, wins in self.windowed_features.items():
            lines.append(f"    {feat:30s} windows={sorted(wins)}")
        lines.append(f"  Total columns : {len(self.get_all_columns())}")
        return "\n".join(lines)

    # ── Cache integration ──────────────────────────────────────────────────────

    def validate_against_cache(
        self,
        cache_dir: Path | str,
        ctx=None,
        logic_file: Optional[str] = None,
        fps: float = 6.0,
        auto_fill: bool = True,
    ) -> List[str]:
        """
        Check that every column in get_all_columns() exists in the manifest.

        If `auto_fill=True` (default) and missing columns are found, calls
        update_cache() to compute and add them automatically (requires ctx and
        logic_file).

        Returns the list of columns that were missing (empty = all good).
        """
        cache_dir = Path(cache_dir)
        manifest = _read_manifest(cache_dir)
        cached_cols = set(manifest.get("columns", {}).keys())
        needed = self.get_all_columns()
        missing = [c for c in needed if c not in cached_cols]

        if not missing:
            print(f"[FeatureSetConfig '{self.name}'] ✓ All {len(needed)} columns present in cache.")
            return []

        print(
            f"[FeatureSetConfig '{self.name}'] "
            f"⚠ {len(missing)} columns missing from cache: {missing}"
        )

        if auto_fill and ctx is not None and logic_file is not None:
            print("  Auto-filling missing columns via update_cache() …")
            update_cache(
                ctx=ctx,
                logic_file=logic_file,
                cache_dir=cache_dir,
                fps=fps,
                new_columns=missing,
            )
            return []

        if auto_fill and (ctx is None or logic_file is None):
            print(
                "  auto_fill=True but ctx/logic_file not provided — "
                "cannot fill automatically.  Run update_cache() manually."
            )
        return missing

    def select_from(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return only the columns declared in this config.
        Missing columns are filled with 0.0 and a warning is printed.
        """
        cols = self.get_all_columns()
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(
                f"[FeatureSetConfig '{self.name}'] "
                f"WARNING: {len(missing)} columns not found in dataframe, "
                f"filling with 0: {missing}"
            )
            for c in missing:
                df = df.copy()
                df[c] = 0.0
        return df[cols].astype("float32")

    def to_dict(self) -> dict:
        """Serialisable representation (for saving alongside model artifacts)."""
        return {
            "name": self.name,
            "description": self.description,
            "base_features": self.base_features,
            "windowed_features": {k: sorted(v) for k, v in self.windowed_features.items()},
            "all_columns": self.get_all_columns(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSetConfig":
        return cls(
            name=d.get("name", "unnamed"),
            description=d.get("description", ""),
            base_features=d.get("base_features", []),
            windowed_features=d.get("windowed_features", {}),
        )

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "FeatureSetConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers used by tp_export.py
# ─────────────────────────────────────────────────────────────────────────────

def _load_features_for_sources(
    cache_dir: Path,
    sources: List[str],
    fsc: "FeatureSetConfig",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and concatenate cached features for a list of sources.

    Returns
    -------
    X    : float32 DataFrame with only fsc.get_all_columns()
    meta : DataFrame with (source, ID, et) — same row order as X
    """
    feature_cols = fsc.get_all_columns()

    X_list, meta_list = [], []
    for src in sources:
        dest = _parquet_path(cache_dir, src)
        if not dest.exists():
            raise FileNotFoundError(
                f"No cache parquet for source '{src}'.  "
                "Run build_full_cache() or update_cache() first."
            )
        # Single read — load everything, then select columns
        df = pd.read_parquet(dest)
        # Fill any requested feature columns not present in this parquet with 0
        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0.0
        meta_list.append(df[_KEY_COLS])
        X_list.append(df[feature_cols].astype("float32"))

    meta = pd.concat(meta_list, ignore_index=True)
    X = pd.concat(X_list, ignore_index=True)
    return X, meta
