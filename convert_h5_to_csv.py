"""Utility to convert multi-scale HDF5 feature dumps to DSMIL-compatible CSV bags.

This script reads per-slide feature files exported as HDF5 containers (the
format used by CLAM/UNI pipelines) and writes one CSV per slide that DSMIL's
`train_tcga.py` loader understands.  When both low-magnification (e.g. 2.5×)
and high-magnification (e.g. 5×) feature dumps are provided, the converter
aligns the patches using the same quad-tree mapping employed in UNI/DSMIL
multi-scale fusion:

* each low-mag patch located at ``(x_low, y_low)`` corresponds to four
  high-mag patches centered on the corners of the doubled receptive field:
  ``[(x_low * ratio, y_low * ratio), (x_low * ratio + patch, y_low * ratio), ...]``.
* optional pixel offsets and a search tolerance can be supplied to compensate
  for rounding differences between extraction pipelines.
* the high-mag vectors matched to a low-mag patch are averaged and concatenated
  with the low-mag vector (``--fusion cat``) by default.  Alternative fusion
  strategies are exposed via CLI arguments.

Example usage (mirrors the mapping described in the question prompt):

```
python convert_h5_to_csv.py \
    --dataset-name FA_PT \
    --label-csv path/to/train.csv path/to/val.csv path/to/test.csv \
    --low-mag-dir C:/.../uniextracted_mag2x_patch224_fp/feats_h5 \
    --high-mag-dir C:/.../uniextracted_mag5x_patch224_fp/feats_h5 \
    --output-dir datasets/FA_PT \
    --ratio 2.0 \
    --patch-size 224 \
    --tolerance 8
```

The command writes ``datasets/FA_PT/<label>/<slide>.csv`` feature bags and a
``datasets/FA_PT/FA_PT.csv`` index with ``path,label`` rows, allowing DSMIL to
train directly from the converted files.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def list_h5(directory: Path) -> Dict[str, Path]:
    """Return a mapping from slide stem to HDF5 path for a directory."""

    h5_files: Dict[str, Path] = {}
    for suffix in ("*.h5", "*.hdf5"):
        for path in directory.glob(suffix):
            h5_files[path.stem] = path
    return h5_files


def load_h5_features(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load feature vectors and coordinates from a CLAM/UNI style HDF5 file."""

    with h5py.File(path, "r") as handle:
        feats = np.asarray(handle["features"], dtype=np.float32)
        coords = np.asarray(handle["coords"], dtype=np.int32)
    return feats, coords


def build_coord_index(coords: np.ndarray) -> Dict[Tuple[int, int], int]:
    """Create a dictionary that maps coordinate tuples to row indices."""

    return {(int(x), int(y)): int(i) for i, (x, y) in enumerate(coords)}


def find_index(coord_to_idx: Mapping[Tuple[int, int], int], coord: Tuple[int, int], tolerance: int) -> Optional[int]:
    """Locate the index of a coordinate with an optional tolerance window."""

    if coord in coord_to_idx:
        return coord_to_idx[coord]
    if tolerance <= 0:
        return None
    x0, y0 = coord
    for dx in range(-tolerance, tolerance + 1):
        for dy in range(-tolerance, tolerance + 1):
            candidate = (x0 + dx, y0 + dy)
            if candidate in coord_to_idx:
                return coord_to_idx[candidate]
    return None


def quad_high_coords(low_coord: Tuple[int, int], ratio: float, patch_size: int, offset: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Generate the 2×2 grid of high-magnification patch origins for a low-mag patch."""

    x_low, y_low = low_coord
    x_base = int(round(x_low * ratio)) + offset[0]
    y_base = int(round(y_low * ratio)) + offset[1]
    return [
        (x_base, y_base),
        (x_base + patch_size, y_base),
        (x_base, y_base + patch_size),
        (x_base + patch_size, y_base + patch_size),
    ]


def fuse_patch(
    feat_low: np.ndarray,
    high_feats: Sequence[np.ndarray],
    fusion: str,
    high_dim: Optional[int],
) -> np.ndarray:
    """Combine low- and high-mag features according to the selected strategy."""

    if fusion == "low":
        return feat_low

    if not high_feats:
        if fusion == "high":
            if high_dim is None:
                raise ValueError("High magnification dimension must be known when using --fusion high without matches")
            return np.zeros(high_dim, dtype=feat_low.dtype)
        if fusion == "cat":
            if high_dim is None:
                raise ValueError("High magnification dimension must be known when using --fusion cat without matches")
            return np.concatenate([feat_low, np.zeros(high_dim, dtype=feat_low.dtype)], axis=0)
        if fusion == "add":
            return feat_low

    high_vector: np.ndarray
    if fusion == "high":
        high_vector = np.mean(np.stack(high_feats, axis=0), axis=0)
        return high_vector

    if fusion == "cat":
        high_vector = np.mean(np.stack(high_feats, axis=0), axis=0)
        return np.concatenate([feat_low, high_vector], axis=0)

    if fusion == "add":
        high_vector = np.mean(np.stack(high_feats, axis=0), axis=0)
        if feat_low.shape != high_vector.shape:
            raise ValueError(
                "Cannot add feature vectors of different dimensionality: "
                f"low={feat_low.shape}, high={high_vector.shape}"
            )
        return feat_low + high_vector

    raise ValueError(f"Unsupported fusion strategy: {fusion}")


@dataclass
class ConverterConfig:
    dataset_name: str
    output_dir: Path
    low_mag_dir: Path
    high_mag_dir: Optional[Path]
    label_csvs: Sequence[Path]
    slide_column: str
    label_column: str
    ratio: float
    patch_size: int
    tolerance: int
    offset_x_high: int
    offset_y_high: int
    fusion: str
    relative_paths: bool
    write_coords: bool
    index_only: bool
    verbose: bool


def load_labels(csv_paths: Sequence[Path], slide_col: str, label_col: str) -> Mapping[str, str]:
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if slide_col not in df.columns or label_col not in df.columns:
            raise KeyError(f"Required columns '{slide_col}' and '{label_col}' not found in {path}")
        frames.append(df[[slide_col, label_col]].copy())
    merged = pd.concat(frames, axis=0).drop_duplicates(subset=slide_col, keep="first")
    duplicates = merged[merged.duplicated(subset=slide_col, keep=False)]
    if not duplicates.empty:
        raise ValueError(f"Found conflicting labels for slides: {duplicates}")
    return dict(zip(merged[slide_col].astype(str), merged[label_col].astype(str)))


def convert_slide(
    slide_id: str,
    label: str,
    config: ConverterConfig,
    low_files: Mapping[str, Path],
    high_files: Mapping[str, Path],
) -> Tuple[Path, np.ndarray]:
    if slide_id not in low_files:
        raise FileNotFoundError(f"Missing low-magnification H5 for slide '{slide_id}'")

    feats_low, coords_low = load_h5_features(low_files[slide_id])
    high_dim: Optional[int] = None
    feats_high: Optional[np.ndarray] = None
    coords_high: Optional[np.ndarray] = None
    high_index: Optional[Dict[Tuple[int, int], int]] = None

    if config.high_mag_dir is not None:
        if slide_id not in high_files:
            raise FileNotFoundError(f"Missing high-magnification H5 for slide '{slide_id}'")
        feats_high, coords_high = load_h5_features(high_files[slide_id])
        high_index = build_coord_index(coords_high)
        high_dim = feats_high.shape[1]

    fused_patches: List[np.ndarray] = []
    quad_offset = (config.offset_x_high, config.offset_y_high)
    for idx_low, (x_low, y_low) in enumerate(coords_low.astype(int)):
        low_vec = feats_low[idx_low]
        high_vectors: List[np.ndarray] = []
        if feats_high is not None and high_index is not None:
            for coord_high in quad_high_coords((x_low, y_low), config.ratio, config.patch_size, quad_offset):
                index = find_index(high_index, coord_high, config.tolerance)
                if index is not None:
                    high_vectors.append(feats_high[index])
        fused = fuse_patch(low_vec, high_vectors, config.fusion, high_dim)
        fused_patches.append(fused)

    return write_slide_csv(slide_id, label, np.stack(fused_patches, axis=0), coords_low, config)


def write_slide_csv(
    slide_id: str,
    label: str,
    features: np.ndarray,
    coords: np.ndarray,
    config: ConverterConfig,
) -> Tuple[Path, np.ndarray]:
    label_dir = config.output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    feature_path = label_dir / f"{slide_id}.csv"
    np.savetxt(feature_path, features, delimiter=",")

    if config.write_coords:
        coords_path = label_dir / f"{slide_id}_coords.csv"
        header = ["x", "y"]
        with coords_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(coords.astype(int))

    return feature_path, features

def run_conversion(config: ConverterConfig) -> None:
    if not config.index_only:
        if not config.low_mag_dir.exists():
            raise FileNotFoundError(f"Low-magnification directory not found: {config.low_mag_dir}")
        if config.high_mag_dir is not None and not config.high_mag_dir.exists():
            raise FileNotFoundError(f"High-magnification directory not found: {config.high_mag_dir}")
    elif not config.output_dir.exists():
        raise FileNotFoundError(
            f"Output directory not found: {config.output_dir}. Run without --index-only first to generate features."
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(config.label_csvs, config.slide_column, config.label_column)
    LOGGER.info("Loaded %d labelled slides", len(labels))

    low_files: Mapping[str, Path] = {}
    high_files: Mapping[str, Path] = {}
    if config.index_only:
        LOGGER.info("Index-only mode: reusing existing slide CSVs")
    else:
        low_files = list_h5(config.low_mag_dir)
        LOGGER.info("Found %d low-magnification H5 files", len(low_files))

        if config.high_mag_dir is not None:
            high_files = list_h5(config.high_mag_dir)
            LOGGER.info("Found %d high-magnification H5 files", len(high_files))

    label_to_int = {label: idx for idx, label in enumerate(sorted(set(labels.values())))}
    dataset_rows: List[Tuple[str, int]] = []

    for slide_id, label in labels.items():
        feature_path = config.output_dir / label / f"{slide_id}.csv"
        if feature_path.exists():
            LOGGER.info("Skipping existing CSV for %s (already exists)", slide_id)
        elif not config.index_only:
            LOGGER.info("Processing slide %s (label=%s)", slide_id, label)
            feature_path, features = convert_slide(slide_id, label, config, low_files, high_files)
            LOGGER.debug("%s -> %s (features: %s)", slide_id, feature_path, features.shape)
        else:
            raise FileNotFoundError(
                f"Expected precomputed feature CSV for slide '{slide_id}' at {feature_path}. "
                "Run without --index-only to generate it."
            )

        label_idx = label_to_int[label]
        dataset_rows.append((str(feature_path.resolve()), label_idx))

    index_path = config.output_dir / f"{config.dataset_name}.csv"
    if config.relative_paths:
        index_base = index_path.parent.resolve()
        dataset_rows = [
            (os.path.relpath(Path(path).resolve(), index_base), label)
            for path, label in dataset_rows
        ]

    with index_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label"])
        writer.writerows(dataset_rows)

    LOGGER.info("Wrote dataset index to %s", index_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> ConverterConfig:
    parser = argparse.ArgumentParser(description="Convert CLAM/UNI H5 feature dumps to DSMIL CSV bags")
    parser.add_argument("--dataset-name", required=True, help="Name of the dataset (used for the index CSV)")
    parser.add_argument("--output-dir", required=True, type=Path, help="Root directory for DSMIL-formatted features")
    parser.add_argument("--low-mag-dir", required=True, type=Path, help="Directory containing low magnification H5 files")
    parser.add_argument("--high-mag-dir", type=Path, help="Directory containing high magnification H5 files")
    parser.add_argument(
        "--label-csv",
        required=True,
        nargs="+",
        type=Path,
        help="One or more CSV files with slide/label columns. Duplicates are deduplicated by the first occurrence.",
    )
    parser.add_argument("--slide-column", default="slide_id", help="Column containing slide identifiers")
    parser.add_argument("--label-column", default="label", help="Column containing label strings")
    parser.add_argument("--ratio", type=float, default=2.0, help="Coordinate scale ratio between low and high magnification")
    parser.add_argument("--patch-size", type=int, default=224, help="Patch size used during extraction (in pixels)")
    parser.add_argument(
        "--tolerance",
        type=int,
        default=0,
        help="Search radius (in pixels) applied when matching high-magnification coordinates",
    )
    parser.add_argument("--offset-x-high", type=int, default=0, help="Pixel offset to add to high-mag X coordinates")
    parser.add_argument("--offset-y-high", type=int, default=0, help="Pixel offset to add to high-mag Y coordinates")
    parser.add_argument(
        "--fusion",
        choices=["low", "high", "cat", "add"],
        default="cat",
        help="How to fuse low/high magnification features (default: concatenate)",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Store paths in the dataset index relative to the index CSV location",
    )
    parser.add_argument(
        "--write-coords",
        action="store_true",
        help="Export companion CSVs with patch coordinates for later visualization",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Skip feature conversion and only (re)write the dataset index CSV using existing slide CSVs",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    return ConverterConfig(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        low_mag_dir=args.low_mag_dir,
        high_mag_dir=args.high_mag_dir,
        label_csvs=args.label_csv,
        slide_column=args.slide_column,
        label_column=args.label_column,
        ratio=args.ratio,
        patch_size=args.patch_size,
        tolerance=args.tolerance,
        offset_x_high=args.offset_x_high,
        offset_y_high=args.offset_y_high,
        fusion=args.fusion,
        relative_paths=args.relative_paths,
        write_coords=args.write_coords,
        index_only=args.index_only,
        verbose=args.verbose,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    _setup_logging(config.verbose)
    run_conversion(config)


if __name__ == "__main__":
    main()
