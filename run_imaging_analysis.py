"""Run imaging_data_analysis.py across all extracted imaging folders and
write the results as a CSV table (thesis-ready) plus a JSON backup (full detail).

Usage:
    python run_imaging_analysis.py
"""

import csv
import json
from pathlib import Path

from data_report.statistical_analysis.local.imaging_data_analysis import (
    DICOM_EXTENSIONS,
    NIFTI_EXTENSIONS,
    RASTER_EXTENSIONS,
    file_size_statistics,
    resolution_statistics,
    distribution_by_modality,
    distribution_by_anatomical_region,
)

IMAGING_DATA_DIR = Path("imaging data")
OUTPUT_DIR = Path("results") / "imaging_analysis"

CSV_COLUMNS = [
    "folder",
    "format",
    "n_files",
    "size_mb",
    "width",
    "height",
    "modality_distribution",
    "anatomical_region_distribution",
]


def dict_to_str(d):
    return ", ".join(f"{k}:{v}" for k, v in sorted(d.items())) if d else ""


def interval_str(lo, hi):
    if lo == "" or hi == "":
        return ""
    return str(lo) if lo == hi else f"{lo}–{hi}"


def detect_format(folder):
    labels = set()
    for file in folder.iterdir():
        if not file.is_file():
            continue
        name = file.name.lower()
        if name.endswith(DICOM_EXTENSIONS):
            labels.add("DICOM")
        elif name.endswith(NIFTI_EXTENSIONS):
            labels.add("NIfTI")
        elif name.endswith(RASTER_EXTENSIONS):
            labels.add(file.suffix.lstrip(".").upper())
    return "+".join(sorted(labels))


def analyze_folder(folder):
    size_stats = file_size_statistics(folder)
    res_stats = resolution_statistics(folder)
    modality = distribution_by_modality(folder)
    region = distribution_by_anatomical_region(folder)

    row = {
        "folder": folder.name,
        "format": detect_format(folder),
        "n_files": size_stats.get("n_files", ""),
        "size_mb": interval_str(size_stats.get("min_size", ""), size_stats.get("max_size", "")),
        "width": interval_str(res_stats.get("min_width", ""), res_stats.get("max_width", "")),
        "height": interval_str(res_stats.get("min_height", ""), res_stats.get("max_height", "")),
        "modality_distribution": dict_to_str(modality),
        "anatomical_region_distribution": dict_to_str(region.get("counts", {})),
    }

    raw = {
        "file_size_statistics": size_stats,
        "resolution_statistics": res_stats,
        "distribution_by_modality": modality,
        "distribution_by_anatomical_region": region,
    }
    return row, raw


def discover_folders():
    folders = []
    png_dir = IMAGING_DATA_DIR / "images_001" / "images"
    if png_dir.is_dir():
        folders.append(png_dir)
    series_dir = IMAGING_DATA_DIR / "dicom_series"
    if series_dir.is_dir():
        folders.extend(sorted(p for p in series_dir.iterdir() if p.is_dir()))
    return folders


def main(folders=None):
    folders = folders if folders is not None else discover_folders()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    raw_by_folder = {}
    for folder in folders:
        print(f"Analyzing {folder} ...")
        row, raw = analyze_folder(folder)
        rows.append(row)
        raw_by_folder[folder.name] = raw

    csv_path = OUTPUT_DIR / "imaging_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    json_path = OUTPUT_DIR / "imaging_summary_full.json"
    with open(json_path, "w") as f:
        json.dump(raw_by_folder, f, indent=2)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
