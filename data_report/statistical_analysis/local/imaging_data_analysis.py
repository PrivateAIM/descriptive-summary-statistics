"""Analysis functions for medical imaging data.

Supports DICOM (``.dcm``), NIfTI (``.nii`` / ``.nii.gz``), and common
raster formats (``.png``, ``.jpg``, ``.jpeg``).  All functions accept a
directory path and scan every matching file within it, returning plain dicts
suitable for JSON serialisation.
"""

from pathlib import Path
from collections import defaultdict
import statistics

import pydicom
import nibabel as nib
from PIL import Image

DICOM_EXTENSIONS = (".dcm",)
"""File extensions recognised as DICOM images."""

NIFTI_EXTENSIONS = (".nii", ".nii.gz")
"""File extensions recognised as NIfTI volumes."""

RASTER_EXTENSIONS = (".png", ".jpg", ".jpeg")
"""File extensions recognised as raster images."""

IMAGE_EXTENSIONS = DICOM_EXTENSIONS + NIFTI_EXTENSIONS + RASTER_EXTENSIONS
"""All imaging extensions handled by this module."""


def file_size_statistics(data_dir):
    """Compute summary size statistics for all imaging files in a directory.

    Only files whose extensions are in ``IMAGE_EXTENSIONS`` are included.

    Args:
        data_dir (str or Path): Directory to scan.

    Returns:
        dict: Keys ``n_files``, ``total_size``, ``average_size``,
            ``min_size``, ``max_size`` (all sizes in megabytes, rounded to
            3 dp). Returns an empty dict if no matching files are found.
    """
    file_sizes = []
    for file in Path(data_dir).iterdir():
        if file.is_file() and file.name.lower().endswith(IMAGE_EXTENSIONS):
            file_sizes.append(file.stat().st_size / (1024 ** 2))

    if not file_sizes:
        return {}

    return {
        "n_files": len(file_sizes),
        "total_size": round(sum(file_sizes), 3),
        "average_size": round(statistics.mean(file_sizes), 3),
        "min_size": round(min(file_sizes), 3),
        "max_size": round(max(file_sizes), 3),
    }


def _dicom_dimensions(path):
    """Read pixel dimensions from a DICOM file header without loading pixel data."""
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    width = getattr(ds, "Columns", None)
    height = getattr(ds, "Rows", None)
    depth = getattr(ds, "NumberOfFrames", None)
    return width, height, depth


def _nifti_dimensions(path):
    """Read spatial dimensions from a NIfTI volume header."""
    shape = nib.load(path).shape
    width = shape[0] if len(shape) > 0 else None
    height = shape[1] if len(shape) > 1 else None
    depth = shape[2] if len(shape) > 2 else None
    return width, height, depth


def _raster_dimensions(path):
    """Read pixel dimensions from a raster image."""
    with Image.open(path) as image:
        width, height = image.size
    return width, height, None


def resolution_statistics(data_dir):
    """Compute spatial resolution statistics for all imaging files in a directory.

    Dimensions are extracted from DICOM headers (Columns/Rows/NumberOfFrames),
    NIfTI headers (shape), and raster files (PIL Image.size).  Physical
    pixel/voxel spacing is intentionally excluded because its units differ
    across formats and cannot be meaningfully compared in a mixed directory.

    Args:
        data_dir (str or Path): Directory containing imaging files.

    Returns:
        dict: Keys ``n_images``, ``min_width``, ``max_width``,
            ``average_width``, ``min_height``, ``max_height``,
            ``average_height``.  For directories containing 3-D volumes,
            ``min_depth``, ``max_depth``, and ``average_depth`` are also
            included (depth values of 1 or None are excluded).
            Returns an empty dict if no images with valid dimensions are found.
    """
    widths, heights, depths = [], [], []

    for file in Path(data_dir).iterdir():
        if not file.is_file():
            continue

        name = file.name.lower()
        if name.endswith(DICOM_EXTENSIONS):
            width, height, depth = _dicom_dimensions(file)
        elif name.endswith(NIFTI_EXTENSIONS):
            width, height, depth = _nifti_dimensions(file)
        elif name.endswith(RASTER_EXTENSIONS):
            width, height, depth = _raster_dimensions(file)
        else:
            continue

        if width is not None:
            widths.append(width)
        if height is not None:
            heights.append(height)
        if depth is not None and depth > 1:
            depths.append(depth)

    if not widths or not heights:
        return {}

    stats = {
        "n_images": len(widths),
        "min_width": min(widths),
        "max_width": max(widths),
        "average_width": round(statistics.mean(widths), 3),
        "min_height": min(heights),
        "max_height": max(heights),
        "average_height": round(statistics.mean(heights), 3),
    }
    if depths:
        stats["min_depth"] = min(depths)
        stats["max_depth"] = max(depths)
        stats["average_depth"] = round(statistics.mean(depths), 3)

    return stats


def distribution_by_modality(data_dir):
    """Count DICOM files by imaging modality.

    Reads the ``Modality`` DICOM tag (e.g. ``"CT"``, ``"MR"``, ``"US"``)
    from each ``.dcm`` file.  NIfTI and raster files carry no equivalent tag
    and are skipped.

    Args:
        data_dir (str or Path): Directory containing DICOM files.

    Returns:
        dict: Mapping of modality label (str) → count (int).
            Files whose ``Modality`` tag is absent are counted under
            ``"Unknown"``.  Returns an empty dict if no DICOM files are found.
    """
    distribution = defaultdict(int)

    for file in Path(data_dir).iterdir():
        if not file.is_file() or not file.name.lower().endswith(DICOM_EXTENSIONS):
            continue
        ds = pydicom.dcmread(file, stop_before_pixels=True)
        modality = getattr(ds, "Modality", None) or "Unknown"
        distribution[modality] += 1

    return dict(distribution)


def distribution_by_anatomical_region(data_dir):
    """Count DICOM files by anatomical region from the ``BodyPartExamined`` tag.

    Note:
        This is a best-effort statistic.  The ``BodyPartExamined`` tag is
        inconsistently populated in real-world DICOM data.  Single-organ
        study directories will always yield a degenerate (single-category)
        distribution; the ``is_degenerate`` field in the return value flags
        this so callers can suppress the result or add a disclaimer.

    Args:
        data_dir (str or Path): Directory containing DICOM files.

    Returns:
        dict: A dict with keys:
            - ``counts`` (dict): Mapping of region label (str) → count (int).
              Files with no ``BodyPartExamined`` tag are counted under
              ``"Unknown"``.
            - ``is_degenerate`` (bool): ``True`` if only one distinct region
              was found.
            - ``reliability_note`` (str): Human-readable caveat.
            Returns an empty dict if no DICOM files are found.
    """
    distribution = defaultdict(int)

    for file in Path(data_dir).iterdir():
        if not file.is_file() or not file.name.lower().endswith(DICOM_EXTENSIONS):
            continue
        ds = pydicom.dcmread(file, stop_before_pixels=True)
        region = getattr(ds, "BodyPartExamined", None) or "Unknown"
        distribution[region] += 1

    if not distribution:
        return {}

    return {
        "counts": dict(distribution),
        "is_degenerate": len(distribution) == 1,
        "reliability_note": (
            "BodyPartExamined is inconsistently populated in real-world DICOM "
            "data; treat this distribution as best-effort, not authoritative."
        ),
    }
