# Imaging Analysis Module — README

## Background

`data_report/statistical_analysis/local/imaging_data_analysis.py` started as an empty stub (mirroring the genomic module before it was implemented). The original checklist under consideration was:

- Total number of annotated images (manual/automatic labels or markings, e.g. segmentation masks)
- File size statistics (min/max/average)
- Resolution statistics (image dimensions or pixel/voxel size)
- Number of patients
- Images per patient
- Distribution by modality
- Distribution by anatomical region
- Distribution by class (label)

Each item was reviewed for plausibility against the imaging libraries already available in the project's environment (`pydicom`, `nibabel`, `SimpleITK`, `opencv`, `Pillow` — i.e. DICOM, NIfTI, and plain raster formats are all in scope).

## Decision: scope cut down to 4 items

After review, the following were **dropped**:

- **Number of patients / images per patient** — dropped over privacy concerns. `PatientID` is sensitive even when present in DICOM headers, and deriving "number of patients" in a federated-analysis context risks leaking identifiable counts/structure about a site's population.
- **Distribution by class (label)** — dropped. No image format carries a classification label natively; it would require an external manifest CSV or folder-per-class convention that doesn't exist yet.
- **Total number of annotated images** — dropped. "Annotated" has no single definition without first picking a convention (manifest column vs. sibling mask file vs. DICOM-SEG object), which hasn't been decided.

**Kept, implemented in full:**
- File size statistics
- Resolution statistics
- Distribution by modality

**Kept, but explicitly best-effort:**
- Distribution by anatomical region — DICOM's `BodyPartExamined` tag is unreliable in real-world data (often blank/inconsistent across scanners) and has no equivalent in NIfTI/raster files. Single-organ studies will also always produce a degenerate (single-category) distribution that isn't informative. Implemented with a built-in check (`is_degenerate` flag + `reliability_note`) rather than presented as a reliable core statistic.

## What was implemented

All four functions are directory-level (operate on a folder of image files for one node/sample), following the same pattern as the genomic module's `file_size_statistics`/`file_format_overview`.

### `file_size_statistics(data_dir)`
Scans the directory for recognized image extensions (`.dcm`, `.nii`, `.nii.gz`, `.png`, `.jpg`, `.jpeg`) and returns `{n_files, total_size, average_size, min_size, max_size}` (sizes in MB). Returns `{}` if no recognized image files are found.

### `resolution_statistics(data_dir)`
Reads dimensions per file, format-aware:
- **DICOM** (`.dcm`) — header-only read (`stop_before_pixels=True`) of `Columns`/`Rows`/`NumberOfFrames`.
- **NIfTI** (`.nii`/`.nii.gz`) — `nibabel`'s lazy-loaded `.shape` (no pixel data read into memory).
- **Raster** (`.png`/`.jpg`/`.jpeg`) — `Pillow`'s `Image.size` (lazy, no decode).

Returns `{n_images, min_width, max_width, average_width, min_height, max_height, average_height}`, plus `min_depth`/`max_depth`/`average_depth` only if any 3D volume was found in the directory. Physical voxel/pixel spacing (mm) was deliberately left out — DICOM and NIfTI expose it in different units/forms, and raster files have none, so it isn't a meaningful aggregate across a mixed-format directory. Returns `{}` if the directory has no images.

### `distribution_by_modality(data_dir)`
Reads the DICOM `Modality` tag (header-only) for every `.dcm` file and counts occurrences (`Unknown` if the tag is present but empty). Non-DICOM files have no modality concept and are skipped rather than miscounted. Returns `{}` if no DICOM files are present.

### `distribution_by_anatomical_region(data_dir)`
Reads the DICOM `BodyPartExamined` tag the same way, but returns a richer shape than the others to surface the reliability caveat:
```python
{
    "counts": {...},
    "is_degenerate": bool,      # True if every file fell into one category
    "reliability_note": "...",  # explains why this isn't authoritative
}
```
Returns `{}` if no DICOM files are present.

## Tests

`data_report/statistical_analysis/local/test_imaging_data_analysis.py` — 14 tests, using synthetic fixtures built at test time (no real patient data, no downloads):
- Minimal valid DICOM files built directly with `pydicom.dataset.FileDataset` (configurable rows/columns/modality/body part/frame count).
- Minimal NIfTI volumes built with `nibabel.Nifti1Image` over a `numpy` array.
- Minimal PNGs built with `Pillow`.

Covers: file size aggregation, 2D vs. 3D resolution stats, NIfTI and raster resolution, mixed-format directories, modality counting (including non-DICOM files being ignored), anatomical-region degenerate vs. non-degenerate cases, missing tags defaulting to `"Unknown"`, and empty-directory `{}` returns for all four functions.

Full project test suite: 300 passed.

## Not done yet (out of scope for this pass)

This module is self-contained, exactly like `genomic_data_analysis.py` before its pipeline wiring step. It is **not** yet:
- Hooked into `detect_file_type` / `stage_genomic_file`-style staging in `data_report/get_data/utils.py`.
- Wired into `DataReportAnalyzer.analysis_method` in `data_report/analyze.py`.

Per the earlier genomic-module integration plan (see `GENOMIC_ANALYSIS_CHANGES.md`), the `analysis_method` rewire to loop over multiple files and dispatch by type was deliberately deferred until both the genomic and imaging modules were ready.

**This is not a hub limitation.** The hub/`flame.star` transport is type-agnostic — it delivers whatever bytes sit at the S3 keys configured for a node, regardless of file type (see `hub_entry.py`, which already references a `VCF_S3_KEYS` precedent from an example script, i.e. genomic-style keys are an established pattern on the hub). The actual reason integration was deferred is that the only example datasets available for testing on the hub were CSV, so there was no genomic or imaging test data to validate the wiring against end-to-end. The imaging functions above are implemented and unit-tested as standalone components; pipeline integration remains untested on real hub-delivered data for that reason, not because of a platform constraint.
