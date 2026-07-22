# Genomic Analysis Module — Changes Log

## Original request

> finish implementing the genomic analysis in genomic_data_analysis.py to include functions for the following things: # Variant counts per sample (number of genomic variants identified in each sample), # Variant type distribution (counts of different variant types (e.g., SNPs, insertions, deletions)). Then test them. Also after that, check the other functions and the whole code and make a review, check if there are errors, bugs or things to be fixed, but do not change anything yet till I confirm and tell you. Start implementing what is missing first

Follow-up request: "Fix all of these issues now" (i.e. apply the fixes identified in the review below).

## Summary of additions

Implemented the two missing functions in `data_report/statistical_analysis/local/genomic_data_analysis.py`:

- **`variant_counts_per_sample(vcf_file)`** — opens the VCF with `cyvcf2` (`gts012=True`) and, for each sample, counts variant records where that sample's genotype is HET or HOM_ALT (i.e. the sample actually carries the variant).
- **`variant_type_distribution(vcf_file)`** — iterates all variants and buckets them into `SNP` / `insertion` / `deletion` / `SV` / `MNP` / `other` using `cyvcf2`'s `is_snp` / `is_indel` / `is_deletion` / `is_sv` / `is_mnp` flags.

Added `data_report/statistical_analysis/local/test_genomic_data_analysis.py` with a synthetic multi-sample VCF fixture and tests for both functions, plus tests for every bug fix below (paired-end filename merging, case-insensitive extensions, gzip FASTQ support, empty-input edge cases for FASTQ/FASTA/BAM). 25 tests total in this file's suite; full project suite (262 tests) passes.

## Summary of fixes (review findings, then applied)

1. **`count_samples` — broken lane-suffix regex.** `r"_L\\d+$"` (double backslash) matched a literal backslash + `d`, never digits, so `_L001` was never stripped. Fixed to `r"_L\d+$"`.
2. **`count_samples` — wrong suffix-stripping order.** Real Illumina filenames end `..._L001_R1_001`; the code stripped `_R[12]$` and `_L\d+$` before `_001$`, so none of them matched on a first pass. Paired-end files like `Sample_S1_L001_R1_001.fastq.gz` / `..._R2_001.fastq.gz` came out as two different "samples" instead of one. Reordered to strip `_001` → `_R[12]` → `_L\d+`.
3. **`count_samples` — case-sensitive extension matching**, inconsistent with `file_format_overview` (which lowercases first). `.FASTQ`/`.VCF` etc. were silently skipped. Fixed by matching against a lowercased copy of the filename while preserving original casing in the returned sample name.
4. **`read_length_statistics` — division before the zero-check.** `average_length = total_bases / total_reads` ran before `if total_reads == 0: return {}`, so an empty FASTQ raised `ZeroDivisionError` instead of returning `{}`. Reordered so the zero-check runs first.
5. **`coverage_statistics` — no guard for empty pileups.** An empty/unaligned BAM produced an empty `depths` list, and `np.min`/`np.max` on an empty array raise `ValueError` (and `np.mean` warns/returns `nan`). Added an `if not depths: return {}` guard, consistent with how other stats functions in the module handle empty input.
6. **Missing gzip support for FASTQ.** `EXTENSIONS`/`FORMATS` both advertise `.fastq.gz`/`.fq.gz`, but `count_reads` and `read_length_statistics` opened files in plain text mode, which fails on gzipped input. Added a shared `_open_fastq()` helper that transparently opens `.gz` files via `gzip.open(..., "rt")`.
7. **`coverage_statistics` — CRAM mode had no way to supply a reference.** Some CRAM files don't embed their reference sequence and need one supplied externally. Added an optional `reference_filename=None` parameter, passed through to `pysam.AlignmentFile` only for `"rc"` mode.
8. **`fasta_handling` was a stub.** It only printed each record's ID and returned `None`, unlike every other function in the module, which returns a structured stats dict. Rewrote it to return `{n_sequences, sequence_ids, min_length, max_length, average_length}`, matching the pattern used by `read_length_statistics`/`file_size_statistics`.
9. **Unused imports.** Removed `import pandas as pd` and `import os` — neither was referenced anywhere in the file.

## Previous version of the code (before any changes)

```python
import pandas as pd
from pathlib import Path
from collections import defaultdict
import statistics
import os
from Bio import SeqIO
import pysam
import re
import numpy as np
from cyvcf2 import VCF

# number of samples
EXTENSIONS = (".fastq", ".fastq.gz", ".fq", ".fq.gz", ".bam", ".vcf", ".vcf.gz")

def count_samples(data_dir):

    sample_names = set()
    for file in Path(data_dir).iterdir():
        if not file.is_file():
            continue

        filename = file.name
        if not filename.endswith(EXTENSIONS):
            continue
        sample_name = filename

        for ext in EXTENSIONS:
            if sample_name.endswith(ext):
                sample_name = sample_name.removesuffix(ext)

        # remove paired-end indicators
        # for example: sample1_R1, sample_R2
        # R1 for forward reads
        # R2 for reverse reads
        sample_name = re.sub(r"_R[12]$", "", sample_name)
        # remove lane suffixes
        # example: _L001
        sample_name = re.sub(r"_L\\d+$", "", sample_name)
        # remove common read suffixes
        sample_name = re.sub(r"_001$", "", sample_name)
        sample_names.add(sample_name)

    return {
        "n_samples": len(sample_names),
        "samples": sorted(sample_names)
    }


FORMATS = {
    ".fasta": "FASTA",
    ".fa": "FASTA",
    ".fasta.gz": "FASTA",
    ".fastq": "FASTQ",
    ".fq": "FASTQ",
    ".fastq.gz": "FASTQ",
    ".fq.gz": "FASTQ",
    ".bam": "BAM",
    ".sam": "SAM",
    ".vcf": "VCF",
    ".vcf.gz": "VCF" }

# File format overview (types of file formats present (e.g., FASTQ, BAM, VCF))
def file_format_overview(data_dir):
    overview = {}
    for file in Path(data_dir).iterdir():
        if not file.is_file():
            continue

        filename = file.name.lower()
        for suffix, format_name in FORMATS.items():
            if filename.endswith(suffix):
                overview[format_name] = overview.get(format_name, 0) + 1
                break
    return overview

# File size statistics
def file_size(file_path):
    path = Path(file_path)
    if path.is_file():
        return path.stat().st_size
    raise ValueError("Path is not a file")

def file_size_statistics(data_directory):
    file_sizes = []
    for file in Path(data_directory).iterdir():
        if file.is_file():
            size = file_size(file) / (1024 ** 2)
            file_sizes.append(size)

    if not file_sizes:
        return {}

    return {
        "n_files": len(file_sizes),
        "total_size": round(sum(file_sizes), 3),
        "average_size": round(statistics.mean(file_sizes), 3),
        "min_size": round(min(file_sizes), 3),
        "max_size": round(max(file_sizes), 3)
    }

# Total reads
def count_reads(fastq_file):
    count = 0
    for read in SeqIO.parse(fastq_file, "fastq"):
        count = count + 1
    return count

# Read length statistics (average and range of sequencing read lengths)

def read_length_statistics(fastq_file):
        total_reads = 0
        total_bases = 0
        min_length = float("inf")
        max_length = 0

        with open(fastq_file) as handle:
            for record in SeqIO.parse(handle, "fastq"):
                read_length = len(record.seq)
                total_reads += 1
                total_bases += read_length
                min_length = min(min_length, read_length)
                max_length = max(max_length, read_length)
        average_length = total_bases / total_reads
        if total_reads == 0:
            return {}
        return {
            "n_reads": total_reads,
            "total_reads": total_reads,
            "min_length": min_length,
            "max_length": max_length,
            "average_length": average_length,
        }




# Coverage depth statistics (average number of times each base is sequenced)
def coverage_statistics(path):
    if path.endswith(".bam"):
        obj = pysam.AlignmentFile(path, "rb")
    elif path.endswith(".sam"):
        obj = pysam.AlignmentFile(path, "r")
    elif path.endswith(".cram"):
        obj = pysam.AlignmentFile(path, "rc")
    else:
        raise ValueError(f"Unsupported file format: {path}")

    depths = []
    for pileup_col in obj.pileup():
        depths.append(pileup_col.n)
    obj.close()
    depths = np.array(depths)

    return {
        "mean_depth": round(float(np.mean(depths)), 3),
        "min_depth": int(np.min(depths)),
        "max_depth": int(np.max(depths)),
        "median_depth": float(np.median(depths))
    }

# Variant counts per sample (number of genomic variants identified in each sample)




# Variant type distribution (counts of different variant types (e.g., SNPs, insertions, deletions))'''


def fasta_handling(fasta_file):
    # The with-statement makes sure that the file is properly closed after reading it.
    # That should all happen automatically if you just use the filename instead.
    with open(fasta_file) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            print(record.id)
    # The SeqRecord class itself is quite simple, and offers the following information as attributes:
    # .seq — The sequence itself, typically a Seq object.
    # .id — The primary ID used to identify the sequence — a string. In most cases this is something like an accession number.
```

## Pipeline integration

The integration plan was split into five steps (A–E). Decision: **steps A–C are isolated and low-risk, so they were implemented per-module as each module is finished — but step D (rewiring `DataReportAnalyzer.analysis_method` to loop over multiple files and dispatch by type) is deliberately deferred** until the imaging data analysis module is also ready, since imaging will need the exact same kind of multi-file-type routing change to that method. Doing it once, after both genomic and imaging dispatchers exist, avoids touching that core pipeline entry point twice and gives a single consolidated test pass instead of two chances to regress the existing CSV path. The JSON-FHIR-to-CSV conversion is a separate upstream ingestion concern and doesn't block this either way, since it should just produce CSV bytes before anything reaches `analysis_method`.

### Step A — extended the file-type detector (done)

`data_report/get_data/utils.py::detect_file_type` previously only recognized `csv`/`excel`/`txt`/`json`/`image`, with a comment marking genomics as a TODO (and was unused/dead code — `analyze.py` imported it but never called it). Added:

- `.fastq` / `.fastq.gz` / `.fq` / `.fq.gz` → `"fastq"`
- `.bam` / `.sam` / `.cram` → `"alignment"`
- `.vcf` / `.vcf.gz` → `"vcf"`
- `.fasta` / `.fa` / `.fasta.gz` → `"fasta"`

Also made matching case-insensitive (lowercases the filename before comparing), consistent with the `count_samples` case-insensitivity fix above.

### Step B — added the genomic dispatcher (done)

Added `analyze_genomic_file(file_path, file_type)` in `genomic_data_analysis.py`, which maps the `detect_file_type` result to the right function(s):

- `"fastq"` → `count_reads` + `read_length_statistics`
- `"alignment"` → `coverage_statistics`
- `"vcf"` → `variant_counts_per_sample` + `variant_type_distribution`
- `"fasta"` → `fasta_handling`
- anything else → raises `ValueError`

It deliberately does **not** catch exceptions internally — per-file error isolation (so one malformed file doesn't kill a node's whole analysis) is a call-site concern that belongs to step D, which is still deferred.

### Step C — added the staging helper (done)

Genomic tools (`pysam`, `cyvcf2`) need a real file on disk, unlike `pandas.read_csv(BytesIO(...))`. Added `stage_genomic_file(filename, node_data, tmp_dir)` in `data_report/get_data/utils.py`, which writes a node's in-memory bytes for `filename` to `tmp_dir`, plus any sibling index file present in `node_data` (`.bai`, `.csi`, `.tbi`, `.crai` — e.g. `sample.bam.bai`), and returns the path to the staged primary file.

### Step D — `analysis_method` rewire (deferred — not a hub limitation)

Not implemented. The plan was to loop over all files in a node's data dict, call `detect_file_type` per filename, keep the existing single-CSV path for `"csv"`, and route genomic types through `stage_genomic_file` + `analyze_genomic_file`, together with the equivalent imaging routing.

This was deferred because the only example datasets available for testing on the hub were CSV — **not** because the hub or the `flame.star` transport is limited to CSV. The hub/`flame.star` layer is type-agnostic: it delivers whatever bytes sit at the S3 keys configured in `query`/`DATA_S3_KEYS` (see `hub_entry.py`, which already references a `VCF_S3_KEYS` precedent from an example script — i.e. genomic-style keys are an established pattern on the hub, just not the ones configured for this deployment). So genomic and imaging analysis were implemented and unit-tested as standalone, self-contained modules, but pipeline integration was never exercised end-to-end on real hub-delivered data, simply because no genomic/imaging test data was ever provided for that purpose.

### Step E — persistence/reporting (deferred)

Saving results to `LOCAL_RESULTS_DIR/node{N}/genomic/` and surfacing them in the PDF/JSON report sections — depends on step D, still deferred for the same reason.

### Tests added

- `tests/test_get_data_utils.py` — `detect_file_type` coverage for every new extension (case-insensitive too) plus `stage_genomic_file` (primary file, sibling index file, no-index case).
- `data_report/statistical_analysis/local/test_genomic_data_analysis.py` — `analyze_genomic_file` coverage for all four genomic types plus the unsupported-type error case.

Full project test suite: 286 passed.
