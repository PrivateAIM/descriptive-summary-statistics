"""Analysis functions for genomic sequencing and variant data.

Supports FASTQ (short-read sequencing), BAM/SAM (aligned reads), VCF
(variant call format), and FASTA (reference / assembled sequences).
Each function targets a single file or directory and returns a plain dict
suitable for JSON serialisation.
"""

from pathlib import Path
from collections import defaultdict
import statistics
import gzip
from Bio import SeqIO
import pysam
import re
import numpy as np
from cyvcf2 import VCF

EXTENSIONS = (".fastq", ".fastq.gz", ".fq", ".fq.gz", ".bam", ".vcf", ".vcf.gz")
"""Genomic file extensions recognised by this module."""

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
    ".vcf.gz": "VCF",
}
"""Mapping from file extension to genomic format label."""


def count_samples(data_dir):
    """Count the number of unique sequencing samples in a directory.

    Sample identity is inferred from filenames: Illumina-style suffixes
    (_001, _R1/_R2, _L001) are stripped before de-duplication, so paired-end
    files for the same sample are counted as one.

    Args:
        data_dir (str or Path): Directory containing genomic files.

    Returns:
        dict: A dict with keys:
            - ``n_samples`` (int): Number of unique sample names found.
            - ``samples`` (list[str]): Sorted list of unique sample names.
    """
    sample_names = set()
    for file in Path(data_dir).iterdir():
        if not file.is_file():
            continue

        filename = file.name
        filename_lower = filename.lower()
        if not filename_lower.endswith(EXTENSIONS):
            continue
        sample_name = filename

        for ext in EXTENSIONS:
            if filename_lower.endswith(ext):
                sample_name = sample_name[: len(sample_name) - len(ext)]
                break

        # Illumina filenames end with suffixes in this order:
        # ..._L001_R1_001 -> strip from the end inward, not R1/L001 first.
        sample_name = re.sub(r"_001$", "", sample_name)
        sample_name = re.sub(r"_R[12]$", "", sample_name)
        sample_name = re.sub(r"_L\d+$", "", sample_name)
        sample_names.add(sample_name)

    return {
        "n_samples": len(sample_names),
        "samples": sorted(sample_names),
    }


def file_format_overview(data_dir):
    """Summarise the genomic file formats present in a directory.

    Args:
        data_dir (str or Path): Directory to scan.

    Returns:
        dict: Keys are format labels (e.g. ``"FASTQ"``, ``"BAM"``, ``"VCF"``)
            and values are the count of files with that format.
            Files with unrecognised extensions are silently skipped.
    """
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


def file_size(file_path):
    """Return the size of a single file in megabytes.

    Args:
        file_path (str or Path): Path to the file.

    Returns:
        float: File size in megabytes.

    Raises:
        ValueError: If ``file_path`` does not point to a regular file.
    """
    path = Path(file_path)
    if path.is_file():
        return path.stat().st_size
    raise ValueError("Path is not a file")


def file_size_statistics(data_directory):
    """Compute summary size statistics for all files in a directory.

    Args:
        data_directory (str or Path): Directory to scan.

    Returns:
        dict: Keys ``n_files``, ``total_size``, ``average_size``, ``min_size``,
            ``max_size`` (all sizes in megabytes, rounded to 3 dp).
            Returns an empty dict if no files are found.
    """
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
        "max_size": round(max(file_sizes), 3),
    }


def _open_fastq(fastq_file):
    """Open a FASTQ file for reading, handling gzip compression transparently."""
    path = str(fastq_file)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def count_reads(fastq_file):
    """Count the total number of reads in a FASTQ file.

    Args:
        fastq_file (str or Path): Path to a plain or gzip-compressed FASTQ file.

    Returns:
        int: Total number of sequencing reads.
    """
    count = 0
    with _open_fastq(fastq_file) as handle:
        for read in SeqIO.parse(handle, "fastq"):
            count = count + 1
    return count


def read_length_statistics(fastq_file):
    """Compute read length statistics from a FASTQ file.

    Args:
        fastq_file (str or Path): Path to a plain or gzip-compressed FASTQ file.

    Returns:
        dict: Keys ``n_reads``, ``total_reads``, ``min_length``,
            ``max_length``, ``average_length`` (lengths in base pairs).
            Returns an empty dict if the file contains no reads.
    """
    total_reads = 0
    total_bases = 0
    min_length = float("inf")
    max_length = 0

    with _open_fastq(fastq_file) as handle:
        for record in SeqIO.parse(handle, "fastq"):
            read_length = len(record.seq)
            total_reads += 1
            total_bases += read_length
            min_length = min(min_length, read_length)
            max_length = max(max_length, read_length)

    if total_reads == 0:
        return {}

    average_length = total_bases / total_reads
    return {
        "n_reads": total_reads,
        "total_reads": total_reads,
        "min_length": min_length,
        "max_length": max_length,
        "average_length": average_length,
    }


def coverage_statistics(path, reference_filename=None):
    """Compute per-base sequencing depth statistics from an alignment file.

    Args:
        path (str): Path to a BAM, SAM, or CRAM alignment file.
        reference_filename (str, optional): Path to the reference FASTA,
            required when ``path`` is a CRAM file.

    Returns:
        dict: Keys ``mean_depth``, ``min_depth``, ``max_depth``,
            ``median_depth`` (all in units of reads per base).
            Returns an empty dict if the file has no pileup columns.

    Raises:
        ValueError: If ``path`` does not have a recognised alignment extension
            (``.bam``, ``.sam``, or ``.cram``).
    """
    if path.endswith(".bam"):
        obj = pysam.AlignmentFile(path, "rb")
    elif path.endswith(".sam"):
        obj = pysam.AlignmentFile(path, "r")
    elif path.endswith(".cram"):
        obj = pysam.AlignmentFile(path, "rc", reference_filename=reference_filename)
    else:
        raise ValueError(f"Unsupported file format: {path}")

    depths = []
    for pileup_col in obj.pileup():
        depths.append(pileup_col.n)
    obj.close()

    if not depths:
        return {}

    depths = np.array(depths)
    return {
        "mean_depth": round(float(np.mean(depths)), 3),
        "min_depth": int(np.min(depths)),
        "max_depth": int(np.max(depths)),
        "median_depth": float(np.median(depths)),
    }


def variant_counts_per_sample(vcf_file):
    """Count the number of variants carried by each sample in a VCF file.

    Only heterozygous (HET) and homozygous-alternate (HOM_ALT) genotypes
    are counted; homozygous-reference and missing calls are ignored.

    Args:
        vcf_file (str or Path): Path to a VCF or bgzipped VCF file.

    Returns:
        dict: Mapping of sample name (str) → variant count (int).
    """
    vcf = VCF(vcf_file, gts012=True)
    counts = {sample: 0 for sample in vcf.samples}

    for variant in vcf:
        for sample, gt_type in zip(vcf.samples, variant.gt_types):
            if gt_type in (1, 2):  # HET or HOM_ALT
                counts[sample] += 1
    vcf.close()

    return counts


def variant_type_distribution(vcf_file):
    """Compute the distribution of variant types in a VCF file.

    Args:
        vcf_file (str or Path): Path to a VCF or bgzipped VCF file.

    Returns:
        dict: Mapping of variant type label (str) → count (int). Labels
            are ``"SNP"``, ``"insertion"``, ``"deletion"``, ``"SV"`` (structural
            variant), ``"MNP"`` (multi-nucleotide polymorphism), ``"other"``.
    """
    vcf = VCF(vcf_file)
    distribution = defaultdict(int)

    for variant in vcf:
        if variant.is_snp:
            distribution["SNP"] += 1
        elif variant.is_indel:
            if variant.is_deletion:
                distribution["deletion"] += 1
            else:
                distribution["insertion"] += 1
        elif variant.is_sv:
            distribution["SV"] += 1
        elif variant.is_mnp:
            distribution["MNP"] += 1
        else:
            distribution["other"] += 1
    vcf.close()

    return dict(distribution)


def fasta_handling(fasta_file):
    """Extract sequence metadata from a FASTA file.

    Args:
        fasta_file (str or Path): Path to a plain FASTA file.

    Returns:
        dict: Keys ``n_sequences``, ``sequence_ids`` (list[str]),
            ``min_length``, ``max_length``, ``average_length`` (all lengths
            in base pairs). Returns an empty dict if the file has no sequences.
    """
    sequence_ids = []
    lengths = []

    with open(fasta_file) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequence_ids.append(record.id)
            lengths.append(len(record.seq))

    if not lengths:
        return {}

    return {
        "n_sequences": len(lengths),
        "sequence_ids": sequence_ids,
        "min_length": min(lengths),
        "max_length": max(lengths),
        "average_length": sum(lengths) / len(lengths),
    }


def analyze_genomic_file(file_path, file_type):
    """Dispatch a genomic file to the appropriate analysis function.

    Routes the file to ``count_reads`` + ``read_length_statistics`` (FASTQ),
    ``coverage_statistics`` (BAM/SAM/CRAM), ``variant_counts_per_sample`` +
    ``variant_type_distribution`` (VCF), or ``fasta_handling`` (FASTA)
    based on ``file_type``.

    Args:
        file_path (str or Path): Path to the genomic file.
        file_type (str): One of ``"fastq"``, ``"alignment"``, ``"vcf"``,
            ``"fasta"``.

    Returns:
        dict: Analysis results; structure depends on ``file_type``.

    Raises:
        ValueError: If ``file_type`` is not one of the recognised values.
    """
    file_path = str(file_path)

    if file_type == "fastq":
        result = {"read_count": count_reads(file_path)}
        result.update(read_length_statistics(file_path))
        return result
    if file_type == "alignment":
        return coverage_statistics(file_path)
    if file_type == "vcf":
        return {
            "variant_counts_per_sample": variant_counts_per_sample(file_path),
            "variant_type_distribution": variant_type_distribution(file_path),
        }
    if file_type == "fasta":
        return fasta_handling(file_path)

    raise ValueError(f"Unsupported genomic file type: {file_type!r}")
