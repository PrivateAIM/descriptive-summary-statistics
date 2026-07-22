"""Run genomic_data_analysis.py across the files in genomic data/ and write
the results as a CSV summary table (thesis-ready) plus a JSON backup (full detail).

Usage:
    python run_genomic_analysis.py
"""

import csv
import json
from pathlib import Path

from data_report.statistical_analysis.local.genomic_data_analysis import (
    read_length_statistics,
    coverage_statistics,
    variant_counts_per_sample,
    variant_type_distribution,
    fasta_handling,
)

GENOMIC_DATA_DIR = Path("genomic data")
OUTPUT_DIR = Path("results") / "genomic_analysis"

FILES = {
    "fastq": GENOMIC_DATA_DIR / "SRR062634_1.fastq",
    "bam": GENOMIC_DATA_DIR / "HG00096_chr20.bam",
    "vcf": GENOMIC_DATA_DIR / "chr20_HG00096_subset.vcf",
    "fasta": GENOMIC_DATA_DIR / "GRCh37_chr20.fa",
}

CSV_COLUMNS = ["file", "format", "size_mb", "key_stats"]


def size_mb(path):
    return round(path.stat().st_size / (1024 ** 2), 3)


def analyze():
    rows = []
    raw = {}

    fastq_stats = read_length_statistics(str(FILES["fastq"]))
    if fastq_stats["min_length"] == fastq_stats["max_length"]:
        length_line = f"Read length: {fastq_stats['min_length']} bp"
    else:
        length_line = f"Read length: {fastq_stats['min_length']}-{fastq_stats['max_length']} bp"
    rows.append({
        "file": FILES["fastq"].name,
        "format": "FASTQ",
        "size_mb": size_mb(FILES["fastq"]),
        "key_stats": "\n".join([
            f"Reads: {fastq_stats['n_reads']:,}",
            length_line,
        ]),
    })
    raw["fastq"] = fastq_stats

    bam_stats = coverage_statistics(str(FILES["bam"]))
    rows.append({
        "file": FILES["bam"].name,
        "format": "BAM",
        "size_mb": size_mb(FILES["bam"]),
        "key_stats": "\n".join([
            f"Mean depth: {bam_stats['mean_depth']}x",
            f"Min depth: {bam_stats['min_depth']}",
            f"Max depth: {bam_stats['max_depth']}",
            f"Median depth: {bam_stats['median_depth']}",
        ]),
    })
    raw["bam"] = bam_stats

    vcf_counts = variant_counts_per_sample(str(FILES["vcf"]))
    vcf_dist = variant_type_distribution(str(FILES["vcf"]))
    n_sites = sum(vcf_dist.values())
    n_sample_variants = sum(vcf_counts.values())
    dist_lines = [
        f"{label}: {vcf_dist[label]:,}"
        for label in sorted(vcf_dist, key=vcf_dist.get, reverse=True)
    ]
    rows.append({
        "file": FILES["vcf"].name,
        "format": "VCF",
        "size_mb": size_mb(FILES["vcf"]),
        "key_stats": "\n".join([
            f"Variant genotypes (HG00096): {n_sample_variants:,}",
            f"Sites classified: {n_sites:,}",
            "Region: 20:1-2,000,000",
            *dist_lines,
        ]),
    })
    raw["vcf"] = {"variant_counts_per_sample": vcf_counts, "variant_type_distribution": vcf_dist}

    fasta_stats = fasta_handling(str(FILES["fasta"]))
    rows.append({
        "file": FILES["fasta"].name,
        "format": "FASTA",
        "size_mb": size_mb(FILES["fasta"]),
        "key_stats": "\n".join([
            f"Sequence length: {fasta_stats['min_length']:,} bp",
            f"Sequence ID: {fasta_stats['sequence_ids'][0]}",
        ]),
    })
    raw["fasta"] = fasta_stats

    return rows, raw


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, raw = analyze()

    csv_path = OUTPUT_DIR / "genomic_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    json_path = OUTPUT_DIR / "genomic_summary_full.json"
    with open(json_path, "w") as f:
        json.dump(raw, f, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
