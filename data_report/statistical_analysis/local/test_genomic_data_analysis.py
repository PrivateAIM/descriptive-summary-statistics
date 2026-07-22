import gzip

import pysam
import pytest

from data_report.statistical_analysis.local.genomic_data_analysis import (
    analyze_genomic_file,
    count_reads,
    count_samples,
    coverage_statistics,
    fasta_handling,
    read_length_statistics,
    variant_counts_per_sample,
    variant_type_distribution,
)

VCF_CONTENT = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1\tsample2
chr1\t100\t.\tA\tG\t50\tPASS\t.\tGT\t0/1\t0/0
chr1\t200\t.\tAT\tA\t50\tPASS\t.\tGT\t1/1\t0/1
chr1\t300\t.\tA\tATT\t50\tPASS\t.\tGT\t0/0\t1/1
chr1\t400\t.\tA\tC\t50\tPASS\t.\tGT\t0/1\t0/1
"""


@pytest.fixture
def vcf_file(tmp_path):
    path = tmp_path / "variants.vcf"
    path.write_text(VCF_CONTENT)
    return str(path)


def test_variant_counts_per_sample(vcf_file):
    result = variant_counts_per_sample(vcf_file)
    assert result == {"sample1": 3, "sample2": 3}


def test_variant_type_distribution(vcf_file):
    result = variant_type_distribution(vcf_file)
    assert result == {"SNP": 2, "deletion": 1, "insertion": 1}


def test_count_samples_merges_paired_end_illumina_names(tmp_path):
    (tmp_path / "SampleA_S1_L001_R1_001.fastq.gz").touch()
    (tmp_path / "SampleA_S1_L001_R2_001.fastq.gz").touch()

    result = count_samples(tmp_path)
    assert result == {"n_samples": 1, "samples": ["SampleA_S1"]}


def test_count_samples_extension_matching_is_case_insensitive(tmp_path):
    (tmp_path / "Sample1.FASTQ").touch()

    result = count_samples(tmp_path)
    assert result == {"n_samples": 1, "samples": ["Sample1"]}


FASTQ_RECORDS = (
    "@read1\nACGTACGT\n+\nIIIIIIII\n"
    "@read2\nACGT\n+\nIIII\n"
)


def test_read_length_statistics(tmp_path):
    fastq_file = tmp_path / "reads.fastq"
    fastq_file.write_text(FASTQ_RECORDS)

    result = read_length_statistics(fastq_file)
    assert result["n_reads"] == 2
    assert result["min_length"] == 4
    assert result["max_length"] == 8
    assert result["average_length"] == 6


def test_read_length_statistics_gzip(tmp_path):
    fastq_file = tmp_path / "reads.fastq.gz"
    with gzip.open(fastq_file, "wt") as handle:
        handle.write(FASTQ_RECORDS)

    result = read_length_statistics(fastq_file)
    assert result["n_reads"] == 2


def test_read_length_statistics_empty_file_returns_empty_dict(tmp_path):
    fastq_file = tmp_path / "empty.fastq"
    fastq_file.write_text("")

    assert read_length_statistics(fastq_file) == {}


def test_count_reads_gzip(tmp_path):
    fastq_file = tmp_path / "reads.fastq.gz"
    with gzip.open(fastq_file, "wt") as handle:
        handle.write(FASTQ_RECORDS)

    assert count_reads(fastq_file) == 2


def test_fasta_handling(tmp_path):
    fasta_file = tmp_path / "sequences.fasta"
    fasta_file.write_text(">seq1\nACGTACGT\n>seq2\nACGT\n")

    result = fasta_handling(fasta_file)
    assert result["n_sequences"] == 2
    assert result["sequence_ids"] == ["seq1", "seq2"]
    assert result["min_length"] == 4
    assert result["max_length"] == 8
    assert result["average_length"] == 6


def test_fasta_handling_empty_file_returns_empty_dict(tmp_path):
    fasta_file = tmp_path / "empty.fasta"
    fasta_file.write_text("")

    assert fasta_handling(fasta_file) == {}


def test_coverage_statistics_empty_bam_returns_empty_dict(tmp_path):
    bam_file = tmp_path / "empty.bam"
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 1000}]}
    with pysam.AlignmentFile(str(bam_file), "wb", header=header) as handle:
        pass
    pysam.index(str(bam_file))

    assert coverage_statistics(str(bam_file)) == {}


def test_analyze_genomic_file_fastq(tmp_path):
    fastq_file = tmp_path / "reads.fastq"
    fastq_file.write_text(FASTQ_RECORDS)

    result = analyze_genomic_file(fastq_file, "fastq")
    assert result["read_count"] == 2
    assert result["n_reads"] == 2


def test_analyze_genomic_file_vcf(vcf_file):
    result = analyze_genomic_file(vcf_file, "vcf")
    assert result["variant_counts_per_sample"] == {"sample1": 3, "sample2": 3}
    assert result["variant_type_distribution"] == {"SNP": 2, "deletion": 1, "insertion": 1}


def test_analyze_genomic_file_fasta(tmp_path):
    fasta_file = tmp_path / "sequences.fasta"
    fasta_file.write_text(">seq1\nACGTACGT\n>seq2\nACGT\n")

    result = analyze_genomic_file(fasta_file, "fasta")
    assert result["n_sequences"] == 2


def test_analyze_genomic_file_alignment(tmp_path):
    bam_file = tmp_path / "empty.bam"
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 1000}]}
    with pysam.AlignmentFile(str(bam_file), "wb", header=header):
        pass
    pysam.index(str(bam_file))

    assert analyze_genomic_file(bam_file, "alignment") == {}


def test_analyze_genomic_file_unsupported_type_raises(tmp_path):
    with pytest.raises(ValueError):
        analyze_genomic_file(tmp_path / "data.csv", "csv")
