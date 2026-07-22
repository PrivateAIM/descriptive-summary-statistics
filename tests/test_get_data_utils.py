import pytest

from data_report.get_data.utils import detect_file_type, stage_genomic_file


@pytest.mark.parametrize(
    "filename,expected_type",
    [
        ("sample.csv", "csv"),
        ("sample.XLSX", "excel"),
        ("notes.txt", "txt"),
        ("payload.json", "json"),
        ("scan.PNG", "image"),
        ("reads.fastq", "fastq"),
        ("reads.fastq.gz", "fastq"),
        ("reads.fq.gz", "fastq"),
        ("alignment.BAM", "alignment"),
        ("alignment.sam", "alignment"),
        ("alignment.cram", "alignment"),
        ("variants.vcf", "vcf"),
        ("variants.VCF.gz", "vcf"),
        ("genome.fasta", "fasta"),
        ("genome.fa", "fasta"),
        ("unknown.xyz", None),
    ],
)
def test_detect_file_type(filename, expected_type):
    assert detect_file_type(filename) == expected_type


def test_stage_genomic_file_writes_primary_file(tmp_path):
    node_data = {"sample.vcf": b"##fileformat=VCFv4.2\n"}

    path = stage_genomic_file("sample.vcf", node_data, tmp_path)

    assert path == tmp_path / "sample.vcf"
    assert path.read_bytes() == node_data["sample.vcf"]


def test_stage_genomic_file_stages_sibling_index(tmp_path):
    node_data = {
        "alignment.bam": b"bam-bytes",
        "alignment.bam.bai": b"index-bytes",
    }

    stage_genomic_file("alignment.bam", node_data, tmp_path)

    assert (tmp_path / "alignment.bam").read_bytes() == b"bam-bytes"
    assert (tmp_path / "alignment.bam.bai").read_bytes() == b"index-bytes"


def test_stage_genomic_file_without_index_does_not_error(tmp_path):
    node_data = {"genome.fasta": b">seq1\nACGT\n"}

    path = stage_genomic_file("genome.fasta", node_data, tmp_path)

    assert path.read_bytes() == b">seq1\nACGT\n"
    assert list(tmp_path.iterdir()) == [path]
