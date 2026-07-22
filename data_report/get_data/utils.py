"""File-type detection and genomic staging utilities for the data-report pipeline.

``detect_file_type`` classifies a filename by extension so that callers can
route raw bytes to the appropriate parser.  ``stage_genomic_file`` writes
genomic file bytes (and any sibling index files) to a temporary directory so
that tools such as pysam or cyvcf2 can open them by path.
"""
from pathlib import Path


def detect_file_type(filename: str) -> str:
    """Return a short string identifying the file type based on its extension.

    Args:

        filename (str): The filename (or path) to classify.  Only the
            extension is examined; the value is compared case-insensitively.

    Returns:

        str: One of ``"csv"``, ``"excel"``, ``"txt"``, ``"json"``,
            ``"image"``, ``"fastq"``, ``"alignment"``, ``"vcf"``,
            ``"fasta"``, or ``None`` if the extension is unrecognised.
    """
    name = filename.lower()
    if name.endswith(".csv"):
        return "csv"
    if name.endswith(".xlsx"):
        return "excel"
    if name.endswith(".txt"):
        return "txt"
    if name.endswith(".json"):
        return "json"
    if name.endswith((".png", ".jpg", ".jpeg")):
        return "image"
    if name.endswith((".fastq", ".fastq.gz", ".fq", ".fq.gz")):
        return "fastq"
    if name.endswith((".bam", ".sam", ".cram")):
        return "alignment"
    if name.endswith((".vcf", ".vcf.gz")):
        return "vcf"
    if name.endswith((".fasta", ".fa", ".fasta.gz")):
        return "fasta"
    return None


GENOMIC_INDEX_SUFFIXES = (".bai", ".csi", ".tbi", ".crai")


# Genomic tools (pysam, cyvcf2) require a real file on disk, unlike pandas'
# BytesIO-based CSV reading -- this stages a node's in-memory file bytes
# (plus any sibling index file, e.g. sample.bam.bai) into a temp directory.
def stage_genomic_file(filename, node_data, tmp_dir):
    """Write a genomic file and its index file(s) to a temporary directory.

    Genomic tools such as pysam and cyvcf2 require real paths on disk; they
    cannot read from in-memory byte streams.  This helper writes the primary
    file and any recognised sibling index files (``*.bai``, ``*.csi``,
    ``*.tbi``, ``*.crai``) from the ``node_data`` byte mapping into
    ``tmp_dir``.

    Args:

        filename (str): Key in ``node_data`` for the primary genomic file
            (e.g. ``"sample.bam"``).
        node_data (dict): Mapping of filename to raw bytes for all files
            belonging to this node.
        tmp_dir (str | Path): Directory where the files should be written.

    Returns:

        Path: Absolute path to the staged primary file.
    """
    tmp_dir = Path(tmp_dir)
    primary_path = tmp_dir / filename
    primary_path.write_bytes(node_data[filename])

    for suffix in GENOMIC_INDEX_SUFFIXES:
        index_name = filename + suffix
        if index_name in node_data:
            (tmp_dir / index_name).write_bytes(node_data[index_name])

    return primary_path


# function to deal with multi-modal data
# function to list the types of files in each dataset folder
# function to store the sources of each dataset and maybe other metadata
