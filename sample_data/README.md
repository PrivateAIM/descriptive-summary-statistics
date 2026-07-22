# Sample Data

Small, real, publicly available files downloaded for local manual testing of `genomic_data_analysis.py` and `imaging_data_analysis.py`, and for citation as test samples. Not used by the automated pytest suite (those use synthetic, in-code fixtures — see `GENOMIC_ANALYSIS_CHANGES.md` / `IMAGING_ANALYSIS_README.md`).

## `vcf/test-vcf-hdr-in.vcf`

- **Source**: [`samtools/htslib`](https://github.com/samtools/htslib), `test/test-vcf-hdr-in.vcf` (downloaded from `https://raw.githubusercontent.com/samtools/htslib/develop/test/test-vcf-hdr-in.vcf`).
- **What it is**: a small (10-variant, single-sample) VCF from htslib's own header-parsing test suite. Despite being a test fixture, it contains a realistic mix of variant types — SNPs (including multi-allelic), a deletion, an insertion, and two no-call/complex records — making it a good minimal demonstration file for `variant_counts_per_sample`/`variant_type_distribution`.
- **License**: htslib's repository license is not a single SPDX-identified license (GitHub reports it as "Other"); check `https://github.com/samtools/htslib/blob/develop/LICENSE` for exact terms before any redistribution.
- **Verified output** (run via `python -c "from data_report.statistical_analysis.local.genomic_data_analysis import *; print(variant_counts_per_sample('sample_data/vcf/test-vcf-hdr-in.vcf')); print(variant_type_distribution('sample_data/vcf/test-vcf-hdr-in.vcf'))"`):
  ```
  variant_counts_per_sample: {'NA00001': 10}
  variant_type_distribution: {'SNP': 5, 'deletion': 2, 'insertion': 1, 'other': 2}
  ```
  (htslib prints harmless `[W::vcf_parse] Contig '...' is not defined in the header` warnings — the file doesn't declare `##contig` lines for chromosomes 1/4/16, which doesn't affect parsing.)

## `dicom/US1_J2KI.dcm`

- **Source**: [`pydicom/pydicom-data`](https://github.com/pydicom/pydicom-data), `data_store/data/US1_J2KI.dcm` (downloaded from `https://raw.githubusercontent.com/pydicom/pydicom-data/master/data_store/data/US1_J2KI.dcm`).
- **What it is**: a real, de-identified ultrasound (modality `US`) DICOM file, JPEG2000-compressed, used by pydicom's own test suite. 640×480 pixels.
- **License**: MIT (per the `pydicom-data` repository).
- **Verified output**:
  ```
  file_size_statistics:               {'n_files': 1, 'total_size': 0.056, 'average_size': 0.056, 'min_size': 0.056, 'max_size': 0.056}
  resolution_statistics:              {'n_images': 1, 'min_width': 640, 'max_width': 640, 'average_width': 640, 'min_height': 480, 'max_height': 480, 'average_height': 480}
  distribution_by_modality:           {'US': 1}
  distribution_by_anatomical_region:  {'counts': {'Unknown': 1}, 'is_degenerate': True, 'reliability_note': 'BodyPartExamined is inconsistently populated in real-world DICOM data; treat this distribution as best-effort, not authoritative.'}
  ```
  Note this file has no `BodyPartExamined` tag at all — a real-world illustration of exactly the reliability caveat documented in `IMAGING_ANALYSIS_README.md` for that statistic.
