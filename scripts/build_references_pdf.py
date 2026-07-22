"""Build a PDF listing all genomic/imaging data sources used, with descriptions,
corresponding files, and BibTeX entries for LaTeX.
"""

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted, HRFlowable,
)
from reportlab.lib import colors

OUT_PATH = "results/references/data_sources_references.pdf"

styles = getSampleStyleSheet()
title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=18, spaceAfter=6)
subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=10, textColor=colors.grey, spaceAfter=18)
h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=4, textColor=colors.HexColor("#1a1a1a"))
label_style = ParagraphStyle("Label", parent=styles["Normal"], fontSize=9.5, textColor=colors.HexColor("#52514e"), spaceAfter=2, leading=13)
body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=6, alignment=TA_LEFT)
code_style = ParagraphStyle("Code", parent=styles["Code"], fontSize=8, leading=10.5, backColor=colors.HexColor("#f5f5f3"), borderPadding=6)
section_style = ParagraphStyle("Section", parent=styles["Heading1"], fontSize=15, spaceBefore=20, spaceAfter=8, textColor=colors.HexColor("#0b0b0b"))

REFERENCES = [
    {
        "section": "Genomic data",
        "title": "1. 1000 Genomes Project — Chromosome 20 Variant Calls (VCF)",
        "file": "chr20_HG00096_subset.vcf",
        "fetched_by": "Fetched by assistant (original file provided was truncated/header-only)",
        "source": "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/",
        "description": (
            "The 1000 Genomes Project catalogued genetic variation across thousands of human "
            "genomes from populations worldwide. This file is a subset of the phase 3 "
            "chromosome 20 release, filtered to sample HG00096 and the region 20:1-2,000,000. "
            "It was extracted from the official FTP release via a remote bcftools query, so the "
            "full ~1.2 GB chromosome file was never downloaded in full."
        ),
        "bibtex": """@misc{1000genomes_vcf,
  author       = {{1000 Genomes Project Consortium}},
  title        = {1000 Genomes Phase 3 Chromosome 20 Variant Calls},
  year         = {2015},
  howpublished = {\\url{https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/}},
  note         = {Subset used: sample HG00096, region 20:1-2{,}000{,}000}
}""",
    },
    {
        "section": "Genomic data",
        "title": "2. 1000 Genomes Project — HG00096 Alignment (BAM)",
        "file": "HG00096_chr20.bam",
        "fetched_by": "Provided directly by user",
        "source": "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00096/alignment/",
        "description": (
            "This file contains sequencing reads for sample HG00096 aligned to chromosome 20 "
            "of the GRCh37 reference genome using BWA. It represents low-coverage whole-genome "
            "sequencing data generated as part of the 1000 Genomes Project's per-sample "
            "alignment releases."
        ),
        "bibtex": """@misc{1000genomes_bam,
  author       = {{1000 Genomes Project Consortium}},
  title        = {HG00096 Low-Coverage Alignment, Chromosome 20 (BAM)},
  year         = {2012},
  howpublished = {\\url{https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00096/alignment/}}
}""",
    },
    {
        "section": "Genomic data",
        "title": "3. European Nucleotide Archive — Raw Reads (FASTQ)",
        "file": "SRR062634_1.fastq",
        "fetched_by": "Provided directly by user",
        "source": "https://www.ebi.ac.uk/ena/browser/view/SRR062634",
        "description": (
            "This file contains raw, unaligned Illumina sequencing reads (mate 1 of a "
            "paired-end run) for sample HG00096, generated as part of the 1000 Genomes "
            "Project's British (GBR) population sequencing effort (study PRJNA41223). It was "
            "retrieved from the European Nucleotide Archive under run accession SRR062634."
        ),
        "bibtex": """@misc{ena_srr062634,
  author       = {{European Nucleotide Archive}},
  title        = {Sequencing Run SRR062634 (Study PRJNA41223)},
  year         = {2010},
  howpublished = {\\url{https://www.ebi.ac.uk/ena/browser/view/SRR062634}}
}""",
    },
    {
        "section": "Genomic data",
        "title": "4. Ensembl — GRCh37 Chromosome 20 Reference (FASTA)",
        "file": "GRCh37_chr20.fa",
        "fetched_by": "Fetched by assistant (original file provided was the wrong genome build, GRCh38)",
        "source": "https://ftp.ensembl.org/pub/grch37/release-113/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.dna.chromosome.20.fa.gz",
        "description": (
            "This file is the chromosome 20 reference sequence for the GRCh37/hg19 human "
            "genome assembly. It matches the alignment build used for the BAM and VCF files "
            "above, and was downloaded from Ensembl's GRCh37 archive."
        ),
        "bibtex": """@misc{ensembl_grch37_chr20,
  author       = {{Ensembl}},
  title        = {Homo sapiens GRCh37 Chromosome 20 Reference Sequence},
  year         = {2015},
  howpublished = {\\url{https://ftp.ensembl.org/pub/grch37/release-113/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.dna.chromosome.20.fa.gz}}
}""",
    },
    {
        "section": "Imaging data — DICOM (The Cancer Imaging Archive)",
        "title": "5. TCIA — RIDER Pilot Collection",
        "file": "dicom_series/series_01, series_02, series_03",
        "fetched_by": "Fetched by assistant via the NBIA API",
        "source": "https://doi.org/10.7937/m87f-mz83",
        "description": (
            "RIDER Pilot is a CT chest imaging collection from The Cancer Imaging Archive's "
            "Reference Image Database to Evaluate Response initiative. The series used here are "
            "CT chest scans from a single subject, retrieved via the NBIA API using series "
            "identifiers from a manifest built at nbia-search."
        ),
        "bibtex": """@misc{tcia_rider_pilot,
  author       = {{The Cancer Imaging Archive}},
  title        = {RIDER Pilot Collection},
  year         = {2023},
  doi          = {10.7937/m87f-mz83},
  howpublished = {\\url{https://doi.org/10.7937/m87f-mz83}}
}""",
    },
    {
        "section": "Imaging data — DICOM (The Cancer Imaging Archive)",
        "title": "6. TCIA — CT COLONOGRAPHY Collection",
        "file": "dicom_series/series_04, series_05, series_06, series_07",
        "fetched_by": "Fetched by assistant via the NBIA API",
        "source": "https://doi.org/10.7937/K9/TCIA.2015.NWTESAY1",
        "description": (
            "CT COLONOGRAPHY is a collection of abdominal CT scans acquired for colorectal "
            "cancer screening research, hosted on The Cancer Imaging Archive. The series used "
            "here cover a single subject's supine and prone colonography acquisitions."
        ),
        "bibtex": """@misc{tcia_ct_colonography,
  author       = {{The Cancer Imaging Archive}},
  title        = {CT COLONOGRAPHY Collection},
  year         = {2013},
  doi          = {10.7937/K9/TCIA.2015.NWTESAY1},
  howpublished = {\\url{https://doi.org/10.7937/K9/TCIA.2015.NWTESAY1}}
}""",
    },
    {
        "section": "Imaging data — DICOM (The Cancer Imaging Archive)",
        "title": "7. TCIA — MIDI-B Synthetic Validation / Test Collections",
        "file": "dicom_series/series_08, series_09, series_13, series_14, series_15",
        "fetched_by": "Fetched by assistant via the NBIA API",
        "source": "https://doi.org/10.7937/cf2p-aw56",
        "description": (
            "MIDI-B is a synthetic medical imaging dataset from The Cancer Imaging Archive's "
            "Medical Imaging De-Identification Benchmark, spanning MR and PET series. It "
            "consists of realistic but artificially generated patient imaging data, so no real "
            "patients are represented."
        ),
        "bibtex": """@misc{tcia_midib_synthetic,
  author       = {{The Cancer Imaging Archive}},
  title        = {MIDI-B Synthetic Validation and Test Collections},
  year         = {2025},
  doi          = {10.7937/cf2p-aw56},
  howpublished = {\\url{https://doi.org/10.7937/cf2p-aw56}}
}""",
    },
    {
        "section": "Imaging data — DICOM (The Cancer Imaging Archive)",
        "title": "8. TCIA — CMB-MML Collection",
        "file": "dicom_series/series_10, series_11, series_12",
        "fetched_by": "Fetched by assistant via the NBIA API",
        "source": "https://doi.org/10.7937/szkb-sw39",
        "description": (
            "CMB-MML is part of the Cancer Moonshot Biobank program hosted on The Cancer "
            "Imaging Archive, contributing X-ray angiography (XA) imaging. The series used "
            "here are single-image spine X-ray acquisitions from one subject."
        ),
        "bibtex": """@misc{tcia_cmb_mml,
  author       = {{The Cancer Imaging Archive}},
  title        = {CMB-MML Collection},
  year         = {2026},
  doi          = {10.7937/szkb-sw39},
  howpublished = {\\url{https://doi.org/10.7937/szkb-sw39}}
}""",
    },
    {
        "section": "Imaging data — DICOM (The Cancer Imaging Archive)",
        "title": "9. TCIA — General Repository Citation",
        "file": "All files in dicom_series/ (series_01-15)",
        "fetched_by": "N/A — repository-level citation",
        "source": "https://doi.org/10.1007/s10278-013-9622-7",
        "description": (
            "The Cancer Imaging Archive (TCIA) is a public repository of de-identified medical "
            "imaging data for cancer research. TCIA's data usage policy requires citing this "
            "general repository paper in addition to each specific collection's own DOI, listed "
            "individually above."
        ),
        "bibtex": """@article{tcia_general,
  author  = {Clark, Kenneth and Vendt, Bruce and Smith, Kirk and Freymann, John and
             Kirby, Justin and Koppel, Paul and Moore, Stephen and Phillips, Stanley and
             Maffitt, David and Pringle, Michael and Tarbox, Lawrence and Prior, Fred},
  title   = {The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public
             Information Repository},
  journal = {Journal of Digital Imaging},
  year    = {2013},
  doi     = {10.1007/s10278-013-9622-7}
}""",
    },
    {
        "section": "Imaging data — PNG",
        "title": "10. NIH ChestX-ray14",
        "file": "images_001/images/*.png (4,999 files)",
        "fetched_by": "Provided directly by user",
        "source": "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737",
        "description": (
            "ChestX-ray14 (originally released as ChestX-ray8) is a large public dataset of "
            "frontal-view chest X-ray images released by the NIH Clinical Center for "
            "weakly-supervised thoracic disease classification research. The images used here "
            "are drawn from the dataset's public Box distribution."
        ),
        "bibtex": """@inproceedings{nih_chestxray14,
  author    = {Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and
               Bagheri, Mohammadhadi and Summers, Ronald M.},
  title     = {ChestX-ray8: Hospital-Scale Chest X-ray Database and Benchmarks on
               Weakly-Supervised Classification and Localization of Common Thorax Diseases},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern
               Recognition (CVPR)},
  year      = {2017},
  howpublished = {\\url{https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737}}
}""",
    },
]


def build():
    doc = SimpleDocTemplate(
        OUT_PATH, pagesize=LETTER,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.85 * inch, rightMargin=0.85 * inch,
    )
    story = []

    story.append(Paragraph("Data Sources and References", title_style))
    story.append(Paragraph(
        "Genomic and imaging data used in this thesis — description, corresponding file(s), "
        "download source, and LaTeX/BibTeX citation for each.",
        subtitle_style,
    ))

    current_section = None
    for ref in REFERENCES:
        if ref["section"] != current_section:
            current_section = ref["section"]
            story.append(Paragraph(current_section, section_style))
            story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#c3c2b7"), spaceAfter=6))

        story.append(Paragraph(ref["title"], h2_style))
        story.append(Paragraph(f"<b>File(s):</b> {ref['file']}", label_style))
        story.append(Paragraph(f"<b>Source:</b> {ref['source']}", label_style))
        story.append(Paragraph(f"<b>Obtained:</b> {ref['fetched_by']}", label_style))
        story.append(Spacer(1, 4))
        story.append(Paragraph(ref["description"], body_style))
        story.append(Spacer(1, 4))
        story.append(Preformatted(ref["bibtex"], code_style))
        story.append(Spacer(1, 8))

    doc.build(story)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
