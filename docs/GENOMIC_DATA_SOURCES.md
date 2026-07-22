# Genomic Data — Sources and Provenance

All files in `genomic data/` correspond to the same individual (1000 Genomes
sample **HG00096**), the same chromosome (**chr20**), and the same reference
build (**GRCh37 / b37**), so that the FASTQ → BAM → VCF → FASTA files used in
the thesis are internally consistent with each other.

## Files in use

| File | Format | Size | Description |
|---|---|---|---|
| `SRR062634_1.fastq` | FASTQ | 6.4 GB | Raw sequencing reads, one of the runs (SRR062634) whose reads were aligned into the BAM below — verified by matching read names (`SRR062634.*`) against BAM read groups. Provided by the user; not re-fetched. |
| `HG00096_chr20.bam` | BAM | 318 MB | Low-coverage whole-chr20 alignment for sample HG00096, aligned to GRCh37/b37. Provided by the user; not re-fetched. **Renamed 2026-07-19** from the original `HG00096.chrom20.ILLUMINA.bwa.GBR.low_coverage.20120522.bam` for readability in the thesis table (content unchanged). Canonical location (1000 Genomes phase 1/3 alignment index): `https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00096/alignment/` |
| `chr20_HG00096_subset.vcf` | VCF | 8.5 MB | **Fetched 2026-07-19.** Subset of the 1000 Genomes phase 3 chr20 call set, filtered to region `20:1-2,000,000` and sample `HG00096` only (59,445 variant records). Source file: `https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr20.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz` — subset extracted with `bcftools view -r 20:1-2000000 -s HG00096 <url>` (remote query over HTTPS, no full download of the ~1.2 GB source file). |
| `GRCh37_chr20.fa` | FASTA | 64 MB | **Fetched 2026-07-19.** GRCh37 chromosome 20 reference sequence, matching the alignment build of the BAM and VCF above (sequence ID `20`, length 63,025,520 bp — matches the `##contig` line in the VCF header exactly). **Renamed 2026-07-19** from `Homo_sapiens.GRCh37.dna.chromosome.20.fa` for readability in the thesis table (content unchanged). Source: `https://ftp.ensembl.org/pub/grch37/release-113/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.dna.chromosome.20.fa.gz` |

## Files moved to `genomic data/unused/`

| File | Reason removed |
|---|---|
| `ALL.chr20.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf` | Truncated during original download — only 64 KB / 255 lines, contains the VCF header but stops before any variant records. `variant_counts_per_sample` / `variant_type_distribution` would return all-zero / empty results on this file. Replaced by `chr20_HG00096_subset.vcf` above. |
| `chr20.fa` | Valid FASTA, but wrong reference build: 64,444,167 bp, which is the **GRCh38/hg38** chr20 length, not the GRCh37/b37 build the BAM and VCF are aligned to (63,025,520 bp). Kept for reference but not used in analysis. Replaced by `Homo_sapiens.GRCh37.dna.chromosome.20.fa` above. |

## Notes

- The VCF subset was deliberately restricted to a 2 Mb region and a single sample (rather than the full 2,504-sample population panel) to keep the file small while still producing a genuine, non-degenerate variant set for the same individual represented in the BAM/FASTQ.
- `bcftools`/`tabix` fetched the subset via HTTP range requests against the remote bgzipped/tabix-indexed VCF, so the full ~1.2 GB source file was never downloaded locally.
