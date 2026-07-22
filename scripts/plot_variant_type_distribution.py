"""Bar chart of the VCF variant type distribution for sample HG00096 (chr20:1-2,000,000).

Single-hue bar chart (dataviz skill: magnitude comparison across a nominal
category -> sequential/single-slot color, not one hue per bar). Linear scale
is intentional: the SNP-dominant skew (57,438 vs. low hundreds/tens for the
other types) is itself the finding, and direct labels on every bar keep the
small values readable despite their small bar heights.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BLUE = "#2a78d6"
SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

DATA = {"SNP": 57438, "Deletion": 1163, "Insertion": 794, "SV": 50}

OUT_DIR = "results/genomic_analysis"


def main():
    labels = list(DATA.keys())
    values = list(DATA.values())

    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = range(len(labels))
    ax.bar(x, values, width=0.5, color=BLUE, zorder=3)

    headroom = max(values) * 0.22
    for xi, v in zip(x, values):
        ax.text(
            xi, v + headroom * 0.18, f"{v:,}",
            ha="center", va="bottom", fontsize=9, color=PRIMARY_INK, zorder=4,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10, color=PRIMARY_INK)
    ax.set_ylim(0, max(values) + headroom)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.tick_params(axis="y", labelsize=9, colors=MUTED_INK, length=0)
    ax.tick_params(axis="x", length=0)

    ax.set_ylabel("Variants", fontsize=10, color=MUTED_INK)
    ax.set_title(
        "Variant type distribution — sample HG00096, chr20:1-2,000,000",
        fontsize=11, color=PRIMARY_INK, pad=14, loc="left",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1)

    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    fig.savefig(f"{OUT_DIR}/variant_type_distribution.png", facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/variant_type_distribution.pdf", facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {OUT_DIR}/variant_type_distribution.png")
    print(f"Wrote {OUT_DIR}/variant_type_distribution.pdf")


if __name__ == "__main__":
    main()
