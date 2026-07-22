"""
Global visual style for all generated figures.

Changing PALETTE here (or calling set_palette / set_theme) before any plots
are generated will propagate everywhere, because primitives.py accesses
style.PALETTE as a module attribute at call time — not at import time.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Default palette
# ---------------------------------------------------------------------------
PALETTE: list[str] = [
    "#4C9BE8",  # blue
    "#F28B30",  # orange
    "#3BB37E",  # green
    "#E85C4C",  # red
    "#A77FD3",  # purple
    "#F7C948",  # yellow
    "#72C9CF",  # teal
    "#D98CB0",  # pink
]

# ---------------------------------------------------------------------------
# Default resolution
# ---------------------------------------------------------------------------
DPI: int = 200

# ---------------------------------------------------------------------------
# Named themes
# ---------------------------------------------------------------------------
THEMES: dict[str, list[str]] = {
    "default": [
        "#4C9BE8", "#F28B30", "#3BB37E", "#E85C4C",
        "#A77FD3", "#F7C948", "#72C9CF", "#D98CB0",
    ],
    # Wong (2011) colorblind-safe palette — safe for deuteranopia and protanopia
    "colorblind": [
        "#0072B2", "#E69F00", "#009E73", "#D55E00",
        "#CC79A7", "#56B4E9", "#F0E442", "#000000",
    ],
    "grayscale": [
        "#222222", "#555555", "#888888", "#AAAAAA",
        "#BBBBBB", "#CCCCCC", "#DDDDDD", "#EEEEEE",
    ],
}

# ---------------------------------------------------------------------------
# Theme switching
# ---------------------------------------------------------------------------

def set_palette(new_palette: list[str]) -> None:
    """Replace the active colour palette used by all subsequent plots.

    Args:
        new_palette (list[str]): Ordered list of hex colour strings. Must
            contain at least as many entries as the maximum number of
            distinct series expected in any single chart.
    """
    global PALETTE
    PALETTE = list(new_palette)


def set_theme(name: str) -> None:
    """Switch the active palette to a named theme.

    Args:
        name (str): Theme identifier; must be a key in THEMES. Built-in
            options are "default", "colorblind", and "grayscale".

    Raises:
        KeyError: If name is not found in THEMES.
    """
    if name not in THEMES:
        raise KeyError(
            f"Unknown theme '{name}'. Available themes: {list(THEMES.keys())}"
        )
    set_palette(THEMES[name])
