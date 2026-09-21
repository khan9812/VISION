"""
Central color palette for visualization outputs.
"""

from __future__ import annotations

from typing import List, Tuple

import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


PALETTE_HEX: Tuple[str, ...] = (
    "#BF4E58",
    "#4971A6",
    "#56A662",
    "#D97652",
    "#F2F2F2",
)

PALETTE_PRIMARY: Tuple[str, ...] = PALETTE_HEX[:4]
PALETTE_NEUTRAL: str = PALETTE_HEX[4]
PALETTE_TEXT_DARK: str = PALETTE_HEX[0]
PALETTE_TEXT_LIGHT: str = PALETTE_HEX[4]


def palette_hex(index: int, include_neutral: bool = False) -> str:
    palette = PALETTE_HEX if include_neutral else PALETTE_PRIMARY
    return palette[index % len(palette)]


def palette_rgb(index: int, include_neutral: bool = False) -> Tuple[float, float, float]:
    return tuple(mcolors.to_rgb(palette_hex(index, include_neutral=include_neutral)))


def palette_bgr255(index: int, include_neutral: bool = False) -> Tuple[int, int, int]:
    r, g, b = palette_rgb(index, include_neutral=include_neutral)
    return (int(b * 255), int(g * 255), int(r * 255))


def palette_sequence(n: int, include_neutral: bool = False) -> List[Tuple[float, float, float]]:
    return [palette_rgb(i, include_neutral=include_neutral) for i in range(max(n, 0))]


def make_value_colormap(name: str = "framework_value_cmap") -> LinearSegmentedColormap:
    # Small -> large mapping: blue -> green -> orange -> red
    return LinearSegmentedColormap.from_list(
        name,
        [PALETTE_HEX[1], PALETTE_HEX[2], PALETTE_HEX[3], PALETTE_HEX[0]],
    )


def rgba(hex_color: str, alpha: float) -> Tuple[float, float, float, float]:
    r, g, b = mcolors.to_rgb(hex_color)
    return (r, g, b, alpha)

