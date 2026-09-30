"""Run two-band dual-window OCT reconstruction and plotting.

Example:
    python run_dual_window_oct.py \
        --data-dir test_data \
        --pixel-map test_data/OCT_pixelMap.mat \
        --output output_dual_window/dual_window_center.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from oct_hsv_python_stage1.dual_window import (
    plot_dual_window_result,
    reconstruct_dual_window_pair,
)


def _paths_from_args(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.s1 is not None or args.s2 is not None:
        if args.s1 is None or args.s2 is None:
            raise ValueError("Provide both --s1 and --s2, or use --data-dir.")
        return Path(args.s1), Path(args.s2), Path(args.pixel_map)

    data_dir = Path(args.data_dir)
    s1_files = sorted(data_dir.glob(args.s1_glob))
    s2_files = sorted(data_dir.glob(args.s2_glob))
    if not s1_files:
        raise FileNotFoundError(f"No S1 files found with {data_dir / args.s1_glob}")
    if len(s1_files) != len(s2_files):
        raise ValueError(f"Found {len(s1_files)} S1 files and {len(s2_files)} S2 files.")

    index = args.slice_index
    if index is None:
        index = len(s1_files) // 2
    if not 0 <= index < len(s1_files):
        raise IndexError(f"--slice-index must be between 0 and {len(s1_files) - 1}.")
    return s1_files[index], s2_files[index], Path(args.pixel_map)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full-band, short-band, and long-band dual-window OCT plots."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--data-dir", help="Folder containing sorted S1/S2 TIFF pairs.")
    input_group.add_argument("--s1", help="Single S1 TIFF path.")
    parser.add_argument("--s2", help="Single S2 TIFF path; required with --s1.")
    parser.add_argument("--pixel-map", required=True, help="MAT file containing pixelMap.")
    parser.add_argument("--s1-glob", default="*S1*.tif*", help="Glob used inside --data-dir.")
    parser.add_argument("--s2-glob", default="*S2*.tif*", help="Glob used inside --data-dir.")
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Zero-based B-scan index for --data-dir. Defaults to center slice.",
    )
    parser.add_argument(
        "--output",
        default="output_dual_window/dual_window_oct.png",
        help="Path for the saved summary figure.",
    )
    parser.add_argument("--c2", type=float, default=38.02, help="Manual dispersion C2.")
    parser.add_argument("--c3", type=float, default=-54.63, help="Manual dispersion C3.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the Matplotlib figure interactively after saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    s1_path, s2_path, pixel_map_path = _paths_from_args(args)
    result = reconstruct_dual_window_pair(
        s1_path,
        s2_path,
        pixel_map_path,
        c2=args.c2,
        c3=args.c3,
    )
    output_path = Path(args.output)
    plot_dual_window_result(result, output_path=output_path)
    suffix = output_path.suffix or ".png"
    saved_paths = [
        output_path.with_name(f"{output_path.stem}_windows_full_band{suffix}"),
        output_path.with_name(f"{output_path.stem}_component_windows{suffix}"),
        output_path.with_name(f"{output_path.stem}_full_band_as_broad{suffix}"),
    ]

    print(f"S1: {s1_path}")
    print(f"S2: {s2_path}")
    print("Saved figures:")
    for saved_path in saved_paths:
        print(f"  {saved_path}")
    print(
        "Dual-window centers: "
        f"short {result.metadata['short_center_nm']:.1f} nm, "
        f"long {result.metadata['long_center_nm']:.1f} nm"
    )
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
