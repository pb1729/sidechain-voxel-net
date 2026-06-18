#!/usr/bin/env python3
"""Plot an RGB slice of a saved density field."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from density_fns import DENSFN_FORWARD_1_CHANNELS, DensFn1, DensityFunction
from density_io import load_density_file


AXIS_TO_DIM = {"X": 0, "Y": 1, "Z": 2}


def _empty_color_map() -> np.ndarray:
    return np.zeros((len(DENSFN_FORWARD_1_CHANNELS), 3), dtype=np.float32)


def _channel_color_map(colors: dict[str, tuple[float, float, float]]) -> np.ndarray:
    color_map = _empty_color_map()
    for channel, color in colors.items():
        color_map[DENSFN_FORWARD_1_CHANNELS.index(channel)] = color
    return color_map


COLOR_MAPS = {
    "atoms": _channel_color_map({
        "backbone_ca": (0.55, 0.55, 0.55), "backbone_c": (0.55, 0.55, 0.55),
        "backbone_n": (0.2, 0.2, 1.0), "backbone_o": (1.0, 0.2, 0.15),
        "sidechain_n": (0.2, 0.2, 1.0), "sidechain_o": (1.0, 0.2, 0.15),
        "sidechain_s": (1.0, 0.85, 0.15),
        "sidechain_c_grey": (0.55, 0.55, 0.55),
        "sidechain_c_blue": (0.55, 0.55, 0.55),
    }),
    "backbone": _channel_color_map({
        "backbone_ca": (0.2, 1.0, 0.2), "backbone_c": (1.0, 0.3, 0.2),
        "backbone_n": (0.2, 0.35, 1.0), "backbone_o": (1.0, 0.2, 0.15),
        "backbone_bond": (0.9, 0.9, 0.9), "chain_start": (0.0, 1.0, 0.7),
        "chain_end": (1.0, 0.1, 0.9),
    }),
    "sidechain": _channel_color_map({
        "sidechain_n": (0.2, 0.35, 1.0), "sidechain_o": (1.0, 0.2, 0.15),
        "sidechain_s": (1.0, 0.85, 0.15),
        "sidechain_c_grey": (0.55, 0.55, 0.55),
        "sidechain_c_blue": (0.0, 0.85, 1.0),
    }),
    "decode": _channel_color_map({
        "backbone_ca": (1.0, 1.0, 1.0), "backbone_o": (1.0, 0.15, 0.1),
        "sidechain_c_grey": (0.55, 0.55, 0.55),
        "sidechain_c_blue": (0.0, 0.75, 1.0),
        "sidechain_n": (0.15, 0.25, 1.0), "sidechain_o": (1.0, 0.15, 0.1),
        "sidechain_s": (1.0, 0.8, 0.1), "chain_start": (0.0, 1.0, 0.35),
        "chain_end": (1.0, 0.0, 0.8),
    }),
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot an RGB slice from a saved density file."
    )
    parser.add_argument("input_path", nargs="?", help="saved density file")
    parser.add_argument(
        "channels", nargs="*", metavar="CHANNEL",
        help="three channel names or indexes to use as RGB",
    )
    parser.add_argument("--axis", choices=tuple(AXIS_TO_DIM), default="Z")
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--color-map", choices=tuple(COLOR_MAPS), default=None)
    parser.add_argument("--list-channels", action="store_true")
    parser.add_argument("--list-color-maps", action="store_true")
    return parser.parse_args(argv)


def _density_slice(
    field: np.ndarray, axis_dim: int, slice_i: int, smoothing: float
) -> np.ndarray:
    if smoothing == 0.0:
        return np.asarray(np.take(field, slice_i, axis=axis_dim), dtype=np.float32)
    coords = np.arange(field.shape[axis_dim], dtype=np.float32)
    weights = np.exp(-0.5 * ((coords - slice_i) / smoothing) ** 2).astype(np.float32)
    shape = [1, 1, 1, 1]
    shape[axis_dim] = field.shape[axis_dim]
    return np.asarray((field * weights.reshape(shape)).sum(axis=axis_dim), dtype=np.float32)


def _rgb_image(density_slice: np.ndarray, color_map: np.ndarray) -> np.ndarray:
    return np.asarray(density_slice @ color_map, dtype=np.float32)


def _parse_channels(values: list[str], densfn: DensityFunction) -> list[int]:
    channels = []
    for value in values:
        try:
            channel_i = int(value)
        except ValueError:
            if not isinstance(densfn, DensFn1):
                raise ValueError("DensFn2 projected channels must be selected by index")
            try:
                channel_i = DENSFN_FORWARD_1_CHANNELS.index(value)
            except ValueError as exc:
                raise ValueError(f"unknown DensFn1 channel {value!r}") from exc
        if channel_i < 0 or channel_i >= densfn.channel_count():
            raise ValueError(
                f"channel index must be in [0, {densfn.channel_count() - 1}], got {channel_i}"
            )
        channels.append(channel_i)
    return channels


def _print_density_stats(field: np.ndarray) -> None:
    mean = float(np.mean(field))
    std = float(np.std(field))
    minimum = float(np.min(field))
    maximum = float(np.max(field))
    rms = float(np.sqrt(np.mean(np.square(field))))
    print(
        f"density stats: mean={mean:.6g} std={std:.6g} min={minimum:.6g} "
        f"max={maximum:.6g} rms={rms:.6g}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.list_channels:
        for i, channel in enumerate(DENSFN_FORWARD_1_CHANNELS):
            print(f"{i}: {channel}")
        return 0
    if args.list_color_maps:
        for name in COLOR_MAPS:
            print(name)
        return 0
    if args.input_path is None:
        raise ValueError("input_path is required unless a listing flag is used")
    if args.smoothing < 0.0:
        raise ValueError("--smoothing must be non-negative")
    if args.noise_level < 0.0:
        raise ValueError("--noise-level must be non-negative")
    if args.vmax is not None and args.vmax <= 0.0:
        raise ValueError("--vmax must be positive")

    grid, field, densfn = load_density_file(args.input_path)
    channels = _parse_channels(args.channels, densfn)
    if args.color_map is not None:
        if not isinstance(densfn, DensFn1):
            raise ValueError("--color-map applies only to DensFn1 density fields")
        if channels:
            raise ValueError("pass either --color-map or three RGB channels")
        color_map = COLOR_MAPS[args.color_map]
        color_desc = f"map={args.color_map}"
    else:
        if len(channels) != 3:
            raise ValueError("exactly three channels are required for RGB plotting")
        color_map = np.zeros((densfn.channel_count(), 3), dtype=np.float32)
        for rgb_i, channel_i in enumerate(channels):
            color_map[channel_i, rgb_i] = 1.0
        color_desc = "RGB=" + ",".join(str(channel) for channel in channels)

    if args.noise_level > 0.0:
        rng = np.random.default_rng(args.seed)
        field = field + rng.normal(0.0, args.noise_level, field.shape).astype(np.float32)
    _print_density_stats(field)

    axis_dim = AXIS_TO_DIM[args.axis]
    initial_slice_i = int(grid.N[axis_dim] // 2)

    def normalized_image(slice_i: int) -> np.ndarray:
        density_slice = _density_slice(field, axis_dim, slice_i, args.smoothing)
        image = _rgb_image(density_slice, color_map)
        vmax = float(np.max(image)) if args.vmax is None else args.vmax
        if vmax <= 0.0:
            vmax = 1.0
        return np.swapaxes(np.clip(image / vmax, 0.0, 1.0), 0, 1)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.18)
    image_artist = ax.imshow(
        normalized_image(initial_slice_i), origin="lower", interpolation="nearest"
    )
    shown_axes = [axis for axis in ("X", "Y", "Z") if axis != args.axis]
    ax.set_xlabel(f"grid {shown_axes[0]}")
    ax.set_ylabel(f"grid {shown_axes[1]}")

    def set_title(slice_i: int) -> None:
        ax.set_title(
            f"{args.input_path} | {args.axis}={slice_i} | "
            f"smoothing={args.smoothing:g} | noise={args.noise_level:g} | {color_desc}"
        )

    def update_slice(value: float) -> None:
        slice_i = int(round(value))
        image_artist.set_data(normalized_image(slice_i))
        set_title(slice_i)
        fig.canvas.draw_idle()

    set_title(initial_slice_i)
    slider_ax = fig.add_axes([0.15, 0.06, 0.7, 0.03])
    slice_slider = Slider(
        slider_ax, f"{args.axis} slice", 0, int(grid.N[axis_dim] - 1),
        valinit=initial_slice_i, valstep=1,
    )
    slice_slider.on_changed(update_slice)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
