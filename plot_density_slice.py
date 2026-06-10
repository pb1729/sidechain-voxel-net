#!/usr/bin/env python3
"""Plot a 2D RGB slice of densfn_forward_1 for a CIF file."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from density_fns import DENSFN_FORWARD_1_CHANNELS, densfn_forward_1
from kspace_ops import blur as kspace_blur, nf_1
from parse_cif import read_protein_cif_with_codes


AXIS_TO_DIM = {
    "X": 0,
    "Y": 1,
    "Z": 2,
}


def _empty_color_map() -> np.ndarray:
    return np.zeros((len(DENSFN_FORWARD_1_CHANNELS), 3), dtype=np.float32)


def _channel_color_map(colors: dict[str, tuple[float, float, float]]) -> np.ndarray:
    color_map = _empty_color_map()
    for channel, color in colors.items():
        color_map[DENSFN_FORWARD_1_CHANNELS.index(channel)] = color
    return color_map


COLOR_MAPS = {
    "atoms": _channel_color_map(
        {
            "backbone_ca": (0.55, 0.55, 0.55),
            "backbone_c": (0.55, 0.55, 0.55),
            "backbone_n": (0.2, 0.2, 1.0),
            "backbone_o": (1.0, 0.2, 0.15),
            "sidechain_n": (0.2, 0.2, 1.0),
            "sidechain_o": (1.0, 0.2, 0.15),
            "sidechain_s": (1.0, 0.85, 0.15),
            "sidechain_c_grey": (0.55, 0.55, 0.55),
            "sidechain_c_blue": (0.55, 0.55, 0.55),
        }
    ),
    "backbone": _channel_color_map(
        {
            "backbone_ca": (0.2, 1.0, 0.2),
            "backbone_c": (1.0, 0.3, 0.2),
            "backbone_n": (0.2, 0.35, 1.0),
            "backbone_o": (1.0, 0.2, 0.15),
            "backbone_bond": (0.9, 0.9, 0.9),
            "chain_start": (0.0, 1.0, 0.7),
            "chain_end": (1.0, 0.1, 0.9),
        }
    ),
    "sidechain": _channel_color_map(
        {
            "sidechain_n": (0.2, 0.35, 1.0),
            "sidechain_o": (1.0, 0.2, 0.15),
            "sidechain_s": (1.0, 0.85, 0.15),
            "sidechain_c_grey": (0.55, 0.55, 0.55),
            "sidechain_c_blue": (0.0, 0.85, 1.0),
        }
    ),
    "decode": _channel_color_map(
        {
            "backbone_ca": (1.0, 1.0, 1.0),
            "backbone_o": (1.0, 0.15, 0.1),
            "sidechain_c_grey": (0.55, 0.55, 0.55),
            "sidechain_c_blue": (0.0, 0.75, 1.0),
            "sidechain_n": (0.15, 0.25, 1.0),
            "sidechain_o": (1.0, 0.15, 0.1),
            "sidechain_s": (1.0, 0.8, 0.1),
            "chain_start": (0.0, 1.0, 0.35),
            "chain_end": (1.0, 0.0, 0.8),
        }
    ),
}


def _parse_channel(channel: str) -> int:
    try:
        channel_i = int(channel)
    except ValueError:
        try:
            return DENSFN_FORWARD_1_CHANNELS.index(channel)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"unknown channel {channel!r}; use one of {DENSFN_FORWARD_1_CHANNELS}"
            ) from exc

    if channel_i < 0 or channel_i >= len(DENSFN_FORWARD_1_CHANNELS):
        raise argparse.ArgumentTypeError(
            f"channel index must be in [0, {len(DENSFN_FORWARD_1_CHANNELS) - 1}]"
        )
    return channel_i


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot an RGB slice through densfn_forward_1 for a CIF file."
    )
    parser.add_argument("cif_path", nargs="?", help="path to the .cif file to plot")
    parser.add_argument(
        "channels",
        nargs="*",
        type=_parse_channel,
        metavar="CHANNEL",
        help="three channel names or indexes to use as RGB",
    )
    parser.add_argument(
        "--axis",
        choices=tuple(AXIS_TO_DIM),
        default="Z",
        help="axis to slice along; default: Z",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="density value mapped to full brightness; default: max of current slice",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
        help="Gaussian standard deviation along the sliced axis before summing; default: 0.0",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.0,
        help="standard deviation of Gaussian noise added to the density field; default: 0.0",
    )
    parser.add_argument(
        "--kspace-blur",
        action="store_true",
        help="show an extra t slider for k-space blurring/noising",
    )
    parser.add_argument(
        "--blur-max",
        type=float,
        default=8.0,
        help="maximum k-space Gaussian blur sigma for --kspace-blur; default: 8.0",
    )
    parser.add_argument(
        "--kspace-scale",
        type=float,
        default=1.5,
        help="FFT padding scale for --kspace-blur; default: 1.5",
    )
    parser.add_argument(
        "--initial-t",
        type=float,
        default=1.0,
        help="initial t value for --kspace-blur; default: 1.0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for optional noise; default: 0",
    )
    parser.add_argument(
        "--color-map",
        choices=tuple(COLOR_MAPS),
        default=None,
        help="pre-baked channel-to-RGB color map; positional RGB channels are ignored",
    )
    parser.add_argument(
        "--list-channels",
        action="store_true",
        help="print available density channels and exit",
    )
    parser.add_argument(
        "--list-color-maps",
        action="store_true",
        help="print available pre-baked color maps and exit",
    )
    return parser.parse_args(argv)


def _density_slice(
    field: np.ndarray,
    axis_dim: int,
    slice_i: int,
    smoothing: float,
) -> np.ndarray:
    if smoothing == 0.0:
        if axis_dim == 0:
            density_slice = field[slice_i, :, :, :]
        elif axis_dim == 1:
            density_slice = field[:, slice_i, :, :]
        else:
            density_slice = field[:, :, slice_i, :]
        return np.asarray(density_slice, dtype=np.float32)

    coords = np.arange(field.shape[axis_dim], dtype=np.float32)
    weights = np.exp(-0.5 * ((coords - slice_i) / smoothing) ** 2).astype(np.float32)
    shape = [1, 1, 1, 1]
    shape[axis_dim] = field.shape[axis_dim]
    weighted_field = field * weights.reshape(shape)
    return np.asarray(weighted_field.sum(axis=axis_dim), dtype=np.float32)


def _individual_channel_color_map(channels: list[int]) -> np.ndarray:
    color_map = _empty_color_map()
    for rgb_i, channel_i in enumerate(channels):
        color_map[channel_i, rgb_i] = 1.0
    return color_map


def _rgb_image(density_slice: np.ndarray, color_map: np.ndarray) -> np.ndarray:
    return np.asarray(density_slice @ color_map, dtype=np.float32)


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
    if args.cif_path is None:
        raise ValueError("cif_path is required unless a listing flag is used")
    if args.color_map is None and len(args.channels) != 3:
        raise ValueError("exactly three channels are required for RGB plotting")
    if args.color_map is not None and len(args.channels) not in (0, 3):
        raise ValueError("pass either --color-map alone or exactly three positional RGB channels")
    if args.smoothing < 0.0:
        raise ValueError("--smoothing must be non-negative")
    if args.noise_level < 0.0:
        raise ValueError("--noise-level must be non-negative")
    if args.blur_max < 0.0:
        raise ValueError("--blur-max must be non-negative")
    if args.kspace_scale <= 0.0:
        raise ValueError("--kspace-scale must be positive")
    if args.initial_t < 0.0 or args.initial_t > 1.0:
        raise ValueError("--initial-t must be between 0 and 1")
    color_map = (
        COLOR_MAPS[args.color_map]
        if args.color_map is not None
        else _individual_channel_color_map(args.channels)
    )

    protein = read_protein_cif_with_codes(args.cif_path)
    grid, field = densfn_forward_1(protein)
    print(f"RMS field value: {(field**2).mean()**0.5}")
    if args.noise_level > 0.0:
        rng = np.random.default_rng(args.seed)
        field = field + rng.normal(0.0, args.noise_level, field.shape).astype(np.float32)
    axis_dim = AXIS_TO_DIM[args.axis]
    initial_slice_i = int(grid.N[axis_dim] // 2)
    field_tensor = None
    kspace_noise = None
    if args.kspace_blur:
        field_tensor = torch.from_numpy(np.moveaxis(field, -1, 0).copy()).float()
        generator = torch.Generator(device=field_tensor.device)
        generator.manual_seed(args.seed)
        _initial_field, kspace_noise = kspace_blur(
            lambda kx, ky, kz: nf_1(torch.tensor(0.0), kx, ky, kz, blur_max=args.blur_max),
            field_tensor,
            scale=args.kspace_scale,
            generator=generator,
            return_noise=True
        )

    def field_at_t(t: float) -> np.ndarray:
        if not args.kspace_blur:
            return field
        assert field_tensor is not None
        assert kspace_noise is not None
        t = torch.as_tensor(t)
        noised = kspace_blur(
            lambda kx, ky, kz: nf_1(t, kx, ky, kz, blur_max=args.blur_max),
            field_tensor,
            scale=args.kspace_scale,
            noise=kspace_noise,
        )
        noised = noised.detach().cpu().numpy()
        return np.ascontiguousarray(np.moveaxis(noised, 0, -1).astype(np.float32))

    def normalized_image(slice_i: int, t: float) -> np.ndarray:
        density_slice = _density_slice(field_at_t(t), axis_dim, slice_i, args.smoothing)
        image = _rgb_image(density_slice, color_map)
        image = np.clip(image, 0.0, 1.0)
        return np.swapaxes(image, 0, 1)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    color_desc = (
        f"map={args.color_map}"
        if args.color_map is not None
        else "RGB=" + ", ".join(DENSFN_FORWARD_1_CHANNELS[i] for i in args.channels)
    )
    state = {"slice_i": initial_slice_i, "t": args.initial_t}
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.27 if args.kspace_blur else 0.18)
    image_artist = ax.imshow(
        normalized_image(state["slice_i"], state["t"]),
        origin="lower",
        interpolation="nearest",
    )
    shown_axes = [axis for axis in ("X", "Y", "Z") if axis != args.axis]
    ax.set_xlabel(f"grid {shown_axes[0]}")
    ax.set_ylabel(f"grid {shown_axes[1]}")

    def set_title() -> None:
        blur_desc = ""
        if args.kspace_blur:
            blur_desc = f" | t={state['t']:.3f} | blur_max={args.blur_max:g}"
        ax.set_title(
            f"{args.cif_path} | {args.axis}={state['slice_i']} | "
            f"smoothing={args.smoothing:g} | noise={args.noise_level:g}"
            f"{blur_desc} | {color_desc}"
        )

    def update_image() -> None:
        image_artist.set_data(normalized_image(state["slice_i"], state["t"]))
        set_title()
        fig.canvas.draw_idle()

    def update_slice(value: float) -> None:
        state["slice_i"] = int(round(value))
        update_image()

    def update_t(value: float) -> None:
        state["t"] = float(value)
        update_image()

    set_title()
    slice_slider_ax = fig.add_axes([0.15, 0.13 if args.kspace_blur else 0.06, 0.7, 0.03])
    slice_slider = Slider(
        slice_slider_ax,
        f"{args.axis} slice",
        0,
        int(grid.N[axis_dim] - 1),
        valinit=initial_slice_i,
        valstep=1,
    )
    slice_slider.on_changed(update_slice)
    if args.kspace_blur:
        t_slider_ax = fig.add_axes([0.15, 0.06, 0.7, 0.03])
        t_slider = Slider(t_slider_ax, "t", 0.0, 1.0, valinit=args.initial_t)
        t_slider.on_changed(update_t)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
