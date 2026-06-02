#!/usr/bin/env python3
"""Round-trip a CIF through density, VAE latent, density, and atom positions."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from compare_density_roundtrip import protein_rmsd, residue_code_mismatches
from density_fns import DENSFN_FORWARD_1_CHANNELS, Grid, densfn_backward_1, densfn_forward_1
from parse_cif import read_protein_cif_with_codes
from train_vae import VAE


AXIS_TO_DIM = {
    "X": 0,
    "Y": 1,
    "Z": 2,
}


def _parse_error_channel(channel: str) -> int | str:
    if channel in ("rms", "max_abs"):
        return channel
    try:
        channel_i = int(channel)
    except ValueError:
        try:
            return DENSFN_FORWARD_1_CHANNELS.index(channel)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"unknown channel {channel!r}; use a channel name/index, rms, or max_abs"
            ) from exc

    if channel_i < 0 or channel_i >= len(DENSFN_FORWARD_1_CHANNELS):
        raise argparse.ArgumentTypeError(
            f"channel index must be in [0, {len(DENSFN_FORWARD_1_CHANNELS) - 1}]"
        )
    return channel_i


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode a CIF as densfn_forward_1 density, pass it through a saved VAE "
            "latent, decode density, then recover atom positions with densfn_backward_1."
        )
    )
    parser.add_argument("cif_path", help="path to the .cif file to round-trip")
    parser.add_argument("vae_path", help="path to a saved train_vae.py checkpoint")
    parser.add_argument(
        "--latent-noise-level",
        type=float,
        default=0.0,
        help="standard deviation of Gaussian noise added to the VAE latent; default: 0.0",
    )
    parser.add_argument(
        "--peak-threshold",
        type=float,
        default=0.55,
        help="density peak threshold passed to densfn_backward_1; default: 0.55",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for grid rotation and optional latent noise; default: None",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device for VAE inference; default: cuda if available else cpu",
    )
    parser.add_argument(
        "--show-density-error",
        action="store_true",
        help="display a scrollable imshow of pointwise decoded-minus-input density error",
    )
    parser.add_argument(
        "--error-axis",
        choices=tuple(AXIS_TO_DIM),
        default="Z",
        help="axis to slice along in --show-density-error; default: Z",
    )
    parser.add_argument(
        "--error-channel",
        type=_parse_error_channel,
        default="rms",
        help=(
            "density error channel to display by name/index, or aggregate rms/max_abs; "
            "default: rms"
        ),
    )
    parser.add_argument(
        "--error-vmax",
        type=float,
        default=1.0,
        help="absolute error value mapped to full color; default: 1.0",
    )
    parser.add_argument(
        "--error-smoothing",
        type=float,
        default=0.0,
        help="Gaussian standard deviation along the sliced axis before summing errors; default: 0.0",
    )
    return parser.parse_args(argv)


def _load_vae(path: str, device: torch.device) -> VAE:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    return VAE.from_dict(checkpoint).to(device).eval()


def _crop_field_to_even_grid(grid: Grid, field: np.ndarray) -> tuple[Grid, np.ndarray]:
    spatial_shape = np.asarray(field.shape[:3], dtype=np.int32)
    even_shape = 2 * (spatial_shape // 2)
    if np.any(even_shape <= 0):
        raise ValueError(f"density grid is too small for VAE stride-2 round-trip: {tuple(spatial_shape)}")
    if np.array_equal(spatial_shape, even_shape):
        return grid, field

    cropped = field[: even_shape[0], : even_shape[1], : even_shape[2], :]
    cropped_grid = Grid(dx=grid.dx, N=even_shape, transform=grid.transform)
    return cropped_grid, np.ascontiguousarray(cropped)


def _field_to_tensor(field: np.ndarray, device: torch.device) -> torch.Tensor:
    channels_first = np.moveaxis(field, -1, 0)
    channels_first = np.ascontiguousarray(channels_first)
    return torch.from_numpy(channels_first).unsqueeze(0).to(device)


def _tensor_to_field(tensor: torch.Tensor) -> np.ndarray:
    density = tensor.squeeze(0).detach().float().cpu().numpy()
    return np.ascontiguousarray(np.moveaxis(density, 0, -1).astype(np.float32))


def _add_latent_noise(
    z: torch.Tensor,
    noise_level: float,
    seed: int | None,
) -> torch.Tensor:
    if noise_level == 0.0:
        return z
    generator = None
    if seed is not None:
        generator = torch.Generator(device=z.device)
        generator.manual_seed(seed)
    noise = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator)
    return z + noise_level * noise


def _comparable_fields(field: np.ndarray, decoded_field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    comparable_shape = tuple(min(a, b) for a, b in zip(field.shape, decoded_field.shape))
    return (
        field[
            : comparable_shape[0],
            : comparable_shape[1],
            : comparable_shape[2],
            : comparable_shape[3],
        ],
        decoded_field[
            : comparable_shape[0],
            : comparable_shape[1],
            : comparable_shape[2],
            : comparable_shape[3],
        ],
    )


def _print_density_error_stats(field: np.ndarray, decoded_field: np.ndarray) -> float:
    error = decoded_field - field
    density_mse = float(np.mean(error ** 2))
    print(f"Density MSE: {density_mse:.8f}")
    print("Density error by channel:")
    print("channel                      input_max decoded_max       mse      rmse       mae   max_abs      bias")
    for channel_i, channel in enumerate(DENSFN_FORWARD_1_CHANNELS):
        input_chan = field[..., channel_i]
        decoded_chan = decoded_field[..., channel_i]
        err_chan = error[..., channel_i]
        mse = float(np.mean(err_chan ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(err_chan)))
        max_abs = float(np.max(np.abs(err_chan)))
        bias = float(np.mean(err_chan))
        input_max = float(np.max(input_chan))
        decoded_max = float(np.max(decoded_chan))
        print(
            f"{channel_i:2d} {channel:<22} "
            f"{input_max:9.4f} {decoded_max:11.4f} "
            f"{mse:9.6f} {rmse:9.6f} {mae:9.6f} {max_abs:9.6f} {bias:9.6f}"
        )
    return density_mse


def _error_volume(error: np.ndarray, channel: int | str) -> tuple[np.ndarray, str, str]:
    if channel == "rms":
        return np.sqrt(np.mean(error ** 2, axis=-1)), "RMS channel error", "viridis"
    if channel == "max_abs":
        return np.max(np.abs(error), axis=-1), "max abs channel error", "viridis"
    return error[..., channel], DENSFN_FORWARD_1_CHANNELS[channel], "coolwarm"


def _density_error_slice(volume: np.ndarray, axis_dim: int, slice_i: int, smoothing: float) -> np.ndarray:
    if smoothing == 0.0:
        if axis_dim == 0:
            return np.asarray(volume[slice_i, :, :], dtype=np.float32)
        if axis_dim == 1:
            return np.asarray(volume[:, slice_i, :], dtype=np.float32)
        return np.asarray(volume[:, :, slice_i], dtype=np.float32)

    coords = np.arange(volume.shape[axis_dim], dtype=np.float32)
    weights = np.exp(-0.5 * ((coords - slice_i) / smoothing) ** 2).astype(np.float32)
    shape = [1, 1, 1]
    shape[axis_dim] = volume.shape[axis_dim]
    return np.asarray((volume * weights.reshape(shape)).sum(axis=axis_dim), dtype=np.float32)


def _show_density_error(
    error: np.ndarray,
    *,
    axis: str,
    channel: int | str,
    vmax: float | None,
    smoothing: float,
    title_prefix: str,
) -> None:
    axis_dim = AXIS_TO_DIM[axis]
    volume, channel_desc, cmap = _error_volume(error, channel)
    abs_vmax = float(np.max(np.abs(volume))) if vmax is None else float(vmax)
    if abs_vmax <= 0.0:
        abs_vmax = 1.0
    initial_slice_i = int(volume.shape[axis_dim] // 2)

    def image(slice_i: int) -> np.ndarray:
        return np.swapaxes(_density_error_slice(volume, axis_dim, slice_i, smoothing), 0, 1)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.18)
    image_artist = ax.imshow(
        image(initial_slice_i),
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0 if channel in ("rms", "max_abs") else -abs_vmax,
        vmax=abs_vmax,
    )
    fig.colorbar(image_artist, ax=ax)
    shown_axes = [shown_axis for shown_axis in ("X", "Y", "Z") if shown_axis != axis]
    ax.set_xlabel(f"grid {shown_axes[0]}")
    ax.set_ylabel(f"grid {shown_axes[1]}")

    def set_title(slice_i: int) -> None:
        ax.set_title(
            f"{title_prefix} | decoded-input | {channel_desc} | "
            f"{axis}={slice_i} | smoothing={smoothing:g}"
        )

    def update_slice(value: float) -> None:
        slice_i = int(round(value))
        image_artist.set_data(image(slice_i))
        set_title(slice_i)
        fig.canvas.draw_idle()

    set_title(initial_slice_i)
    slider_ax = fig.add_axes([0.15, 0.06, 0.7, 0.03])
    slice_slider = Slider(
        slider_ax,
        f"{axis} slice",
        0,
        int(volume.shape[axis_dim] - 1),
        valinit=initial_slice_i,
        valstep=1,
    )
    slice_slider.on_changed(update_slice)
    plt.show()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.latent_noise_level < 0.0:
        raise ValueError("--latent-noise-level must be non-negative")
    if args.peak_threshold < 0.0:
        raise ValueError("--peak-threshold must be non-negative")
    if args.error_smoothing < 0.0:
        raise ValueError("--error-smoothing must be non-negative")
    if args.error_vmax is not None and args.error_vmax <= 0.0:
        raise ValueError("--error-vmax must be positive")

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    protein = read_protein_cif_with_codes(args.cif_path)
    grid, field = densfn_forward_1(protein, rng=rng)
    grid, field = _crop_field_to_even_grid(grid, field)

    vae = _load_vae(args.vae_path, device)
    if vae.conf.chan_dens_fields != len(DENSFN_FORWARD_1_CHANNELS):
        raise ValueError(
            "VAE density channel count does not match densfn_forward_1: "
            f"{vae.conf.chan_dens_fields} != {len(DENSFN_FORWARD_1_CHANNELS)}"
        )

    x = _field_to_tensor(field, device)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=vae.conf.autocast):
            z = vae.enc(x)
            z_noised = _add_latent_noise(z, args.latent_noise_level, args.seed)
            x_pred = vae.dec(z_noised)

    decoded_field = _tensor_to_field(x_pred)
    field_cmp, decoded_field_cmp = _comparable_fields(field, decoded_field)
    density_mse = _print_density_error_stats(field_cmp, decoded_field_cmp)
    print(f"Input density shape: {tuple(field.shape)}")
    print(f"Latent shape: {tuple(z.shape)}")
    print(f"Decoded density shape: {tuple(decoded_field.shape)}")
    print(f"Latent noise level: {args.latent_noise_level}")
    if args.show_density_error:
        _show_density_error(
            decoded_field_cmp - field_cmp,
            axis=args.error_axis,
            channel=args.error_channel,
            vmax=args.error_vmax,
            smoothing=args.error_smoothing,
            title_prefix=args.cif_path,
        )

    decoded_grid = Grid(
        dx=grid.dx,
        N=np.asarray(decoded_field.shape[:3], dtype=np.int32),
        transform=grid.transform,
    )
    try:
        decoded = densfn_backward_1(decoded_grid, decoded_field, peak_threshold=args.peak_threshold)
    except Exception as exc:
        print(f"Atom decode failed: {type(exc).__name__}: {exc}")
        return 1

    rmsd, atom_count, warnings = protein_rmsd(protein, decoded)
    code_mismatches = residue_code_mismatches(protein, decoded)

    print(f"RMSD: {rmsd:.6f} A")
    print(f"Compared atoms: {atom_count}")
    print(f"Reference chains/residues: {len(protein)}/{sum(len(chain) for chain in protein)}")
    print(f"Decoded chains/residues: {len(decoded)}/{sum(len(chain) for chain in decoded)}")
    if code_mismatches:
        print("Residue code mismatches:")
        for mismatch in code_mismatches:
            print(f"- {mismatch}")
    if warnings:
        print("Warnings:")
        for warning in warnings[:20]:
            print(f"- {warning}")
        if len(warnings) > 20:
            print(f"- ... {len(warnings) - 20} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
