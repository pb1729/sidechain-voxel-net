#!/usr/bin/env python3
"""Sample one density field from a saved direct flow model and decode it to mmCIF."""

from __future__ import annotations

import argparse
import contextlib
import os
import string
import sys

import numpy as np
import torch

from density_fns import DENSFN_FORWARD_1_CHANNELS, Grid, densfn_backward_1
from parse_cif import AMINO_ACID_CODES, ATOM_IDENTITY_ELEMENTS, ProteinWithCodes, Z2CoordPair
from plot_density_slice import AXIS_TO_DIM, COLOR_MAPS, _density_slice, _rgb_image
from train_direct_flow_model import FlowModel, SAVE_PATH


ONE_TO_THREE = {one: three for three, one in AMINO_ACID_CODES.items()}


def _parse_shape(value: str) -> tuple[int, int, int]:
    parts = value.lower().replace(",", "x").split("x")
    if len(parts) == 1:
        dims = (int(parts[0]),) * 3
    elif len(parts) == 3:
        dims = tuple(int(part) for part in parts)
    else:
        raise argparse.ArgumentTypeError("shape must be N or XxYxZ")
    if any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return dims


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample batch=1 from a saved train_direct_flow_model.py checkpoint."
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        default=SAVE_PATH,
        help=f"path to saved flow checkpoint; default: {SAVE_PATH}",
    )
    parser.add_argument(
        "--shape",
        type=_parse_shape,
        default=(64, 64, 64),
        help="sample grid shape as N or XxYxZ; default: 64x64x64",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=32,
        help="Euler integration steps for flow inference; default: 32",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="torch random seed for sampling; default: None",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device for sampling; default: cuda if available else cpu",
    )
    parser.add_argument(
        "--peak-threshold",
        type=float,
        default=0.55,
        help="density peak threshold passed to densfn_backward_1; default: 0.55",
    )
    parser.add_argument(
        "--dx",
        type=float,
        default=1.0,
        help="grid spacing in Angstroms for backconversion; default: 1.0",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="do not open the slideable density map before backconversion",
    )
    parser.add_argument(
        "--axis",
        choices=tuple(AXIS_TO_DIM),
        default="Z",
        help="axis to slice along in the density display; default: Z",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
        help="Gaussian standard deviation along the sliced axis before summing; default: 0.0",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=1.0,
        help="density value mapped to full display brightness; default: 1.0",
    )
    parser.add_argument(
        "--color-map",
        choices=tuple(COLOR_MAPS),
        default="decode",
        help="pre-baked channel-to-RGB color map for display; default: decode",
    )
    return parser.parse_args(argv)


def _load_flow_model(path: str, device: torch.device) -> FlowModel:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    with contextlib.redirect_stdout(sys.stderr):
        return FlowModel.from_dict(checkpoint).to(device)


def _sample_density(
    flowmodel: FlowModel,
    shape: tuple[int, int, int],
    *,
    steps: int,
    seed: int | None,
    device: torch.device,
) -> torch.Tensor:
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    eps = torch.randn(
        (1, flowmodel.conf.chan_dens, *shape),
        device=device,
        generator=generator,
    )
    with torch.no_grad():
        return flowmodel.infer(eps, steps=steps)


def _tensor_to_field(tensor: torch.Tensor) -> np.ndarray:
    density = tensor.squeeze(0).detach().float().cpu().numpy()
    return np.ascontiguousarray(np.moveaxis(density, 0, -1).astype(np.float32))


def _show_density(
    field: np.ndarray,
    *,
    axis: str,
    smoothing: float,
    vmax: float,
    color_map_name: str,
    title_prefix: str,
) -> None:
    axis_dim = AXIS_TO_DIM[axis]
    initial_slice_i = int(field.shape[axis_dim] // 2)
    color_map = COLOR_MAPS[color_map_name]

    def normalized_image(slice_i: int) -> np.ndarray:
        density_slice = _density_slice(field, axis_dim, slice_i, smoothing)
        image = _rgb_image(density_slice, color_map)
        image = np.clip(image / vmax, 0.0, 1.0)
        return np.swapaxes(image, 0, 1)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.18)
    image_artist = ax.imshow(
        normalized_image(initial_slice_i),
        origin="lower",
        interpolation="nearest",
    )
    shown_axes = [shown_axis for shown_axis in ("X", "Y", "Z") if shown_axis != axis]
    ax.set_xlabel(f"grid {shown_axes[0]}")
    ax.set_ylabel(f"grid {shown_axes[1]}")

    def set_title(slice_i: int) -> None:
        ax.set_title(
            f"{title_prefix} | {axis}={slice_i} | "
            f"smoothing={smoothing:g} | map={color_map_name}"
        )

    def update_slice(value: float) -> None:
        slice_i = int(round(value))
        image_artist.set_data(normalized_image(slice_i))
        set_title(slice_i)
        fig.canvas.draw_idle()

    set_title(initial_slice_i)
    slider_ax = fig.add_axes([0.15, 0.06, 0.7, 0.03])
    slice_slider = Slider(
        slider_ax,
        f"{axis} slice",
        0,
        int(field.shape[axis_dim] - 1),
        valinit=initial_slice_i,
        valstep=1,
    )
    slice_slider.on_changed(update_slice)
    plt.show()


def _chain_id(chain_i: int) -> str:
    alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits
    if chain_i < len(alphabet):
        return alphabet[chain_i]
    return f"X{chain_i}"


def _coords_for_cif(coord_or_pair) -> tuple[np.ndarray, ...]:
    if isinstance(coord_or_pair, Z2CoordPair):
        return coord_or_pair.a, coord_or_pair.b
    return (coord_or_pair,)


def _atom_element(atom_name: str) -> str:
    if atom_name in ATOM_IDENTITY_ELEMENTS:
        return ATOM_IDENTITY_ELEMENTS[atom_name]
    return "".join(char for char in atom_name if char.isalpha())[:1].upper() or "?"


def protein_to_cif_text(protein: ProteinWithCodes, *, data_name: str = "sampled_flow") -> str:
    lines = [
        f"data_{data_name}",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    atom_id = 1
    for chain_i, chain in enumerate(protein):
        asym_id = _chain_id(chain_i)
        for residue_i, (aa, residue) in enumerate(chain, start=1):
            comp_id = ONE_TO_THREE.get(aa, "UNK")
            for atom_name, coord_or_pair in residue.items():
                coords = _coords_for_cif(coord_or_pair)
                for alt_i, coord in enumerate(coords):
                    alt_id = "." if len(coords) == 1 else string.ascii_uppercase[alt_i]
                    x, y, z = np.asarray(coord, dtype=np.float32)
                    element = _atom_element(atom_name)
                    lines.append(
                        " ".join(
                            (
                                "ATOM",
                                str(atom_id),
                                element,
                                atom_name,
                                alt_id,
                                comp_id,
                                asym_id,
                                str(residue_i),
                                "?",
                                f"{x:.3f}",
                                f"{y:.3f}",
                                f"{z:.3f}",
                                "1.00",
                                "0.00",
                                str(residue_i),
                                comp_id,
                                asym_id,
                                atom_name,
                                "1",
                            )
                        )
                    )
                    atom_id += 1
    lines.append("#")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.peak_threshold < 0.0:
        raise ValueError("--peak-threshold must be non-negative")
    if args.dx <= 0.0:
        raise ValueError("--dx must be positive")
    if args.smoothing < 0.0:
        raise ValueError("--smoothing must be non-negative")
    if args.vmax <= 0.0:
        raise ValueError("--vmax must be positive")

    device = torch.device(args.device)
    print(f"loading {args.model_path} on {device}", file=sys.stderr)
    flowmodel = _load_flow_model(args.model_path, device)
    if flowmodel.conf.chan_dens != len(DENSFN_FORWARD_1_CHANNELS):
        raise ValueError(
            "flow density channel count does not match densfn_forward_1: "
            f"{flowmodel.conf.chan_dens} != {len(DENSFN_FORWARD_1_CHANNELS)}"
        )

    print(f"sampling batch=1 shape={args.shape} steps={args.steps}", file=sys.stderr)
    sampled = _sample_density(
        flowmodel,
        args.shape,
        steps=args.steps,
        seed=args.seed,
        device=device,
    )
    field = _tensor_to_field(sampled)
    print(
        "sample density stats: "
        f"min={float(field.min()):.4f} max={float(field.max()):.4f} mean={float(field.mean()):.4f}",
        file=sys.stderr,
    )

    if not args.no_display:
        _show_density(
            field,
            axis=args.axis,
            smoothing=args.smoothing,
            vmax=args.vmax,
            color_map_name=args.color_map,
            title_prefix=args.model_path,
        )

    grid = Grid(dx=float(args.dx), N=np.asarray(field.shape[:3], dtype=np.int32), transform=None)
    try:
        decoded = densfn_backward_1(grid, field, peak_threshold=args.peak_threshold)
    except Exception as exc:
        print(f"backconversion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    atom_count = sum(
        len(_coords_for_cif(coord_or_pair))
        for chain in decoded
        for _aa, residue in chain
        for coord_or_pair in residue.values()
    )
    print(
        f"backconversion succeeded: chains={len(decoded)} "
        f"residues={sum(len(chain) for chain in decoded)} atoms={atom_count}",
        file=sys.stderr,
    )
    print(protein_to_cif_text(decoded), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
