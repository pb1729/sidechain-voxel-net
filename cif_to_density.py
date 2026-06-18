#!/usr/bin/env python3
"""Convert an mmCIF protein structure to a saved density field."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from density_fns import DensFn1, DensFn2, DensityFunction
from density_io import save_density_file
from parse_cif import read_protein_cif_with_codes


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a CIF structure to a metadata-bearing density file."
    )
    parser.add_argument("input_path", help="input .cif file")
    parser.add_argument("output_path", help="output density file")
    parser.add_argument(
        "--densfn", choices=("1", "2"), required=True,
        help="density function to use",
    )
    parser.add_argument("--dx", type=float, default=1.0, help="coarse grid spacing in Angstroms")
    parser.add_argument("--padding", type=float, default=2.0, help="grid padding in Angstroms")
    parser.add_argument(
        "--grid-seed", type=int, default=None,
        help="random seed for density-grid orientation",
    )

    densfn2 = parser.add_argument_group("DensFn2 parameters")
    densfn2.add_argument("--stride", type=int, default=4)
    densfn2.add_argument("--atom-gaussian-radius", type=float, default=1.0)
    densfn2.add_argument("--atom-gaussian-sigma", type=float, default=None)
    densfn2.add_argument("--output-channels", type=int, default=32)
    densfn2.add_argument("--projection-seed", type=int, default=0)
    densfn2.add_argument("--random-conv-radius", type=float, default=4.0)
    densfn2.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    densfn2.add_argument("--conv-method", choices=("direct", "fft"), default="fft")
    densfn2.add_argument("--fft-phase-batch-size", type=int, default=8)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.dx <= 0.0:
        raise ValueError("--dx must be positive")
    if args.padding < 0.0:
        raise ValueError("--padding must be non-negative")
    if args.densfn == "1":
        return
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.atom_gaussian_radius <= 0.0:
        raise ValueError("--atom-gaussian-radius must be positive")
    if args.atom_gaussian_sigma is not None and args.atom_gaussian_sigma <= 0.0:
        raise ValueError("--atom-gaussian-sigma must be positive")
    if args.output_channels <= 0:
        raise ValueError("--output-channels must be positive")
    if args.random_conv_radius <= 0.0:
        raise ValueError("--random-conv-radius must be positive")
    if args.fft_phase_batch_size is not None and args.fft_phase_batch_size <= 0:
        raise ValueError("--fft-phase-batch-size must be positive")


def _make_density_function(args: argparse.Namespace) -> DensityFunction:
    if args.densfn == "1":
        return DensFn1()
    return DensFn2(
        stride=args.stride,
        atom_gaussian_radius=args.atom_gaussian_radius,
        atom_gaussian_sigma=args.atom_gaussian_sigma,
        output_channels=args.output_channels,
        seed=args.projection_seed,
        random_conv_radius=args.random_conv_radius,
        device=args.device,
        conv_method=args.conv_method,
        fft_phase_batch_size=args.fft_phase_batch_size,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    _validate_args(args)
    densfn = _make_density_function(args)
    protein = read_protein_cif_with_codes(args.input_path)
    rng = None if args.grid_seed is None else np.random.default_rng(args.grid_seed)
    grid, field = densfn.forward(
        protein, dx=args.dx, padding=args.padding, rng=rng
    )
    save_density_file(args.output_path, grid, field, densfn)
    print(
        f"saved {args.output_path}: densfn={type(densfn).__name__} "
        f"shape={field.shape} dx={grid.dx:g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
