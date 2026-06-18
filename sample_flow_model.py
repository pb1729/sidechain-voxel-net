#!/usr/bin/env python3
"""Sample a saved flow model and write one density field."""

from __future__ import annotations

import argparse
import contextlib
import sys

import numpy as np
import torch

from density_fns import Grid
from density_io import save_density_file
from train_flow_model import FlowModel


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
        description="Sample batch=1 from a flow checkpoint and save the density field."
    )
    parser.add_argument("model_path", help="path to saved flow checkpoint")
    parser.add_argument("output_path", help="output density file")
    parser.add_argument(
        "--shape", type=_parse_shape, default=(64, 64, 64),
        help="sample grid shape as N or XxYxZ; default: 64x64x64",
    )
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dx", type=float, default=1.0, help="coarse grid spacing in Angstroms")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="sampling device; default: cuda if available else cpu",
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
) -> np.ndarray:
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    noise = torch.randn(
        (1, flowmodel.conf.densfn.channel_count(), *shape),
        device=device,
        generator=generator,
    )
    with torch.no_grad():
        sampled = flowmodel.infer(noise, steps=steps)
    channels_first = sampled.squeeze(0).detach().float().cpu().numpy()
    return np.ascontiguousarray(np.moveaxis(channels_first, 0, -1).astype(np.float32))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.dx <= 0.0:
        raise ValueError("--dx must be positive")

    device = torch.device(args.device)
    print(f"loading {args.model_path} on {device}", file=sys.stderr)
    flowmodel = _load_flow_model(args.model_path, device)
    print(f"sampling batch=1 shape={args.shape} steps={args.steps}", file=sys.stderr)
    field = _sample_density(
        flowmodel, args.shape, steps=args.steps, seed=args.seed, device=device
    )
    grid = Grid(dx=args.dx, N=np.asarray(field.shape[:3], dtype=np.int32), transform=None)
    save_density_file(args.output_path, grid, field, flowmodel.conf.densfn)
    print(f"saved {args.output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
