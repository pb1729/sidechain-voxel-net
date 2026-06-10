#!/usr/bin/env python3
"""Save radial k-space power spectrum bins for density training data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from density_dataset import CHAN_DENSFN_1, make_density_batch_loader
from kspace_ops import fft3d
from train_direct_flow_model import CIF_DATASET_PATH


def parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Estimate radial k-space power bins from density batches."
  )
  parser.add_argument(
    "dataset",
    nargs="?",
    default=CIF_DATASET_PATH,
    help=f"CIF dataset path; default: {CIF_DATASET_PATH}",
  )
  parser.add_argument("--batch", type=int, default=2, help="loader batch size; default: 2")
  parser.add_argument("--batches", type=int, default=32, help="number of batches to scan; default: 32")
  parser.add_argument("--bins", type=int, default=96, help="radial k bins; default: 96")
  parser.add_argument("--crop-size", type=int, default=48, help="center crop size before FFT; default: 48")
  parser.add_argument("--kspace-scale", type=float, default=1.5, help="FFT padding scale; default: 1.5")
  parser.add_argument("--seed", type=int, default=0, help="dataset shuffle seed; default: 0")
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--out", type=Path, required=True, help="JSON output path")
  return parser.parse_args(argv)


def center_crop(x: torch.Tensor, crop_size: int) -> torch.Tensor:
  if crop_size <= 0:
    return x
  slices = [slice(None), slice(None)]
  for dim in x.shape[-3:]:
    if dim <= crop_size:
      start = 0
      stop = dim
    else:
      start = (dim - crop_size)//2
      stop = start + crop_size
    slices.append(slice(start, stop))
  return x[tuple(slices)]


def radial_bin_stats(
    power: torch.Tensor,
    radius: torch.Tensor,
    bin_edges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  bin_i = torch.bucketize(radius.reshape(-1), bin_edges) - 1
  valid = (bin_i >= 0) & (bin_i < bin_edges.numel() - 1)
  bin_i = bin_i[valid]
  flat_power = power.reshape(-1)[valid]
  sums = torch.zeros(bin_edges.numel() - 1, device=power.device, dtype=torch.float64)
  counts = torch.zeros_like(sums)
  sums.scatter_add_(0, bin_i, flat_power.to(torch.float64))
  counts.scatter_add_(0, bin_i, torch.ones_like(flat_power, dtype=torch.float64))
  return sums, counts


def main(argv: list[str] | None = None) -> int:
  args = parse_args(sys.argv[1:] if argv is None else argv)
  if args.batch <= 0:
    raise ValueError("--batch must be positive")
  if args.batches <= 0:
    raise ValueError("--batches must be positive")
  if args.bins <= 1:
    raise ValueError("--bins must be greater than 1")
  if args.kspace_scale <= 0.0:
    raise ValueError("--kspace-scale must be positive")

  device = torch.device(args.device)
  loader = make_density_batch_loader(args.dataset, args.batch, seed=args.seed)

  bin_edges = None
  power_sums = None
  counts = None
  total_sq = 0.0
  total_n = 0
  scanned_batches = 0
  scanned_samples = 0
  shape = None

  for batch_i, x in enumerate(loader):
    if batch_i >= args.batches:
      break
    x = center_crop(x.to(device=device, dtype=torch.float32), args.crop_size)
    shape = tuple(int(dim) for dim in x.shape[-3:])
    x_fft, (kx, ky, kz) = fft3d(x, scale=args.kspace_scale)
    radius = torch.sqrt(kx**2 + ky**2 + kz**2)
    power = (x_fft.abs()**2).mean(dim=(0, 1))
    if bin_edges is None:
      bin_edges = torch.linspace(0.0, float(radius.max()), args.bins + 1, device=device)
      power_sums = torch.zeros(args.bins, device=device, dtype=torch.float64)
      counts = torch.zeros(args.bins, device=device, dtype=torch.float64)
    sums_i, counts_i = radial_bin_stats(power, radius, bin_edges)
    power_sums += sums_i
    counts += counts_i
    total_sq += float(x.square().sum().detach().cpu())
    total_n += int(x.numel())
    scanned_batches += 1
    scanned_samples += int(x.shape[0])
    print(f"batch {batch_i + 1}/{args.batches} shape={shape}", file=sys.stderr)

  if scanned_batches == 0 or bin_edges is None or power_sums is None or counts is None:
    raise RuntimeError("no batches were scanned")

  radial_power = power_sums/torch.clamp(counts, min=1)
  k_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
  raw_rms = (total_sq/total_n)**0.5
  scalar_intensity_ratio = 1.0/raw_rms if raw_rms > 0 else float("inf")

  result = {
    "kind": "radial_power_spectrum_bins",
    "dataset": str(args.dataset),
    "batches": scanned_batches,
    "samples": scanned_samples,
    "channels": CHAN_DENSFN_1,
    "crop_shape": shape,
    "kspace_scale": args.kspace_scale,
    "raw_rms": raw_rms,
    "scalar_intensity_ratio": scalar_intensity_ratio,
    "radial_bins": {
      "k_centers": [float(v) for v in k_centers.detach().cpu()],
      "power": [float(v) for v in radial_power.detach().cpu()],
      "counts": [int(v) for v in counts.detach().cpu()],
    },
  }

  args.out.write_text(json.dumps(result, indent=2) + "\n")
  print(f"wrote {args.out}", file=sys.stderr)
  print(json.dumps(result, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
