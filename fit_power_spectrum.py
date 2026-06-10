#!/usr/bin/env python3
"""Fit and plot radial k-space power spectrum bins."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Fit a power-law spectrum from save_power_spectrum_bins.py JSON output."
  )
  parser.add_argument("bins_json", type=Path, help="radial power bin JSON")
  parser.add_argument("--plot-out", type=Path, default=None, help="optional image output path")
  parser.add_argument("--fit-out", type=Path, default=None, help="optional fit JSON output path")
  parser.add_argument("--title", default=None, help="optional plot title")
  parser.add_argument("--no-display", action="store_true", help="do not open an interactive window")
  parser.add_argument(
    "--min-count",
    type=int,
    default=1,
    help="minimum samples in a radial bin to include in fit; default: 1",
  )
  parser.add_argument(
    "--min-k",
    type=float,
    default=0.0,
    help="minimum radial frequency to include in fit; default: 0",
  )
  parser.add_argument(
    "--max-k",
    type=float,
    default=None,
    help="maximum radial frequency to include in fit; default: no limit",
  )
  return parser.parse_args(argv)


def load_bins(path: Path) -> dict:
  with path.open() as f:
    data = json.load(f)
  if "radial_bins" not in data:
    raise ValueError(f"{path} does not contain radial_bins")
  return data


def power_law(k: np.ndarray, amp: float, exponent: float) -> np.ndarray:
  return amp*np.maximum(k, 1e-30)**(-exponent)


def fit_power_law(
    k: np.ndarray,
    power: np.ndarray,
    counts: np.ndarray,
    *,
    min_count: int,
    min_k: float,
    max_k: float | None,
) -> dict:
  valid = (
    np.isfinite(k)
    & np.isfinite(power)
    & (power > 0.0)
    & (counts >= min_count)
    & (k > min_k)
  )
  if max_k is not None:
    valid &= k <= max_k
  if valid.sum() < 2:
    raise ValueError("need at least two valid bins to fit a power law")

  x = np.log(k[valid])
  y = np.log(power[valid])
  w = counts[valid].astype(np.float64)
  w = w/w.mean()

  # Weighted least squares for log(power) = log_amp - exponent*log(k).
  design = np.stack([np.ones_like(x), -x], axis=1)
  lhs = design.T @ (w[:, None]*design)
  rhs = design.T @ (w*y)
  log_amp, exponent = np.linalg.solve(lhs, rhs)
  pred_log = log_amp - exponent*x
  log_mse = float(np.average((pred_log - y)**2, weights=w))

  amp = float(np.exp(log_amp))
  exponent = float(exponent)
  pred = power_law(k[valid], amp, exponent)
  rel = power[valid]/pred
  return {
    "model": "amp * k**(-exponent)",
    "amp": amp,
    "exponent": exponent,
    "log_mse": log_mse,
    "fit_bins": int(valid.sum()),
    "min_k": float(k[valid].min()),
    "max_k": float(k[valid].max()),
    "median_ratio": float(np.median(rel)),
    "p10_ratio": float(np.quantile(rel, 0.1)),
    "p90_ratio": float(np.quantile(rel, 0.9)),
  }


def write_fit(data: dict, fit: dict, out: Path) -> None:
  result = {
    "kind": "radial_power_spectrum_fit",
    "source": {
      key: data[key]
      for key in (
        "dataset",
        "batches",
        "samples",
        "channels",
        "crop_shape",
        "kspace_scale",
        "raw_rms",
        "scalar_intensity_ratio",
      )
      if key in data
    },
    "fit": fit,
  }
  out.write_text(json.dumps(result, indent=2) + "\n")
  print(f"wrote {out}", file=sys.stderr)


def plot_fit(
    data: dict,
    fit: dict,
    k: np.ndarray,
    power: np.ndarray,
    counts: np.ndarray,
    *,
    title: str | None,
    out: Path | None,
    display: bool,
) -> None:
  valid = (counts > 0) & np.isfinite(k) & np.isfinite(power) & (power > 0.0)
  k_valid = k[valid]
  power_valid = power[valid]
  counts_valid = counts[valid]
  k_fit = np.linspace(float(k_valid.min()), float(k_valid.max()), 512)
  power_fit = power_law(k_fit, float(fit["amp"]), float(fit["exponent"]))

  os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
  import matplotlib.pyplot as plt

  fig, (ax_power, ax_ratio) = plt.subplots(
    2,
    1,
    figsize=(8, 7),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
  )
  point_sizes = 8 + 32*np.sqrt(counts_valid/counts_valid.max())
  ax_power.scatter(k_valid, power_valid, s=point_sizes, alpha=0.75, label="radial bins")
  ax_power.plot(k_fit, power_fit, color="black", linewidth=1.8, label=fit["model"])
  ax_power.set_xscale("log")
  ax_power.set_yscale("log")
  ax_power.set_ylabel("mean |FFT(density)|^2")
  ax_power.grid(True, which="both", alpha=0.25)
  ax_power.legend()

  ratio = power_valid/power_law(k_valid, float(fit["amp"]), float(fit["exponent"]))
  ax_ratio.axhline(1.0, color="black", linewidth=1.0)
  ax_ratio.scatter(k_valid, ratio, s=point_sizes, alpha=0.75)
  ax_ratio.set_xscale("log")
  ax_ratio.set_yscale("log")
  ax_ratio.set_xlabel("radial frequency |k|")
  ax_ratio.set_ylabel("bin / fit")
  ax_ratio.grid(True, which="both", alpha=0.25)

  if title is None:
    title = (
      f"{data.get('dataset', 'power spectrum')} | "
      f"samples={data.get('samples', '?')} | "
      f"amp={fit['amp']:.4g} | exponent={fit['exponent']:.4g}"
    )
  fig.suptitle(title)
  fig.tight_layout()

  if out is not None:
    fig.savefig(out, dpi=160)
    print(f"wrote {out}", file=sys.stderr)
  if display:
    plt.show()


def main(argv: list[str] | None = None) -> int:
  args = parse_args(sys.argv[1:] if argv is None else argv)
  data = load_bins(args.bins_json)
  bins = data["radial_bins"]
  k = np.asarray(bins["k_centers"], dtype=np.float64)
  power = np.asarray(bins["power"], dtype=np.float64)
  counts = np.asarray(bins["counts"], dtype=np.float64)

  fit = fit_power_law(
    k,
    power,
    counts,
    min_count=args.min_count,
    min_k=args.min_k,
    max_k=args.max_k,
  )
  print(json.dumps({"fit": fit}, indent=2))
  if args.fit_out is not None:
    write_fit(data, fit, args.fit_out)
  if args.plot_out is not None or not args.no_display:
    plot_fit(
      data,
      fit,
      k,
      power,
      counts,
      title=args.title,
      out=args.plot_out,
      display=not args.no_display,
    )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
