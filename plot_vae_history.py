#!/usr/bin/env python3
"""Plot scalar training history from a saved VAE checkpoint."""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any

import torch


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot scalar history series from a train_vae.py saved checkpoint."
    )
    parser.add_argument("checkpoint_path", help="path to a saved VAE .pt file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="output image path; default: do not save",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="history keys to plot; default: all scalar numeric keys",
    )
    parser.add_argument(
        "--same-axes",
        action="store_true",
        help="plot all selected metrics on one shared set of axes",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="use a logarithmic y-axis for positive-valued series",
    )
    return parser.parse_args(argv)


def _scalar_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def _history_series(history: dict[str, Any]) -> dict[str, tuple[list[int], list[float]]]:
    series = {}
    for name, records in history.items():
        xs = []
        ys = []
        for record in records:
            if not isinstance(record, (tuple, list)) or len(record) != 2:
                continue
            step, value = record
            step_value = _scalar_float(step)
            y_value = _scalar_float(value)
            if step_value is None or y_value is None:
                continue
            xs.append(int(step_value))
            ys.append(y_value)
        if xs:
            series[name] = (xs, ys)
    return series


def _load_checkpoint(path: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path!r} did not contain a dictionary checkpoint")
    if "history" not in checkpoint:
        raise ValueError(f"{path!r} does not contain a 'history' entry")
    if not isinstance(checkpoint["history"], dict):
        raise ValueError(f"{path!r} has a non-dictionary 'history' entry")
    return checkpoint


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    checkpoint = _load_checkpoint(args.checkpoint_path)
    series = _history_series(checkpoint["history"])
    if not series:
        raise ValueError("checkpoint history has no scalar numeric series to plot")

    if args.metrics is None:
        selected_names = sorted(series)
    else:
        missing = [name for name in args.metrics if name not in series]
        if missing:
            available = ", ".join(sorted(series))
            raise ValueError(f"unknown or non-scalar metric(s): {missing}; available: {available}")
        selected_names = args.metrics

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    if args.same_axes:
        fig, ax = plt.subplots(figsize=(9, 5))
        axes = [ax]
        for name in selected_names:
            xs, ys = series[name]
            ax.plot(xs, ys, marker=".", linewidth=1.2, label=name)
        ax.set_xlabel("step")
        ax.set_ylabel("value")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(
            len(selected_names),
            1,
            figsize=(9, max(3, 2.6 * len(selected_names))),
            sharex=True,
            squeeze=False,
        )
        axes = [row[0] for row in axes]
        for ax, name in zip(axes, selected_names):
            xs, ys = series[name]
            ax.plot(xs, ys, marker=".", linewidth=1.2)
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("step")

    if args.log_y:
        has_nonpositive = any(y <= 0.0 for name in selected_names for y in series[name][1])
        if has_nonpositive:
            print("warning: --log-y skipped because a selected series includes non-positive values", file=sys.stderr)
        else:
            for ax in axes:
                ax.set_yscale("log")

    fig.suptitle(f"VAE history: {args.checkpoint_path}")
    fig.tight_layout()

    if args.output is not None:
        fig.savefig(args.output, dpi=160)
        print(f"saved {args.output}")
        plt.close(fig)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
