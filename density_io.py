"""Serialization for density fields and their reconstruction metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from density_fns import (
    DensityFunction,
    Grid,
    density_function_from_dict,
    density_function_to_dict,
)


def save_density_file(
    path: str | Path,
    grid: Grid,
    field: np.ndarray,
    densfn: DensityFunction,
) -> None:
    field = np.ascontiguousarray(np.asarray(field, dtype=np.float32))
    if field.ndim != 4:
        raise ValueError(f"density field must have shape (X, Y, Z, channels), got {field.shape}")
    if tuple(field.shape[:3]) != grid.shape:
        raise ValueError(f"field shape {field.shape[:3]} does not match grid shape {grid.shape}")
    if field.shape[-1] != densfn.channel_count():
        raise ValueError(
            f"field has {field.shape[-1]} channels but {type(densfn).__name__} "
            f"expects {densfn.channel_count()}"
        )
    torch.save(
        {
            "format": "sidechain-voxel-density-v1",
            "field": torch.from_numpy(field),
            "grid": {
                "dx": float(grid.dx),
                "N": np.asarray(grid.N, dtype=np.int32).tolist(),
                "transform": None if grid.transform is None else np.asarray(grid.transform).tolist(),
            },
            "densfn": density_function_to_dict(densfn),
        },
        path,
    )


def load_density_file(path: str | Path) -> tuple[Grid, np.ndarray, DensityFunction]:
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict) or data.get("format") != "sidechain-voxel-density-v1":
        raise ValueError(f"{path} is not a supported saved density file")

    grid_data = data["grid"]
    transform = grid_data["transform"]
    grid = Grid(
        dx=float(grid_data["dx"]),
        N=np.asarray(grid_data["N"], dtype=np.int32),
        transform=None if transform is None else np.asarray(transform, dtype=np.float32),
    )
    stored_field = data["field"]
    if isinstance(stored_field, torch.Tensor):
        stored_field = stored_field.detach().cpu().numpy()
    field = np.ascontiguousarray(np.asarray(stored_field, dtype=np.float32))
    densfn = density_function_from_dict(data["densfn"])
    if field.ndim != 4 or tuple(field.shape[:3]) != grid.shape:
        raise ValueError(f"invalid field shape {field.shape} for grid shape {grid.shape}")
    if field.shape[-1] != densfn.channel_count():
        raise ValueError(
            f"saved field has {field.shape[-1]} channels but saved density function expects "
            f"{densfn.channel_count()}"
        )
    return grid, field, densfn
