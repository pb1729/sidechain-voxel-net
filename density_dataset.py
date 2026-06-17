"""PyTorch loaders for CIF density batches."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from density_fns import DensFn1, DensityFunction
from parse_cif import read_protein_cif_with_codes


DEFAULT_DENSFN = DensFn1()
CHAN_DENSFN_1 = DEFAULT_DENSFN.channel_count()


def is_holdout(filename: str | Path, holdout_percent: int | float) -> bool:
    """Return whether a filename belongs to a deterministic holdout split."""
    if holdout_percent < 0 or holdout_percent > 100:
        raise ValueError("holdout_percent must be between 0 and 100")
    digest = hashlib.sha256(Path(filename).name.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return bucket < holdout_percent


def _as_file_list(cif_paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(cif_paths, (str, Path)):
        path = Path(cif_paths)
        if path.is_dir():
            files = sorted(path.glob("*.cif"))
        else:
            files = [path]
    else:
        files = [Path(path) for path in cif_paths]
    if not files:
        raise ValueError("no CIF files found")
    return files


def cif_to_density_tensor(
    cif_path: str | Path,
    *,
    densfn: DensityFunction = DEFAULT_DENSFN,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Read one CIF and return its density tensor as (chan, H, W, L)."""
    protein = read_protein_cif_with_codes(cif_path)
    _grid, density = densfn.forward(protein, rng=rng)
    density = np.moveaxis(density, -1, 0)
    return torch.from_numpy(np.ascontiguousarray(density))


def _random_padding(size: int, target_size: int, rng: np.random.Generator) -> tuple[int, int]:
    total_padding = int(target_size - size)
    if total_padding < 0:
        raise ValueError("target size must be at least the tensor size")
    left_padding = int(rng.integers(0, total_padding + 1))
    return left_padding, total_padding - left_padding

def _slice_to_even(arr):
  """ make grid dimensions all even """
  *_, gx, gy, gz = arr.shape
  gx, gy, gz = 2*(gx//2), 2*(gy//2), 2*(gz//2)
  return arr[..., :gx, :gy, :gz]


def _filter_holdout_files(
    files: list[Path],
    holdout_percent: int | float,
    holdout: bool,
) -> list[Path]:
    if holdout_percent < 0 or holdout_percent > 100:
        raise ValueError("holdout_percent must be between 0 and 100")
    filtered = [
        path for path in files
        if is_holdout(path, holdout_percent) == holdout
    ]
    if not filtered:
        split_name = "holdout" if holdout else "training"
        raise ValueError(f"no CIF files found in {split_name} split")
    return filtered


def make_density_batch(
    cif_paths: Iterable[str | Path],
    *,
    densfn: DensityFunction = DEFAULT_DENSFN,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Convert CIF files to a zero-padded density batch.

    Each sample is encoded as (chan, H, W, L). Samples are padded to the largest
    H/W/L in the batch, with each axis' padding split randomly between the two
    sides, then stacked to (batch, chan, H_max, W_max, L_max).
    """
    if rng is None:
        rng = np.random.default_rng()

    tensors = [cif_to_density_tensor(path, densfn=densfn, rng=rng) for path in cif_paths]
    if not tensors:
        raise ValueError("cannot make a batch from zero CIF files")

    channels = tensors[0].shape[0]
    if any(tensor.shape[0] != channels for tensor in tensors):
        raise ValueError("all density tensors must have the same channel count")

    max_shape = tuple(max(tensor.shape[axis] for tensor in tensors) for axis in (1, 2, 3))
    batch = tensors[0].new_zeros((len(tensors), channels, *max_shape))

    for batch_i, tensor in enumerate(tensors):
        slices = [batch_i, slice(None)]
        for axis, target_size in enumerate(max_shape, start=1):
            left_padding, _right_padding = _random_padding(tensor.shape[axis], target_size, rng)
            slices.append(slice(left_padding, left_padding + tensor.shape[axis]))
        batch[tuple(slices)] = tensor

    batch = _slice_to_even(batch)

    return batch


class CifDensityBatchDataset(IterableDataset):
    """Iterable dataset yielding shuffled, padded CIF density batches."""

    def __init__(
        self,
        cif_paths: str | Path | Iterable[str | Path],
        batch_size: int,
        *,
        seed: int | None = None,
        densfn: DensityFunction = DEFAULT_DENSFN,
        drop_last: bool = True,
        holdout_percent: int | float = 0,
        holdout: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.files = _filter_holdout_files(_as_file_list(cif_paths), holdout_percent, holdout)
        self.batch_size = int(batch_size)
        self.seed = seed
        self.densfn = densfn
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = get_worker_info()
        files = self.files
        worker_seed = self.seed
        if worker_info is not None:
            files = files[worker_info.id::worker_info.num_workers]
            if worker_seed is not None:
                worker_seed += worker_info.id

        rng = np.random.default_rng(worker_seed)
        shuffled_indices = rng.permutation(len(files))
        shuffled_files = [files[int(i)] for i in shuffled_indices]

        for start in range(0, len(shuffled_files), self.batch_size):
            batch_files = shuffled_files[start:start + self.batch_size]
            if len(batch_files) < self.batch_size and self.drop_last:
                continue
            try:
                batch = make_density_batch(batch_files, densfn=self.densfn, rng=rng)
            except: continue
            yield batch

def make_density_batch_loader(
    cif_paths: str | Path | Iterable[str | Path],
    batch_size: int,
    *,
    seed: int | None = None,
    densfn: DensityFunction = DEFAULT_DENSFN,
    drop_last: bool = True,
    num_workers: int = 0,
    holdout_percent: int | float = 0,
    holdout: bool = False,
) -> torch.utils.data.DataLoader:
    """Create a PyTorch DataLoader that yields density batches directly."""
    dataset = CifDensityBatchDataset(
        cif_paths,
        batch_size,
        seed=seed,
        densfn=densfn,
        drop_last=drop_last,
        holdout_percent=holdout_percent,
        holdout=holdout,
    )
    return torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=num_workers)
