from collections.abc import Sequence

import torch
import torch.nn.functional as F


__all__ = ["gaussian_blur_3d"]


def gaussian_blur_3d(
    grid: torch.Tensor,
    sigma: float | Sequence[float],
    *,
    truncate: float = 4.0,
    radius: int | Sequence[int] | None = None,
) -> torch.Tensor:
    """Apply a Gaussian blur to a tensor shaped ``(..., gx, gy, gz)``.

    The output has the same shape as ``grid``. Zero padding is used at the
    boundary and the kernel is not renormalized near edges, so mass is allowed
    to fall off the grid.

    Args:
        grid: Input tensor with at least three dimensions.
        sigma: Gaussian standard deviation in grid units. Use a single float
            for isotropic blur, or ``(sx, sy, sz)`` for per-axis blur.
        truncate: Kernel radius is ``round(truncate * sigma)`` when ``radius``
            is not provided.
        radius: Optional explicit kernel radius. Use a single int for all axes,
            or ``(rx, ry, rz)`` per axis.

    Returns:
        Blurred tensor with the same shape, dtype, and device as ``grid``.
    """
    if grid.ndim < 3:
        raise ValueError(f"grid must have at least 3 dimensions, got {grid.ndim}")
    if not torch.is_floating_point(grid):
        raise TypeError(f"grid must be a floating point tensor, got {grid.dtype}")

    sigmas = _as_3_tuple(sigma, "sigma", float)
    if any(s < 0 for s in sigmas):
        raise ValueError(f"sigma values must be non-negative, got {sigmas}")

    if radius is None:
        radii = tuple(int(round(truncate * s)) for s in sigmas)
    else:
        radii = _as_3_tuple(radius, "radius", int)
    if any(r < 0 for r in radii):
        raise ValueError(f"radius values must be non-negative, got {radii}")

    out = grid.reshape((-1, 1) + tuple(grid.shape[-3:]))

    for axis, (s, r) in enumerate(zip(sigmas, radii)):
        if s == 0 or r == 0:
            continue
        kernel_1d = _gaussian_kernel_1d(s, r, dtype=grid.dtype, device=grid.device)
        out = _conv_along_axis(out, kernel_1d, axis)

    return out.reshape(grid.shape)


def _as_3_tuple(value, name: str, item_type):
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 3:
            raise ValueError(f"{name} must be a scalar or a length-3 sequence")
        return tuple(item_type(v) for v in value)
    return (item_type(value), item_type(value), item_type(value))


def _gaussian_kernel_1d(
    sigma: float,
    radius: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    kernel_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    offsets = torch.arange(-radius, radius + 1, dtype=kernel_dtype, device=device)
    kernel = torch.exp(-0.5 * (offsets / sigma).square())
    kernel = kernel / kernel.sum()
    return kernel.to(dtype=dtype)


def _conv_along_axis(volume: torch.Tensor, kernel_1d: torch.Tensor, axis: int) -> torch.Tensor:
    kernel_shape = [1, 1, 1, 1, 1]
    kernel_shape[2 + axis] = kernel_1d.numel()
    weight = kernel_1d.reshape(kernel_shape)

    padding = [0, 0, 0]
    padding[axis] = kernel_1d.numel() // 2

    return F.conv3d(volume, weight, padding=tuple(padding))
