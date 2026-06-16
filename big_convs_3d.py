from collections.abc import Sequence

import torch
import torch.nn.functional as F

from util import must_be


__all__ = ["gaussian_blur_3d", "affine_gauss_conv_3d"]


def _gaussian_kernel_1d(
    sigma: float|torch.Tensor, radius:int,
    poly: tuple[float, ...]|None = None, use_rms_norm: bool = False,
    dtype: torch.dtype|None = None, device: torch.device|None = None
) -> torch.Tensor:
  """ sigma: (chan) OR ()
      ans: (chan, 2*radius + 1) OR (2*radius + 1)
      returns a gaussian (or gaussian times a polynomial if poly is specified) """
  if type(sigma) == float:
    assert dtype is not None and device is not None, "device and dtype must be given if sigma is float"
    sigma = torch.tensor(sigma, dtype=dtype, device=device)
  else:
    assert dtype is None and device is None, "if sigma is not float, dtype and device cannot be over-ridden"
    dtype, device = sigma.dtype, sigma.device
  offsets = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
  kernel = torch.exp(-0.5 * (offsets / sigma[..., None]).square())
  if poly is not None:
    p = 0.
    for term in poly[::-1]:
      p *= offsets
      p += term
    kernel = kernel*p
  if use_rms_norm:
    kernel = kernel*torch.rsqrt(kernel.square().sum(-1, keepdim=True))
  else:
    kernel = kernel/abs(kernel).sum(-1, keepdim=True)
  return kernel.to(dtype=dtype)


def affine_gauss_conv_3d(
  grid: torch.Tensor,
  sigma: torch.Tensor,
  W: torch.Tensor,
) -> torch.Tensor:
  """ Convolution where the shape basis is: (1, x, y, z)*exp(-1/2 sigma^2)
      The value used for sigma can vary by channel.
      grid: (batch, chan_in, gx, gy, gz)
      sigma: (chan_out)
      W: (4, chan_out, chan_in)
      ans: (batch, chan_out, gx, gy, gz) """
  batch, chan_in, gx, gy, gz = grid.shape
  chan_out, = sigma.shape
  must_be[4], must_be[chan_out], must_be[chan_in] = W.shape
  sigma_max = torch.max(sigma)
  radius = int(round(4*sigma_max.item()))
  blur_1 = _gaussian_kernel_1d(sigma, radius)
  blur_v = _gaussian_kernel_1d(sigma, radius, poly=(0., 1.))
  blur_x = torch.concatenate([blur_1, blur_v, blur_1, blur_1], dim=0)[:, None, :, None, None]
  blur_y = torch.concatenate([blur_1, blur_1, blur_v, blur_1], dim=0)[:, None, None, :, None]
  blur_z = torch.concatenate([blur_1, blur_1, blur_1, blur_v], dim=0)[:, None, None, None, :]
  ans = F.conv3d(grid, W.reshape(4*chan_out, chan_in, 1, 1, 1))
  ans = F.conv3d(ans, blur_x, padding=(radius, 0, 0), groups=4*chan_out)
  ans = F.conv3d(ans, blur_y, padding=(0, radius, 0), groups=4*chan_out)
  ans = F.conv3d(ans, blur_z, padding=(0, 0, radius), groups=4*chan_out)
  return ans.reshape(batch, 4, chan_out, gx, gy, gz).sum(1)
  


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


def _conv_along_axis(volume: torch.Tensor, kernel_1d: torch.Tensor, axis: int) -> torch.Tensor:
    kernel_shape = [1, 1, 1, 1, 1]
    kernel_shape[2 + axis] = kernel_1d.numel()
    weight = kernel_1d.reshape(kernel_shape)

    padding = [0, 0, 0]
    padding[axis] = kernel_1d.numel() // 2

    return F.conv3d(volume, weight, padding=tuple(padding))
