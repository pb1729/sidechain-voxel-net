import torch


def next_pow2(x:float) -> int:
  z = int(x)
  for shift in [1, 2, 4, 8, 16]:
    z |= z >> shift
  return z + 1


def balanced_freqrange(N:int, device):
  assert N % 2 == 0
  halfN = N // 2
  return ((torch.arange(N, device=device) + halfN) % N) - halfN


def fft3d(arr, scale:float=1.5):
  *_, gx, gy, gz = arr.shape
  Gx, Gy, Gz = next_pow2(gx*scale), next_pow2(gy*scale), next_pow2(gz*scale)
  arr = torch.fft.fft(arr, n=Gx, dim=-3, norm="ortho")
  kx = (2*torch.pi/Gx)*balanced_freqrange(Gx, arr.device)[:, None, None]
  arr = torch.fft.fft(arr, n=Gy, dim=-2, norm="ortho")
  ky = (2*torch.pi/Gy)*balanced_freqrange(Gy, arr.device)[None, :, None]
  arr = torch.fft.fft(arr, n=Gz, dim=-1, norm="ortho")
  kz = (2*torch.pi/Gz)*balanced_freqrange(Gz, arr.device)[None, None, :]
  return arr, (kx, ky, kz)


def ifft3d(arr, gxyz=None):
  arr = torch.fft.ifft(arr, dim=-3, norm="ortho")
  arr = torch.fft.ifft(arr, dim=-2, norm="ortho")
  arr = torch.fft.ifft(arr, dim=-1, norm="ortho")
  if gxyz is not None:
    gx, gy, gz = gxyz
    arr = arr[..., :gx, :gy, :gz]
  return arr.real


def batch_only_t(t, ref):
  t = torch.as_tensor(t, device=ref.device, dtype=ref.dtype)
  if t.ndim == 0:
    return t
  if t.ndim == 1:
    t = t[:, None]
  else:
    t = t.reshape(t.shape[0], -1)[:, :1]
  return t[:, :, None, None, None]


def _noise_like(arr_fft, noise, generator):
  if noise is None:
    return torch.randn(
        arr_fft.shape,
        dtype=arr_fft.dtype,
        device=arr_fft.device,
        generator=generator,
    )
  return noise


def blur(
    noise_func,
    arr,
    scale:float = 1.5,
    noise=None,
    generator: torch.Generator | None = None,
    return_noise: bool = False,
):
  *_, gx, gy, gz = arr.shape
  arr_fft, (kx, ky, kz) = fft3d(arr, scale=scale)
  alpha, sigma = noise_func(kx, ky, kz)
  epsilon = _noise_like(arr_fft, noise, generator)
  arr_fft = alpha*arr_fft + sigma*epsilon
  arr = ifft3d(arr_fft, (gx, gy, gz))
  if return_noise:
    return arr, epsilon
  return arr


def blur_and_velocity(
    noise_func,
    velocity_func,
    arr,
    scale:float = 1.5,
    noise=None,
    generator: torch.Generator | None = None,
    return_noise: bool = False,
):
  *_, gx, gy, gz = arr.shape
  arr_fft, (kx, ky, kz) = fft3d(arr, scale=scale)
  alpha, sigma = noise_func(kx, ky, kz)
  d_alpha, d_sigma = velocity_func(kx, ky, kz)
  epsilon = _noise_like(arr_fft, noise, generator)
  noised = ifft3d(alpha*arr_fft + sigma*epsilon, (gx, gy, gz))
  velocity = ifft3d(d_alpha*arr_fft + d_sigma*epsilon, (gx, gy, gz))
  if return_noise:
    return noised, velocity, epsilon
  return noised, velocity


def nf_1(t, kx, ky, kz, blur_max:float=8.0, intensity_ratio:float=40.0):
  t = batch_only_t(t, kx)
  theta = torch.pi*t/2
  alpha = torch.cos(theta)
  blur = blur_max*torch.sin(theta)**2
  mag = kx**2 + ky**2 + kz**2
  alpha_k = alpha*torch.exp(-0.5*mag*(blur**2))
  sigma_k = torch.sqrt(torch.clamp(1.0 - alpha_k**2, min=0.0))
  return intensity_ratio*alpha_k, sigma_k

# trajectory differentiation:
# v = d/dt (x_mix)

def v_nf_1(t, kx, ky, kz, blur_max:float=8.0, intensity_ratio:float=40.0):
  t = batch_only_t(t, kx)
  theta = torch.pi*t/2
  d_theta = torch.pi/2
  alpha = torch.cos(theta)
  d_alpha = -d_theta*torch.sin(theta)
  blur = blur_max*torch.sin(theta)**2
  d_blur = blur_max*d_theta*torch.sin(2*theta)
  mag = kx**2 + ky**2 + kz**2
  blur_exp = torch.exp(-0.5*mag*(blur**2))
  alpha_k = alpha*blur_exp
  d_alpha_k = blur_exp*(d_alpha - alpha*mag*blur*d_blur)
  sigma_k_sq = torch.clamp(1.0 - alpha_k**2, min=0.0)
  sigma_k = torch.sqrt(sigma_k_sq)
  d_sigma_k = torch.where(
      sigma_k > 0,
      -alpha_k*d_alpha_k/sigma_k,
      torch.zeros_like(sigma_k),
  )
  return intensity_ratio*d_alpha_k, d_sigma_k


def velocity_scale(
    t,
    gxyz: tuple[int, int, int],
    device=None,
    dtype=None,
    scale: float = 1.5,
    blur_max: float = 8.0,
    intensity_ratio: float = 1.0,
    signal_power_amp: float | None = None,
    signal_power_exponent: float | None = None,
    signal_power_k_min: float = 1e-6,
    eps: float = 1e-4,
):
  if device is None:
    device = t.device if torch.is_tensor(t) else None
  if dtype is None:
    dtype = t.dtype if torch.is_tensor(t) and t.is_floating_point() else torch.float32
  gx, gy, gz = gxyz
  Gx, Gy, Gz = next_pow2(gx*scale), next_pow2(gy*scale), next_pow2(gz*scale)
  kx = (2*torch.pi/Gx)*balanced_freqrange(Gx, device)[:, None, None]
  ky = (2*torch.pi/Gy)*balanced_freqrange(Gy, device)[None, :, None]
  kz = (2*torch.pi/Gz)*balanced_freqrange(Gz, device)[None, None, :]
  kx, ky, kz = kx.to(dtype), ky.to(dtype), kz.to(dtype)
  t = batch_only_t(t, kx)
  theta = torch.pi*t/2
  d_theta = torch.pi/2
  alpha = torch.cos(theta)
  d_alpha = -d_theta*torch.sin(theta)
  blur = blur_max*torch.sin(theta)**2
  d_blur = blur_max*d_theta*torch.sin(2*theta)
  mag = kx**2 + ky**2 + kz**2
  blur_exp = torch.exp(-0.5*mag*(blur**2))
  alpha_k = alpha*blur_exp
  d_alpha_k = blur_exp*(d_alpha - alpha*mag*blur*d_blur)
  if signal_power_amp is None or signal_power_exponent is None:
    signal_scale_sq = (d_alpha_k**2).mean(dim=(-3, -2, -1), keepdim=True)
  else:
    k = torch.sqrt(mag).clamp_min(signal_power_k_min)
    signal_power = signal_power_amp*k**(-signal_power_exponent)
    signal_scale_sq = (signal_power*d_alpha_k**2).sum(
      dim=(-3, -2, -1),
      keepdim=True,
    )/(gx*gy*gz)
  sigma_k_sq = torch.clamp(1.0 - alpha_k**2, min=0.0)
  sigma_k = torch.sqrt(sigma_k_sq)
  d_sigma_k = torch.where(
      sigma_k > 0,
      -alpha_k*d_alpha_k/sigma_k,
      torch.zeros_like(sigma_k),
  )
  scale_sq = (
      intensity_ratio**2*signal_scale_sq
      + 0.5*(d_sigma_k**2).mean(dim=(-3, -2, -1), keepdim=True)
  )
  return torch.sqrt(torch.clamp(scale_sq, min=eps**2))
