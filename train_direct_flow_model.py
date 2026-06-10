from dataclasses import dataclass
import json

import torch
from torch import nn

from umup_layers.residual_layers import (
  get_param_groups, scale_param_lrs, UMUPConv3d, UMUPTanhGated, UMUPResiduals)
from util import must_be, annotate_path
from source_save import get_current_source, source_dict_diff
from kspace_ops import blur_and_velocity, nf_1, v_nf_1, velocity_scale
from train_vae import VAE


CIF_DATASET_PATH = "cath-cif"
SAVE_PATH = "models/flownet_3.pt"
POWER_SPECTRUM_FIT_PATH = "models/power_law_fit.json"


@dataclass
class FlowConfig:
  batch:int
  chan_dens:int
  chan_L_list:list[tuple[int, int]]
  lr:float = 0.1
  autocast:bool = False
  kspace_scale:float = 1.5
  kspace_blur_max:float = 8.0
  kspace_intensity_ratio:float = 20.0
  kspace_power_spectrum_path:str | None = None
  

class LinearReadinWithBias(nn.Module):
  def __init__(self, chan_in:int, chan_out:int):
    super().__init__()
    self.bias = nn.Parameter(torch.empty(chan_out))
    nn.init.zeros_(self.bias)
    self.lin = UMUPConv3d(chan_in, chan_out, (1, 1, 1), (0, 0, 0), readin=True)
  def forward(self, x):
    return self.lin(x, with_bias=self.bias)

def crop_and_add(x, dx):
  """ Take a grid dx that is maybe large than x and add it to x.
      x: (batch, chan, dim_x, dim_y, dim_z)
      dx: (batch, chan, dim_x + qx, dim_y + qy, dim_z + qz)
      qx, qy, qz in {0, 1} """
  *_, dim_x, dim_y, dim_z = x.shape
  return x + dx[..., :dim_x, :dim_y, :dim_z]

def avgpool_8(x):
  kernel = 0.125 + torch.zeros(1, 1, 2, 2, 2, device=x.device)
  return torch.nn.functional.conv3d(x, kernel, stride=(2, 2, 2))

class UNet3d(nn.Module):
  def __init__(self, conf:FlowConfig):
    super().__init__()
    self.N = len(conf.chan_L_list)
    chan_0 = conf.chan_L_list[0][0]
    self.readin = LinearReadinWithBias(conf.chan_dens, chan_0)
    self.input_residuals = nn.ModuleList([
      UMUPResiduals(L, lambda j: UMUPTanhGated(chan, t_depend=True))
      for chan, L in conf.chan_L_list
    ])
    self.compressors = nn.ModuleList([
      UMUPConv3d(chan1, chan2, (2, 2, 2), (0, 0, 0), stride=(2, 2, 2))
      for (chan1, L1), (chan2, L2) in zip(conf.chan_L_list[:-1], conf.chan_L_list[1:])
    ])
    self.expanders = nn.ModuleList([
      UMUPConv3d(chan1, chan2, (2, 2, 2), (0, 0, 0), stride=(2, 2, 2), transpose=True)
      for (chan1, L1), (chan2, L2) in zip(conf.chan_L_list[:-1], conf.chan_L_list[1:])
    ])
    self.output_residuals = nn.ModuleList([
      UMUPResiduals(L, lambda j: UMUPTanhGated(chan, t_depend=True))
      for chan, L in conf.chan_L_list
    ])
    self.readout = UMUPConv3d(chan_0, conf.chan_dens, (1, 1, 1), (0, 0, 0), readout=True)
  def forward(self, x, t):
    """ x: (batch, chan_latent, H, W, L)
        t: (batch, 1, H, W, L) """
    activations = [None]*self.N
    times = [None]*self.N
    activations[0] = self.readin(x)
    times[0] = t
    for i in range(self.N):
      activations[i] = self.input_residuals[i](activations[i], times[i])
      if i + 1 < self.N:
        activations[i + 1] = self.compressors[i](activations[i])
        times[i + 1] = avgpool_8(times[i])
    for i in range(self.N)[::-1]:
      if i + 1 < self.N:
        activations[i] = crop_and_add(
          activations[i],
          self.expanders[i](activations[i + 1], output_padding=1))
      activations[i] = self.output_residuals[i](activations[i], times[i])
    return self.readout(activations[0])


class FlowModel:
  def __init__(self, conf:FlowConfig):
    self.history = {}
    self.conf = conf
    self.model = UNet3d(conf)
    self.source = get_current_source()
    self.power_spectrum_fit = self._load_power_spectrum_fit(conf.kspace_power_spectrum_path)
    self.optim = None
  def to(self, device):
    self.model.to(device)
    return self
  def record(self, i:int, nm:str, val):
    if nm not in self.history:
      self.history[nm] = []
    self.history[nm].append((i, val))
  def to_dict(self):
    return {
      "history": self.history,
      "conf": vars(self.conf),
      "model": self.model.state_dict(),
      "source": self.source,
    }
  @staticmethod
  def from_dict(d):
    ans = FlowModel(FlowConfig(**d["conf"]))
    ans.history = d["history"]
    ans.model.load_state_dict(d["model"])
    curr_source = ans.source
    ans.source = d["source"]
    source_diff = source_dict_diff(ans.source, curr_source)
    # print alerts to source changes
    for add in source_diff["added"]:
      print("added:", add)
    for rem in source_diff["removed"]:
      print("removed:", rem)
    for chg in source_diff["changed"]:
      print("changed:", chg)
      print(source_diff["changed"][chg]["diff"])
    # return the answer
    return ans
  @staticmethod
  def _load_power_spectrum_fit(path):
    if path is None:
      return None
    with open(path) as f:
      data = json.load(f)
    fit = data["fit"]
    if fit.get("model") != "amp * k**(-exponent)":
      raise ValueError(f"unsupported power spectrum model in {path}: {fit.get('model')}")
    return {
      "amp": float(fit["amp"]),
      "exponent": float(fit["exponent"]),
      "min_k": float(fit.get("min_k", 1e-6)),
    }
  def _noise_func(self, t):
    return lambda kx, ky, kz: nf_1(
      t,
      kx,
      ky,
      kz,
      blur_max=self.conf.kspace_blur_max,
      intensity_ratio=self.conf.kspace_intensity_ratio,
    )
  def _velocity_func(self, t):
    return lambda kx, ky, kz: v_nf_1(
      t,
      kx,
      ky,
      kz,
      blur_max=self.conf.kspace_blur_max,
      intensity_ratio=self.conf.kspace_intensity_ratio,
    )
  def _expand_t(self, t, x):
    return t[:, None, None, None, None].expand(-1, 1, *x.shape[-3:])
  def _velocity_scale(self, t, gxyz, device, dtype=torch.float32):
    fit = self.power_spectrum_fit
    return velocity_scale(
      t,
      gxyz,
      device=device,
      dtype=dtype,
      scale=self.conf.kspace_scale,
      blur_max=self.conf.kspace_blur_max,
      intensity_ratio=self.conf.kspace_intensity_ratio,
      signal_power_amp=None if fit is None else fit["amp"],
      signal_power_exponent=None if fit is None else fit["exponent"],
      signal_power_k_min=1e-6 if fit is None else fit["min_k"],
    )
  def step(self, i:int, x):
    """ MUTATES self """
    if self.optim is None:
      self.optim = torch.optim.Adam(
        scale_param_lrs(self.conf.lr, get_param_groups(self.model)),
        betas=(0.9, 0.999)
      )
    # make tensors smaller so we don't run out of memory
    gx, gy, gz = x.shape[-3:]
    crop_x, crop_y, crop_z = max(1, (gx - 48)//2), max(1, (gy - 48)//2), max(1, (gz - 48)//2)
    x = x[..., crop_x:-crop_x, crop_y:-crop_y, crop_z:-crop_z]
    t = torch.rand(x.shape[0], device=x.device)
    with torch.autocast(device_type=x.device.type, enabled=False):
      x_mix, v = blur_and_velocity(
        self._noise_func(t),
        self._velocity_func(t),
        x.float(),
        scale=self.conf.kspace_scale,
      )
      v_scale = self._velocity_scale(t, x.shape[-3:], x.device)
    t_grid = self._expand_t(t, x)
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=self.conf.autocast):
      v_pred = self.model(x_mix, t_grid)
      sqerr = (v_pred - v/v_scale)**2
      loss = sqerr.mean()
    # update
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()
    self.record(i, "loss", loss.item())
    self.record(i, "grid_dims", x.shape[-3:])
  def infer(self, ε, steps=32):
    self.model.eval()
    batch, must_be[self.conf.chan_dens], gx, gy, gz = ε.shape
    for i in range(steps):
      t_value = 1.0 - (0.5 + i)/steps
      t_batch = t_value + torch.zeros(batch, device=ε.device)
      t = self._expand_t(t_batch, ε)
      with torch.autocast(device_type=ε.device.type, dtype=torch.bfloat16, enabled=self.conf.autocast):
        v = self.model(ε, t)
      v = v*self._velocity_scale(t_batch, (gx, gy, gz), ε.device).to(v.dtype)
      ε = ε - v*(1/steps)
    return ε/self.conf.kspace_intensity_ratio


if __name__ == "__main__":
  from density_dataset import CHAN_DENSFN_1, make_density_batch_loader
  batch = 2
  chan_L_list = [(64, 5), (128, 5), (256, 5), (512, 4)]
  autocast = True
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("device:", device)
  print(SAVE_PATH)
  conf = FlowConfig(
    batch,
    CHAN_DENSFN_1,
    chan_L_list,
    autocast=autocast,
    kspace_power_spectrum_path=POWER_SPECTRUM_FIT_PATH,
  )
  flowmodel = FlowModel(conf).to(device)
  i = 0
  for epoch in range(40):
    flowmodel.record(i, "start_epoch", epoch)
    dataloader = make_density_batch_loader(CIF_DATASET_PATH, conf.batch)
    for x in dataloader:
      x = x.to(device)
      flowmodel.step(i, x)
      _, loss = flowmodel.history["loss"][-1]
      print(f"{i}  loss={loss}")
      if i % 10 == 0:
        torch.save(flowmodel.to_dict(), SAVE_PATH)
      i += 1
    torch.save(flowmodel.to_dict(), annotate_path(SAVE_PATH, f"epoch_{epoch}"))
  torch.save(flowmodel.to_dict(), SAVE_PATH)
