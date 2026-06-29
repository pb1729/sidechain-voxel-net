from dataclasses import dataclass, replace
import json

import torch
from torch import nn

from umup_layers.residual_layers import ( LrScale,
  get_param_groups, scale_param_lrs, UMUPConv3d, rms_norm_3d_no_affine, UMUPResiduals, embed_t)
from util import must_be, annotate_path
from source_save import get_current_source, source_dict_diff
from big_convs_3d import gaussian_blur_3d, affine_gauss_conv_3d
from density_fns import (
  DensFn1, DensFn2, DensityFunction,
  density_function_from_dict, density_function_to_dict,
)


CIF_DATASET_PATH = "cath-cif"
SAVE_PATH = "models/flownet_10.pt"


@dataclass
class FlowConfig:
  batch:int
  densfn:DensityFunction
  chan_L_list:list[tuple[int, int]]
  blur_list:list[float]
  lr:float = 0.05
  intensity_ratio:float = 2000.
  autocast:bool = False
  

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

class LRConv3d(nn.Module):
  """ Conv that mixes long-range and local updates. """
  def __init__(self, chan_in:int, chan_out:int):
    super().__init__()
    self.conv = UMUPConv3d(chan_in, chan_out, (3, 3, 3), (1, 1, 1))
    self.conv_blur = BlurConv3d(chan_in, [2. + 10.*j/chan_out for j in range(chan_out)])
  def forward(self, x, with_bias=None):
    return self.conv(x, with_bias=with_bias) + self.conv_blur(x)

class BlurConv3d(nn.Module):
  def __init__(self, chan_in:int, sigmas:list[float]):
    super().__init__()
    chan_out = len(sigmas)
    self.weight = nn.Parameter(torch.randn(4, chan_out, chan_in))
    self.sigmas = nn.parameter.Buffer(torch.tensor(sigmas))
    self.lr_scale = LrScale(4*chan_in)
    self.scale = (4*chan_in)**-0.5
  def parameters_with_lr_scalings(self):
    return [(self.lr_scale, self.weight)]
  def forward(self, x):
    return self.scale*affine_gauss_conv_3d(x, self.sigmas, self.weight)

class UMUPTanhGatedLR(nn.Module):
  def __init__(self, chan_res:int, chan_gate:int|None=None, alpha_tanh:float=1.6, t_depend:bool=False):
    super().__init__()
    if chan_gate is None:
      chan_gate = chan_res
    self.alpha_tanh = alpha_tanh
    self.t_depend = t_depend
    self.tanh_offsets = nn.Parameter(torch.empty(chan_gate))
    nn.init.zeros_(self.tanh_offsets)
    self.conv1 = LRConv3d(chan_res, chan_gate)
    self.conv2 = LRConv3d(chan_res, chan_gate)
    self.conv3 = UMUPConv3d(chan_gate, chan_res, (1, 1, 1), (0, 0, 0))
    if self.t_depend:
      self.t_emb = UMUPConv3d(16,     chan_gate, (1, 1, 1), (0, 0, 0))
  def forward(self, x, t=None):
    # pre-norm
    x = rms_norm_3d_no_affine(x)
    # convolutions
    conv = self.conv1(x)
    gate = torch.tanh(self.conv2(x, with_bias=self.tanh_offsets))*self.alpha_tanh
    if self.t_depend:
      gate = gate*(12/13) + self.t_emb(embed_t(t))*(5/13)
    return self.conv3(conv*gate)

class UNet3d(nn.Module):
  def __init__(self, conf:FlowConfig):
    super().__init__()
    self.N = len(conf.chan_L_list)
    chan_0 = conf.chan_L_list[0][0]
    self.readin = LinearReadinWithBias(conf.densfn.channel_count(), chan_0)
    self.input_residuals = nn.ModuleList([
      UMUPResiduals(L, lambda j: UMUPTanhGatedLR(chan, t_depend=True))
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
      UMUPResiduals(L, lambda j: UMUPTanhGatedLR(chan, t_depend=True))
      for chan, L in conf.chan_L_list
    ])
    self.readout = UMUPConv3d(
      chan_0, conf.densfn.channel_count(), (1, 1, 1), (0, 0, 0), readout=True)
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
    self.optim = None
  def to(self, device):
    self.model.to(device)
    if isinstance(self.conf.densfn, DensFn2):
      self.conf.densfn = replace(self.conf.densfn, device=str(device))
    return self
  def record(self, i:int, nm:str, val):
    if nm not in self.history:
      self.history[nm] = []
    self.history[nm].append((i, val))
  def to_dict(self):
    conf = vars(self.conf).copy()
    conf["densfn"] = density_function_to_dict(self.conf.densfn)
    return {
      "history": self.history,
      "conf": conf,
      "model": self.model.state_dict(),
      "source": self.source,
    }
  @staticmethod
  def from_dict(d):
    conf = d["conf"].copy()
    if isinstance(conf["densfn"], dict):
      conf["densfn"] = density_function_from_dict(conf["densfn"])
    ans = FlowModel(FlowConfig(**conf))
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
    x = x*self.conf.intensity_ratio
    ε = torch.randn_like(x)
    t = torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device).expand(-1, 1, *x.shape[-3:])
    with torch.autocast(device_type=x.device.type, enabled=False):
      x_mix = t*x + (1. - t)*ε
      v = x - ε
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=self.conf.autocast):
      v_pred = self.model(x_mix, t)
      sqerrs = []
      sqerrs.append(((v_pred - v)**2).mean())
      for blur in self.conf.blur_list:
        v = gaussian_blur_3d(v, blur)
        v_pred = gaussian_blur_3d(v_pred, blur)
        sqerrs.append(((v_pred - v)**2).mean())
      loss = sum(sqerrs)
    # update
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()
    # record metrics
    self.record(i, "blur_sqerrs", [sqerr.item() for sqerr in sqerrs])
    self.record(i, "loss", loss.item())
    self.record(i, "grid_dims", x.shape[-3:])
  def infer(self, ε, steps=32):
    self.model.eval()
    batch, must_be[self.conf.densfn.channel_count()], gx, gy, gz = ε.shape
    for i in range(steps):
      t_value = (0.5 + i)/steps
      t = t_value + torch.zeros(batch, device=ε.device)
      t = t[:, None, None, None, None].expand(-1, 1, *ε.shape[-3:])
      with torch.autocast(device_type=ε.device.type, dtype=torch.bfloat16, enabled=self.conf.autocast):
        v = self.model(ε, t)
      ε = ε + v*(1/steps)
    return ε/self.conf.intensity_ratio


if __name__ == "__main__":
  from density_dataset import make_density_batch_loader
  batch = 1
  densfn = DensFn2(atom_gaussian_radius=1.5, output_channels=24, device="cuda", conv_method="fft", fft_phase_batch_size=2)
  chan_L_list = [(64, 5), (128, 5), (256, 5), (512, 4)]
  autocast = True
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("device:", device)
  print(SAVE_PATH)
  conf = FlowConfig(
    batch,
    densfn,
    chan_L_list,
    [2., 4., 6., 8.],
    autocast=autocast,
  )
  flowmodel = FlowModel(conf).to(device)
  i = 0
  for epoch in range(40):
    flowmodel.record(i, "start_epoch", epoch)
    dataloader = make_density_batch_loader(CIF_DATASET_PATH, conf.batch, densfn=conf.densfn)
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
