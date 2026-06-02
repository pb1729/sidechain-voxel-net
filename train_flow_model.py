from dataclasses import dataclass

import torch
from torch import nn

from umup_layers.residual_layers import (
  get_param_groups, scale_param_lrs, UMUPConv3d, UMUPTanhGated, UMUPResiduals)
from util import must_be, annotate_path
from source_save import get_current_source, source_dict_diff
from train_vae import VAE


CIF_DATASET_PATH = "cath-cif"
SAVE_PATH = "models/flownet_8.pt"


@dataclass
class FlowConfig:
  batch:int
  chan_latent:int
  chan_L_list:list[tuple[int, int]]
  vae_path:str
  lr:float = 0.1
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

class UNet3d(nn.Module):
  def __init__(self, conf:FlowConfig):
    super().__init__()
    self.N = len(conf.chan_L_list)
    chan_0 = conf.chan_L_list[0][0]
    self.readin = LinearReadinWithBias(conf.chan_latent, chan_0)
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
    self.readout = UMUPConv3d(chan_0, conf.chan_latent, (1, 1, 1), (0, 0, 0), readout=True)
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
    self.vae = VAE.from_dict(torch.load(conf.vae_path)).eval() # vae is inference only
    self.optim = None
    assert self.vae.conf.chan_out == self.conf.chan_latent
  def to(self, device):
    self.model.to(device)
    self.vae.to(device)
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
  def step(self, i:int, x):
    """ MUTATES self """
    if self.optim is None:
      self.optim = torch.optim.Adam(
        scale_param_lrs(self.conf.lr, get_param_groups(self.model)),
        betas=(0.9, 0.999)
      )
    # encode and decode
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=self.conf.autocast):
      z = self.vae.enc(x)
      t = torch.rand(self.conf.batch)**2 # bias towards low t
      t = t[:, None, None, None, None].expand(-1, 1, *z.shape[-3:])
      ε = torch.randn_like(z)
      z_mix = t*z + (1. - t)*ε
      v = z - ε
      v_pred = self.model(z_mix, t)
      loss = ((v_pred - v)**2).mean()
    # update
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()
    self.record(i, "loss", loss.item())
    self.record(i, "grid_dims", z.shape[-3:])


if __name__ == "__main__":
  from density_dataset import make_density_batch_loader
  batch = 32
  chan_latent = 16
  chan_L_list = [(64, 5), (128, 5), (256, 5), (512, 4)]
  vae_path = "models/vae_8.pt"
  autocast = True
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("device:", device)
  print(SAVE_PATH)
  conf = FlowConfig(batch, chan_latent, chan_L_list, vae_path, autocast=autocast)
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


