from dataclasses import dataclass

import torch
from torch import nn

from umup_layers.residual_layers import (
  get_param_groups, scale_param_lrs, UMUPConv3d, UMUPTanhGated, UMUPResiduals)
from util import must_be, annotate_path
from source_save import get_current_source, source_dict_diff


CIF_DATASET_PATH = "cath-cif"
SAVE_PATH = "models/vae_14.pt"


@dataclass
class VAEConfig:
  batch:int
  chan_dens_fields:int
  chan_1:int
  L_1:int
  chan_2:int
  L_2:int
  chan_out:int
  sigma_z:float = 1.0
  lr:float = 0.1
  λ:float = 1e-6
  autocast:bool = False

class VAEEncoder(nn.Module):
  def __init__(self, conf:VAEConfig):
    super().__init__()
    self.readin = UMUPConv3d(conf.chan_dens_fields, conf.chan_1, (1, 1, 1), (0, 0, 0), readin=True)
    self.layers_1 = UMUPResiduals(conf.L_1, lambda j: UMUPTanhGated(conf.chan_1))
    self.compress = UMUPConv3d(conf.chan_1, conf.chan_2, (2, 2, 2), (0, 0, 0), stride=(2, 2, 2))
    self.layers_2 = UMUPResiduals(conf.L_2, lambda j: UMUPTanhGated(conf.chan_2))
    self.readout = UMUPConv3d(conf.chan_2, conf.chan_out, (1, 1, 1), (0, 0, 0))
  def forward(self, x):
    x = self.readin(x)
    x = self.layers_1(x)
    x = self.compress(x)
    x = self.layers_2(x)
    x = self.readout(x)
    return x

class VAEDecoder(nn.Module):
  def __init__(self, conf:VAEConfig):
    super().__init__()
    self.readin = UMUPConv3d(conf.chan_out, conf.chan_2, (1, 1, 1), (0, 0, 0))
    self.layers_2 = UMUPResiduals(conf.L_2, lambda j: UMUPTanhGated(conf.chan_2))
    self.expand = UMUPConv3d(conf.chan_1, conf.chan_2, (2, 2, 2), (0, 0, 0), stride=(2, 2, 2), transpose=True)
    self.layers_1 = UMUPResiduals(conf.L_1, lambda j: UMUPTanhGated(conf.chan_1))
    self.readout = UMUPConv3d(conf.chan_1, conf.chan_dens_fields, (1, 1, 1), (0, 0, 0), readout=True)
  def forward(self, x):
    x = self.readin(x)
    x = self.layers_2(x)
    x = self.expand(x)
    x = self.layers_1(x)
    x = self.readout(x)
    return x

class VAE:
  def __init__(self, conf:VAEConfig):
    self.history = {}
    self.conf = conf
    self.enc = VAEEncoder(conf)
    self.dec = VAEDecoder(conf)
    self.source = get_current_source()
    self.optim = None
  def to(self, device):
    self.enc.to(device)
    self.dec.to(device)
    return self
  def eval(self):
    self.enc.eval()
    self.dec.eval()
    return self
  def record(self, i:int, nm:str, val):
    if nm not in self.history:
      self.history[nm] = []
    self.history[nm].append((i, val))
  def to_dict(self):
    return {
      "history": self.history,
      "conf": vars(self.conf),
      "enc": self.enc.state_dict(),
      "dec": self.dec.state_dict(),
      "source": self.source,
    }
  @staticmethod
  def from_dict(d):
    ans = VAE(VAEConfig(**d["conf"]))
    ans.history = d["history"]
    ans.enc.load_state_dict(d["enc"])
    ans.dec.load_state_dict(d["dec"])
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
        scale_param_lrs(self.conf.lr, get_param_groups(self.enc, self.dec)),
        betas=(0.9, 0.999)
      )
    # make tensors smaller so we don't run out of memory
    gx, gy, gz = x.shape[-3:]
    crop_x, crop_y, crop_z = max(1, (gx - 40)//2), max(1, (gy - 40)//2), max(1, (gz - 40)//2)
    x = x[..., crop_x:-crop_x, crop_y:-crop_y, crop_z:-crop_z]
    # encode and decode
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=self.conf.autocast):
      z = self.enc(x)
      z_noised = z + self.conf.sigma_z*torch.randn_like(z)
      x_pred = self.dec(z_noised)
      weights = torch.tanh(10*abs(x).sum(1, keepdim=True)) + 0.05
      mserr = (((x_pred - x)**2)*weights).mean() / weights.mean()
      mslat = (z**2).mean()
      loss = mserr + self.conf.λ*mslat
    # update
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()
    self.record(i, "loss", loss.item())
    self.record(i, "mserr", mserr.item())
    self.record(i, "mslat", mslat.item())
    self.record(i, "batch_size", x.shape[0])
    self.record(i, "grid_dims", x.shape[-3:])
    self.record(i, "ms_x_pred", ((weights*x_pred**2).mean()/weights.mean()).item())
    self.record(i, "input_ms", ((weights*x**2).mean()/weights.mean()).item())


if __name__ == "__main__":
  from density_dataset import make_density_batch_loader
  from density_fns import DensFn1
  batch = 8
  densfn = DensFn1()
  chan_1 = 64
  chan_2 = 96
  L = 3
  chan_latent = 16
  sigma_z = 0.1
  autocast = True
  holdout_percent = 10
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("device:", device)
  print(SAVE_PATH)
  conf = VAEConfig(batch, densfn.channel_count(), chan_1, L, chan_2, L, chan_latent, autocast=autocast, sigma_z=sigma_z)
  vae = VAE(conf).to(device)
  i = 0
  for epoch in range(1):
    vae.record(i, "start_epoch", epoch)
    dataloader = make_density_batch_loader(CIF_DATASET_PATH, conf.batch,
      densfn=densfn, holdout_percent=holdout_percent, holdout=False)
    for x in dataloader:
      x = x.to(device)
      vae.step(i, x)
      _, loss = vae.history["loss"][-1]
      _, mserr = vae.history["mserr"][-1]
      _, mslat = vae.history["mslat"][-1]
      _, input_ms = vae.history["input_ms"][-1]
      _, ms_x_pred = vae.history["ms_x_pred"][-1]
      print(f"{i} loss={loss}, rmslat={mslat**0.5}, input_rms={input_ms**0.5}, rms_x_pred={ms_x_pred**0.5}")
      if i % 10 == 0:
        torch.save(vae.to_dict(), SAVE_PATH)
      i += 1
    torch.save(vae.to_dict(), annotate_path(SAVE_PATH, f"epoch_{epoch}"))
  torch.save(vae.to_dict(), SAVE_PATH)
