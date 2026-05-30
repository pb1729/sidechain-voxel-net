from dataclasses import dataclass
from collections import defaultdict

import torch
from torch import nn


# Unit maximal update parametrization: Make it such that layers
# have unit-variance parameters, and produce unit-variance activations.
# Note: we currently do not target unit-scale gradients; this can be
# added later if it turns out to be necessary.


@dataclass(frozen=True)
class LrScale:
  """ keeps track of param groups of different fan-in (should be 1/sqrt(fan_in))
      use inverse square, as it's usually an integer """
  fan_in: int # effective fan-in for this dataclass
  @property
  def value(self):
    # learning rate scaling factor
    return self.fan_in**-0.5

def get_param_groups(*models):
  # get params
  params = []
  for model in models:
    for name, module in model.named_modules():
      if hasattr(module, "parameters_with_lr_scalings"):
        params.extend(module.parameters_with_lr_scalings())
      else: # don't scale lr if we don't have the data to do so
        params.extend([
          (LrScale(1), param)
          for param in module.parameters(recurse=False)
        ])
  # group by lr scaling
  groups = defaultdict(list)
  for lr_scale, param in params:
    groups[lr_scale].append(param)
  return groups

def scale_param_lrs(base_lr:float, params:defaultdict):
  return [
    {
      "params": params[lr_scale],
      "lr": base_lr*lr_scale.value,
    }
    for lr_scale in params
  ]

class UMUPLinear(nn.Module):
  """ UMUP Linear layer with no bias. """
  def __init__(self, chan_in:int, chan_out:int, readin:bool=False, readout:bool=False):
    super().__init__()
    self.weight = nn.Parameter(torch.empty(out_features, in_features))
    nn.init.normal_(self.weight, mean=0.0, std=1.0)
    if readout:
      self.scale = chan_in**-1.0
      self.lr_scale = LrScale(1)
    else:
      self.scale = chan_in**-0.5
      if readin:
        self.lr_scale = LrScale(chan_out)
      else:
        self.lr_scale = LrScale(chan_in)
  def parameters_with_lr_scalings(self):
    return [(self.lr_scale, self.weight)]
  def forward(self, x):
    return nn.functional.linear(x, self.weight, None)*self.scale

class UMUPConv3d(nn.Module):
  """ UMUP Conv3d layer with no bias. """
  def __init__(self, chan_in:int, chan_out:int,
      kernsz:tuple[int, int, int], padding:tuple[int, int, int], stride:tuple[int, int, int]=(1, 1, 1),
      transpose:bool=False,
      readin:bool=False, readout:bool=False
  ):
    super().__init__()
    kx, ky, kz = kernsz
    k = kx*ky*kz # fan-in associated with kernel
    self.padding = padding
    self.stride = stride
    self.transpose = transpose
    self.weight = nn.Parameter(torch.empty(chan_out, chan_in, kx, ky, kz))
    if self.transpose: # stride reduces fan-in for conv transpose
      k //= self.stride[0]*self.stride[1]*self.stride[2]
    nn.init.normal_(self.weight, mean=0.0, std=1.0)
    if readout:
      self.scale = (k*chan_in)**-1.0
      self.lr_scale = LrScale(1)
    else:
      self.scale = (k*chan_in)**-0.5
      if readin:
        self.lr_scale = LrScale(k*chan_out)
      else:
        self.lr_scale = LrScale(k*chan_in)
  def parameters_with_lr_scalings(self):
    return [(self.lr_scale, self.weight)]
  def forward(self, x, with_bias=None, output_padding=0):
    if self.transpose:
      return self.scale*nn.functional.conv_transpose3d(x, self.weight, bias=with_bias,
        padding=self.padding, stride=self.stride, output_padding=output_padding)
    else:
      return self.scale*nn.functional.conv3d(x, self.weight, bias=with_bias,
        padding=self.padding, stride=self.stride)

def rms_norm_3d_no_affine(x, ε:float=1e-6):
  return x*torch.rsqrt(x.pow(2).mean(1, keepdim=True) + ε)

def embed_t(t):
  """ t: (batch, 1, H, W, L)
      ans: (batch, 16, H, W, L) """
  n = torch.arange(16, device=t.device)[:, None, None, None]
  return torch.cos((1 + torch.sqrt(n))*1.5707963267948966*(t + 10.))
  
class UMUPTanhGated(nn.Module):
  def __init__(self, chan_res:int, chan_gate:int|None=None, alpha_tanh:float=1.6, t_depend:bool=False):
    super().__init__()
    if chan_gate is None:
      chan_gate = chan_res
    self.alpha_tanh = alpha_tanh
    self.t_depend = t_depend
    self.tanh_offsets = nn.Parameter(torch.empty(chan_gate))
    nn.init.zeros_(self.tanh_offsets)
    self.conv1 = UMUPConv3d(chan_res, chan_gate, (3, 3, 3), (1, 1, 1))
    self.conv2 = UMUPConv3d(chan_res, chan_gate, (3, 3, 3), (1, 1, 1))
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

class UMUPResiduals(nn.Module):
  # Note: we don't have the 1/sqrt(L) learning rate factor recommended by Yang right now...
  def __init__(self, L:int, make_layer):
    """ create a residual network of L layers constructed by make_layer(j) for the jth layer """
    super().__init__()
    self.layers = nn.ModuleList([
      make_layer(j)
      for j in range(L)
    ])
  def forward(self, x, t=None):
    n = 1
    for layer in self.layers:
      α = (n/(n + 1))**0.5
      β = (1/(n + 1))**0.5
      x = α*x + β*layer(x, t)
      n = n + 1
    return x


if __name__ == "__main__":
  batch = 4
  chan = 16
  L = 7
  grid = (10, 10, 10)
  model = UMUPResiduals(L, lambda j: UMUPTanhGated(chan))
  optimizer = torch.optim.Adam(
    scale_param_lrs(0.01, get_param_groups(model)),
    betas=(0.9, 0.999)
  )
  for pg in optimizer.param_groups:
    print(pg["lr"], ":", len(pg["params"]))
  x = torch.randn(batch, chan, *grid)
  t = torch.rand(batch, 1, *grid)
  y = model(x, t)
  print("y rms", (y**2).mean().item()**0.5)
  loss = 0.5*(y**2).mean()
  print("loss", loss.item())
  loss.backward()
  for name, param in model.named_parameters():
    print(name, (param.grad**2).mean().item()**0.5)





