from dataclasses import dataclass
import itertools
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from util import must_be
from parse_cif import ATOM_IDENTITY_ELEMENTS, ProteinWithCodes, Z2CoordPair


def NPArr(dtype, *shape):
  """ Type annotation for np array of given dtype and shape. """
  return np.ndarray[tuple[tuple([Literal[dim] for dim in shape])], np.dtype[dtype]]

def vec3(x, y ,z):
  return np.array([x, y, z])


DENSFN_FORWARD_1_CHANNELS = (
  "backbone_ca",
  "backbone_c",
  "backbone_n",
  "backbone_o",
  "sidechain_n",
  "sidechain_o",
  "sidechain_s",
  "sidechain_c_grey",
  "sidechain_c_blue",
  "backbone_bond",
  "chain_start",
  "chain_end",
)
DENSFN_FORWARD_1_CHANNEL_INDEX = {
  channel: i for i, channel in enumerate(DENSFN_FORWARD_1_CHANNELS)
}
DENSFN_FORWARD_2_CHANNELS = (
  "backbone_ca",
  "backbone_c",
  "backbone_n",
  "backbone_o",
  "sidechain_n",
  "sidechain_o",
  "sidechain_s",
  "sidechain_c_grey",
  "sidechain_c_blue",
)
DENSFN_FORWARD_2_CHANNEL_INDEX = {
  channel: i for i, channel in enumerate(DENSFN_FORWARD_2_CHANNELS)
}
DENSFN_FORWARD_2_CHANNELS_IN_FORWARD_1 = tuple(
  DENSFN_FORWARD_1_CHANNEL_INDEX[channel]
  for channel in DENSFN_FORWARD_2_CHANNELS
)


@dataclass(frozen=True)
class DensFn1:
  def channel_count(self) -> int:
    return len(DENSFN_FORWARD_1_CHANNELS)

  def forward(
      self,
      protein: ProteinWithCodes,
      *,
      grid: "Grid | None" = None,
      dx: float = 1.0,
      padding: float = 2.0,
      rng: np.random.Generator | None = None,
  ) -> tuple["Grid", NPArr(np.float32, "Nx", "Ny", "Nz", "chan")]:
    return densfn_forward_1(
      protein,
      grid=grid,
      dx=dx,
      padding=padding,
      rng=rng,
    )

  def backward(
      self,
      grid: "Grid",
      field: NPArr(np.float32, "Nx", "Ny", "Nz", "chan"),
      *,
      peak_threshold: float = 0.55,
  ) -> ProteinWithCodes:
    return densfn_backward_1(grid, field, peak_threshold=peak_threshold)


@dataclass(frozen=True)
class DensFn2:
  stride: int = 4
  atom_gaussian_radius: float = 1.0
  atom_gaussian_sigma: float | None = None
  output_channels: int = 32
  seed: int = 0
  random_conv_radius: float = 4.0
  device: str | torch.device | None = None
  conv_method: Literal["direct", "fft"] = "fft"
  fft_phase_batch_size: int | None = 8

  def channel_count(self) -> int:
    return self.output_channels

  def forward(
      self,
      protein: ProteinWithCodes,
      *,
      grid: "Grid | None" = None,
      dx: float = 1.0,
      padding: float = 2.0,
      rng: np.random.Generator | None = None,
  ) -> tuple["Grid", NPArr(np.float32, "Nx", "Ny", "Nz", "output_channels")]:
    return densfn_forward_2(
      protein,
      grid=grid,
      dx=dx,
      padding=padding,
      rng=rng,
      stride=self.stride,
      atom_gaussian_radius=self.atom_gaussian_radius,
      atom_gaussian_sigma=self.atom_gaussian_sigma,
      output_channels=self.output_channels,
      seed=self.seed,
      random_conv_radius=self.random_conv_radius,
      device=self.device,
      conv_method=self.conv_method,
      fft_phase_batch_size=self.fft_phase_batch_size,
    )

  def backward(
      self,
      grid: "Grid",
      field: NPArr(np.float32, "Nx", "Ny", "Nz", "output_channels"),
      *,
      peak_rel_threshold: float = 0.25,
      peak_abs_threshold: float | None = None,
      peak_min_distance: float = 0.7,
      max_peaks_per_channel: int = 512,
  ) -> ProteinWithCodes:
    return densfn_backward_2(
      grid,
      field,
      stride=self.stride,
      atom_gaussian_radius=self.atom_gaussian_radius,
      atom_gaussian_sigma=self.atom_gaussian_sigma,
      seed=self.seed,
      random_conv_radius=self.random_conv_radius,
      device=self.device,
      conv_method=self.conv_method,
      fft_phase_batch_size=self.fft_phase_batch_size,
      peak_rel_threshold=peak_rel_threshold,
      peak_abs_threshold=peak_abs_threshold,
      peak_min_distance=peak_min_distance,
      max_peaks_per_channel=max_peaks_per_channel,
    )


DensityFunction = DensFn1 | DensFn2


GREY_CARBON = "grey"
BLUE_CARBON = "blue"


SIDECHAIN_CARBON_COLORS = {
  "A": {"CB": GREY_CARBON},
  "R": {"CB": GREY_CARBON, "CG": BLUE_CARBON, "CD": GREY_CARBON, "CZ": BLUE_CARBON},
  "N": {"CB": GREY_CARBON, "CG": BLUE_CARBON},
  "D": {"CB": GREY_CARBON, "CG": BLUE_CARBON},
  "C": {"CB": GREY_CARBON},
  "Q": {"CB": GREY_CARBON, "CG": BLUE_CARBON, "CD": GREY_CARBON},
  "E": {"CB": GREY_CARBON, "CG": BLUE_CARBON, "CD": GREY_CARBON},
  "G": {},
  "H": {"CB": GREY_CARBON, "CG": BLUE_CARBON, "CD2": GREY_CARBON, "CE1": BLUE_CARBON},
  "I": {"CB": GREY_CARBON, "CG1": BLUE_CARBON, "CG2": BLUE_CARBON, "CD1": GREY_CARBON},
  "L": {"CB": GREY_CARBON, "CG": BLUE_CARBON, "CD1": GREY_CARBON, "CD2": GREY_CARBON},
  "K": {"CB": GREY_CARBON, "CG": BLUE_CARBON, "CD": GREY_CARBON, "CE": BLUE_CARBON},
  "M": {"CB": GREY_CARBON, "CG": BLUE_CARBON, "CE": GREY_CARBON},
  "F": {"CB": GREY_CARBON, "CG": BLUE_CARBON, "CD": GREY_CARBON, "CE": BLUE_CARBON, "CZ": GREY_CARBON},
  "P": {"CB": GREY_CARBON, "CG": BLUE_CARBON, "CD": GREY_CARBON},
  "S": {"CB": GREY_CARBON},
  "T": {"CB": GREY_CARBON, "CG2": BLUE_CARBON},
  "W": {
    "CB": GREY_CARBON,
    "CG": BLUE_CARBON,
    "CD1": GREY_CARBON,
    "CD2": GREY_CARBON,
    "CE2": BLUE_CARBON,
    "CE3": BLUE_CARBON,
    "CZ2": GREY_CARBON,
    "CZ3": GREY_CARBON,
    "CH2": BLUE_CARBON,
  },
  "Y": {"CB": GREY_CARBON, "CG": BLUE_CARBON, "CD": GREY_CARBON, "CE": BLUE_CARBON, "CZ": GREY_CARBON},
  "V": {"CB": GREY_CARBON, "CG1": BLUE_CARBON, "CG2": BLUE_CARBON},
}


SIDECHAIN_ATOM_TEMPLATES = {
  "A": {"CB": "sidechain_c_grey"},
  "R": {"CB": "sidechain_c_grey", "CG": "sidechain_c_blue", "CD": "sidechain_c_grey", "CZ": "sidechain_c_blue", "NE": "sidechain_n", "NH": "sidechain_n"},
  "N": {"CB": "sidechain_c_grey", "CG": "sidechain_c_blue", "OD1": "sidechain_o", "ND2": "sidechain_n"},
  "D": {"CB": "sidechain_c_grey", "CG": "sidechain_c_blue", "OD": "sidechain_o"},
  "C": {"CB": "sidechain_c_grey", "SG": "sidechain_s"},
  "Q": {"CB": "sidechain_c_grey", "CG": "sidechain_c_blue", "CD": "sidechain_c_grey", "OE1": "sidechain_o", "NE2": "sidechain_n"},
  "E": {"CB": "sidechain_c_grey", "CG": "sidechain_c_blue", "CD": "sidechain_c_grey", "OE": "sidechain_o"},
  "G": {},
  "H": {"CB": "sidechain_c_grey", "CG": "sidechain_c_blue", "ND1": "sidechain_n", "CD2": "sidechain_c_grey", "CE1": "sidechain_c_blue", "NE2": "sidechain_n"},
  "I": {"CB": "sidechain_c_grey", "CG1": "sidechain_c_blue", "CG2": "sidechain_c_blue", "CD1": "sidechain_c_grey"},
  "L": {"CB": "sidechain_c_grey", "CG": "sidechain_c_blue", "CD1": "sidechain_c_grey", "CD2": "sidechain_c_grey"},
  "K": {"CB": "sidechain_c_grey", "CG": "sidechain_c_blue", "CD": "sidechain_c_grey", "CE": "sidechain_c_blue", "NZ": "sidechain_n"},
  "M": {"CB": "sidechain_c_grey", "CG": "sidechain_c_blue", "SD": "sidechain_s", "CE": "sidechain_c_grey"},
  "F": {"CB": "sidechain_c_grey", "CG": "sidechain_c_blue", "CD": "sidechain_c_grey", "CE": "sidechain_c_blue", "CZ": "sidechain_c_grey"},
  "P": {"CB": "sidechain_c_grey", "CG": "sidechain_c_blue", "CD": "sidechain_c_grey"},
  "S": {"CB": "sidechain_c_grey", "OG": "sidechain_o"},
  "T": {"CB": "sidechain_c_grey", "OG1": "sidechain_o", "CG2": "sidechain_c_blue"},
  "W": {"CB": "sidechain_c_grey", "CG": "sidechain_c_blue", "CD1": "sidechain_c_grey", "CD2": "sidechain_c_grey", "NE1": "sidechain_n", "CE2": "sidechain_c_blue", "CE3": "sidechain_c_blue", "CZ2": "sidechain_c_grey", "CZ3": "sidechain_c_grey", "CH2": "sidechain_c_blue"},
  "Y": {"CB": "sidechain_c_grey", "CG": "sidechain_c_blue", "CD": "sidechain_c_grey", "CE": "sidechain_c_blue", "CZ": "sidechain_c_grey", "OH": "sidechain_o"},
  "V": {"CB": "sidechain_c_grey", "CG1": "sidechain_c_blue", "CG2": "sidechain_c_blue"},
}


SIDECHAIN_Z2_ATOMS = {
  "R": {"NH"},
  "D": {"OD"},
  "E": {"OE"},
  "F": {"CD", "CE"},
  "Y": {"CD", "CE"},
}


SIDECHAIN_BONDS = {
  "A": (("CA", "CB"),),
  "R": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"), ("CZ", "NH")),
  "N": (("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")),
  "D": (("CA", "CB"), ("CB", "CG"), ("CG", "OD")),
  "C": (("CA", "CB"), ("CB", "SG")),
  "Q": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")),
  "E": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE")),
  "G": (),
  "H": (("CA", "CB"), ("CB", "CG"), ("CG", "ND1"), ("CG", "CD2"), ("ND1", "CE1"), ("CD2", "NE2"), ("CE1", "NE2")),
  "I": (("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")),
  "L": (("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")),
  "K": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")),
  "M": (("CA", "CB"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE")),
  "F": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "CZ")),
  "P": (("CA", "CB"), ("CB", "CG"), ("CG", "CD")),
  "S": (("CA", "CB"), ("CB", "OG")),
  "T": (("CA", "CB"), ("CB", "OG1"), ("CB", "CG2")),
  "W": (
    ("CA", "CB"),
    ("CB", "CG"),
    ("CG", "CD1"),
    ("CG", "CD2"),
    ("CD1", "NE1"),
    ("NE1", "CE2"),
    ("CE2", "CD2"),
    ("CD2", "CE3"),
    ("CE2", "CZ2"),
    ("CE3", "CZ3"),
    ("CZ2", "CH2"),
    ("CZ3", "CH2"),
  ),
  "Y": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "CZ"), ("CZ", "OH")),
  "V": (("CA", "CB"), ("CB", "CG1"), ("CB", "CG2")),
}


def _sidechain_atom_count(aa, atom_name):
  return 2 if atom_name in SIDECHAIN_Z2_ATOMS.get(aa, set()) else 1


SIDECHAIN_TEMPLATE_COUNTS = {
  aa: {
    channel: sum(
      _sidechain_atom_count(aa, atom_name)
      for atom_name, atom_channel in atoms.items()
      if atom_channel == channel
    )
    for channel in ("sidechain_n", "sidechain_o", "sidechain_s", "sidechain_c_grey", "sidechain_c_blue")
  }
  for aa, atoms in SIDECHAIN_ATOM_TEMPLATES.items()
}

@dataclass
class Grid:
  dx: float
  N:  NPArr(np.int32, 3)
  transform: NPArr(np.float32, 4, 4) | None = None
  @property
  def shape(self):
    return tuple(self.N)
  @property
  def L(self):
    return self.dx*self.N


@dataclass(frozen=True)
class _Peak:
  index_coord: NPArr(np.float32, 3)
  coord: NPArr(np.float32, 3)
  value: float


@dataclass(frozen=True)
class DensityCandidate:
  channel: str
  coord: NPArr(np.float32, 3)
  value: float = 1.0


@dataclass(frozen=True)
class _BackboneResidue:
  n: NPArr(np.float32, 3)
  ca: NPArr(np.float32, 3)
  c: NPArr(np.float32, 3)
  o: NPArr(np.float32, 3) | None = None


def random_rotation(rng: np.random.Generator | None = None) -> NPArr(np.float64, 3, 3):
  """Draw a rotation matrix from Haar measure on SO(3)."""
  if rng is None:
    rng = np.random.default_rng()
  
  q = rng.standard_normal(4)
  q = q/np.linalg.norm(q)

  x, y, z, w = q

  return 2*np.array([
      [0.5 - (y * y + z * z), (x * y - z * w), (x * z + y * w)],
      [(x * y + z * w), 0.5 - (x * x + z * z), (y * z - x * w)],
      [(x * z - y * w), (y * z + x * w), 0.5 - (x * x + y * y)],
  ])


def _protein_positions(protein: ProteinWithCodes) -> NPArr(np.float32, "natom", 3):
  positions = []
  for chain in protein:
    for _aa, residue in chain:
      for coord_or_pair in residue.values():
        positions.extend(_coords_for_density(coord_or_pair))
  if not positions:
    raise ValueError("protein does not contain any atom coordinates")
  return np.asarray(positions, dtype=np.float32)


def box_with_grid_for_protein(
    protein: ProteinWithCodes,
    *,
    dx: float = 1.0,
    padding: float = 2.0,
    rng: np.random.Generator | None = None,
) -> Grid:
  """Make a randomly oriented grid covering all atoms in a parsed protein."""
  if dx <= 0.0:
    raise ValueError("dx must be positive")
  if padding < 0.0:
    raise ValueError("padding must be non-negative")

  positions = _protein_positions(protein)
  rotation = random_rotation(rng)
  rotated_positions = positions @ rotation.T

  min_corner = rotated_positions.min(axis=0)
  max_corner = rotated_positions.max(axis=0)
  padded_min_corner = min_corner - padding
  padded_size = (max_corner - min_corner) + 2.0 * padding
  N = np.ceil(padded_size / dx).astype(np.int32)
  N = np.maximum(N, 1)

  transform = np.eye(4, dtype=np.float32)
  transform[:3, :3] = rotation.astype(np.float32)
  transform[:3, 3] = (-padded_min_corner).astype(np.float32)

  return Grid(dx=float(dx), N=N, transform=transform)


def coord_to_grid_index(grid: Grid, coord: NPArr(np.float32, 3)) -> NPArr(np.float32, 3):
  """Map an original CIF Cartesian coordinate to a floating point grid index."""
  coord = np.asarray(coord, dtype=np.float32)
  must_be[3], = coord.shape

  if grid.transform is None:
    grid_coord = coord
  else:
    homogeneous_coord = np.concatenate([coord, np.ones(1, dtype=coord.dtype)])
    grid_coord = (grid.transform @ homogeneous_coord)[:3]
  return grid_coord / grid.dx


def grid_index_to_coord(grid: Grid, index_coord: NPArr(np.float32, 3)) -> NPArr(np.float32, 3):
  """Map a floating point grid index coordinate to original CIF Cartesian coordinates."""
  index_coord = np.asarray(index_coord, dtype=np.float32)
  must_be[3], = index_coord.shape
  grid_coord = index_coord * grid.dx

  if grid.transform is None:
    return grid_coord
  homogeneous_coord = np.concatenate([grid_coord, np.ones(1, dtype=grid_coord.dtype)])
  return (np.linalg.inv(grid.transform) @ homogeneous_coord)[:3].astype(np.float32)


def density_field(grid: Grid, chan: int) -> NPArr(np.float32, "Nx", "Ny", "Nz", "chan"):
  """Create an empty density field with shape (Nx, Ny, Nz, chan)."""
  if chan <= 0:
    raise ValueError("chan must be positive")
  return np.zeros((*grid.shape, chan), dtype=np.float32)


def add_atom_densities(
    field: NPArr(np.float32, "Nx", "Ny", "Nz", "chan"),
    grid: Grid,
    positions: NPArr(np.float32, "natom", 3),
    vectors: NPArr(np.float32, "natom", "chan"),
) -> NPArr(np.float32, "Nx", "Ny", "Nz", "chan"):
  """Add tent-function atom densities to ``field`` in-place.

  ``positions`` are original CIF Cartesian coordinates. Each row of ``vectors``
  gives that atom's contribution strength for every density channel.
  """
  positions = np.asarray(positions, dtype=np.float32)
  vectors = np.asarray(vectors, dtype=np.float32)

  natom, must_be[3] = positions.shape
  must_be[natom], chan = vectors.shape
  must_be[(*grid.shape, chan)] = field.shape

  for position, vector in zip(positions, vectors):
    index = coord_to_grid_index(grid, position)
    index_floor = np.floor(index).astype(np.int32)
    index_frac = index - index_floor

    for dx_i, wx in ((0, 1.0 - index_frac[0]), (1, index_frac[0])):
      ix = index_floor[0] + dx_i
      if ix < 0 or ix >= grid.N[0]:
        continue
      for dy_i, wy in ((0, 1.0 - index_frac[1]), (1, index_frac[1])):
        iy = index_floor[1] + dy_i
        if iy < 0 or iy >= grid.N[1]:
          continue
        for dz_i, wz in ((0, 1.0 - index_frac[2]), (1, index_frac[2])):
          iz = index_floor[2] + dz_i
          if iz < 0 or iz >= grid.N[2]:
            continue
          field[ix, iy, iz, :] += (wx * wy * wz) * vector

  return field


def _coords_for_density(coord_or_pair):
  if isinstance(coord_or_pair, Z2CoordPair):
    return (coord_or_pair.a, coord_or_pair.b)
  return (coord_or_pair,)


def _add_density_sample(positions, vectors, coord, channel, value=1.0):
  vector = np.zeros(len(DENSFN_FORWARD_1_CHANNELS), dtype=np.float32)
  vector[channel] = value
  positions.append(np.asarray(coord, dtype=np.float32))
  vectors.append(vector)


def _backbone_bond_midpoints(chain):
  midpoints = []
  for residue_i, (_aa, residue) in enumerate(chain):
    if "N" in residue and "CA" in residue:
      midpoints.append(0.5 * (residue["N"] + residue["CA"]))
    if "CA" in residue and "C" in residue:
      midpoints.append(0.5 * (residue["CA"] + residue["C"]))
    if residue_i + 1 < len(chain) and "C" in residue and "N" in chain[residue_i + 1][1]:
      midpoints.append(0.5 * (residue["C"] + chain[residue_i + 1][1]["N"]))
  return midpoints


def _chain_end_weight(offset_from_end, count):
  if count == 1:
    return 4.0
  return float(np.linspace(4.0, 0.0, count, dtype=np.float32)[offset_from_end])


def _densfn_forward_1_samples(protein: ProteinWithCodes):
  density_positions = []
  density_vectors = []

  backbone_channels = {
    "CA": DENSFN_FORWARD_1_CHANNEL_INDEX["backbone_ca"],
    "C": DENSFN_FORWARD_1_CHANNEL_INDEX["backbone_c"],
    "N": DENSFN_FORWARD_1_CHANNEL_INDEX["backbone_n"],
    "O": DENSFN_FORWARD_1_CHANNEL_INDEX["backbone_o"],
  }
  sidechain_element_channels = {
    "N": DENSFN_FORWARD_1_CHANNEL_INDEX["sidechain_n"],
    "O": DENSFN_FORWARD_1_CHANNEL_INDEX["sidechain_o"],
    "S": DENSFN_FORWARD_1_CHANNEL_INDEX["sidechain_s"],
  }
  carbon_channels = {
    GREY_CARBON: DENSFN_FORWARD_1_CHANNEL_INDEX["sidechain_c_grey"],
    BLUE_CARBON: DENSFN_FORWARD_1_CHANNEL_INDEX["sidechain_c_blue"],
  }

  for chain in protein:
    for aa, residue in chain:
      carbon_colors = SIDECHAIN_CARBON_COLORS[aa]
      for atom_identity, coord_or_pair in residue.items():
        if atom_identity in backbone_channels:
          channel = backbone_channels[atom_identity]
        elif atom_identity == "OXT":
          continue
        else:
          element = ATOM_IDENTITY_ELEMENTS[atom_identity]
          if element == "C":
            try:
              channel = carbon_channels[carbon_colors[atom_identity]]
            except KeyError as exc:
              raise ValueError(
                f"no sidechain carbon color for amino acid {aa}, atom {atom_identity}"
              ) from exc
          else:
            try:
              channel = sidechain_element_channels[element]
            except KeyError as exc:
              raise ValueError(
                f"no density channel for amino acid {aa}, atom {atom_identity}"
              ) from exc

        for coord in _coords_for_density(coord_or_pair):
          _add_density_sample(density_positions, density_vectors, coord, channel)

    bond_midpoints = _backbone_bond_midpoints(chain)
    for midpoint in bond_midpoints:
      _add_density_sample(
        density_positions,
        density_vectors,
        midpoint,
        DENSFN_FORWARD_1_CHANNEL_INDEX["backbone_bond"],
      )

    chain_marker_count = min(8, len(bond_midpoints))
    for i in range(chain_marker_count):
      start_weight = _chain_end_weight(i, chain_marker_count)
      end_weight = _chain_end_weight(i, chain_marker_count)
      _add_density_sample(
        density_positions,
        density_vectors,
        bond_midpoints[i],
        DENSFN_FORWARD_1_CHANNEL_INDEX["chain_start"],
        start_weight,
      )
      _add_density_sample(
        density_positions,
        density_vectors,
        bond_midpoints[-1 - i],
        DENSFN_FORWARD_1_CHANNEL_INDEX["chain_end"],
        end_weight,
      )

  return density_positions, density_vectors


def densfn_forward_1(
    protein: ProteinWithCodes,
    *,
    grid: Grid | None = None,
    dx: float = 1.0,
    padding: float = 2.0,
    rng: np.random.Generator | None = None,
) -> tuple[Grid, NPArr(np.float32, "Nx", "Ny", "Nz", "chan")]:
  """Map a parsed protein to the first density-field representation."""
  if grid is None:
    grid = box_with_grid_for_protein(protein, dx=dx, padding=padding, rng=rng)
  field = density_field(grid, len(DENSFN_FORWARD_1_CHANNELS))
  density_positions, density_vectors = _densfn_forward_1_samples(protein)

  if density_positions:
    add_atom_densities(
      field,
      grid,
      np.asarray(density_positions, dtype=np.float32),
      np.asarray(density_vectors, dtype=np.float32),
    )

  return grid, field


def add_gaussian_atom_densities(
    field: NPArr(np.float32, "Nx", "Ny", "Nz", "chan"),
    grid: Grid,
    positions: NPArr(np.float32, "natom", 3),
    vectors: NPArr(np.float32, "natom", "chan"),
    *,
    radius: float,
    sigma: float | None = None,
) -> NPArr(np.float32, "Nx", "Ny", "Nz", "chan"):
  """Add truncated Gaussian atom densities to ``field`` in-place."""
  if radius <= 0.0:
    raise ValueError("radius must be positive")
  if sigma is None:
    sigma = 0.5 * radius
  if sigma <= 0.0:
    raise ValueError("sigma must be positive")

  positions = np.asarray(positions, dtype=np.float32)
  vectors = np.asarray(vectors, dtype=np.float32)

  natom, must_be[3] = positions.shape
  must_be[natom], chan = vectors.shape
  must_be[(*grid.shape, chan)] = field.shape

  radius_grid = int(np.ceil(radius / grid.dx))
  inv_two_sigma2 = 0.5 / (sigma * sigma)
  for position, vector in zip(positions, vectors):
    index = coord_to_grid_index(grid, position)
    center = np.round(index).astype(np.int32)
    lo = np.maximum(center - radius_grid, 0)
    hi = np.minimum(center + radius_grid + 1, grid.N)
    if np.any(lo >= hi):
      continue

    coords = np.indices(tuple(hi - lo), dtype=np.float32)
    dist2 = np.zeros(tuple(hi - lo), dtype=np.float32)
    for axis in range(3):
      offset_angstrom = (coords[axis] + lo[axis] - index[axis]) * grid.dx
      dist2 += offset_angstrom * offset_angstrom
    weights = np.exp(-dist2 * inv_two_sigma2).astype(np.float32)
    weights[dist2 > radius * radius] = 0.0
    weight_sum = float(weights.sum())
    if weight_sum == 0.0:
      continue
    weights /= weight_sum
    field[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2], :] += weights[..., None] * vector

  return field


def random_radial_conv_kernel_3d(
    chan_in: int,
    chan_out: int,
    *,
    dx: float,
    radius: float,
    seed: int,
) -> NPArr(np.float32, "chan_out", "chan_in", "kx", "ky", "kz"):
  """Create a deterministic random 3D conv kernel with a radial cutoff."""
  if chan_in <= 0:
    raise ValueError("chan_in must be positive")
  if chan_out <= 0:
    raise ValueError("chan_out must be positive")
  if dx <= 0.0:
    raise ValueError("dx must be positive")
  if radius <= 0.0:
    raise ValueError("radius must be positive")

  radius_grid = int(np.ceil(radius / dx))
  offsets = np.arange(-radius_grid, radius_grid + 1, dtype=np.float32) * dx
  x, y, z = np.meshgrid(offsets, offsets, offsets, indexing="ij")
  dist = np.sqrt(x * x + y * y + z * z)
  window_sigma = 0.5 * radius
  window = np.exp(-0.5 * np.square(dist / window_sigma)).astype(np.float32)
  window[dist > radius] = 0.0

  rng = np.random.default_rng(seed)
  kernel = rng.standard_normal(
    (chan_out, chan_in, *window.shape),
    dtype=np.float32,
  )
  kernel *= window[None, None, :, :, :]

  expected_sq_norm = chan_in * float(np.sum(window * window))
  if expected_sq_norm > 0.0:
    kernel *= np.float32(1.0 / np.sqrt(expected_sq_norm))
  return kernel


def _as_3_int_tuple(value, name: str):
  if isinstance(value, tuple):
    if len(value) != 3:
      raise ValueError(f"{name} must be an int or length-3 tuple")
    return tuple(int(v) for v in value)
  return (int(value), int(value), int(value))


def _fft_conv3d_full(input_tensor: torch.Tensor, kernel_tensor: torch.Tensor) -> torch.Tensor:
  """Full linear 3D convolution summed over input channels."""
  batch, chan_in, gx, gy, gz = input_tensor.shape
  chan_out, must_be[chan_in], kx, ky, kz = kernel_tensor.shape
  fft_shape = (gx + kx - 1, gy + ky - 1, gz + kz - 1)

  input_fft = torch.fft.rfftn(input_tensor, s=fft_shape, dim=(-3, -2, -1))
  kernel_fft = torch.fft.rfftn(kernel_tensor, s=fft_shape, dim=(-3, -2, -1))
  output_fft = (
    input_fft[:, None, :, :, :, :] * kernel_fft[None, :, :, :, :, :]
  ).sum(dim=2)
  return torch.fft.irfftn(output_fft, s=fft_shape, dim=(-3, -2, -1))


def _fft_conv3d_full_batched_phases(
    input_phases: list[torch.Tensor],
    kernel_phases: list[torch.Tensor],
) -> list[torch.Tensor]:
  """Full linear 3D convolutions for matched input/kernel phase pairs."""
  if len(input_phases) != len(kernel_phases):
    raise ValueError("input_phases and kernel_phases must have the same length")
  if not input_phases:
    return []

  batch, chan_in = input_phases[0].shape[:2]
  chan_out, must_be[chan_in] = kernel_phases[0].shape[:2]
  input_shapes = [phase.shape[-3:] for phase in input_phases]
  kernel_shapes = [phase.shape[-3:] for phase in kernel_phases]
  fft_shape = tuple(
    max(input_shape[axis] + kernel_shape[axis] - 1
        for input_shape, kernel_shape in zip(input_shapes, kernel_shapes))
    for axis in range(3)
  )

  input_stack = torch.zeros(
    (len(input_phases), batch, chan_in, *fft_shape),
    dtype=input_phases[0].dtype,
    device=input_phases[0].device,
  )
  kernel_stack = torch.zeros(
    (len(kernel_phases), chan_out, chan_in, *fft_shape),
    dtype=kernel_phases[0].dtype,
    device=kernel_phases[0].device,
  )
  for phase_i, input_phase in enumerate(input_phases):
    input_stack[
      phase_i,
      :,
      :,
      :input_phase.shape[-3],
      :input_phase.shape[-2],
      :input_phase.shape[-1],
    ] = input_phase
  for phase_i, kernel_phase in enumerate(kernel_phases):
    kernel_stack[
      phase_i,
      :,
      :,
      :kernel_phase.shape[-3],
      :kernel_phase.shape[-2],
      :kernel_phase.shape[-1],
    ] = kernel_phase

  input_fft = torch.fft.rfftn(input_stack, dim=(-3, -2, -1))
  kernel_fft = torch.fft.rfftn(kernel_stack, dim=(-3, -2, -1))
  output_fft = (
    input_fft[:, :, None, :, :, :, :] * kernel_fft[:, None, :, :, :, :, :]
  ).sum(dim=3)
  full_stack = torch.fft.irfftn(output_fft, s=fft_shape, dim=(-3, -2, -1))

  return [
    full_stack[
      phase_i,
      :,
      :,
      :input_shapes[phase_i][0] + kernel_shapes[phase_i][0] - 1,
      :input_shapes[phase_i][1] + kernel_shapes[phase_i][1] - 1,
      :input_shapes[phase_i][2] + kernel_shapes[phase_i][2] - 1,
    ]
    for phase_i in range(len(input_phases))
  ]


def conv3d_strided_fft_polyphase(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    *,
    padding: int | tuple[int, int, int],
    stride: int | tuple[int, int, int],
    phase_batch_size: int | None = None,
) -> torch.Tensor:
  """Compute ``F.conv3d(input_tensor, weight, padding=padding, stride=stride)``.

  This uses a polyphase decomposition so striding is handled before the FFT
  convolutions. It is intended for large, non-separable kernels on CPU.
  """
  if input_tensor.ndim != 5:
    raise ValueError(f"input_tensor must have shape (batch, chan, x, y, z), got {input_tensor.shape}")
  if weight.ndim != 5:
    raise ValueError(f"weight must have shape (chan_out, chan_in, kx, ky, kz), got {weight.shape}")
  batch, chan_in, gx, gy, gz = input_tensor.shape
  chan_out, must_be[chan_in], kx, ky, kz = weight.shape
  padding = _as_3_int_tuple(padding, "padding")
  stride = _as_3_int_tuple(stride, "stride")
  if any(s <= 0 for s in stride):
    raise ValueError(f"stride values must be positive, got {stride}")
  if any(p < 0 for p in padding):
    raise ValueError(f"padding values must be non-negative, got {padding}")
  if phase_batch_size is not None and phase_batch_size <= 0:
    raise ValueError("phase_batch_size must be positive if specified")

  output_shape = tuple(
    (spatial + 2 * pad - kernel) // step + 1
    for spatial, pad, kernel, step in zip((gx, gy, gz), padding, (kx, ky, kz), stride)
  )
  if any(size <= 0 for size in output_shape):
    raise ValueError(f"conv output shape must be positive, got {output_shape}")
  output = torch.zeros(
    (batch, chan_out, *output_shape),
    dtype=input_tensor.dtype,
    device=input_tensor.device,
  )

  phase_records = []
  kernel_sizes = (kx, ky, kz)
  for phase_x in range(stride[0]):
    for phase_y in range(stride[1]):
      for phase_z in range(stride[2]):
        phases = (phase_x, phase_y, phase_z)
        input_phase = input_tensor[
          :,
          :,
          phase_x::stride[0],
          phase_y::stride[1],
          phase_z::stride[2],
        ]
        kernel_indices_by_axis = []
        offset_ranges = []
        for axis, (kernel_size, pad, step, phase) in enumerate(zip(kernel_sizes, padding, stride, phases)):
          kernel_indices = [
            kernel_i
            for kernel_i in range(kernel_size)
            if (kernel_i - pad) % step == phase
          ]
          if not kernel_indices:
            break
          offsets = np.asarray(
            [(kernel_i - pad - phase) // step for kernel_i in kernel_indices],
            dtype=np.int32,
          )
          kernel_indices_by_axis.append(kernel_indices)
          offset_ranges.append((int(offsets.min()), int(offsets.max())))
        else:
          phase_weight = weight[
            :,
            :,
            kernel_indices_by_axis[0],
            :,
            :,
          ][:, :, :, kernel_indices_by_axis[1], :][:, :, :, :, kernel_indices_by_axis[2]]
          # The strided correlation phase is y[n] = sum_a k[a] x[n + a].
          # Reverse offsets to express it as a linear convolution.
          conv_kernel = torch.flip(phase_weight, dims=(-3, -2, -1))
          crop_start = tuple(offset_range[1] for offset_range in offset_ranges)
          phase_records.append((input_phase, conv_kernel, crop_start))

  if phase_batch_size is None:
    phase_batch_size = max(1, len(phase_records))
  for chunk_start in range(0, len(phase_records), phase_batch_size):
    chunk = phase_records[chunk_start:chunk_start + phase_batch_size]
    full_phases = _fft_conv3d_full_batched_phases(
      [record[0] for record in chunk],
      [record[1] for record in chunk],
    )
    for (_input_phase, _conv_kernel, crop_start), full in zip(chunk, full_phases):
      source_slices = []
      dest_slices = []
      for axis in range(3):
        source_start = max(crop_start[axis], 0)
        source_stop = min(crop_start[axis] + output_shape[axis], full.shape[2 + axis])
        if source_start >= source_stop:
          break
        dest_start = source_start - crop_start[axis]
        dest_stop = dest_start + (source_stop - source_start)
        source_slices.append(slice(source_start, source_stop))
        dest_slices.append(slice(dest_start, dest_stop))
      else:
        output[
          :,
          :,
          dest_slices[0],
          dest_slices[1],
          dest_slices[2],
        ] += full[
          :,
          :,
          source_slices[0],
          source_slices[1],
          source_slices[2],
        ]

  return output


def conv_transpose3d_strided_fft_polyphase(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    *,
    padding: int | tuple[int, int, int],
    stride: int | tuple[int, int, int],
    output_padding: int | tuple[int, int, int] = 0,
    phase_batch_size: int | None = None,
) -> torch.Tensor:
  """Compute ``F.conv_transpose3d`` for the densfn2 strided-conv geometry."""
  if input_tensor.ndim != 5:
    raise ValueError(f"input_tensor must have shape (batch, chan, x, y, z), got {input_tensor.shape}")
  if weight.ndim != 5:
    raise ValueError(f"weight must have shape (chan_in, chan_out, kx, ky, kz), got {weight.shape}")
  batch, chan_in, gx, gy, gz = input_tensor.shape
  must_be[chan_in], chan_out, kx, ky, kz = weight.shape
  padding = _as_3_int_tuple(padding, "padding")
  stride = _as_3_int_tuple(stride, "stride")
  output_padding = _as_3_int_tuple(output_padding, "output_padding")
  if any(s <= 0 for s in stride):
    raise ValueError(f"stride values must be positive, got {stride}")
  if any(p < 0 for p in padding):
    raise ValueError(f"padding values must be non-negative, got {padding}")
  if any(op < 0 or op >= s for op, s in zip(output_padding, stride)):
    raise ValueError(f"output_padding must satisfy 0 <= output_padding < stride, got {output_padding}")
  if phase_batch_size is not None and phase_batch_size <= 0:
    raise ValueError("phase_batch_size must be positive if specified")

  output_shape = tuple(
    (spatial - 1) * step - 2 * pad + kernel + op
    for spatial, step, pad, kernel, op in zip((gx, gy, gz), stride, padding, (kx, ky, kz), output_padding)
  )
  if any(size <= 0 for size in output_shape):
    raise ValueError(f"conv_transpose output shape must be positive, got {output_shape}")
  output = torch.zeros(
    (batch, chan_out, *output_shape),
    dtype=input_tensor.dtype,
    device=input_tensor.device,
  )

  phase_records = []
  kernel_sizes = (kx, ky, kz)
  for phase_x in range(stride[0]):
    for phase_y in range(stride[1]):
      for phase_z in range(stride[2]):
        phases = (phase_x, phase_y, phase_z)
        kernel_indices_by_axis = []
        output_offsets = []
        for kernel_size, pad, step, phase in zip(kernel_sizes, padding, stride, phases):
          kernel_indices = [
            kernel_i
            for kernel_i in range(kernel_size)
            if (kernel_i - pad) % step == phase
          ]
          if not kernel_indices:
            break
          offsets = np.asarray(
            [(kernel_i - pad - phase) // step for kernel_i in kernel_indices],
            dtype=np.int32,
          )
          kernel_indices_by_axis.append(kernel_indices)
          output_offsets.append((int(offsets.min()), int(offsets.max())))
        else:
          phase_weight = weight[
            :,
            :,
            kernel_indices_by_axis[0],
            :,
            :,
          ][:, :, :, kernel_indices_by_axis[1], :][:, :, :, :, kernel_indices_by_axis[2]]
          # Convert transposed-correlation phase accumulation into linear conv.
          conv_kernel = phase_weight.permute(1, 0, 2, 3, 4)
          output_phase_shape = tuple(
            (output_shape[axis] + stride[axis] - 1 - phases[axis]) // stride[axis]
            for axis in range(3)
          )
          crop_start = tuple(-offset_range[0] for offset_range in output_offsets)
          phase_records.append((conv_kernel, crop_start, phases, output_phase_shape))

  if phase_batch_size is None:
    phase_batch_size = max(1, len(phase_records))
  for chunk_start in range(0, len(phase_records), phase_batch_size):
    chunk = phase_records[chunk_start:chunk_start + phase_batch_size]
    full_phases = _fft_conv3d_full_batched_phases(
      [input_tensor for _record in chunk],
      [record[0] for record in chunk],
    )
    for (_conv_kernel, crop_start, phases, output_phase_shape), full in zip(chunk, full_phases):
      phase_output = torch.zeros(
        (batch, chan_out, *output_phase_shape),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
      )
      source_slices = []
      dest_slices = []
      for axis in range(3):
        source_start = max(crop_start[axis], 0)
        source_stop = min(crop_start[axis] + output_phase_shape[axis], full.shape[2 + axis])
        if source_start >= source_stop:
          break
        dest_start = source_start - crop_start[axis]
        dest_stop = dest_start + (source_stop - source_start)
        source_slices.append(slice(source_start, source_stop))
        dest_slices.append(slice(dest_start, dest_stop))
      else:
        phase_output[
          :,
          :,
          dest_slices[0],
          dest_slices[1],
          dest_slices[2],
        ] = full[
          :,
          :,
          source_slices[0],
          source_slices[1],
          source_slices[2],
        ]
        output[
          :,
          :,
          phases[0]::stride[0],
          phases[1]::stride[1],
          phases[2]::stride[2],
        ] += phase_output

  return output


def densfn_forward_2(
    protein: ProteinWithCodes,
    *,
    grid: Grid | None = None,
    dx: float = 1.0,
    padding: float = 2.0,
    rng: np.random.Generator | None = None,
    stride: int = 4,
    atom_gaussian_radius: float = 1.0,
    atom_gaussian_sigma: float | None = None,
    output_channels: int = 32,
    seed: int = 0,
    random_conv_radius: float = 4.0,
    device: str | torch.device | None = None,
    conv_method: Literal["direct", "fft"] = "direct",
    fft_phase_batch_size: int | None = 8,
) -> tuple[Grid, NPArr(np.float32, "Nx", "Ny", "Nz", "output_channels")]:
  """Map a protein to a fine Gaussian density followed by random strided conv.

  The returned grid is the coarse output grid. Internally, atoms are sampled on
  a grid with spacing ``grid.dx / stride`` and then projected back to the coarse
  spacing by a deterministic random convolution with stride ``stride``.
  """
  if stride <= 0:
    raise ValueError("stride must be positive")
  if output_channels <= 0:
    raise ValueError("output_channels must be positive")
  if grid is None:
    grid = box_with_grid_for_protein(protein, dx=dx, padding=padding, rng=rng)

  fine_grid = Grid(
    dx=grid.dx / stride,
    N=(np.asarray(grid.N, dtype=np.int32) * stride).astype(np.int32),
    transform=grid.transform,
  )
  fine_field = density_field(fine_grid, len(DENSFN_FORWARD_2_CHANNELS))
  density_positions, density_vectors = _densfn_forward_1_samples(protein)
  if density_positions:
    density_vectors = np.asarray(density_vectors, dtype=np.float32)
    density_vectors = density_vectors[:, DENSFN_FORWARD_2_CHANNELS_IN_FORWARD_1]
    atom_sample_mask = np.any(density_vectors != 0.0, axis=1)
    add_gaussian_atom_densities(
      fine_field,
      fine_grid,
      np.asarray(density_positions, dtype=np.float32)[atom_sample_mask],
      density_vectors[atom_sample_mask],
      radius=atom_gaussian_radius,
      sigma=atom_gaussian_sigma,
    )

  kernel = random_radial_conv_kernel_3d(
    len(DENSFN_FORWARD_2_CHANNELS),
    output_channels,
    dx=fine_grid.dx,
    radius=random_conv_radius,
    seed=seed,
  )
  if device is None:
    conv_device = torch.device("cpu")
  else:
    conv_device = torch.device(device)
  fine_tensor = torch.from_numpy(
    np.moveaxis(fine_field, -1, 0)[None, :, :, :, :]
  ).to(conv_device)
  kernel_tensor = torch.from_numpy(kernel).to(conv_device)
  radius_grid = kernel.shape[-1] // 2
  with torch.no_grad():
    if conv_method == "direct":
      output = F.conv3d(
        fine_tensor,
        kernel_tensor,
        padding=radius_grid,
        stride=stride,
      )
    elif conv_method == "fft":
      output = conv3d_strided_fft_polyphase(
        fine_tensor,
        kernel_tensor,
        padding=radius_grid,
        stride=stride,
        phase_batch_size=fft_phase_batch_size,
      )
    else:
      raise ValueError(f"unknown conv_method {conv_method!r}")
  field = np.moveaxis(output.squeeze(0).cpu().numpy(), 0, -1)
  must_be[(*grid.shape, output_channels)] = field.shape
  return grid, np.ascontiguousarray(field.astype(np.float32))


def _fine_grid_for_stride(grid: Grid, stride: int) -> Grid:
  if stride <= 0:
    raise ValueError("stride must be positive")
  return Grid(
    dx=grid.dx / stride,
    N=(np.asarray(grid.N, dtype=np.int32) * stride).astype(np.int32),
    transform=grid.transform,
  )


def _gaussian_matched_filter_3d(
    score_tensor: torch.Tensor,
    fine_grid: Grid,
    *,
    radius: float,
    sigma: float | None = None,
) -> torch.Tensor:
  if radius <= 0.0:
    raise ValueError("radius must be positive")
  if sigma is None:
    sigma = 0.5 * radius
  if sigma <= 0.0:
    raise ValueError("sigma must be positive")

  _batch, chan, _gx, _gy, _gz = score_tensor.shape
  radius_grid = int(np.ceil(radius / fine_grid.dx))
  offsets = (
    torch.arange(
      -radius_grid,
      radius_grid + 1,
      dtype=score_tensor.dtype,
      device=score_tensor.device,
    ) * fine_grid.dx
  )
  x, y, z = torch.meshgrid(offsets, offsets, offsets, indexing="ij")
  dist2 = x * x + y * y + z * z
  weights = torch.exp(-0.5 * dist2 / (sigma * sigma))
  weights = torch.where(
    dist2 <= radius * radius,
    weights,
    torch.zeros((), dtype=weights.dtype, device=weights.device),
  )
  weights = weights / weights.sum()
  kernel = weights.reshape(1, 1, *weights.shape).expand(chan, 1, -1, -1, -1)
  return F.conv3d(score_tensor, kernel, padding=radius_grid, groups=chan)


def densfn_backward_2_score_fields(
    grid: Grid,
    field: NPArr(np.float32, "Nx", "Ny", "Nz", "output_channels"),
    *,
    stride: int = 4,
    atom_gaussian_radius: float = 1.0,
    atom_gaussian_sigma: float | None = None,
    seed: int = 0,
    random_conv_radius: float = 4.0,
    device: str | torch.device | None = None,
    conv_method: Literal["direct", "fft"] = "direct",
    fft_phase_batch_size: int | None = 8,
) -> tuple[Grid, NPArr(np.float32, "Nx_fine", "Ny_fine", "Nz_fine", "chan")]:
  """Map a random-projected density field back to fine semantic score fields."""
  field = np.asarray(field, dtype=np.float32)
  output_channels = field.shape[-1]
  must_be[(*grid.shape, output_channels)] = field.shape
  fine_grid = _fine_grid_for_stride(grid, stride)

  kernel = random_radial_conv_kernel_3d(
    len(DENSFN_FORWARD_2_CHANNELS),
    output_channels,
    dx=fine_grid.dx,
    radius=random_conv_radius,
    seed=seed,
  )
  if device is None:
    conv_device = torch.device("cpu")
  else:
    conv_device = torch.device(device)

  field_tensor = torch.from_numpy(
    np.moveaxis(field, -1, 0)[None, :, :, :, :]
  ).to(conv_device)
  kernel_tensor = torch.from_numpy(kernel).to(conv_device)
  radius_grid = kernel.shape[-1] // 2
  with torch.no_grad():
    if conv_method == "direct":
      score_tensor = F.conv_transpose3d(
        field_tensor,
        kernel_tensor,
        padding=radius_grid,
        stride=stride,
        output_padding=stride - 1,
      )
    elif conv_method == "fft":
      score_tensor = conv_transpose3d_strided_fft_polyphase(
        field_tensor,
        kernel_tensor,
        padding=radius_grid,
        stride=stride,
        output_padding=stride - 1,
        phase_batch_size=fft_phase_batch_size,
      )
    else:
      raise ValueError(f"unknown conv_method {conv_method!r}")
    must_be[(1, len(DENSFN_FORWARD_2_CHANNELS), *fine_grid.shape)] = score_tensor.shape
    score_tensor = _gaussian_matched_filter_3d(
      score_tensor,
      fine_grid,
      radius=atom_gaussian_radius,
      sigma=atom_gaussian_sigma,
    )

  score_field = np.moveaxis(score_tensor.squeeze(0).cpu().numpy(), 0, -1)
  return fine_grid, np.ascontiguousarray(score_field.astype(np.float32))


def _extract_matched_score_peaks(
    grid: Grid,
    channel_field: NPArr(np.float32, "Nx", "Ny", "Nz"),
    *,
    rel_threshold: float = 0.25,
    abs_threshold: float | None = None,
    min_distance: float = 0.7,
    max_peaks: int = 512,
) -> list[_Peak]:
  if rel_threshold < 0.0:
    raise ValueError("rel_threshold must be non-negative")
  if min_distance < 0.0:
    raise ValueError("min_distance must be non-negative")
  if max_peaks <= 0:
    raise ValueError("max_peaks must be positive")

  channel_field = np.asarray(channel_field, dtype=np.float32)
  channel_max = float(channel_field.max(initial=0.0))
  if abs_threshold is None:
    threshold = rel_threshold * channel_max
  else:
    threshold = abs_threshold
  if threshold <= 0.0:
    threshold = np.nextafter(np.float32(0.0), np.float32(1.0)).item()

  candidate_flat = np.flatnonzero(channel_field >= threshold)
  if candidate_flat.size == 0:
    return []
  candidate_scores = channel_field.reshape(-1)[candidate_flat]
  order = np.argsort(candidate_scores)[::-1]

  suppress_radius = int(np.ceil(min_distance / grid.dx))
  suppressed = np.zeros(channel_field.shape, dtype=bool)
  peaks = []
  for flat_i in candidate_flat[order]:
    index = np.asarray(np.unravel_index(int(flat_i), channel_field.shape), dtype=np.int32)
    if suppressed[tuple(index)]:
      continue
    value = float(channel_field[tuple(index)])
    index_coord = index.astype(np.float32)
    peaks.append(_Peak(index_coord, grid_index_to_coord(grid, index_coord), value))

    lo = np.maximum(index - suppress_radius, 0)
    hi = np.minimum(index + suppress_radius + 1, np.asarray(channel_field.shape, dtype=np.int32))
    suppressed[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    if len(peaks) >= max_peaks:
      break
  return peaks


def _extract_densfn_forward_2_peaks(
    grid: Grid,
    score_field: NPArr(np.float32, "Nx", "Ny", "Nz", "chan"),
    *,
    rel_threshold: float = 0.25,
    abs_threshold: float | None = None,
    min_distance: float = 0.7,
    max_peaks_per_channel: int = 512,
    channels: tuple[str, ...] | None = None,
) -> dict[str, list[_Peak]]:
  peaks = {}
  peak_channels = channels or (
    "backbone_ca",
    "backbone_c",
    "backbone_n",
    "backbone_o",
    "sidechain_n",
    "sidechain_o",
    "sidechain_s",
    "sidechain_c_grey",
    "sidechain_c_blue",
  )
  for channel in peak_channels:
    channel_i = DENSFN_FORWARD_2_CHANNEL_INDEX[channel]
    peaks[channel] = _extract_matched_score_peaks(
      grid,
      score_field[..., channel_i],
      rel_threshold=rel_threshold,
      abs_threshold=abs_threshold,
      min_distance=min_distance,
      max_peaks=max_peaks_per_channel,
    )
  return peaks


def densfn_backward_2(
    grid: Grid,
    field: NPArr(np.float32, "Nx", "Ny", "Nz", "output_channels"),
    *,
    stride: int = 4,
    atom_gaussian_radius: float = 1.0,
    atom_gaussian_sigma: float | None = None,
    seed: int = 0,
    random_conv_radius: float = 4.0,
    device: str | torch.device | None = None,
    conv_method: Literal["direct", "fft"] = "direct",
    fft_phase_batch_size: int | None = 8,
    peak_rel_threshold: float = 0.25,
    peak_abs_threshold: float | None = None,
    peak_min_distance: float = 0.7,
    max_peaks_per_channel: int = 512,
) -> ProteinWithCodes:
  """Initial decoder for densfn_forward_2 fields using matched-filter peaks."""
  fine_grid, score_field = densfn_backward_2_score_fields(
    grid,
    field,
    stride=stride,
    atom_gaussian_radius=atom_gaussian_radius,
    atom_gaussian_sigma=atom_gaussian_sigma,
    seed=seed,
    random_conv_radius=random_conv_radius,
    device=device,
    conv_method=conv_method,
    fft_phase_batch_size=fft_phase_batch_size,
  )
  peaks = _extract_densfn_forward_2_peaks(
    fine_grid,
    score_field,
    rel_threshold=peak_rel_threshold,
    abs_threshold=peak_abs_threshold,
    min_distance=peak_min_distance,
    max_peaks_per_channel=max_peaks_per_channel,
  )
  return densfn_peaks_to_protein(
    fine_grid,
    peaks,
  )


def _tent_weights(index_coord: NPArr(np.float32, 3), grid_shape) -> tuple[list[tuple[int, int, int]], np.ndarray]:
  index_floor = np.floor(index_coord).astype(np.int32)
  index_frac = index_coord - index_floor
  indices = []
  weights = []
  for dx_i, wx in ((0, 1.0 - index_frac[0]), (1, index_frac[0])):
    ix = index_floor[0] + dx_i
    if ix < 0 or ix >= grid_shape[0]:
      continue
    for dy_i, wy in ((0, 1.0 - index_frac[1]), (1, index_frac[1])):
      iy = index_floor[1] + dy_i
      if iy < 0 or iy >= grid_shape[1]:
        continue
      for dz_i, wz in ((0, 1.0 - index_frac[2]), (1, index_frac[2])):
        iz = index_floor[2] + dz_i
        if iz < 0 or iz >= grid_shape[2]:
          continue
        indices.append((int(ix), int(iy), int(iz)))
        weights.append(float(wx * wy * wz))
  return indices, np.asarray(weights, dtype=np.float32)


def _subtract_tent(residual, index_coord, amplitude=1.0):
  indices, weights = _tent_weights(index_coord, residual.shape)
  for (ix, iy, iz), weight in zip(indices, weights):
    residual[ix, iy, iz] -= amplitude * weight


def _tent_model_window(index_coord, lo, shape):
  coords = np.indices(shape, dtype=np.float32)
  model = np.ones(shape, dtype=np.float32)
  for axis in range(3):
    global_axis_coords = coords[axis] + lo[axis]
    model *= np.maximum(0.0, 1.0 - np.abs(global_axis_coords - index_coord[axis]))
  return model


def _refine_peak_index_coord(residual, initial_index_coord, offset):
  lo = np.maximum(np.floor(initial_index_coord).astype(np.int32) - 1, 0)
  hi = np.minimum(np.ceil(initial_index_coord).astype(np.int32) + 2, residual.shape)
  window = np.maximum(residual[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], 0.0)
  if float(window.sum()) <= 0.0:
    return initial_index_coord.astype(np.float32)
  coords = np.indices(window.shape, dtype=np.float32)
  global_coords = [coords[axis] + lo[axis] for axis in range(3)]

  lower = np.zeros(3, dtype=np.float32)
  upper = np.asarray(residual.shape, dtype=np.float32) - 1.0
  for axis in range(3):
    if offset[axis] == 0.5:
      lower[axis] = initial_index_coord[axis] - 0.5
      upper[axis] = initial_index_coord[axis] + 0.5
    else:
      lower[axis] = initial_index_coord[axis] - 0.75
      upper[axis] = initial_index_coord[axis] + 0.75
  lower = np.maximum(lower, 0.0)
  upper = np.minimum(upper, np.asarray(residual.shape, dtype=np.float32) - 1.0)

  def score(index_coord):
    model = np.ones(window.shape, dtype=np.float32)
    for axis in range(3):
      model *= np.maximum(0.0, 1.0 - np.abs(global_coords[axis] - index_coord[axis]))
    model_norm = float(np.sqrt(np.sum(model * model)))
    if model_norm == 0.0:
      return -np.inf
    return float(np.sum(window * model) / model_norm)

  best = np.clip(initial_index_coord.astype(np.float32), lower, upper)
  best_score = score(best)
  for step in (0.25, 0.1, 0.04, 0.015, 0.006):
    improved = True
    while improved:
      improved = False
      for axis in range(3):
        for direction in (-1.0, 1.0):
          candidate = best.copy()
          candidate[axis] = np.clip(candidate[axis] + direction * step, lower[axis], upper[axis])
          candidate_score = score(candidate)
          if candidate_score > best_score:
            best = candidate
            best_score = candidate_score
            improved = True
  return best.astype(np.float32)


def _extract_channel_peaks(
    grid: Grid,
    channel_field: NPArr(np.float32, "Nx", "Ny", "Nz"),
    *,
    threshold: float = 0.55,
    max_peaks: int | None = None,
) -> list[_Peak]:
  residual = np.asarray(channel_field, dtype=np.float32).copy()
  peaks: list[_Peak] = []

  sqrt_residual = np.sqrt(np.maximum(residual, 0.0))
  inv_sqrt2 = 1.0 / np.sqrt(2.0)
  candidates = []
  for offset_bits in range(8):
    kernels = []
    offsets = []
    for axis in range(3):
      if offset_bits & (1 << axis):
        kernels.append((inv_sqrt2, inv_sqrt2))
        offsets.append(0.5)
      else:
        kernels.append((1.0, 0.0))
        offsets.append(0.0)

    score = np.zeros(tuple(np.asarray(residual.shape) - 1), dtype=np.float32)
    for dx_i in (0, 1):
      wx = kernels[0][dx_i]
      if wx == 0.0:
        continue
      for dy_i in (0, 1):
        wy = kernels[1][dy_i]
        if wy == 0.0:
          continue
        for dz_i in (0, 1):
          wz = kernels[2][dz_i]
          if wz == 0.0:
            continue
          score += (
            wx * wy * wz
          ) * sqrt_residual[
            dx_i:dx_i + score.shape[0],
            dy_i:dy_i + score.shape[1],
            dz_i:dz_i + score.shape[2],
          ]

    flat_score = score.reshape(-1)
    candidate_limit = flat_score.size
    if max_peaks is not None:
      candidate_limit = min(candidate_limit, max(256, max_peaks * 4))
    if candidate_limit < flat_score.size:
      candidate_flat_indices = np.argpartition(flat_score, -candidate_limit)[-candidate_limit:]
      candidate_flat_indices = candidate_flat_indices[flat_score[candidate_flat_indices] >= threshold]
    else:
      candidate_flat_indices = np.flatnonzero(flat_score >= threshold)
    for flat_candidate_i in candidate_flat_indices:
      lo = np.asarray(np.unravel_index(int(flat_candidate_i), score.shape), dtype=np.int32)
      offset = np.asarray(offsets, dtype=np.float32)
      candidates.append((
        float(score[tuple(lo)]),
        lo,
        offset,
        tuple(kernels),
        lo.astype(np.float32) + offset,
      ))

  candidates.sort(key=lambda candidate: candidate[0], reverse=True)
  filtered_candidates = []
  suppressed = np.zeros(tuple(np.asarray(residual.shape) - 1), dtype=bool)
  for candidate in candidates:
    candidate_lo = candidate[1]
    if suppressed[tuple(candidate_lo)]:
      continue
    filtered_candidates.append(candidate)
    lo = np.maximum(candidate_lo - 1, 0)
    hi = np.minimum(candidate_lo + 2, suppressed.shape)
    suppressed[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    if max_peaks is not None and len(filtered_candidates) >= max_peaks * 2:
      break

  for _initial_score, best_lo, best_offset, kernels, initial_index_coord in filtered_candidates:
    if max_peaks is not None and len(peaks) >= max_peaks:
      break
    peak_value = 0.0
    for dx_i in (0, 1):
      for dy_i in (0, 1):
        for dz_i in (0, 1):
          weight = kernels[0][dx_i] * kernels[1][dy_i] * kernels[2][dz_i]
          if weight != 0.0:
            peak_value += weight * float(
              np.sqrt(max(0.0, residual[
                best_lo[0] + dx_i,
                best_lo[1] + dy_i,
                best_lo[2] + dz_i,
              ]))
            )
    if peak_value < threshold:
      continue

    index_coord = _refine_peak_index_coord(residual, initial_index_coord, best_offset)
    peaks.append(_Peak(index_coord, grid_index_to_coord(grid, index_coord), peak_value))
    _subtract_tent(residual, index_coord)
    residual[residual < 0.0] = 0.0

  return peaks


def _extract_densfn_forward_1_peaks(
    grid: Grid,
    field: NPArr(np.float32, "Nx", "Ny", "Nz", "chan"),
    *,
    threshold: float = 0.55,
    channels: tuple[str, ...] | None = None,
) -> dict[str, list[_Peak]]:
  peaks = {}
  peak_channels = channels or (
    "backbone_ca",
    "backbone_c",
    "backbone_n",
    "backbone_o",
    "sidechain_n",
    "sidechain_o",
    "sidechain_s",
    "sidechain_c_grey",
    "sidechain_c_blue",
  )
  for channel in peak_channels:
    channel_i = DENSFN_FORWARD_1_CHANNEL_INDEX[channel]
    channel_field = field[..., channel_i]
    expected_count = max(0, int(round(float(channel_field.sum()))))
    max_peaks = expected_count + max(16, expected_count // 4)
    peaks[channel] = _extract_channel_peaks(
      grid,
      channel_field,
      threshold=threshold,
      max_peaks=max_peaks,
    )
  return peaks


def _dist(a, b) -> float:
  return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _assemble_backbone_residues(
    n_peaks: list[_Peak],
    ca_peaks: list[_Peak],
    c_peaks: list[_Peak],
    o_peaks: list[_Peak] | None = None,
    *,
    n_ca_len: float = 1.46,
    ca_c_len: float = 1.53,
    c_o_len: float = 1.23,
    tolerance: float = 0.75,
) -> list[_BackboneResidue]:
  if not n_peaks or not ca_peaks or not c_peaks:
    return []
  n_coords = np.asarray([peak.coord for peak in n_peaks], dtype=np.float32)
  ca_coords = np.asarray([peak.coord for peak in ca_peaks], dtype=np.float32)
  c_coords = np.asarray([peak.coord for peak in c_peaks], dtype=np.float32)
  n_ca_dists = np.linalg.norm(ca_coords[:, None, :] - n_coords[None, :, :], axis=2)
  ca_c_dists = np.linalg.norm(ca_coords[:, None, :] - c_coords[None, :, :], axis=2)
  candidates = []
  for ca_i in range(len(ca_peaks)):
    n_candidates = np.flatnonzero(np.abs(n_ca_dists[ca_i] - n_ca_len) <= tolerance)
    c_candidates = np.flatnonzero(np.abs(ca_c_dists[ca_i] - ca_c_len) <= tolerance)
    for n_i in n_candidates:
      n_ca_dist = float(n_ca_dists[ca_i, n_i])
      for c_i in c_candidates:
        ca_c_dist = float(ca_c_dists[ca_i, c_i])
        score = abs(n_ca_dist - n_ca_len) + abs(ca_c_dist - ca_c_len)
        candidates.append((score, ca_i, int(n_i), int(c_i)))

  candidates.sort()
  used_ca = set()
  used_n = set()
  used_c = set()
  residues = []
  for _score, ca_i, n_i, c_i in candidates:
    if ca_i in used_ca or n_i in used_n or c_i in used_c:
      continue
    used_ca.add(ca_i)
    used_n.add(n_i)
    used_c.add(c_i)
    residues.append(_BackboneResidue(n_peaks[n_i].coord, ca_peaks[ca_i].coord, c_peaks[c_i].coord))

  if not o_peaks or not residues:
    return residues

  c_coords = np.asarray([residue.c for residue in residues], dtype=np.float32)
  o_coords = np.asarray([peak.coord for peak in o_peaks], dtype=np.float32)
  c_o_dists = np.linalg.norm(c_coords[:, None, :] - o_coords[None, :, :], axis=2)
  o_candidates = []
  for residue_i in range(len(residues)):
    for o_i in np.flatnonzero(np.abs(c_o_dists[residue_i] - c_o_len) <= tolerance):
      dist = float(c_o_dists[residue_i, o_i])
      o_candidates.append((abs(dist - c_o_len), residue_i, int(o_i)))
  o_candidates.sort()

  oxygen_by_residue = {}
  used_o = set()
  for _score, residue_i, o_i in o_candidates:
    if residue_i in oxygen_by_residue or o_i in used_o:
      continue
    oxygen_by_residue[residue_i] = o_peaks[o_i].coord
    used_o.add(o_i)

  return [
    _BackboneResidue(residue.n, residue.ca, residue.c, oxygen_by_residue.get(residue_i))
    for residue_i, residue in enumerate(residues)
  ]


def _link_backbone_residues(
    residues: list[_BackboneResidue],
    *,
    peptide_len: float = 1.33,
    ca_ca_len: float = 3.8,
    tolerance: float = 0.9,
    bond_field=None,
    grid: Grid | None = None,
) -> list[list[_BackboneResidue]]:
  if not residues:
    return []
  c_coords = np.asarray([residue.c for residue in residues], dtype=np.float32)
  n_coords = np.asarray([residue.n for residue in residues], dtype=np.float32)
  ca_coords = np.asarray([residue.ca for residue in residues], dtype=np.float32)
  peptide_dists = np.linalg.norm(c_coords[:, None, :] - n_coords[None, :, :], axis=2)
  ca_ca_dists = np.linalg.norm(ca_coords[:, None, :] - ca_coords[None, :, :], axis=2)
  np.fill_diagonal(peptide_dists, np.inf)
  np.fill_diagonal(ca_ca_dists, np.inf)
  candidate_srcs, candidate_dsts = np.nonzero(
    (np.abs(peptide_dists - peptide_len) <= tolerance)
    & (np.abs(ca_ca_dists - ca_ca_len) <= 1.5)
  )
  candidates = []
  for src_i, dst_i in zip(candidate_srcs, candidate_dsts):
    peptide_dist = float(peptide_dists[src_i, dst_i])
    ca_ca_dist = float(ca_ca_dists[src_i, dst_i])
    score = abs(peptide_dist - peptide_len) + 0.25 * abs(ca_ca_dist - ca_ca_len)
    if bond_field is not None and grid is not None:
      score -= 0.2 * _sample_density_at_coord(grid, bond_field, 0.5 * (residues[src_i].c + residues[dst_i].n))
    candidates.append((score, int(src_i), int(dst_i)))

  candidates.sort()
  successor: dict[int, int] = {}
  predecessor: dict[int, int] = {}
  for _score, src_i, dst_i in candidates:
    if src_i in successor or dst_i in predecessor:
      continue
    successor[src_i] = dst_i
    predecessor[dst_i] = src_i

  starts = [i for i in range(len(residues)) if i not in predecessor]
  if not starts and residues:
    raise ValueError("cyclic backbone chain detected: no chain start residue")

  chains = []
  visited = set()
  for start in starts:
    chain = []
    current = start
    chain_seen = set()
    while current is not None:
      if current in chain_seen:
        raise ValueError("cyclic backbone chain detected")
      if current in visited:
        raise ValueError("backbone residue appears in multiple chains")
      chain_seen.add(current)
      visited.add(current)
      chain.append(residues[current])
      current = successor.get(current)
    chains.append(chain)

  if len(visited) != len(residues):
    raise ValueError("cyclic backbone chain detected in unvisited residues")
  chains.sort(key=lambda chain: (len(chain), chain[0].n[0] if chain else 0.0))
  return chains


def _sample_density_at_coord(grid: Grid, density, coord) -> float:
  index_coord = coord_to_grid_index(grid, coord)
  indices, weights = _tent_weights(index_coord, density.shape)
  value = 0.0
  for (ix, iy, iz), weight in zip(indices, weights):
    value += weight * float(density[ix, iy, iz])
  return value


def _flatten_backbone_chains(chains):
  return [
    (chain_i, residue_i, residue)
    for chain_i, chain in enumerate(chains)
    for residue_i, residue in enumerate(chain)
  ]


def _assign_sidechain_peaks_to_residues(chains, peaks, max_ca_distance=8.0):
  sidechain_channels = tuple(SIDECHAIN_TEMPLATE_COUNTS["G"])
  flat_residues = _flatten_backbone_chains(chains)
  assignments = {
    (chain_i, residue_i): {channel: [] for channel in sidechain_channels}
    for chain_i, residue_i, _residue in flat_residues
  }
  if not flat_residues:
    return assignments

  ca_coords = np.asarray([residue.ca for _chain_i, _residue_i, residue in flat_residues], dtype=np.float32)
  peak_records = []
  for channel in sidechain_channels:
    for peak in peaks.get(channel, []):
      peak_records.append((channel, peak))
  if not peak_records:
    return assignments

  peak_coords = np.asarray([peak.coord for _channel, peak in peak_records], dtype=np.float32)
  grey_channel = "sidechain_c_grey"
  cb_candidates = [
    peak_i for peak_i, (channel, _peak) in enumerate(peak_records)
    if channel == grey_channel
  ]
  cb_edges = []
  for peak_i in cb_candidates:
    dists = np.linalg.norm(ca_coords - peak_coords[peak_i], axis=1)
    for residue_flat_i, dist in enumerate(dists):
      if dist <= 2.4:
        cb_edges.append((float(dist), residue_flat_i, peak_i))
  cb_edges.sort()
  residue_for_cb_peak = {}
  used_residues = set()
  used_cb_peaks = set()
  for _dist_value, residue_flat_i, peak_i in cb_edges:
    if residue_flat_i in used_residues or peak_i in used_cb_peaks:
      continue
    used_residues.add(residue_flat_i)
    used_cb_peaks.add(peak_i)
    residue_for_cb_peak[peak_i] = residue_flat_i

  n_peaks = len(peak_records)
  parent = list(range(n_peaks))

  def find(i):
    while parent[i] != i:
      parent[i] = parent[parent[i]]
      i = parent[i]
    return i

  def union(i, j):
    root_i = find(i)
    root_j = find(j)
    if root_i != root_j:
      parent[root_j] = root_i

  dists = np.linalg.norm(peak_coords[:, None, :] - peak_coords[None, :, :], axis=2)
  for i in range(n_peaks):
    for j in range(i + 1, n_peaks):
      if dists[i, j] <= 2.05:
        union(i, j)

  components = {}
  for peak_i in range(n_peaks):
    components.setdefault(find(peak_i), []).append(peak_i)

  for component_peak_indices in components.values():
    component_cb_residues = [
      residue_for_cb_peak[peak_i]
      for peak_i in component_peak_indices
      if peak_i in residue_for_cb_peak
    ]
    if len(component_cb_residues) == 1:
      residue_flat_i = component_cb_residues[0]
      chain_i, residue_i, _residue = flat_residues[residue_flat_i]
      key = (chain_i, residue_i)
      for peak_i in component_peak_indices:
        channel, peak = peak_records[peak_i]
        if _dist(peak.coord, ca_coords[residue_flat_i]) <= max_ca_distance:
          assignments[key][channel].append(peak)
    elif len(component_cb_residues) > 1:
      cb_coords = np.asarray([ca_coords[residue_i] for residue_i in component_cb_residues], dtype=np.float32)
      for peak_i in component_peak_indices:
        channel, peak = peak_records[peak_i]
        nearest_i = int(np.argmin(np.linalg.norm(cb_coords - peak.coord, axis=1)))
        residue_flat_i = component_cb_residues[nearest_i]
        if _dist(peak.coord, ca_coords[residue_flat_i]) <= max_ca_distance:
          chain_i, residue_i, _residue = flat_residues[residue_flat_i]
          assignments[(chain_i, residue_i)][channel].append(peak)
  return assignments


def _infer_residue_code(channel_assignments):
  observed_counts = {
    channel: len(peaks)
    for channel, peaks in channel_assignments.items()
  }
  best = None
  for aa, expected_counts in SIDECHAIN_TEMPLATE_COUNTS.items():
    count_score = sum(
      abs(observed_counts[channel] - expected_counts[channel])
      for channel in observed_counts
    )
    if best is None or count_score < best[0]:
      best = (count_score, aa)
  return best[1]


def _signed_pair_volume(origin, ref_a, ref_b, coord_a, coord_b):
  return float(np.linalg.det(np.stack([ref_a - origin, ref_b - origin, coord_a - coord_b], axis=0)))


def _orient_named_pair(coords_by_atom, atom_a, atom_b, origin_atom, ref_a_atom, ref_b_atom, target_sign):
  if not all(atom in coords_by_atom for atom in (atom_a, atom_b, origin_atom, ref_a_atom, ref_b_atom)):
    return
  signed_volume = _signed_pair_volume(
    coords_by_atom[origin_atom],
    coords_by_atom[ref_a_atom],
    coords_by_atom[ref_b_atom],
    coords_by_atom[atom_a],
    coords_by_atom[atom_b],
  )
  if signed_volume * target_sign < 0.0:
    coords_by_atom[atom_a], coords_by_atom[atom_b] = coords_by_atom[atom_b], coords_by_atom[atom_a]


def _apply_sidechain_naming_tiebreakers(aa, coords_by_atom):
  # The graph score cannot distinguish identical terminal methyls. Use a local
  # handedness convention so these names are at least deterministic from geometry.
  if aa == "V":
    _orient_named_pair(coords_by_atom, "CG1", "CG2", "CB", "CA", "C", -1.0)
  elif aa == "L":
    _orient_named_pair(coords_by_atom, "CD1", "CD2", "CG", "CB", "CA", -1.0)


def _assigned_sidechain_atom_coords_by_graph(aa, channel_assignments, residue):
  if any(atom_name in SIDECHAIN_Z2_ATOMS.get(aa, set()) for atom_name in SIDECHAIN_ATOM_TEMPLATES[aa]):
    return None
  if any(
      len(channel_assignments[channel]) != SIDECHAIN_TEMPLATE_COUNTS[aa][channel]
      for channel in channel_assignments
  ):
    return None

  atoms_by_channel = {}
  perms_by_channel = []
  for channel in channel_assignments:
    atom_names = [
      atom_name
      for atom_name, atom_channel in SIDECHAIN_ATOM_TEMPLATES[aa].items()
      if atom_channel == channel
    ]
    if not atom_names:
      continue
    atoms_by_channel[channel] = atom_names
    peak_coords = [peak.coord for peak in channel_assignments[channel]]
    perms_by_channel.append((channel, list(itertools.permutations(peak_coords))))

  best = None
  for perm_tuple in itertools.product(*(perms for _channel, perms in perms_by_channel)):
    coords_by_atom = {"N": residue.n, "CA": residue.ca, "C": residue.c}
    for (channel, _perms), perm in zip(perms_by_channel, perm_tuple):
      for atom_name, coord in zip(atoms_by_channel[channel], perm):
        coords_by_atom[atom_name] = coord

    score = 0.0
    for atom_a, atom_b in SIDECHAIN_BONDS[aa]:
      if atom_a not in coords_by_atom or atom_b not in coords_by_atom:
        score += 100.0
        continue
      bond_len = _dist(coords_by_atom[atom_a], coords_by_atom[atom_b])
      score += (bond_len - 1.5) ** 2
    if best is None or score < best[0]:
      best = (score, coords_by_atom)

  if best is None:
    return None
  _apply_sidechain_naming_tiebreakers(aa, best[1])
  return {
    atom_name: coord
    for atom_name, coord in best[1].items()
    if atom_name not in ("N", "CA", "C")
  }


def _sidechain_atoms_for_assignment(aa, channel_assignments, residue):
  graph_atoms = _assigned_sidechain_atom_coords_by_graph(aa, channel_assignments, residue)
  if graph_atoms is not None:
    return graph_atoms

  atoms = {}
  used_by_channel = {channel: 0 for channel in channel_assignments}
  sorted_by_channel = {
    channel: sorted(peaks, key=lambda peak: _dist(peak.coord, residue.ca))
    for channel, peaks in channel_assignments.items()
  }

  for atom_name, channel in SIDECHAIN_ATOM_TEMPLATES[aa].items():
    count = _sidechain_atom_count(aa, atom_name)
    start = used_by_channel[channel]
    stop = start + count
    channel_peaks = sorted_by_channel[channel]
    if stop > len(channel_peaks):
      continue
    coords = [peak.coord for peak in channel_peaks[start:stop]]
    used_by_channel[channel] = stop
    if count == 2:
      atoms[atom_name] = Z2CoordPair(coords[0], coords[1])
    else:
      atoms[atom_name] = coords[0]
  return atoms


def densfn_peaks_to_protein(
    grid: Grid,
    peaks: dict[str, list[_Peak]],
    *,
    bond_field: NPArr(np.float32, "Nx", "Ny", "Nz") | None = None,
) -> ProteinWithCodes:
  """Assemble ProteinWithCodes from semantic channel peak candidates."""
  residues = _assemble_backbone_residues(
    peaks.get("backbone_n", []),
    peaks.get("backbone_ca", []),
    peaks.get("backbone_c", []),
    peaks.get("backbone_o", []),
  )
  chains = _link_backbone_residues(
    residues,
    bond_field=bond_field,
    grid=grid if bond_field is not None else None,
  )
  sidechain_assignments = _assign_sidechain_peaks_to_residues(chains, peaks)
  decoded = []
  for chain_i, chain in enumerate(chains):
    decoded_chain = []
    for residue_i, residue in enumerate(chain):
      assignments = sidechain_assignments[(chain_i, residue_i)]
      aa = _infer_residue_code(assignments)
      decoded_chain.append((
        aa,
        {
          "N": residue.n,
          "CA": residue.ca,
          "C": residue.c,
          **({"O": residue.o} if residue.o is not None else {}),
          **_sidechain_atoms_for_assignment(aa, assignments, residue),
        },
      ))
    decoded.append(decoded_chain)
  return decoded


def densfn_candidates_to_peaks(
    grid: Grid,
    candidates: list[DensityCandidate],
) -> dict[str, list[_Peak]]:
  """Convert explicit channel/position candidates into reverser peak lists."""
  peaks: dict[str, list[_Peak]] = {}
  for candidate in candidates:
    if candidate.channel not in DENSFN_FORWARD_1_CHANNEL_INDEX:
      raise ValueError(f"unknown density channel {candidate.channel!r}")
    coord = np.asarray(candidate.coord, dtype=np.float32)
    must_be[3], = coord.shape
    index_coord = coord_to_grid_index(grid, coord)
    peaks.setdefault(candidate.channel, []).append(_Peak(
      index_coord.astype(np.float32),
      coord,
      float(candidate.value),
    ))
  return peaks


def densfn_candidates_to_protein(
    grid: Grid,
    candidates: list[DensityCandidate],
    *,
    bond_field: NPArr(np.float32, "Nx", "Ny", "Nz") | None = None,
) -> ProteinWithCodes:
  """Assemble ProteinWithCodes from explicit semantic atom candidates."""
  return densfn_peaks_to_protein(
    grid,
    densfn_candidates_to_peaks(grid, candidates),
    bond_field=bond_field,
  )


def densfn_backward_1(
    grid: Grid,
    field: NPArr(np.float32, "Nx", "Ny", "Nz", "chan"),
    *,
    peak_threshold: float = 0.55,
) -> ProteinWithCodes:
  """Decode a densfn_forward_1 field into ProteinWithCodes."""
  field = np.asarray(field, dtype=np.float32)
  must_be[(*grid.shape, len(DENSFN_FORWARD_1_CHANNELS))] = field.shape
  peaks = _extract_densfn_forward_1_peaks(
    grid,
    field,
    threshold=peak_threshold,
    channels=(
      "backbone_ca",
      "backbone_c",
      "backbone_n",
      "backbone_o",
      "sidechain_n",
      "sidechain_o",
      "sidechain_s",
      "sidechain_c_grey",
      "sidechain_c_blue",
    ),
  )
  return densfn_peaks_to_protein(
    grid,
    peaks,
    bond_field=field[..., DENSFN_FORWARD_1_CHANNEL_INDEX["backbone_bond"]],
  )
