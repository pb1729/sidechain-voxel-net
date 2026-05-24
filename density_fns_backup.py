from dataclasses import dataclass
from typing import Literal

import numpy as np

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
class _BackboneResidue:
  n: NPArr(np.float32, 3)
  ca: NPArr(np.float32, 3)
  c: NPArr(np.float32, 3)


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

  density_positions = []
  density_vectors = []

  backbone_channels = {
    "CA": DENSFN_FORWARD_1_CHANNEL_INDEX["backbone_ca"],
    "C": DENSFN_FORWARD_1_CHANNEL_INDEX["backbone_c"],
    "N": DENSFN_FORWARD_1_CHANNEL_INDEX["backbone_n"],
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
        elif atom_identity in {"O", "OXT"}:
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

  if density_positions:
    add_atom_densities(
      field,
      grid,
      np.asarray(density_positions, dtype=np.float32),
      np.asarray(density_vectors, dtype=np.float32),
    )

  return grid, field


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
    *,
    n_ca_len: float = 1.46,
    ca_c_len: float = 1.53,
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
  return residues


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


def densfn_backward_1(
    grid: Grid,
    field: NPArr(np.float32, "Nx", "Ny", "Nz", "chan"),
    *,
    peak_threshold: float = 0.55,
) -> ProteinWithCodes:
  """Decode the backbone of a densfn_forward_1 field into ProteinWithCodes.

  This first version only returns unknown-residue ("X") backbone atoms N, CA,
  and C. The peak extraction helper also supports sidechain atom channels, but
  sidechain assignment is intentionally deferred.
  """
  field = np.asarray(field, dtype=np.float32)
  must_be[(*grid.shape, len(DENSFN_FORWARD_1_CHANNELS))] = field.shape
  peaks = _extract_densfn_forward_1_peaks(
    grid,
    field,
    threshold=peak_threshold,
    channels=("backbone_ca", "backbone_c", "backbone_n"),
  )
  residues = _assemble_backbone_residues(
    peaks["backbone_n"],
    peaks["backbone_ca"],
    peaks["backbone_c"],
  )
  chains = _link_backbone_residues(
    residues,
    bond_field=field[..., DENSFN_FORWARD_1_CHANNEL_INDEX["backbone_bond"]],
    grid=grid,
  )
  return [
    [
      ("X", {"N": residue.n, "CA": residue.ca, "C": residue.c})
      for residue in chain
    ]
    for chain in chains
  ]
