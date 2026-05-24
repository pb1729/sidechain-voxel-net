#!/usr/bin/env python3
"""Compare a CIF protein to its densfn_forward_1/densfn_backward_1 round trip."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from density_fns import densfn_backward_1, densfn_forward_1
from parse_cif import ProteinWithCodes, Z2CoordPair, read_protein_cif_with_codes


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode/decode a CIF with densfn_forward_1/backward_1 and report RMSD."
    )
    parser.add_argument("cif_path", help="path to the .cif file to compare")
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.0,
        help="standard deviation of Gaussian noise added to the density field; default: 0.0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for grid rotation and optional noise; default: None",
    )
    return parser.parse_args(argv)


def _coord_options(coord_or_pair):
    if isinstance(coord_or_pair, Z2CoordPair):
        return (coord_or_pair.a, coord_or_pair.b)
    return (coord_or_pair,)


def _add_atom_sse(
    ref_coord_or_pair,
    decoded_coord_or_pair,
) -> tuple[float, int]:
    ref_coords = _coord_options(ref_coord_or_pair)
    decoded_coords = _coord_options(decoded_coord_or_pair)

    if len(ref_coords) == 1 and len(decoded_coords) == 1:
        diff = ref_coords[0] - decoded_coords[0]
        return float(diff @ diff), 1

    if len(ref_coords) != 2 or len(decoded_coords) != 2:
        raise ValueError("cannot compare a Z2CoordPair to a single coordinate")

    direct = (
        float(np.sum((ref_coords[0] - decoded_coords[0]) ** 2))
        + float(np.sum((ref_coords[1] - decoded_coords[1]) ** 2))
    )
    swapped = (
        float(np.sum((ref_coords[0] - decoded_coords[1]) ** 2))
        + float(np.sum((ref_coords[1] - decoded_coords[0]) ** 2))
    )
    return min(direct, swapped), 2


SWAPPABLE_RESIDUE_ATOM_PAIRS = {
    # These methyl pairs are technically distinguishable by parity. We treat
    # them as swappable for density round-trip RMSD, but this should be
    # considered when writing decoded structures back to an output format.
    "L": (("CD1", "CD2"),),
    "V": (("CG1", "CG2"),),
}


def _add_coord_pair_sse(ref_a, ref_b, decoded_a, decoded_b) -> tuple[float, int]:
    direct = (
        float(np.sum((ref_a - decoded_a) ** 2))
        + float(np.sum((ref_b - decoded_b) ** 2))
    )
    swapped = (
        float(np.sum((ref_a - decoded_b) ** 2))
        + float(np.sum((ref_b - decoded_a) ** 2))
    )
    return min(direct, swapped), 2


def _residue_atom_names(ref_code, ref_residue, decoded_code, decoded_residue):
    if ref_code == "X" or decoded_code == "X":
        return [atom for atom in ("N", "CA", "C") if atom in ref_residue and atom in decoded_residue]
    return sorted(set(ref_residue) & set(decoded_residue))


def _flatten_protein(protein: ProteinWithCodes):
    return [
        (chain_i, residue_i, code, residue)
        for chain_i, chain in enumerate(protein)
        for residue_i, (code, residue) in enumerate(chain)
    ]


def _match_residues(reference: ProteinWithCodes, decoded: ProteinWithCodes):
    ref_flat = _flatten_protein(reference)
    decoded_flat = _flatten_protein(decoded)
    candidates = []
    for ref_i, (_ref_chain_i, _ref_residue_i, _ref_code, ref_residue) in enumerate(ref_flat):
        if "CA" not in ref_residue:
            continue
        for decoded_i, (_dec_chain_i, _dec_residue_i, _dec_code, decoded_residue) in enumerate(decoded_flat):
            if "CA" not in decoded_residue:
                continue
            candidates.append((_add_atom_sse(ref_residue["CA"], decoded_residue["CA"])[0], ref_i, decoded_i))

    candidates.sort()
    used_ref = set()
    used_decoded = set()
    matches = []
    for _score, ref_i, decoded_i in candidates:
        if ref_i in used_ref or decoded_i in used_decoded:
            continue
        used_ref.add(ref_i)
        used_decoded.add(decoded_i)
        matches.append((ref_flat[ref_i], decoded_flat[decoded_i]))
    matches.sort(key=lambda pair: (pair[0][0], pair[0][1]))
    return matches, len(ref_flat), len(decoded_flat)


def protein_rmsd(
    reference: ProteinWithCodes,
    decoded: ProteinWithCodes,
) -> tuple[float, int, list[str]]:
    total_sse = 0.0
    total_atoms = 0
    warnings = []

    if len(reference) != len(decoded):
        warnings.append(f"chain count differs: reference={len(reference)} decoded={len(decoded)}")

    matches, reference_residue_count, decoded_residue_count = _match_residues(reference, decoded)
    if reference_residue_count != decoded_residue_count:
        warnings.append(
            f"residue count differs: reference={reference_residue_count} decoded={decoded_residue_count}"
        )
    if len(matches) != min(reference_residue_count, decoded_residue_count):
        warnings.append(
            f"matched {len(matches)} residue pairs out of "
            f"{reference_residue_count} reference and {decoded_residue_count} decoded residues"
        )

    for (ref_chain_i, ref_residue_i, ref_code, ref_residue), (
        decoded_chain_i,
        decoded_residue_i,
        decoded_code,
        decoded_residue,
    ) in matches:
        atom_names = _residue_atom_names(ref_code, ref_residue, decoded_code, decoded_residue)
        required_atoms = ("N", "CA", "C") if "X" in (ref_code, decoded_code) else tuple(ref_residue)
        missing_ref = sorted(set(required_atoms) - set(ref_residue))
        missing_decoded = sorted(set(required_atoms) - set(decoded_residue))
        if missing_ref:
            warnings.append(f"reference chain {ref_chain_i} residue {ref_residue_i} missing {missing_ref}")
        if missing_decoded:
            warnings.append(
                f"decoded chain {decoded_chain_i} residue {decoded_residue_i} missing {missing_decoded}"
            )

        swappable_atoms = set()
        if ref_code == decoded_code:
            for atom_a, atom_b in SWAPPABLE_RESIDUE_ATOM_PAIRS.get(ref_code, ()):
                if all(atom in ref_residue and atom in decoded_residue for atom in (atom_a, atom_b)):
                    sse, count = _add_coord_pair_sse(
                        ref_residue[atom_a],
                        ref_residue[atom_b],
                        decoded_residue[atom_a],
                        decoded_residue[atom_b],
                    )
                    total_sse += sse
                    total_atoms += count
                    swappable_atoms.update((atom_a, atom_b))

        for atom_name in atom_names:
            if atom_name in swappable_atoms:
                continue
            sse, count = _add_atom_sse(ref_residue[atom_name], decoded_residue[atom_name])
            total_sse += sse
            total_atoms += count

    if total_atoms == 0:
        raise ValueError("no comparable atoms found")
    return float(np.sqrt(total_sse / total_atoms)), total_atoms, warnings


def residue_code_mismatches(reference: ProteinWithCodes, decoded: ProteinWithCodes) -> list[str]:
    matches, _reference_residue_count, _decoded_residue_count = _match_residues(reference, decoded)
    mismatches = []
    for (ref_chain_i, ref_residue_i, ref_code, _ref_residue), (
        decoded_chain_i,
        decoded_residue_i,
        decoded_code,
        _decoded_residue,
    ) in matches:
        if ref_code != decoded_code:
            mismatches.append(
                f"reference chain {ref_chain_i} residue {ref_residue_i}: "
                f"{ref_code} -> {decoded_code} "
                f"(decoded chain {decoded_chain_i} residue {decoded_residue_i})"
            )
    return mismatches


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.noise_level < 0.0:
        raise ValueError("--noise-level must be non-negative")

    seed = args.seed if args.seed is not None else np.random.randint(2**63)
    rng = np.random.default_rng(args.seed)
    protein = read_protein_cif_with_codes(args.cif_path)
    grid, field = densfn_forward_1(protein, rng=rng)
    if args.noise_level > 0.0:
        field = field + rng.normal(0.0, args.noise_level, field.shape).astype(np.float32)

    decoded = densfn_backward_1(grid, field)
    rmsd, atom_count, warnings = protein_rmsd(protein, decoded)
    code_mismatches = residue_code_mismatches(protein, decoded)

    print(f"RMSD: {rmsd:.6f} A")
    print(f"Compared atoms: {atom_count}")
    print(f"Reference chains/residues: {len(protein)}/{sum(len(chain) for chain in protein)}")
    print(f"Decoded chains/residues: {len(decoded)}/{sum(len(chain) for chain in decoded)}")
    if code_mismatches:
        print("Residue code mismatches:")
        for mismatch in code_mismatches:
            print(f"- {mismatch}")
    if warnings:
        print("Warnings:")
        for warning in warnings[:20]:
            print(f"- {warning}")
        if len(warnings) > 20:
            print(f"- ... {len(warnings) - 20} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
