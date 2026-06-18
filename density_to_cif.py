#!/usr/bin/env python3
"""Decode a saved density field and write an mmCIF file."""

from __future__ import annotations

import argparse
from pathlib import Path
import string
import sys

import numpy as np

from density_io import load_density_file
from parse_cif import AMINO_ACID_CODES, ATOM_IDENTITY_ELEMENTS, ProteinWithCodes, Z2CoordPair


ONE_TO_THREE = {one: three for three, one in AMINO_ACID_CODES.items()}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode a saved density field using its stored density function."
    )
    parser.add_argument("input_path", help="saved density file")
    parser.add_argument("output_path", help="output .cif file")
    return parser.parse_args(argv)


def _chain_id(chain_i: int) -> str:
    alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return alphabet[chain_i] if chain_i < len(alphabet) else f"X{chain_i}"


def _coords_for_cif(coord_or_pair) -> tuple[np.ndarray, ...]:
    if isinstance(coord_or_pair, Z2CoordPair):
        return coord_or_pair.a, coord_or_pair.b
    return (coord_or_pair,)


def _atom_element(atom_name: str) -> str:
    if atom_name in ATOM_IDENTITY_ELEMENTS:
        return ATOM_IDENTITY_ELEMENTS[atom_name]
    return "".join(char for char in atom_name if char.isalpha())[:1].upper() or "?"


def protein_to_cif_text(protein: ProteinWithCodes, *, data_name: str) -> str:
    lines = [
        f"data_{data_name}", "#", "loop_", "_atom_site.group_PDB", "_atom_site.id",
        "_atom_site.type_symbol", "_atom_site.label_atom_id", "_atom_site.label_alt_id",
        "_atom_site.label_comp_id", "_atom_site.label_asym_id", "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code", "_atom_site.Cartn_x", "_atom_site.Cartn_y",
        "_atom_site.Cartn_z", "_atom_site.occupancy", "_atom_site.B_iso_or_equiv",
        "_atom_site.auth_seq_id", "_atom_site.auth_comp_id", "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id", "_atom_site.pdbx_PDB_model_num",
    ]
    atom_id = 1
    for chain_i, chain in enumerate(protein):
        asym_id = _chain_id(chain_i)
        for residue_i, (aa, residue) in enumerate(chain, start=1):
            comp_id = ONE_TO_THREE.get(aa, "UNK")
            for atom_name, coord_or_pair in residue.items():
                coords = _coords_for_cif(coord_or_pair)
                for alt_i, coord in enumerate(coords):
                    alt_id = "." if len(coords) == 1 else string.ascii_uppercase[alt_i]
                    x, y, z = np.asarray(coord, dtype=np.float32)
                    lines.append(
                        " ".join((
                            "ATOM", str(atom_id), _atom_element(atom_name), atom_name, alt_id,
                            comp_id, asym_id, str(residue_i), "?", f"{x:.3f}", f"{y:.3f}",
                            f"{z:.3f}", "1.00", "0.00", str(residue_i), comp_id, asym_id,
                            atom_name, "1",
                        ))
                    )
                    atom_id += 1
    lines.append("#")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    grid, field, densfn = load_density_file(args.input_path)
    decoded = densfn.backward(grid, field)
    data_name = Path(args.output_path).stem.replace("-", "_") or "decoded_density"
    Path(args.output_path).write_text(
        protein_to_cif_text(decoded, data_name=data_name), encoding="utf-8"
    )
    atom_count = sum(
        len(_coords_for_cif(coord_or_pair))
        for chain in decoded
        for _aa, residue in chain
        for coord_or_pair in residue.values()
    )
    print(
        f"wrote {args.output_path}: chains={len(decoded)} "
        f"residues={sum(len(chain) for chain in decoded)} atoms={atom_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
