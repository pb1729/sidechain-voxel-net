import os
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np


AMINO_ACID_CODES = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

ATOMIC_NUMBERS = {
    "H": 1,
    "HE": 2,
    "LI": 3,
    "BE": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NE": 10,
    "NA": 11,
    "MG": 12,
    "AL": 13,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "AR": 18,
}


ATOM_IDENTITY_ELEMENTS = {
    "N": "N",
    "CA": "C",
    "C": "C",
    "O": "O",
    "OXT": "O",
    "CB": "C",
    "CG": "C",
    "CG1": "C",
    "CG2": "C",
    "CD": "C",
    "CD1": "C",
    "CD2": "C",
    "CE": "C",
    "CE1": "C",
    "CE2": "C",
    "CE3": "C",
    "CZ": "C",
    "CZ2": "C",
    "CZ3": "C",
    "CH2": "C",
    "ND1": "N",
    "ND2": "N",
    "NE": "N",
    "NE1": "N",
    "NE2": "N",
    "NZ": "N",
    "NH": "N",
    "NH1": "N",
    "NH2": "N",
    "OD": "O",
    "OD1": "O",
    "OD2": "O",
    "OE": "O",
    "OE1": "O",
    "OE2": "O",
    "OG": "O",
    "OG1": "O",
    "OH": "O",
    "SD": "S",
    "SG": "S",
}


SYMMETRIC_ATOM_PAIRS = {
    "ARG": (("NH", "NH1", "NH2"),),
    "ASP": (("OD", "OD1", "OD2"),),
    "GLU": (("OE", "OE1", "OE2"),),
    "PHE": (("CD", "CD1", "CD2"), ("CE", "CE1", "CE2")),
    "TYR": (("CD", "CD1", "CD2"), ("CE", "CE1", "CE2")),
}


@dataclass(frozen=True)
class Z2CoordPair:
    """ Pair of coordinates of indistinguishable atoms.
        NOTE: Sometimes a single symmetric flip will swap multiple atoms at the same time.
        This class does not account for this, and will treat them as separately flippable. """
    a: np.ndarray
    b: np.ndarray


CoordValue = np.ndarray | Z2CoordPair
Protein = list[list[dict[str, CoordValue]]]
ProteinWithCodes = list[list[tuple[str, dict[str, CoordValue]]]]


def _tokenize_cif(text: str) -> list[str]:
    tokens: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(";"):
            block = [line[1:]]
            i += 1
            while i < len(lines) and not lines[i].startswith(";"):
                block.append(lines[i])
                i += 1
            tokens.append("\n".join(block))
            i += 1
            continue

        pos = 0
        while pos < len(line):
            char = line[pos]
            if char == "#":
                break
            if char.isspace():
                pos += 1
                continue
            if char in ("'", '"'):
                quote = char
                pos += 1
                start = pos
                while pos < len(line) and line[pos] != quote:
                    pos += 1
                tokens.append(line[start:pos])
                pos += 1
                continue
            start = pos
            while pos < len(line) and not line[pos].isspace() and line[pos] != "#":
                pos += 1
            tokens.append(line[start:pos])
        i += 1
    return tokens


def _normalize_element(symbol: str) -> str:
    symbol = symbol.strip()
    if not symbol or symbol in {".", "?"}:
        raise ValueError("missing atom element symbol")
    return symbol.upper()



def _parse_atom_site_loop(tokens: list[str]) -> tuple[list[str], list[list[str]]]:
    i = 0
    while i < len(tokens):
        if tokens[i].lower() != "loop_":
            i += 1
            continue

        i += 1
        columns: list[str] = []
        while i < len(tokens) and tokens[i].startswith("_"):
            columns.append(tokens[i])
            i += 1

        if not any(column.startswith("_atom_site.") for column in columns):
            while i < len(tokens) and tokens[i].lower() != "loop_":
                i += 1
            continue

        if not all(column.startswith("_atom_site.") for column in columns):
            raise ValueError("found a mixed-category loop containing _atom_site columns")

        rows: list[list[str]] = []
        while i < len(tokens) and tokens[i].lower() != "loop_" and not tokens[i].startswith("_"):
            row = tokens[i : i + len(columns)]
            if len(row) != len(columns):
                raise ValueError("atom_site loop ended with a partial row")
            rows.append(row)
            i += len(columns)
        return columns, rows

    raise ValueError("no _atom_site loop found")



def _column_index(columns: list[str], name: str) -> int:
    try:
        return columns.index(name)
    except ValueError as exc:
        raise ValueError(f"required CIF column {name} is missing") from exc


def _optional_column_index(columns: list[str], name: str) -> int | None:
    try:
        return columns.index(name)
    except ValueError:
        return None


def _first_existing_column(columns: list[str], names: tuple[str, ...]) -> int:
    for name in names:
        try:
            return columns.index(name)
        except ValueError:
            pass
    raise ValueError(f"required CIF column, one of {names}, is missing")


def atom_identity_to_atomic_number(atom_identity: str) -> int:
    try:
        element = ATOM_IDENTITY_ELEMENTS[atom_identity]
    except KeyError as exc:
        raise ValueError(f"unknown protein atom identity {atom_identity!r}") from exc
    return ATOMIC_NUMBERS[element]


def _pack_symmetric_pairs(resname: str, atoms: dict[str, np.ndarray]) -> dict[str, CoordValue]:
    packed: dict[str, CoordValue] = {}
    paired_atom_names: set[str] = set()

    for pair_identity, atom_a, atom_b in SYMMETRIC_ATOM_PAIRS.get(resname, ()):
        if atom_a in atoms and atom_b in atoms:
            packed[pair_identity] = Z2CoordPair(atoms[atom_a], atoms[atom_b])
            paired_atom_names.update((atom_a, atom_b))

    for atom_name, coord in atoms.items():
        if atom_name not in paired_atom_names:
            packed[atom_name] = coord

    return packed


def _split_chain_on_peptide_breaks(
    chain: list[tuple[str, dict[str, CoordValue]]],
    max_peptide_bond: float,
) -> list[list[tuple[str, dict[str, CoordValue]]]]:
    split_chains: list[list[tuple[str, dict[str, CoordValue]]]] = []
    current_chain: list[tuple[str, dict[str, CoordValue]]] = []
    previous_residue: dict[str, CoordValue] | None = None

    def finish_chain() -> None:
        if current_chain:
            split_chains.append(current_chain.copy())
            current_chain.clear()

    for residue_entry in chain:
        _aa, residue = residue_entry
        has_backbone = all(atom_name in residue for atom_name in ("N", "CA", "C", "O"))
        has_peptide_break = (
            has_backbone
            and previous_residue is not None
            and "C" in previous_residue
            and np.linalg.norm(previous_residue["C"] - residue["N"]) > max_peptide_bond
        )

        if not has_backbone or has_peptide_break:
            finish_chain()

        if has_backbone:
            current_chain.append(residue_entry)
            previous_residue = residue
        else:
            previous_residue = None

    finish_chain()
    return split_chains


def read_protein_cif_with_codes(
    path: str | os.PathLike[str],
    max_peptide_bond: float = 2.0,
) -> ProteinWithCodes:
    tokens = _tokenize_cif(Path(path).read_text())
    columns, rows = _parse_atom_site_loop(tokens)

    group_i = _optional_column_index(columns, "_atom_site.group_PDB")
    element_i = _column_index(columns, "_atom_site.type_symbol")
    atom_i = _first_existing_column(
        columns, ("_atom_site.auth_atom_id", "_atom_site.label_atom_id")
    )
    comp_i = _first_existing_column(
        columns, ("_atom_site.auth_comp_id", "_atom_site.label_comp_id")
    )
    chain_i = _first_existing_column(
        columns, ("_atom_site.auth_asym_id", "_atom_site.label_asym_id")
    )
    seq_i = _first_existing_column(
        columns, ("_atom_site.auth_seq_id", "_atom_site.label_seq_id")
    )
    ins_i = _optional_column_index(columns, "_atom_site.pdbx_PDB_ins_code")
    alt_i = _optional_column_index(columns, "_atom_site.label_alt_id")
    x_i = _column_index(columns, "_atom_site.Cartn_x")
    y_i = _column_index(columns, "_atom_site.Cartn_y")
    z_i = _column_index(columns, "_atom_site.Cartn_z")

    residues: "OrderedDict[tuple[str, str, str], dict[str, object]]" = OrderedDict()
    for row in rows:
        if group_i is not None and row[group_i] != "ATOM":
            continue
        resname = row[comp_i].upper()
        if resname not in AMINO_ACID_CODES:
            continue
        if _normalize_element(row[element_i]) == "H":
            continue
        if alt_i is not None and row[alt_i] not in {".", "?", "A"}:
            continue

        ins_code = "" if ins_i is None or row[ins_i] in {".", "?"} else row[ins_i]
        residue_key = (row[chain_i], row[seq_i], ins_code)
        residue = residues.setdefault(
            residue_key, {"chain": row[chain_i], "resname": resname, "atoms": OrderedDict()}
        )
        atoms = residue["atoms"]
        assert isinstance(atoms, OrderedDict)
        atom_name = row[atom_i]
        if atom_name in atoms:
            continue
        atoms[atom_name] = np.asarray(
            [float(row[x_i]), float(row[y_i]), float(row[z_i])], dtype=np.float32
        )

    chains: dict[str, list[tuple[str, dict[str, CoordValue]]]] = OrderedDict()
    for residue in residues.values():
        chain = residue["chain"]
        resname = residue["resname"]
        atoms = residue["atoms"]
        assert isinstance(chain, str)
        assert isinstance(resname, str)
        assert isinstance(atoms, OrderedDict)
        chains.setdefault(chain, []).append(
            (AMINO_ACID_CODES[resname], _pack_symmetric_pairs(resname, atoms))
        )

    split_chains = [
        split_chain
        for chain in chains.values()
        for split_chain in _split_chain_on_peptide_breaks(chain, max_peptide_bond)
    ]

    sorted_chains = sorted(
        split_chains,
        key=lambda chain: "".join(one_letter for one_letter, _residue in chain),
    )
    return sorted_chains


def read_protein_cif(
    path: str | os.PathLike[str],
    max_peptide_bond: float = 2.0,
) -> Protein:
    return [
        [residue for _one_letter, residue in chain]
        for chain in read_protein_cif_with_codes(path, max_peptide_bond=max_peptide_bond)
    ]


def protein_to_display_arrays(
    protein: Protein,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    atomic_numbers: list[int] = []
    positions: list[np.ndarray] = []
    ribbons: list[np.ndarray] = []

    for chain in protein:
        ribbon: list[int] = []
        for residue in chain:
            atom_indices: dict[str, int] = {}
            for atom_identity, coord_or_pair in residue.items():
                atomic_number = atom_identity_to_atomic_number(atom_identity)
                coords = (
                    (coord_or_pair.a, coord_or_pair.b)
                    if isinstance(coord_or_pair, Z2CoordPair)
                    else (coord_or_pair,)
                )
                for coord in coords:
                    atom_indices.setdefault(atom_identity, len(positions))
                    atomic_numbers.append(atomic_number)
                    positions.append(coord)

            if all(atom_name in atom_indices for atom_name in ("N", "CA", "C", "O")):
                ribbon.extend(
                    [atom_indices["N"], atom_indices["CA"], atom_indices["C"], atom_indices["O"]]
                )
            else:
                if len(ribbon) >= 8:
                    ribbons.append(np.asarray(ribbon, dtype=int))
                ribbon = []

        if len(ribbon) >= 8:
            ribbons.append(np.asarray(ribbon, dtype=int))

    if not positions:
        raise ValueError("the CIF did not contain any standard protein atoms")

    return (
        np.asarray(atomic_numbers, dtype=int),
        np.asarray(positions, dtype=np.float32),
        ribbons,
    )


def _ribbon_indices(
    rows: list[list[str]], columns: list[str], max_peptide_bond: float = 2.0
) -> list[np.ndarray]:
    try:
        chain_i = columns.index("_atom_site.auth_asym_id")
    except ValueError:
        chain_i = _column_index(columns, "_atom_site.label_asym_id")
    try:
        seq_i = columns.index("_atom_site.auth_seq_id")
    except ValueError:
        seq_i = _column_index(columns, "_atom_site.label_seq_id")
    try:
        ins_i = columns.index("_atom_site.pdbx_PDB_ins_code")
    except ValueError:
        ins_i = None
    try:
        atom_i = columns.index("_atom_site.auth_atom_id")
    except ValueError:
        atom_i = _column_index(columns, "_atom_site.label_atom_id")
    x_i = _column_index(columns, "_atom_site.Cartn_x")
    y_i = _column_index(columns, "_atom_site.Cartn_y")
    z_i = _column_index(columns, "_atom_site.Cartn_z")

    residues: "OrderedDict[tuple[str, str, str], dict[str, tuple[int, np.ndarray]]]" = (
        OrderedDict()
    )
    for atom_index, row in enumerate(rows):
        ins_code = "" if ins_i is None or row[ins_i] in {".", "?"} else row[ins_i]
        residue_key = (row[chain_i], row[seq_i], ins_code)
        coord = np.asarray([float(row[x_i]), float(row[y_i]), float(row[z_i])])
        residues.setdefault(residue_key, {})[row[atom_i]] = (atom_index, coord)

    ribbons: list[np.ndarray] = []
    current_chain = None
    current_ribbon: list[int] = []
    previous_atoms: dict[str, tuple[int, np.ndarray]] | None = None

    def finish_ribbon() -> None:
        if len(current_ribbon) >= 8:
            ribbons.append(np.asarray(current_ribbon, dtype=int))
        current_ribbon.clear()

    for (chain, _seq, _ins), atoms in residues.items():
        has_backbone = all(atom_name in atoms for atom_name in ("N", "CA", "C", "O"))
        starts_new_chain = current_chain is not None and chain != current_chain
        has_peptide_break = (
            has_backbone
            and previous_atoms is not None
            and "C" in previous_atoms
            and np.linalg.norm(previous_atoms["C"][1] - atoms["N"][1]) > max_peptide_bond
        )

        if starts_new_chain or not has_backbone or has_peptide_break:
            finish_ribbon()

        current_chain = chain
        if has_backbone:
            current_ribbon.extend(
                [atoms["N"][0], atoms["CA"][0], atoms["C"][0], atoms["O"][0]]
            )
            previous_atoms = atoms
        else:
            previous_atoms = None

    finish_ribbon()
    return ribbons



def read_cif(path: str | os.PathLike[str]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    tokens = _tokenize_cif(Path(path).read_text())
    columns, rows = _parse_atom_site_loop(tokens)

    element_i = _column_index(columns, "_atom_site.type_symbol")
    x_i = _column_index(columns, "_atom_site.Cartn_x")
    y_i = _column_index(columns, "_atom_site.Cartn_y")
    z_i = _column_index(columns, "_atom_site.Cartn_z")

    atomic_numbers = np.zeros(len(rows), dtype=int)
    positions = np.zeros((len(rows), 3), dtype=float)
    for i, row in enumerate(rows):
        element = _normalize_element(row[element_i])
        try:
            atomic_numbers[i] = ATOMIC_NUMBERS[element]
        except KeyError as exc:
            raise ValueError(
                f"unsupported element {element!r}; atoms-display radii are defined "
                "for atomic numbers 1-18 in this checkout"
            ) from exc
        positions[i] = [float(row[x_i]), float(row[y_i]), float(row[z_i])]

    if len(rows) == 0:
        raise ValueError("the _atom_site loop did not contain any atoms")

    return atomic_numbers, positions, _ribbon_indices(rows, columns)




