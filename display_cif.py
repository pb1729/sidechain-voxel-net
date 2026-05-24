#!/usr/bin/env python3
"""Display an mmCIF structure with atoms-display."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from density_fns import box_with_grid_for_protein
from parse_cif import protein_to_display_arrays, read_protein_cif_with_codes


def _grid_coords_to_display_coords(grid, grid_coords, center):
    homogeneous_coords = np.concatenate(
        [grid_coords, np.ones((grid_coords.shape[0], 1), dtype=grid_coords.dtype)],
        axis=1,
    )
    if grid.transform is None:
        original_coords = grid_coords
    else:
        original_coords = (np.linalg.inv(grid.transform) @ homogeneous_coords.T).T[:, :3]
    return original_coords - center


def _grid_box_lines(
    grid,
    center,
    *,
    grid_color=(0.35, 0.45, 0.55),
    edge_color=(1.0, 0.78, 0.22),
):
    L = np.asarray(grid.L, dtype=np.float32)
    vertices = []
    colors = []

    def add_segment(a, b, color):
        vertices.extend([a, b])
        colors.extend([color, color])

    for axis in range(3):
        other_axes = [i for i in range(3) if i != axis]
        for boundary_values in np.ndindex(2, 2):
            a = np.zeros(3, dtype=np.float32)
            b = np.zeros(3, dtype=np.float32)
            a[axis] = 0.0
            b[axis] = L[axis]
            for other_axis, boundary_value in zip(other_axes, boundary_values):
                a[other_axis] = L[other_axis] if boundary_value else 0.0
                b[other_axis] = L[other_axis] if boundary_value else 0.0
            add_segment(a, b, edge_color)

        fixed_axis, stepped_axis = other_axes
        for fixed_value in (0.0, L[fixed_axis]):
            for step in range(1, int(grid.N[stepped_axis])):
                a = np.zeros(3, dtype=np.float32)
                b = np.zeros(3, dtype=np.float32)
                a[axis] = 0.0
                b[axis] = L[axis]
                a[fixed_axis] = fixed_value
                b[fixed_axis] = fixed_value
                a[stepped_axis] = step * grid.dx
                b[stepped_axis] = step * grid.dx
                add_segment(a, b, grid_color)
        for fixed_value in (0.0, L[stepped_axis]):
            for step in range(1, int(grid.N[fixed_axis])):
                a = np.zeros(3, dtype=np.float32)
                b = np.zeros(3, dtype=np.float32)
                a[axis] = 0.0
                b[axis] = L[axis]
                a[stepped_axis] = fixed_value
                b[stepped_axis] = fixed_value
                a[fixed_axis] = step * grid.dx
                b[fixed_axis] = step * grid.dx
                add_segment(a, b, grid_color)

    vertices = np.asarray(vertices, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    return _grid_coords_to_display_coords(grid, vertices, center), colors




def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read an mmCIF file and display it with atoms-display."
    )
    parser.add_argument("cif_path", help="path to the .cif file to display")
    parser.add_argument(
        "--no-ribbons",
        action="store_true",
        help="draw only atoms, without protein backbone ribbons",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="do not draw the density grid wireframe",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    protein_with_codes = read_protein_cif_with_codes(args.cif_path)
    protein = [
        [residue for _one_letter, residue in chain]
        for chain in protein_with_codes
    ]
    atomic_numbers, positions, ribbons = protein_to_display_arrays(protein)
    center = positions.mean(axis=0)
    positions -= center

    from atoms_display import launch_atom_display

    display = launch_atom_display(atomic_numbers, positions, None if args.no_ribbons else ribbons)
    if not args.no_grid:
        grid = box_with_grid_for_protein(protein_with_codes)
        print(f"N: {grid.N}")
        line_vertices, line_colors = _grid_box_lines(grid, center)
        display.add_lines(line_vertices, line_colors)
    print(f"Displaying {len(atomic_numbers)} atoms from {args.cif_path}. Press Ctrl-C to quit.")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
