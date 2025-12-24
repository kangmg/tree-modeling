"""
Atomic Christmas Tree Model
- Base: Ag simple cubic slab
- Trunk: Fe HCP structure
- Leaves: Carbon PAH molecules in spiral (benzene → coronene)
- Decorations: Halogen substitution (F, Cl, Br)
- Star: Au cluster on top
"""

import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory
from typing import List, Tuple


# =============================================================================
# Chemical Constants (Angstroms)
# =============================================================================
BOND_LENGTHS = {
    'Ag-Ag': 2.89,      # Simple cubic Ag
    'Fe-Fe': 2.52,      # HCP Fe (a parameter)
    'Fe-Fe_c': 4.06,    # HCP Fe (c parameter)
    'C-C_aromatic': 1.40,
    'C-H': 1.09,
    'C-F': 1.35,
    'C-Cl': 1.77,
    'C-Br': 1.94,
    'Au-Au': 2.88,
}


# =============================================================================
# Part 1: Ag Simple Cubic Slab (Base)
# =============================================================================
def create_ag_slab(nx: int = 10, ny: int = 10) -> Atoms:
    """
    Create a single-layer Ag simple cubic slab.

    Parameters:
        nx, ny: Number of atoms in x and y directions

    Returns:
        Atoms object with Ag slab
    """
    a = BOND_LENGTHS['Ag-Ag']
    positions = []

    # Center the slab at origin
    offset_x = (nx - 1) * a / 2
    offset_y = (ny - 1) * a / 2

    for i in range(nx):
        for j in range(ny):
            x = i * a - offset_x
            y = j * a - offset_y
            z = 0.0
            positions.append([x, y, z])

    atoms = Atoms('Ag' * (nx * ny), positions=positions)
    return atoms


# =============================================================================
# Part 2: Fe HCP Pillar (Trunk)
# =============================================================================
def create_fe_hcp_pillar(n_layers: int = 12, atoms_per_layer: int = 7) -> Tuple[Atoms, List[Atoms]]:
    """
    Create Fe HCP pillar structure.
    Returns both the complete pillar and list of layer-by-layer Atoms for animation.

    HCP structure: ABAB stacking
    - A layer: atoms at (0,0) and hexagonal positions
    - B layer: atoms shifted by (a/2, a/(2*sqrt(3)))

    Parameters:
        n_layers: Number of HCP layers
        atoms_per_layer: Approximate atoms per layer (uses hexagonal pattern)

    Returns:
        (complete_pillar, [layer1, layer2, ...])
    """
    a = BOND_LENGTHS['Fe-Fe']
    c = BOND_LENGTHS['Fe-Fe_c']

    # Create a small hexagonal cluster for each layer
    # Central atom + 6 surrounding atoms for a nice pillar
    def get_hex_layer(z: float, is_B_layer: bool = False) -> List[List[float]]:
        positions = []

        # Shift for B layer in HCP
        if is_B_layer:
            shift_x = a / 2
            shift_y = a / (2 * np.sqrt(3))
        else:
            shift_x = 0
            shift_y = 0

        # Central atom
        positions.append([shift_x, shift_y, z])

        # 6 surrounding atoms in hexagonal arrangement
        for i in range(6):
            angle = i * np.pi / 3
            x = a * np.cos(angle) + shift_x
            y = a * np.sin(angle) + shift_y
            positions.append([x, y, z])

        return positions

    all_positions = []
    layer_atoms_list = []

    for layer_idx in range(n_layers):
        z = layer_idx * (c / 2) + 2.5  # Start above the Ag slab
        is_B = (layer_idx % 2 == 1)

        layer_positions = get_hex_layer(z, is_B)
        all_positions.extend(layer_positions)

        # Create Atoms object for this layer
        layer_atoms = Atoms('Fe' * len(layer_positions), positions=layer_positions)
        layer_atoms_list.append(layer_atoms)

    complete_pillar = Atoms('Fe' * len(all_positions), positions=all_positions)

    return complete_pillar, layer_atoms_list


# =============================================================================
# Part 3: PAH Molecules (Leaves)
# =============================================================================
def create_benzene() -> Atoms:
    """Create benzene molecule (C6H6) lying flat in xy-plane."""
    cc = BOND_LENGTHS['C-C_aromatic']
    ch = BOND_LENGTHS['C-H']

    # Hexagonal carbon ring
    c_positions = []
    h_positions = []

    radius_c = cc  # Distance from center to C
    radius_h = radius_c + ch  # Distance from center to H

    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6  # Start at 30 degrees
        # Carbon positions
        c_positions.append([
            radius_c * np.cos(angle),
            radius_c * np.sin(angle),
            0.0
        ])
        # Hydrogen positions (radially outward)
        h_positions.append([
            radius_h * np.cos(angle),
            radius_h * np.sin(angle),
            0.0
        ])

    positions = c_positions + h_positions
    symbols = ['C'] * 6 + ['H'] * 6

    return Atoms(symbols=symbols, positions=positions)


def create_naphthalene() -> Atoms:
    """Create naphthalene molecule (C10H8) - two fused benzene rings."""
    cc = BOND_LENGTHS['C-C_aromatic']
    ch = BOND_LENGTHS['C-H']

    # Build naphthalene with proper geometry
    # Two fused hexagons sharing an edge
    h = cc * np.sqrt(3) / 2  # Height of equilateral triangle

    c_positions = [
        # Left ring
        [-cc - cc/2, h, 0],
        [-cc, 0, 0],
        [-cc/2, -h, 0],
        # Shared edge
        [cc/2, -h, 0],
        [cc, 0, 0],
        [cc/2, h, 0],
        [-cc/2, h, 0],
        # Right ring extension
        [cc + cc/2, h, 0],
        [cc + cc, 0, 0],
        [cc + cc/2, -h, 0],
    ]

    # Hydrogen positions (on outer carbons)
    h_positions = [
        [-cc - cc/2 - ch*np.cos(np.pi/6), h + ch*np.sin(np.pi/6), 0],
        [-cc - ch, 0, 0],
        [-cc/2 - ch*np.cos(np.pi/6), -h - ch*np.sin(np.pi/6), 0],
        [cc/2 + ch*np.cos(np.pi/6), -h - ch*np.sin(np.pi/6), 0],
        [cc + cc/2 + ch*np.cos(np.pi/6), h + ch*np.sin(np.pi/6), 0],
        [cc + cc + ch, 0, 0],
        [cc + cc/2 + ch*np.cos(np.pi/6), -h - ch*np.sin(np.pi/6), 0],
        [-cc - cc/2 - ch*np.cos(np.pi/6), -h - ch*np.sin(np.pi/6), 0],
    ]

    positions = c_positions + h_positions
    symbols = ['C'] * 10 + ['H'] * 8

    return Atoms(symbols=symbols, positions=positions)


def create_pyrene() -> Atoms:
    """Create pyrene molecule (C16H10) - four fused benzene rings."""
    cc = BOND_LENGTHS['C-C_aromatic']
    ch = BOND_LENGTHS['C-H']
    h = cc * np.sqrt(3) / 2

    # Pyrene: 4 fused rings in a specific pattern
    c_positions = [
        # Row 1 (top)
        [-cc, 2*h, 0],
        [0, 2*h, 0],
        [cc, 2*h, 0],
        # Row 2
        [-1.5*cc, h, 0],
        [-0.5*cc, h, 0],
        [0.5*cc, h, 0],
        [1.5*cc, h, 0],
        # Row 3
        [-1.5*cc, -h, 0],
        [-0.5*cc, -h, 0],
        [0.5*cc, -h, 0],
        [1.5*cc, -h, 0],
        # Row 4 (bottom)
        [-cc, -2*h, 0],
        [0, -2*h, 0],
        [cc, -2*h, 0],
        # Inner carbons
        [0, h, 0],
        [0, -h, 0],
    ]

    # Hydrogens on outer edge
    h_positions = [
        [-cc, 2*h + ch, 0],
        [cc, 2*h + ch, 0],
        [-1.5*cc - ch*np.cos(np.pi/6), h + ch*np.sin(np.pi/6), 0],
        [1.5*cc + ch*np.cos(np.pi/6), h + ch*np.sin(np.pi/6), 0],
        [-1.5*cc - ch*np.cos(np.pi/6), -h - ch*np.sin(np.pi/6), 0],
        [1.5*cc + ch*np.cos(np.pi/6), -h - ch*np.sin(np.pi/6), 0],
        [-cc, -2*h - ch, 0],
        [cc, -2*h - ch, 0],
        [-2*cc, 0, 0],
        [2*cc, 0, 0],
    ]

    positions = c_positions + h_positions
    symbols = ['C'] * 16 + ['H'] * 10

    return Atoms(symbols=symbols, positions=positions)


def create_coronene() -> Atoms:
    """Create coronene molecule (C24H12) - seven fused benzene rings (hexagonal)."""
    cc = BOND_LENGTHS['C-C_aromatic']
    ch = BOND_LENGTHS['C-H']

    c_positions = []
    h_positions = []

    # Inner ring (6 carbons)
    inner_radius = cc
    for i in range(6):
        angle = i * np.pi / 3
        c_positions.append([
            inner_radius * np.cos(angle),
            inner_radius * np.sin(angle),
            0.0
        ])

    # Middle ring (6 carbons between inner spokes)
    middle_radius = cc * np.sqrt(3)
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        c_positions.append([
            middle_radius * np.cos(angle),
            middle_radius * np.sin(angle),
            0.0
        ])

    # Outer ring (12 carbons)
    outer_radius = 2 * cc
    for i in range(6):
        angle = i * np.pi / 3
        c_positions.append([
            outer_radius * np.cos(angle),
            outer_radius * np.sin(angle),
            0.0
        ])

    outer_radius2 = cc * (1 + np.sqrt(3))
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        c_positions.append([
            outer_radius2 * np.cos(angle),
            outer_radius2 * np.sin(angle),
            0.0
        ])

    # Hydrogens on outer edge
    h_outer_radius = outer_radius + ch
    for i in range(6):
        angle = i * np.pi / 3
        h_positions.append([
            h_outer_radius * np.cos(angle),
            h_outer_radius * np.sin(angle),
            0.0
        ])

    h_outer_radius2 = outer_radius2 + ch
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        h_positions.append([
            h_outer_radius2 * np.cos(angle),
            h_outer_radius2 * np.sin(angle),
            0.0
        ])

    positions = c_positions + h_positions
    symbols = ['C'] * 24 + ['H'] * 12

    return Atoms(symbols=symbols, positions=positions)


def create_pah_spiral_leaves(pillar_height: float, n_spirals: int = 3) -> Tuple[Atoms, List[Atoms]]:
    """
    Create PAH molecules arranged in a spiral from top to bottom.
    Size increases as we go down (benzene at top → coronene at bottom).

    Parameters:
        pillar_height: Height of the Fe pillar (z-coordinate of top)
        n_spirals: Number of complete spiral rotations

    Returns:
        (complete_leaves, [pah_group1, pah_group2, ...])
    """
    # PAH molecules from smallest to largest
    pah_creators = [
        ('benzene', create_benzene),
        ('benzene', create_benzene),
        ('naphthalene', create_naphthalene),
        ('naphthalene', create_naphthalene),
        ('pyrene', create_pyrene),
        ('pyrene', create_pyrene),
        ('pyrene', create_pyrene),
        ('coronene', create_coronene),
        ('coronene', create_coronene),
        ('coronene', create_coronene),
    ]

    n_molecules = len(pah_creators)
    all_atoms = Atoms()
    group_list = []

    # Spiral parameters
    start_z = pillar_height + 1.0  # Start above the pillar
    end_z = 5.0  # End above the base
    start_radius = 3.0  # Small radius at top
    end_radius = 12.0  # Large radius at bottom

    for i, (name, creator) in enumerate(pah_creators):
        # Calculate position in spiral
        t = i / (n_molecules - 1) if n_molecules > 1 else 0

        # Z position (top to bottom)
        z = start_z - t * (start_z - end_z)

        # Radial position (increases going down)
        radius = start_radius + t * (end_radius - start_radius)

        # Angular position (spiral)
        angle = t * n_spirals * 2 * np.pi

        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        # Create and position the molecule
        mol = creator()

        # Rotate molecule to face outward and tilt slightly
        positions = mol.get_positions()

        # Rotate around z-axis to face outward
        rot_z = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        # Tilt molecule (rotate around x-axis)
        tilt = np.pi / 6  # 30 degree tilt
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(tilt), -np.sin(tilt)],
            [0, np.sin(tilt), np.cos(tilt)]
        ])

        # Apply rotations
        positions = positions @ rot_z.T @ rot_x.T

        # Translate to position
        positions[:, 0] += x
        positions[:, 1] += y
        positions[:, 2] += z

        mol.set_positions(positions)

        all_atoms += mol
        group_list.append(mol.copy())

    return all_atoms, group_list


# =============================================================================
# Part 4: Halogen Decorations
# =============================================================================
def add_halogen_decorations(atoms: Atoms, n_substitutions: int = 3) -> Tuple[Atoms, List[Tuple[int, str]]]:
    """
    Randomly substitute H atoms with F, Cl, or Br.

    Parameters:
        atoms: Atoms object containing the tree
        n_substitutions: Number of each halogen to add (3 F, 3 Cl, 3 Br)

    Returns:
        (modified_atoms, [(index, new_symbol), ...])
    """
    atoms = atoms.copy()
    symbols = list(atoms.get_chemical_symbols())
    positions = atoms.get_positions().copy()

    # Find all hydrogen indices
    h_indices = [i for i, s in enumerate(symbols) if s == 'H']

    if len(h_indices) < n_substitutions * 3:
        n_substitutions = len(h_indices) // 3

    # Randomly select hydrogens to replace
    np.random.seed(42)  # For reproducibility
    selected = np.random.choice(h_indices, size=n_substitutions * 3, replace=False)

    halogens = ['F'] * n_substitutions + ['Cl'] * n_substitutions + ['Br'] * n_substitutions
    np.random.shuffle(halogens)

    substitutions = []

    for idx, halogen in zip(selected, halogens):
        old_symbol = symbols[idx]
        symbols[idx] = halogen

        # Adjust bond length (find nearest carbon and extend)
        pos = positions[idx]

        # Find nearest carbon
        c_indices = [i for i, s in enumerate(symbols) if s == 'C']
        if c_indices:
            c_positions = positions[c_indices]
            distances = np.linalg.norm(c_positions - pos, axis=1)
            nearest_c_idx = c_indices[np.argmin(distances)]

            # Direction from C to H
            direction = pos - positions[nearest_c_idx]
            direction = direction / np.linalg.norm(direction)

            # New bond length
            new_length = BOND_LENGTHS[f'C-{halogen}']
            positions[idx] = positions[nearest_c_idx] + direction * new_length

        substitutions.append((idx, halogen))

    # Create new atoms object
    new_atoms = Atoms(symbols=''.join(symbols), positions=positions)

    return new_atoms, substitutions


# =============================================================================
# Part 5: Au Star
# =============================================================================
def create_au_star(center_z: float) -> Atoms:
    """
    Create a small Au cluster (star shape) at the top.
    Uses a pentagonal bipyramid structure (7 atoms).

    Parameters:
        center_z: Z-coordinate for the center of the star

    Returns:
        Atoms object with Au cluster
    """
    au_au = BOND_LENGTHS['Au-Au']

    positions = []

    # Central atom
    positions.append([0, 0, center_z])

    # Pentagon around center
    for i in range(5):
        angle = i * 2 * np.pi / 5 + np.pi / 2  # Start pointing up
        positions.append([
            au_au * np.cos(angle),
            au_au * np.sin(angle),
            center_z
        ])

    # Top point of star
    positions.append([0, 0, center_z + au_au])

    return Atoms('Au' * 7, positions=positions)


# =============================================================================
# Part 6: Assembly and Animation
# =============================================================================
def build_christmas_tree(output_file: str = "christmas_tree.traj") -> Atoms:
    """
    Build the complete atomic Christmas tree and save animation trajectory.

    Animation frames:
    1. Ag slab (all at once)
    2. Fe pillar (layer by layer)
    3. Carbon leaves (PAH group by group)
    4. Halogen decorations (3 at a time)
    5. Au star (all at once)

    Parameters:
        output_file: Output trajectory filename

    Returns:
        Complete Atoms object
    """
    traj = Trajectory(output_file, 'w')
    current_atoms = Atoms()

    print("Building Atomic Christmas Tree...")
    print("=" * 50)

    # ----- Stage 1: Ag Slab (Base) -----
    print("\n[Stage 1] Creating Ag slab base...")
    ag_slab = create_ag_slab(nx=10, ny=10)
    current_atoms += ag_slab
    traj.write(current_atoms)
    print(f"  Added {len(ag_slab)} Ag atoms")

    # ----- Stage 2: Fe Pillar (Trunk) -----
    print("\n[Stage 2] Growing Fe HCP pillar...")
    fe_pillar, fe_layers = create_fe_hcp_pillar(n_layers=12, atoms_per_layer=7)

    for i, layer in enumerate(fe_layers):
        current_atoms += layer
        traj.write(current_atoms)
        print(f"  Layer {i+1}: Added {len(layer)} Fe atoms")

    pillar_height = max(current_atoms.positions[:, 2])

    # ----- Stage 3: Au Star (Top) -----
    print("\n[Stage 3] Placing Au star on top...")
    au_star = create_au_star(center_z=pillar_height + 2.0)
    current_atoms += au_star
    traj.write(current_atoms)
    print(f"  Added {len(au_star)} Au atoms")

    # ----- Stage 4: Carbon Leaves -----
    print("\n[Stage 4] Growing carbon spiral leaves...")
    leaves, pah_groups = create_pah_spiral_leaves(pillar_height=pillar_height, n_spirals=2)

    for i, pah in enumerate(pah_groups):
        current_atoms += pah
        traj.write(current_atoms)
        n_c = sum(1 for s in pah.get_chemical_symbols() if s == 'C')
        n_h = sum(1 for s in pah.get_chemical_symbols() if s == 'H')
        print(f"  PAH {i+1}: Added C{n_c}H{n_h}")

    # ----- Stage 5: Halogen Decorations -----
    print("\n[Stage 5] Adding halogen decorations...")
    decorated_atoms, substitutions = add_halogen_decorations(current_atoms, n_substitutions=3)

    # Group substitutions by 3
    for i in range(0, len(substitutions), 3):
        batch = substitutions[i:i+3]
        for idx, halogen in batch:
            current_atoms[idx].symbol = halogen
            current_atoms.positions[idx] = decorated_atoms.positions[idx]
        traj.write(current_atoms)
        print(f"  Batch {i//3 + 1}: Substituted {[h for _, h in batch]}")

    traj.close()

    print("\n" + "=" * 50)
    print(f"Christmas tree complete!")
    print(f"Total atoms: {len(current_atoms)}")
    print(f"Trajectory saved to: {output_file}")

    # Print composition
    symbols = current_atoms.get_chemical_symbols()
    from collections import Counter
    composition = Counter(symbols)
    print(f"Composition: {dict(composition)}")

    return current_atoms


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    tree = build_christmas_tree("christmas_tree.traj")

    # Also save as xyz for easy viewing
    from ase.io import write
    write("christmas_tree.xyz", tree)
    print(f"\nAlso saved as: christmas_tree.xyz")
