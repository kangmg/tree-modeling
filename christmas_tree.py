"""
Atomic Christmas Tree Model
- Base: Ag simple cubic slab
- Trunk: Fe HCP structure
- Leaves: Helicene-based spiral carbon structure
- Decorations: Halogen substitution (F, Cl, Br)
- Star: Au13 icosahedral cluster on top
"""

import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TreeConfig:
    """Configuration parameters for the atomic Christmas tree."""
    # Base (Ag slab)
    slab_nx: int = 25
    slab_ny: int = 25

    # Trunk (Fe HCP pillar)
    pillar_layers: int = 30

    # Leaves (Helicene spiral)
    n_helix_turns: float = 5.0        # Number of helix rotations
    n_rings_per_turn: int = 12        # Benzene rings per turn
    helix_start_radius: float = 6.0   # Radius at top (small)
    helix_end_radius: float = 30.0    # Radius at bottom (large)

    # Decorations (Halogen substitution) - ~25% of H atoms
    n_decorations_per_halogen: int = 20  # Number of each F, Cl, Br (60 total)

    # Random seed for reproducibility
    random_seed: int = 42

    # Output
    output_file: str = "christmas_tree.traj"


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
def create_ag_slab(nx: int = 15, ny: int = 15) -> Atoms:
    """
    Create a single-layer Ag simple cubic slab.
    """
    a = BOND_LENGTHS['Ag-Ag']
    positions = []

    offset_x = (nx - 1) * a / 2
    offset_y = (ny - 1) * a / 2

    for i in range(nx):
        for j in range(ny):
            x = i * a - offset_x
            y = j * a - offset_y
            z = 0.0
            positions.append([x, y, z])

    return Atoms('Ag' * (nx * ny), positions=positions)


# =============================================================================
# Part 2: Fe HCP Pillar (Trunk)
# =============================================================================
def create_fe_hcp_pillar(n_layers: int = 18) -> Tuple[Atoms, List[Atoms]]:
    """
    Create Fe HCP pillar structure with ABAB stacking.
    """
    a = BOND_LENGTHS['Fe-Fe']
    c = BOND_LENGTHS['Fe-Fe_c']

    def get_hex_layer(z: float, is_B_layer: bool = False) -> List[List[float]]:
        positions = []
        if is_B_layer:
            shift_x = a / 2
            shift_y = a / (2 * np.sqrt(3))
        else:
            shift_x = 0
            shift_y = 0

        # Central atom
        positions.append([shift_x, shift_y, z])

        # 6 surrounding atoms
        for i in range(6):
            angle = i * np.pi / 3
            x = a * np.cos(angle) + shift_x
            y = a * np.sin(angle) + shift_y
            positions.append([x, y, z])

        return positions

    all_positions = []
    layer_atoms_list = []

    for layer_idx in range(n_layers):
        z = layer_idx * (c / 2) + 2.5
        is_B = (layer_idx % 2 == 1)

        layer_positions = get_hex_layer(z, is_B)
        all_positions.extend(layer_positions)

        layer_atoms = Atoms('Fe' * len(layer_positions), positions=layer_positions)
        layer_atoms_list.append(layer_atoms)

    complete_pillar = Atoms('Fe' * len(all_positions), positions=all_positions)
    return complete_pillar, layer_atoms_list


# =============================================================================
# Part 3: Helicene-based Spiral Leaves
# =============================================================================
def create_benzene_ring(center: np.ndarray, normal: np.ndarray,
                        tangent: np.ndarray, add_hydrogens: bool = True) -> Tuple[List, List]:
    """
    Create a single benzene ring at given position with orientation.

    Parameters:
        center: Center position of the ring
        normal: Normal vector to the ring plane
        tangent: Direction for positioning (radial outward)
        add_hydrogens: Whether to add H atoms on outer edge

    Returns:
        (carbon_positions, hydrogen_positions)
    """
    cc = BOND_LENGTHS['C-C_aromatic']
    ch = BOND_LENGTHS['C-H']

    # Normalize vectors
    normal = normal / np.linalg.norm(normal)
    tangent = tangent / np.linalg.norm(tangent)

    # Create orthonormal basis
    binormal = np.cross(normal, tangent)
    binormal = binormal / np.linalg.norm(binormal)
    tangent = np.cross(binormal, normal)  # Ensure orthogonal

    # Rotation matrix from local to global
    R = np.column_stack([tangent, binormal, normal])

    # Carbon positions in local coordinates (hexagon)
    c_positions = []
    h_positions = []
    radius_c = cc
    radius_h = radius_c + ch

    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        local_c = np.array([radius_c * np.cos(angle), radius_c * np.sin(angle), 0])
        global_c = center + R @ local_c
        c_positions.append(global_c)

        if add_hydrogens:
            local_h = np.array([radius_h * np.cos(angle), radius_h * np.sin(angle), 0])
            global_h = center + R @ local_h
            h_positions.append(global_h)

    return c_positions, h_positions


def create_helicene_spiral_leaves(
    pillar_height: float,
    n_helix_turns: float = 3.0,
    n_rings_per_turn: int = 8,
    start_radius: float = 4.0,
    end_radius: float = 18.0
) -> Tuple[Atoms, List[Atoms]]:
    """
    Create helicene-inspired spiral leaves around the pillar.

    Benzene rings are placed in a continuous helix pattern,
    with radius increasing from top to bottom (cone shape).

    Parameters:
        pillar_height: Height of the Fe pillar
        n_helix_turns: Number of complete helix rotations
        n_rings_per_turn: Number of benzene rings per turn
        start_radius: Helix radius at top
        end_radius: Helix radius at bottom

    Returns:
        (complete_leaves, [ring_group1, ring_group2, ...])
    """
    total_rings = int(n_helix_turns * n_rings_per_turn)

    # Helix parameters
    start_z = pillar_height + 2.0
    end_z = 4.0

    all_atoms = Atoms()
    group_list = []

    for i in range(total_rings):
        t = i / max(1, total_rings - 1)

        # Position along helix
        angle = t * n_helix_turns * 2 * np.pi
        z = start_z - t * (start_z - end_z)
        radius = start_radius + t * (end_radius - start_radius)

        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        center = np.array([x, y, z])

        # Orientation: ring faces outward and tilts down
        # Normal is tilted outward (radial) and slightly down
        radial = np.array([np.cos(angle), np.sin(angle), 0])
        down_tilt = 0.3  # Tilt factor
        normal = radial + np.array([0, 0, -down_tilt])
        normal = normal / np.linalg.norm(normal)

        # Tangent along helix direction
        tangent = np.array([-np.sin(angle), np.cos(angle), 0])

        # Create benzene ring
        c_pos, h_pos = create_benzene_ring(center, normal, tangent, add_hydrogens=True)

        positions = c_pos + h_pos
        symbols = ['C'] * 6 + ['H'] * 6

        ring_atoms = Atoms(symbols=symbols, positions=positions)
        all_atoms += ring_atoms
        group_list.append(ring_atoms.copy())

    return all_atoms, group_list


# =============================================================================
# Part 4: Halogen Decorations
# =============================================================================
def add_halogen_decorations(
    atoms: Atoms,
    n_substitutions: int = 8,
    random_seed: int = 42
) -> Tuple[Atoms, List[Tuple[int, str]]]:
    """
    Randomly substitute H atoms with F, Cl, or Br.
    """
    atoms = atoms.copy()
    symbols = list(atoms.get_chemical_symbols())
    positions = atoms.get_positions().copy()

    h_indices = [i for i, s in enumerate(symbols) if s == 'H']

    if len(h_indices) < n_substitutions * 3:
        n_substitutions = len(h_indices) // 3

    np.random.seed(random_seed)
    selected = np.random.choice(h_indices, size=n_substitutions * 3, replace=False)

    halogens = ['F'] * n_substitutions + ['Cl'] * n_substitutions + ['Br'] * n_substitutions
    np.random.shuffle(halogens)

    substitutions = []

    for idx, halogen in zip(selected, halogens):
        symbols[idx] = halogen
        pos = positions[idx]

        # Find nearest carbon and adjust bond length
        c_indices = [i for i, s in enumerate(symbols) if s == 'C']
        if c_indices:
            c_positions = positions[c_indices]
            distances = np.linalg.norm(c_positions - pos, axis=1)
            nearest_c_idx = c_indices[np.argmin(distances)]

            direction = pos - positions[nearest_c_idx]
            direction = direction / np.linalg.norm(direction)

            new_length = BOND_LENGTHS[f'C-{halogen}']
            positions[idx] = positions[nearest_c_idx] + direction * new_length

        substitutions.append((idx, halogen))

    new_atoms = Atoms(symbols=''.join(symbols), positions=positions)
    return new_atoms, substitutions


# =============================================================================
# Part 5: Au13 Icosahedral Star
# =============================================================================
def create_au_star(center_z: float) -> Atoms:
    """
    Create Au13 icosahedral cluster - a spherical, chemically stable structure.

    The icosahedron has:
    - 1 central atom
    - 12 surface atoms at vertices

    Parameters:
        center_z: Z-coordinate for the center of the cluster

    Returns:
        Atoms object with Au13 cluster
    """
    # Au-Au distance in cluster (slightly shorter than bulk)
    au_au = BOND_LENGTHS['Au-Au'] * 0.95

    positions = []

    # Central atom
    positions.append([0, 0, center_z])

    # Icosahedron vertices
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # Scale factor to get correct Au-Au distance
    scale = au_au / np.sqrt(1 + phi**2)

    # 12 vertices of icosahedron (3 orthogonal golden rectangles)
    vertices = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]

    for v in vertices:
        pos = np.array(v) * scale
        pos[2] += center_z  # Shift to correct z position
        positions.append(pos.tolist())

    return Atoms('Au' * 13, positions=positions)


# =============================================================================
# Part 6: Assembly and Animation
# =============================================================================
def build_christmas_tree(config: Optional[TreeConfig] = None) -> Atoms:
    """
    Build the complete atomic Christmas tree and save animation trajectory.

    Animation frames:
    1. Ag slab (all at once)
    2. Fe pillar (layer by layer)
    3. Au star (all at once)
    4. Carbon leaves (ring by ring)
    5. Halogen decorations (3 at a time)
    """
    if config is None:
        config = TreeConfig()

    traj = Trajectory(config.output_file, 'w')
    current_atoms = Atoms()

    print("Building Atomic Christmas Tree...")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Slab: {config.slab_nx}x{config.slab_ny}")
    print(f"  Pillar layers: {config.pillar_layers}")
    print(f"  Helix turns: {config.n_helix_turns}")
    print(f"  Rings per turn: {config.n_rings_per_turn}")
    print(f"  Decorations per halogen: {config.n_decorations_per_halogen}")
    print("=" * 50)

    # ----- Stage 1: Ag Slab (Base) -----
    print("\n[Stage 1] Creating Ag slab base...")
    ag_slab = create_ag_slab(nx=config.slab_nx, ny=config.slab_ny)
    current_atoms += ag_slab
    traj.write(current_atoms)
    print(f"  Added {len(ag_slab)} Ag atoms")

    # ----- Stage 2: Fe Pillar (Trunk) -----
    print("\n[Stage 2] Growing Fe HCP pillar...")
    fe_pillar, fe_layers = create_fe_hcp_pillar(n_layers=config.pillar_layers)

    for i, layer in enumerate(fe_layers):
        current_atoms += layer
        traj.write(current_atoms)
        print(f"  Layer {i+1}: Added {len(layer)} Fe atoms")

    pillar_height = max(current_atoms.positions[:, 2])

    # ----- Stage 3: Au Star (Top) -----
    print("\n[Stage 3] Placing Au13 icosahedral star on top...")
    au_star = create_au_star(center_z=pillar_height + 3.5)
    current_atoms += au_star
    traj.write(current_atoms)
    print(f"  Added {len(au_star)} Au atoms (icosahedral cluster)")

    # ----- Stage 4: Carbon Leaves (Helicene spiral) -----
    print("\n[Stage 4] Growing helicene spiral leaves...")
    leaves, ring_groups = create_helicene_spiral_leaves(
        pillar_height=pillar_height,
        n_helix_turns=config.n_helix_turns,
        n_rings_per_turn=config.n_rings_per_turn,
        start_radius=config.helix_start_radius,
        end_radius=config.helix_end_radius
    )

    for i, ring in enumerate(ring_groups):
        current_atoms += ring
        traj.write(current_atoms)
        if (i + 1) % 5 == 0 or i == len(ring_groups) - 1:
            print(f"  Ring {i+1}/{len(ring_groups)}: Added benzene ring")

    # ----- Stage 5: Halogen Decorations -----
    print("\n[Stage 5] Adding halogen decorations...")
    decorated_atoms, substitutions = add_halogen_decorations(
        current_atoms,
        n_substitutions=config.n_decorations_per_halogen,
        random_seed=config.random_seed
    )

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
    print(f"Trajectory saved to: {config.output_file}")

    # Print composition
    from collections import Counter
    symbols = current_atoms.get_chemical_symbols()
    composition = Counter(symbols)
    print(f"Composition: {dict(composition)}")

    return current_atoms


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # Default configuration (larger tree)
    config = TreeConfig()

    # Custom configuration example:
    # config = TreeConfig(
    #     slab_nx=20,
    #     slab_ny=20,
    #     pillar_layers=25,
    #     n_helix_turns=4.0,
    #     n_rings_per_turn=10,
    #     helix_start_radius=5.0,
    #     helix_end_radius=25.0,
    #     n_decorations_per_halogen=12,
    #     output_file="big_tree.traj"
    # )

    tree = build_christmas_tree(config)

    # Also save as xyz
    from ase.io import write
    xyz_file = config.output_file.replace('.traj', '.xyz')
    write(xyz_file, tree)
    print(f"\nAlso saved as: {xyz_file}")
