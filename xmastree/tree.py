"""
Atomic Christmas Tree Model
- Base: Ag simple cubic slab
- Trunk: Fe HCP structure
- Leaves: Helicene-based spiral carbon structure
- Decorations: Halogen substitution (F, Cl, Br)
- Star: Au55 cluster on top
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

    # Decorations (Halogen substitution) - ~1/3 of H atoms
    n_decorations_per_halogen: int = 150  # Number of each F, Cl, Br (~450 total, ~1/3 of H)

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
# Part 3: PAH Molecule Spiral Leaves (Chemically Valid Discrete Molecules)
# =============================================================================

def _create_naphthalene() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create naphthalene (C10H8) - 2 fused benzene rings.

    Naphthalene is a well-known, stable PAH molecule.
    Returns (carbon_positions, hydrogen_positions) in local coordinates centered at origin.
    """
    # Naphthalene coordinates from reference structure
    c_positions = [
        [2.4044, 0.7559, 0.0000],
        [2.4328, -0.6584, 0.0000],
        [1.2672, -1.3753, 0.0000],
        [0.0142, -0.7050, 0.0000],
        [-0.0142, 0.7048, 0.0000],
        [1.2108, 1.4252, 0.0000],
        [-1.2672, 1.3754, 0.0000],
        [-2.4328, 0.6585, 0.0000],
        [-2.4043, -0.7558, 0.0000],
        [-1.2108, -1.4254, 0.0000],
    ]

    h_positions = [
        [3.3509, 1.3062, 0.0000],
        [3.4006, -1.1703, 0.0000],
        [1.2810, -2.4710, 0.0000],
        [1.1803, 2.5206, 0.0000],
        [-1.2808, 2.4710, 0.0000],
        [-3.4008, 1.1701, 0.0000],
        [-3.3508, -1.3060, 0.0000],
        [-1.1805, -2.5207, 0.0000],
    ]

    return np.array(c_positions), np.array(h_positions)


def _create_anthracene() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create anthracene (C14H10) - 3 linearly fused benzene rings.

    Anthracene is a well-known, stable PAH molecule.
    Returns (carbon_positions, hydrogen_positions) in local coordinates centered at origin.
    """
    # Anthracene coordinates from reference structure
    c_positions = [
        [3.6609, 0.5848, 0.0000],
        [3.6110, -0.8397, 0.0000],
        [2.4165, -1.4895, 0.0000],
        [1.1870, -0.7528, 0.0000],
        [2.5148, 1.3166, 0.0000],
        [1.2368, 0.6679, 0.0000],
        [-0.0491, -1.4032, 0.0000],
        [-1.2369, -0.6678, 0.0000],
        [0.0492, 1.4033, 0.0000],
        [-1.1871, 0.7528, 0.0000],
        [-2.5148, -1.3167, 0.0000],
        [-3.6609, -0.5848, 0.0000],
        [-3.6110, 0.8395, 0.0000],
        [-2.4165, 1.4895, 0.0000],
    ]

    h_positions = [
        [4.6397, 1.0755, 0.0000],
        [4.5529, -1.3980, 0.0000],
        [2.3680, -2.5843, 0.0000],
        [2.5432, 2.4122, 0.0000],
        [-0.0876, -2.4995, 0.0000],
        [0.0876, 2.4996, 0.0000],
        [-2.5431, -2.4122, 0.0000],
        [-4.6397, -1.0756, 0.0000],
        [-4.5531, 1.3975, 0.0000],
        [-2.3682, 2.5844, 0.0000],
    ]

    return np.array(c_positions), np.array(h_positions)


def _create_benzene() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create benzene (C6H6) - single aromatic ring.

    Returns (carbon_positions, hydrogen_positions) in local coordinates.
    """
    cc = BOND_LENGTHS['C-C_aromatic']
    ch = BOND_LENGTHS['C-H']

    c_positions = []
    h_positions = []

    for i in range(6):
        angle = i * np.pi / 3
        # Carbon
        x_c = cc * np.cos(angle)
        y_c = cc * np.sin(angle)
        c_positions.append([x_c, y_c, 0])

        # Hydrogen (radially outward)
        x_h = (cc + ch) * np.cos(angle)
        y_h = (cc + ch) * np.sin(angle)
        h_positions.append([x_h, y_h, 0])

    return np.array(c_positions), np.array(h_positions)


def create_fused_helicene_spiral(
    pillar_height: float,
    n_helix_turns: float = 5.0,
    base_rings_per_turn: int = 12,
    start_radius: float = 6.0,
    end_radius: float = 30.0
) -> Tuple[Atoms, List[Atoms]]:
    """
    Create chemically valid PAH molecule spiral around the tree.

    Uses real, stable PAH molecules (anthracene, naphthalene, benzene) arranged
    in a spiral pattern. Molecules are properly spaced to avoid overlap.

    Parameters:
        pillar_height: Height of the Fe pillar
        n_helix_turns: Number of complete helix rotations
        base_rings_per_turn: Base molecules per turn
        start_radius: Helix radius at top (small)
        end_radius: Helix radius at bottom (large)

    Returns:
        (complete_leaves, [layer_group1, layer_group2, ...])
    """
    cc = BOND_LENGTHS['C-C_aromatic']

    # Helix parameters
    start_z = pillar_height + 2.0
    end_z = 4.0

    all_c_positions = []
    all_h_positions = []
    group_list = []

    # Pre-create molecule templates
    anthracene_c, anthracene_h = _create_anthracene()
    naphthalene_c, naphthalene_h = _create_naphthalene()
    benzene_c, benzene_h = _create_benzene()

    # Molecule sizes (approximate diameter for spacing)
    anthracene_size = 9.5   # ~9.5 Å long (from reference coordinates)
    naphthalene_size = 7.0  # ~7.0 Å long
    benzene_size = 5.0      # ~5.0 Å diameter

    # Number of layers in the spiral (increase for denser coverage)
    n_layers = int(n_helix_turns * base_rings_per_turn * 2)  # Double the layers

    # Keep track of all placed molecule centers to avoid overlap
    placed_centers = []

    for layer_idx in range(n_layers):
        t = layer_idx / max(1, n_layers - 1)

        # Current height and radius
        z = start_z - t * (start_z - end_z)
        radius = start_radius + t * (end_radius - start_radius)

        # Choose molecule type based on radius (larger molecules at larger radius)
        if radius > 20:
            mol_c, mol_h = anthracene_c.copy(), anthracene_h.copy()
            mol_size = anthracene_size
        elif radius > 12:
            mol_c, mol_h = naphthalene_c.copy(), naphthalene_h.copy()
            mol_size = naphthalene_size
        else:
            mol_c, mol_h = benzene_c.copy(), benzene_h.copy()
            mol_size = benzene_size

        # Adaptive minimum spacing based on molecule size and 3D orientation
        # Molecules are tilted, so they don't overlap as much as flat placement
        min_spacing = mol_size * 0.55  # 55% of molecule size (tilted molecules overlap less)

        # Number of molecules around this layer (proportional to circumference)
        # Use tighter packing for denser coverage
        circumference = 2 * np.pi * radius
        n_mols = max(4, int(circumference / (mol_size * 0.9)))

        # Base angular position (helix rotation) with staggering
        # Alternate layers are offset by half the angular spacing
        angular_offset = (layer_idx % 2) * np.pi / n_mols  # Stagger every other layer
        base_theta = t * n_helix_turns * 2 * np.pi + angular_offset

        layer_c_positions = []
        layer_h_positions = []

        for mol_idx in range(n_mols):
            theta = base_theta + mol_idx * 2 * np.pi / n_mols

            # Molecule center position
            cx = radius * np.cos(theta)
            cy = radius * np.sin(theta)
            center = np.array([cx, cy, z])

            # Check if too close to existing molecules
            too_close = False
            for pc in placed_centers:
                if np.linalg.norm(center - pc) < min_spacing:
                    too_close = True
                    break

            if too_close:
                continue

            placed_centers.append(center)

            # Orient molecule: face outward with slight downward tilt
            radial = np.array([np.cos(theta), np.sin(theta), 0])
            tilt = 0.4  # radians downward
            normal = np.cos(tilt) * radial + np.sin(tilt) * np.array([0, 0, -1])
            normal = normal / np.linalg.norm(normal)

            # Tangent (perpendicular to radial in xy plane)
            tangent = np.array([-np.sin(theta), np.cos(theta), 0])

            # Binormal
            binormal = np.cross(normal, tangent)
            binormal = binormal / np.linalg.norm(binormal)
            tangent = np.cross(binormal, normal)

            # Rotation matrix
            R = np.column_stack([tangent, binormal, normal])

            # Transform molecule coordinates
            for c_local in mol_c:
                c_global = center + R @ c_local
                layer_c_positions.append(c_global)

            for h_local in mol_h:
                h_global = center + R @ h_local
                layer_h_positions.append(h_global)

        if layer_c_positions:
            all_c_positions.extend(layer_c_positions)
            all_h_positions.extend(layer_h_positions)

            # Create Atoms for this layer (for animation)
            n_c = len(layer_c_positions)
            n_h = len(layer_h_positions)
            layer_atoms = Atoms(
                symbols=['C'] * n_c + ['H'] * n_h,
                positions=layer_c_positions + layer_h_positions
            )
            group_list.append(layer_atoms)

    # Final collision check and cleanup
    all_c_positions, all_h_positions = _remove_close_atoms(
        all_c_positions, all_h_positions
    )

    # Combine all
    all_positions = all_c_positions + all_h_positions
    all_symbols = ['C'] * len(all_c_positions) + ['H'] * len(all_h_positions)

    complete_leaves = Atoms(symbols=all_symbols, positions=all_positions)

    return complete_leaves, group_list


def _remove_close_atoms(
    c_positions: List[np.ndarray],
    h_positions: List[np.ndarray],
    c_c_min: float = 1.2,
    h_h_min: float = 1.5,
    c_h_min: float = 0.9
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Remove atoms that violate minimum distance constraints.

    This handles any remaining overlaps from molecule placement.
    """
    if not c_positions:
        return c_positions, h_positions

    c_arr = np.array(c_positions)
    h_arr = np.array(h_positions) if h_positions else np.array([]).reshape(0, 3)

    # Check C-C distances
    keep_c = np.ones(len(c_arr), dtype=bool)
    for i in range(len(c_arr)):
        if not keep_c[i]:
            continue
        for j in range(i + 1, len(c_arr)):
            if keep_c[j]:
                dist = np.linalg.norm(c_arr[i] - c_arr[j])
                if dist < c_c_min:
                    keep_c[j] = False

    filtered_c = [c_positions[i] for i in range(len(c_positions)) if keep_c[i]]
    filtered_c_arr = np.array(filtered_c) if filtered_c else np.array([]).reshape(0, 3)

    # Check H-H and C-H distances
    keep_h = np.ones(len(h_arr), dtype=bool)
    for i in range(len(h_arr)):
        if not keep_h[i]:
            continue

        # H-H check
        for j in range(i + 1, len(h_arr)):
            if keep_h[j]:
                dist = np.linalg.norm(h_arr[i] - h_arr[j])
                if dist < h_h_min:
                    keep_h[j] = False

        # C-H check (H should be bonded to exactly one C at ~1.09 Å)
        if len(filtered_c_arr) > 0:
            c_dists = np.linalg.norm(filtered_c_arr - h_arr[i], axis=1)
            min_c_dist = np.min(c_dists)
            # If too close to any C (not its bonded C), remove
            # Bonded distance is ~1.09, so < 0.9 is definitely wrong
            if min_c_dist < c_h_min:
                keep_h[i] = False

    filtered_h = [h_positions[i] for i in range(len(h_positions)) if keep_h[i]]

    return filtered_c, filtered_h


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
# Part 5: Au55 Star Cluster
# =============================================================================
def create_au_star(center_z: float) -> Atoms:
    """
    Create Au55 cluster - a star-shaped gold nanoparticle (Mackay icosahedron).

    Parameters:
        center_z: Z-coordinate for the center of the cluster

    Returns:
        Atoms object with Au55 cluster
    """
    # Au55 cluster coordinates (Mackay icosahedron - magic number cluster)
    positions = [
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000003, 1.42118658, 2.29966110],
        [0.00000003, 1.42118672, -2.29966124],
        [0.00000003, -1.42118668, 2.29966118],
        [0.00000003, -1.42118681, -2.29966131],
        [1.42118658, 2.29966110, 0.00000003],
        [-1.42118668, 2.29966118, 0.00000003],
        [1.42118672, -2.29966124, 0.00000003],
        [-1.42118681, -2.29966131, 0.00000003],
        [2.29966110, 0.00000003, 1.42118658],
        [2.29966118, 0.00000003, -1.42118668],
        [-2.29966124, 0.00000003, 1.42118672],
        [-2.29966131, 0.00000003, -1.42118681],
        [-4.58474231, 0.00000015, -2.83354193],
        [0.00000015, 2.83354149, 4.58474200],
        [0.00000015, 2.83354183, -4.58474215],
        [0.00000015, -2.83354159, 4.58474217],
        [0.00000015, -2.83354193, -4.58474231],
        [2.83354149, 4.58474200, 0.00000015],
        [-2.83354159, 4.58474217, 0.00000015],
        [2.83354183, -4.58474215, 0.00000015],
        [-2.83354193, -4.58474231, 0.00000015],
        [4.58474200, 0.00000015, 2.83354149],
        [4.58474217, 0.00000015, -2.83354159],
        [-4.58474215, 0.00000015, 2.83354183],
        [-0.00000001, 4.68676501, 0.00000016],
        [-0.00000001, -4.68676522, 0.00000016],
        [-4.68676522, 0.00000016, -0.00000001],
        [4.68676501, 0.00000016, -0.00000001],
        [0.00000016, -0.00000001, -4.68676522],
        [0.00000016, -0.00000001, 4.68676501],
        [1.44822101, -3.79150916, 2.34326576],
        [-1.44822095, -3.79150921, 2.34326583],
        [1.44822103, -3.79150937, -2.34326561],
        [-1.44822098, -3.79150941, -2.34326568],
        [3.79150929, -2.34326559, -1.44822084],
        [3.79150924, -2.34326552, 1.44822089],
        [-3.79150941, -2.34326568, -1.44822098],
        [-3.79150937, -2.34326561, 1.44822103],
        [1.44822089, 3.79150924, -2.34326552],
        [1.44822087, 3.79150903, 2.34326566],
        [-1.44822084, 3.79150929, -2.34326559],
        [-1.44822082, 3.79150908, 2.34326573],
        [-2.34326561, 1.44822103, -3.79150937],
        [2.34326576, 1.44822101, -3.79150916],
        [-2.34326568, -1.44822098, -3.79150941],
        [2.34326583, -1.44822095, -3.79150921],
        [-3.79150916, 2.34326576, 1.44822101],
        [-3.79150921, 2.34326583, -1.44822095],
        [-2.34326559, -1.44822084, 3.79150929],
        [2.34326573, -1.44822082, 3.79150908],
        [2.34326566, 1.44822087, 3.79150903],
        [-2.34326552, 1.44822089, 3.79150924],
        [3.79150908, 2.34326573, -1.44822082],
        [3.79150903, 2.34326566, 1.44822087],
    ]

    # Shift to center_z position
    positions = np.array(positions)
    positions[:, 2] += center_z

    return Atoms('Au' * 55, positions=positions.tolist())


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
    print("\n[Stage 3] Placing Au55 star on top...")
    au_star = create_au_star(center_z=pillar_height + 3.5)
    current_atoms += au_star
    traj.write(current_atoms)
    print(f"  Added {len(au_star)} Au atoms (star cluster)")

    # ----- Stage 4: Carbon Leaves (Fused helicene spiral) -----
    print("\n[Stage 4] Growing fused helicene spiral leaves...")
    leaves, layer_groups = create_fused_helicene_spiral(
        pillar_height=pillar_height,
        n_helix_turns=config.n_helix_turns,
        base_rings_per_turn=config.n_rings_per_turn,
        start_radius=config.helix_start_radius,
        end_radius=config.helix_end_radius
    )

    for i, layer in enumerate(layer_groups):
        current_atoms += layer
        traj.write(current_atoms)
        n_c = sum(1 for s in layer.get_chemical_symbols() if s == 'C')
        n_rings = n_c // 6
        if (i + 1) % 10 == 0 or i == len(layer_groups) - 1:
            print(f"  Layer {i+1}/{len(layer_groups)}: {n_rings} fused rings")

    # ----- Stage 5: Halogen Decorations -----
    print("\n[Stage 5] Adding halogen decorations...")
    decorated_atoms, substitutions = add_halogen_decorations(
        current_atoms,
        n_substitutions=config.n_decorations_per_halogen,
        random_seed=config.random_seed
    )

    # Group substitutions by 10
    for i in range(0, len(substitutions), 10):
        batch = substitutions[i:i+10]
        for idx, halogen in batch:
            current_atoms[idx].symbol = halogen
            current_atoms.positions[idx] = decorated_atoms.positions[idx]
        traj.write(current_atoms)
        print(f"  Batch {i//10 + 1}: Substituted {len(batch)} atoms")

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
