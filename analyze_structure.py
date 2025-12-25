"""
Analyze the chemical validity of the Christmas tree structure.
Check for:
1. Too-close atom pairs (< 1.0 Å)
2. Proper bond distances
3. Ring geometry
"""

import numpy as np
from xmastree import TreeConfig, build_christmas_tree
from xmastree.tree import create_fused_helicene_spiral
from collections import Counter

# Minimum allowed distances (Angstroms)
MIN_DISTANCES = {
    ('C', 'C'): 1.20,   # Single C-C bond is ~1.54, aromatic ~1.40
    ('C', 'H'): 0.90,   # C-H bond ~1.09
    ('H', 'H'): 1.50,   # Non-bonded H-H should be > 2.0
    'default': 0.80     # Absolute minimum
}

def get_min_distance(sym1, sym2):
    key = tuple(sorted([sym1, sym2]))
    return MIN_DISTANCES.get(key, MIN_DISTANCES['default'])

def analyze_close_contacts(atoms, threshold=1.5):
    """Find all atom pairs closer than threshold."""
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    n = len(atoms)

    close_contacts = []

    # Check all pairs (slow but thorough for analysis)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            min_allowed = get_min_distance(symbols[i], symbols[j])

            if dist < threshold:
                close_contacts.append({
                    'i': i, 'j': j,
                    'sym_i': symbols[i], 'sym_j': symbols[j],
                    'distance': dist,
                    'min_allowed': min_allowed,
                    'is_violation': dist < min_allowed
                })

    return close_contacts

def analyze_leaves_only():
    """Analyze just the leaves structure for issues."""
    print("=" * 60)
    print("ANALYZING HELICENE SPIRAL LEAVES STRUCTURE")
    print("=" * 60)

    # Create just the leaves
    leaves, _ = create_fused_helicene_spiral(
        pillar_height=60.0,
        n_helix_turns=5.0,
        base_rings_per_turn=12,
        start_radius=6.0,
        end_radius=30.0
    )

    symbols = leaves.get_chemical_symbols()
    n_c = sum(1 for s in symbols if s == 'C')
    n_h = sum(1 for s in symbols if s == 'H')

    print(f"\nLeaves composition:")
    print(f"  Carbon atoms: {n_c}")
    print(f"  Hydrogen atoms: {n_h}")
    print(f"  Expected rings: {n_c // 6}")
    print(f"  H per ring: {n_h / (n_c / 6):.1f} (should be 4 for fused, 6 for isolated)")

    # Check for close contacts
    print(f"\nChecking for close contacts (< 1.5 Å)...")
    contacts = analyze_close_contacts(leaves, threshold=1.5)

    if not contacts:
        print("  No close contacts found!")
    else:
        violations = [c for c in contacts if c['is_violation']]
        print(f"  Found {len(contacts)} pairs < 1.5 Å")
        print(f"  Of these, {len(violations)} are violations (below min allowed distance)")

        # Group by atom type
        cc_violations = [c for c in violations if c['sym_i'] == 'C' and c['sym_j'] == 'C']
        ch_violations = [c for c in violations if 'H' in (c['sym_i'], c['sym_j']) and 'C' in (c['sym_i'], c['sym_j'])]
        hh_violations = [c for c in violations if c['sym_i'] == 'H' and c['sym_j'] == 'H']

        print(f"\n  C-C violations: {len(cc_violations)}")
        print(f"  C-H violations: {len(ch_violations)}")
        print(f"  H-H violations: {len(hh_violations)}")

        # Show some examples
        if violations:
            print(f"\n  Example violations:")
            for v in violations[:10]:
                print(f"    Atoms {v['i']}-{v['j']} ({v['sym_i']}-{v['sym_j']}): {v['distance']:.3f} Å (min: {v['min_allowed']:.2f} Å)")

    # Check aromatic C-C bond distances
    print("\nAnalyzing C-C distances (should be ~1.40 Å for aromatic bonds)...")
    positions = leaves.get_positions()
    c_indices = [i for i, s in enumerate(symbols) if s == 'C']

    cc_bonds = []
    for i, ci in enumerate(c_indices):
        for cj in c_indices[i+1:]:
            dist = np.linalg.norm(positions[ci] - positions[cj])
            if 1.2 < dist < 1.6:  # Likely a C-C bond
                cc_bonds.append(dist)

    if cc_bonds:
        print(f"  C-C bond distances in range [1.2, 1.6] Å:")
        print(f"    Count: {len(cc_bonds)}")
        print(f"    Mean: {np.mean(cc_bonds):.3f} Å")
        print(f"    Min: {np.min(cc_bonds):.3f} Å")
        print(f"    Max: {np.max(cc_bonds):.3f} Å")
        print(f"    Std: {np.std(cc_bonds):.3f} Å")

    return leaves, contacts

if __name__ == "__main__":
    leaves, contacts = analyze_leaves_only()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    violations = [c for c in contacts if c['is_violation']]
    if violations:
        print(f"⚠️  FOUND {len(violations)} CHEMICAL VALIDITY VIOLATIONS")
        print("   The current implementation has overlapping atoms!")
        print("\n   Key issues:")
        print("   1. Benzene rings are placed independently, not truly fused")
        print("   2. Multiple rings at same height can overlap")
        print("   3. No collision detection between rings")
    else:
        print("✓ No violations found")
