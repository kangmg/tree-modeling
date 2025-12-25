"""
xmastree - Atomic Christmas Tree Generator

A chemically accurate atomic-scale Christmas tree model using ASE.

Components:
- Ag simple cubic slab (base)
- Fe HCP pillar (trunk)
- PAH molecules spiral (leaves: benzene, naphthalene, anthracene)
- Halogen decorations (I, Cl, Br)
- Au55 Mackay icosahedral cluster (star)
"""

from .tree import TreeConfig, build_christmas_tree, create_twinkling_trajectory
from .viewer import viewer, ATOM_COLORS, ATOM_RADII

__version__ = "1.0.0"
__all__ = [
    "TreeConfig",
    "build_christmas_tree",
    "create_twinkling_trajectory",
    "viewer",
    "ATOM_COLORS",
    "ATOM_RADII",
]
