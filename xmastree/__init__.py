"""
xmastree - Atomic Christmas Tree Generator

A chemically accurate atomic-scale Christmas tree model using ASE.

Components:
- Ag simple cubic slab (base)
- Fe HCP pillar (trunk)
- Fused helicene spiral (leaves)
- Halogen decorations (F, Cl, Br)
- Au13 icosahedral cluster (star)
"""

from .tree import TreeConfig, build_christmas_tree
from .viewer import viewer, ATOM_COLORS, ATOM_RADII

__version__ = "1.0.0"
__all__ = [
    "TreeConfig",
    "build_christmas_tree",
    "viewer",
    "ATOM_COLORS",
    "ATOM_RADII",
]
