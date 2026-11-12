"""
Geometry subpackage: JAX-accelerated SDF primitives, subpixel smoothing, and structure definitions.
"""

from .shapes import (
    sdf_box,
    sdf_cylinder_z,
    sdf_difference,
    sdf_extruded_polygon_z,
    sdf_extruded_star_polygon_z,
    sdf_extruded_triangle_z,
    sdf_intersection,
    sdf_slab_z,
    sdf_union,
)
from .smoothing import evaluate_permittivity_grid
from .structures import build_structure_permittivity, get_structure_prefix

__all__ = [
    "sdf_cylinder_z",
    "sdf_slab_z",
    "sdf_box",
    "sdf_extruded_triangle_z",
    "sdf_extruded_polygon_z",
    "sdf_extruded_star_polygon_z",
    "sdf_union",
    "sdf_intersection",
    "sdf_difference",
    "evaluate_permittivity_grid",
    "build_structure_permittivity",
    "get_structure_prefix",
]
