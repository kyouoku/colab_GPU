"""
Solver subpackage: Mirror symmetry handling and FDTDX (JAX/GPU) simulation runner.
"""

from .fdtdx_runner import (
    FDTDXSimulationResult,
    check_gpu_environment,
    clear_jit_cache,
    run_fdtdx_simulation,
)
from .symmetry import (
    apply_e_mirror_bc,
    apply_h_mirror_bc,
    build_fdtdx_boundary_override,
    expand_sources_for_full_domain_symmetry,
    parity_label,
    unfold_2d_slice_z,
)

__all__ = [
    "FDTDXSimulationResult",
    "check_gpu_environment",
    "clear_jit_cache",
    "run_fdtdx_simulation",
    "apply_e_mirror_bc",
    "apply_h_mirror_bc",
    "build_fdtdx_boundary_override",
    "expand_sources_for_full_domain_symmetry",
    "parity_label",
    "unfold_2d_slice_z",
]
