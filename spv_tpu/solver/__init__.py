"""
Solver subpackage: Mirror symmetry handling and TPU-optimized 3D FDTD runner.
"""

from .symmetry import (
    apply_e_mirror_bc,
    apply_e_mirror_bc_zyx,
    apply_h_mirror_bc,
    apply_h_mirror_bc_zyx,
    build_fdtdx_boundary_override,
    expand_sources_for_full_domain_symmetry,
    parity_label,
    unfold_2d_slice_z,
)
from .tpu_runner import (
    FDTDXSimulationResult,
    TPUSimulationResult,
    check_gpu_environment,
    check_tpu_environment,
    clear_jit_cache,
    configure_tpu_xla_cache,
    run_fdtdx_simulation,
    run_tpu_simulation,
)

__all__ = [
    "TPUSimulationResult",
    "FDTDXSimulationResult",
    "check_tpu_environment",
    "check_gpu_environment",
    "clear_jit_cache",
    "configure_tpu_xla_cache",
    "run_tpu_simulation",
    "run_fdtdx_simulation",
    "apply_e_mirror_bc",
    "apply_e_mirror_bc_zyx",
    "apply_h_mirror_bc",
    "apply_h_mirror_bc_zyx",
    "build_fdtdx_boundary_override",
    "expand_sources_for_full_domain_symmetry",
    "parity_label",
    "unfold_2d_slice_z",
]
