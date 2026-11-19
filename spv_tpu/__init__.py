"""
spv_tpu: TPU-optimized JAX FDTD + Harminv CLI package for optical cavity mode analysis.
"""

from .config import (
    DEFAULT_BACKEND,
    DEFAULT_CFG,
    align_grid_for_tpu,
    derive_parameters,
    detect_available_gpu_memory_gb,
    detect_available_tpu_memory_gb,
    detect_tpu_hardware,
    estimate_gpu_memory_gb,
    estimate_tpu_memory_gb,
    get_default_config,
    min_feature,
)

__all__ = [
    "DEFAULT_CFG",
    "DEFAULT_BACKEND",
    "get_default_config",
    "min_feature",
    "derive_parameters",
    "align_grid_for_tpu",
    "detect_tpu_hardware",
    "detect_available_tpu_memory_gb",
    "detect_available_gpu_memory_gb",
    "estimate_tpu_memory_gb",
    "estimate_gpu_memory_gb",
]
