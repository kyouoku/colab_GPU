"""
spv_gpu: FDTDX (JAX/GPU) + Harminv CLI package for optical cavity mode analysis.
"""

from .config import (
    DEFAULT_BACKEND,
    DEFAULT_CFG,
    derive_parameters,
    estimate_gpu_memory_gb,
    get_default_config,
    min_feature,
)

__all__ = [
    "DEFAULT_CFG",
    "DEFAULT_BACKEND",
    "get_default_config",
    "min_feature",
    "derive_parameters",
    "estimate_gpu_memory_gb",
]
