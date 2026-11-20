"""ConsensusAlign: pooled unpaired alignment and native encoder interfaces."""

import os

# Set before importing torch, as in the numerical solver's public release.
os.environ.setdefault("MKL_CBWR", "AVX2,STRICT")
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")

from .core import ConsensusAlign, NativeMap, WPConfig
from .data import load_features

__all__ = ["ConsensusAlign", "NativeMap", "WPConfig", "load_features"]
