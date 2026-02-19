"""
molscreen: SMILES-based drug candidate screening tool with RDKit + ML

This package provides:
- Molecular property calculations using RDKit
- Solubility prediction using QSAR models
- Lipinski's Rule of Five compliance checking
- Molecular similarity search and diversity picking
- HTML/JSON report generation
- CLI and Python API interfaces
"""

__version__ = "0.1.0"
__author__ = "Autonomous Agent"

from molscreen.properties import (
    calculate_properties,
    check_lipinski,
)
from molscreen.similarity import (
    compute_fingerprint,
    tanimoto_similarity,
    rank_by_similarity,
    find_diverse_subset,
)

__all__ = [
    "calculate_properties",
    "check_lipinski",
    "compute_fingerprint",
    "tanimoto_similarity",
    "rank_by_similarity",
    "find_diverse_subset",
]
