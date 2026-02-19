"""
molscreen: SMILES-based drug candidate screening tool with RDKit + ML

This package provides:
- Molecular property calculations using RDKit
- Solubility prediction using QSAR models
- Lipinski's Rule of Five compliance checking
- Scaffold analysis and diversity scoring
- HTML/JSON report generation
- CLI and Python API interfaces
"""

__version__ = "0.1.0"
__author__ = "Autonomous Agent"

from molscreen.properties import (
    calculate_properties,
    check_lipinski,
)
from molscreen.scaffold import (
    get_murcko_scaffold,
    get_generic_scaffold,
    group_by_scaffold,
    scaffold_diversity_score,
)

__all__ = [
    "calculate_properties",
    "check_lipinski",
    "get_murcko_scaffold",
    "get_generic_scaffold",
    "group_by_scaffold",
    "scaffold_diversity_score",
]
