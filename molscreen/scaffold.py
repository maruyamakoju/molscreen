"""
Scaffold analysis module for molecular structure analysis.

This module provides functions to:
- Extract Bemis-Murcko scaffolds from molecules
- Generate generic scaffolds (all atoms converted to carbon)
- Group molecules by scaffold structure
- Calculate scaffold diversity scores
"""

from typing import List, Dict, Optional
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from molscreen.properties import MoleculeError


def get_murcko_scaffold(smiles: str) -> Optional[str]:
    """
    Extract Bemis-Murcko scaffold from a molecule.

    The Bemis-Murcko scaffold is the core ring system and linker atoms
    of a molecule, with all side chains removed.

    Args:
        smiles: SMILES representation of the molecule

    Returns:
        Scaffold SMILES string, or None if:
        - SMILES is invalid
        - Molecule has no scaffold (acyclic molecule)

    Example:
        >>> get_murcko_scaffold("Cc1ccccc1")  # Toluene
        'c1ccccc1'
        >>> get_murcko_scaffold("CCO")  # Ethanol (acyclic)
        None
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)

        # Empty string means no scaffold (acyclic molecule)
        if not scaffold_smiles:
            return None

        return scaffold_smiles

    except Exception:
        # Handle any RDKit errors gracefully
        return None


def get_generic_scaffold(smiles: str) -> Optional[str]:
    """
    Extract generic scaffold with all atoms converted to carbon.

    Generic scaffolds replace all heteroatoms with carbon, making it
    easier to identify structurally similar molecules regardless of
    heteroatom substitutions.

    Args:
        smiles: SMILES representation of the molecule

    Returns:
        Generic scaffold SMILES string, or None if:
        - SMILES is invalid
        - Molecule has no scaffold (acyclic molecule)

    Example:
        >>> get_generic_scaffold("c1ccccc1")  # Benzene
        'C1CCCCC1'
        >>> get_generic_scaffold("c1ncccc1")  # Pyridine
        'C1CCCCC1'
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)

        # Empty string means no scaffold (acyclic molecule)
        if not scaffold_smiles:
            return None

        # Make generic (all atoms -> carbon)
        generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        generic_smiles = Chem.MolToSmiles(generic)

        return generic_smiles

    except Exception:
        # Handle any RDKit errors gracefully
        return None


def group_by_scaffold(smiles_list: List[str]) -> Dict[str, List[str]]:
    """
    Group molecules by their Bemis-Murcko scaffold.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        Dictionary mapping scaffold SMILES to list of molecule SMILES.
        Molecules without scaffolds (acyclic) are grouped under 'no_scaffold' key.
        Invalid SMILES are skipped.

    Example:
        >>> group_by_scaffold(["Cc1ccccc1", "c1ccccc1", "CCO"])
        {'c1ccccc1': ['Cc1ccccc1', 'c1ccccc1'], 'no_scaffold': ['CCO']}
    """
    groups: Dict[str, List[str]] = {}

    for smiles in smiles_list:
        scaffold = get_murcko_scaffold(smiles)

        if scaffold is None:
            # Check if molecule is invalid or just acyclic
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # Invalid SMILES - skip it
                continue
            else:
                # Valid but acyclic molecule
                key = 'no_scaffold'
        else:
            key = scaffold

        if key not in groups:
            groups[key] = []
        groups[key].append(smiles)

    return groups


def scaffold_diversity_score(smiles_list: List[str]) -> float:
    """
    Calculate scaffold diversity score for a set of molecules.

    The diversity score is the ratio of unique scaffolds to total molecules:
    - 0.0: All molecules have the same scaffold (low diversity)
    - 1.0: All molecules have different scaffolds (high diversity)

    Args:
        smiles_list: List of SMILES strings

    Returns:
        Diversity score between 0.0 and 1.0
        Returns 0.0 for empty lists or all invalid molecules

    Example:
        >>> scaffold_diversity_score(["Cc1ccccc1", "CCc1ccccc1"])  # Same scaffold
        0.5
        >>> scaffold_diversity_score(["c1ccccc1", "C1CCCCC1"])  # Different scaffolds
        1.0
    """
    if not smiles_list:
        return 0.0

    # Get scaffolds for all valid molecules
    scaffolds = []
    for smiles in smiles_list:
        scaffold = get_murcko_scaffold(smiles)

        # Include acyclic molecules (None) as a scaffold type
        # But skip invalid molecules
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Use 'no_scaffold' as placeholder for None
            scaffolds.append(scaffold if scaffold is not None else 'no_scaffold')

    # Handle case where all molecules are invalid
    if not scaffolds:
        return 0.0

    # Calculate diversity: unique scaffolds / total valid molecules
    unique_scaffolds = len(set(scaffolds))
    diversity = unique_scaffolds / len(scaffolds)

    return diversity
