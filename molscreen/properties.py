"""
Molecular property calculations using RDKit.

This module provides functions to:
- Parse SMILES strings
- Calculate basic molecular properties (MW, logP, HBD, HBA)
- Check Lipinski's Rule of Five compliance
"""

from typing import Dict, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


class MoleculeError(Exception):
    """Exception raised for invalid molecules."""
    pass


def smiles_to_mol(smiles: str) -> Chem.Mol:
    """
    Convert SMILES string to RDKit molecule object.

    Args:
        smiles: SMILES representation of the molecule

    Returns:
        RDKit Mol object

    Raises:
        MoleculeError: If SMILES is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise MoleculeError(f"Invalid SMILES: {smiles}")
    return mol


def calculate_properties(smiles: str) -> Dict[str, float]:
    """
    Calculate basic molecular properties from SMILES.

    Properties calculated:
    - MW: Molecular weight (g/mol)
    - LogP: Octanol-water partition coefficient
    - HBD: Number of hydrogen bond donors
    - HBA: Number of hydrogen bond acceptors

    Args:
        smiles: SMILES representation of the molecule

    Returns:
        Dictionary containing molecular properties

    Raises:
        MoleculeError: If SMILES is invalid

    Example:
        >>> props = calculate_properties("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
        >>> props['MW']
        180.159...
    """
    mol = smiles_to_mol(smiles)

    properties = {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'HBA': Lipinski.NumHAcceptors(mol),
    }

    return properties


def check_lipinski(smiles: Optional[str] = None,
                   properties: Optional[Dict[str, float]] = None) -> Dict[str, bool]:
    """
    Check Lipinski's Rule of Five compliance.

    Lipinski's Rule of Five states that a drug-like molecule should have:
    - Molecular weight <= 500 Da
    - LogP <= 5
    - Hydrogen bond donors <= 5
    - Hydrogen bond acceptors <= 10

    Args:
        smiles: SMILES representation (required if properties not provided)
        properties: Pre-calculated properties (if None, will calculate from SMILES)

    Returns:
        Dictionary with:
        - Individual rule compliance (MW_ok, LogP_ok, HBD_ok, HBA_ok)
        - Overall compliance (passes_lipinski)

    Raises:
        ValueError: If neither smiles nor properties are provided
        MoleculeError: If SMILES is invalid

    Example:
        >>> result = check_lipinski("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
        >>> result['passes_lipinski']
        True
    """
    if properties is None:
        if smiles is None:
            raise ValueError("Either smiles or properties must be provided")
        properties = calculate_properties(smiles)

    # Check each Lipinski criterion
    rules = {
        'MW_ok': properties['MW'] <= 500,
        'LogP_ok': properties['LogP'] <= 5,
        'HBD_ok': properties['HBD'] <= 5,
        'HBA_ok': properties['HBA'] <= 10,
    }

    # Overall compliance: all rules must pass
    rules['passes_lipinski'] = all([
        rules['MW_ok'],
        rules['LogP_ok'],
        rules['HBD_ok'],
        rules['HBA_ok']
    ])

    return rules


def get_molecule_summary(smiles: str) -> Dict:
    """
    Get a comprehensive summary of molecular properties and Lipinski compliance.

    Args:
        smiles: SMILES representation of the molecule

    Returns:
        Dictionary containing:
        - smiles: Input SMILES string
        - properties: Molecular properties
        - lipinski: Lipinski rule compliance

    Raises:
        MoleculeError: If SMILES is invalid

    Example:
        >>> summary = get_molecule_summary("CC(=O)Oc1ccccc1C(=O)O")
        >>> summary['lipinski']['passes_lipinski']
        True
    """
    properties = calculate_properties(smiles)
    lipinski_results = check_lipinski(properties=properties)

    return {
        'smiles': smiles,
        'properties': properties,
        'lipinski': lipinski_results
    }
