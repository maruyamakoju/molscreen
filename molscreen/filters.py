"""
Pharmacokinetic filter module for drug-likeness screening.

This module provides filters for assessing drug-likeness and toxicity risks:
- Lipinski's Rule of Five (oral bioavailability)
- Veber rules (oral bioavailability)
- PAINS (Pan-Assay Interference Compounds) detection
"""

from typing import List, Dict, Any, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, FilterCatalog

from molscreen.properties import smiles_to_mol, calculate_properties, MoleculeError


def lipinski_filter(smiles: str) -> Dict[str, Any]:
    """
    Apply Lipinski's Rule of Five filter.

    Lipinski's Rule of Five criteria for oral bioavailability:
    - Molecular weight ≤ 500 Da
    - LogP ≤ 5
    - Hydrogen bond donors ≤ 5
    - Hydrogen bond acceptors ≤ 10

    Args:
        smiles: SMILES representation of the molecule

    Returns:
        Dictionary with:
        - passes (bool): True if all criteria pass
        - violations (List[str]): List of violated criteria
        - values (Dict[str, float]): Calculated descriptor values

    Raises:
        MoleculeError: If SMILES is invalid

    Example:
        >>> result = lipinski_filter("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
        >>> result['passes']
        True
        >>> result = lipinski_filter("C" * 100)  # Very large molecule
        >>> result['passes']
        False
        >>> "MW > 500" in result['violations']
        True
    """
    # Calculate properties
    props = calculate_properties(smiles)

    # Check each criterion
    violations = []
    if props['MW'] > 500:
        violations.append("MW > 500")
    if props['LogP'] > 5:
        violations.append("LogP > 5")
    if props['HBD'] > 5:
        violations.append("HBD > 5")
    if props['HBA'] > 10:
        violations.append("HBA > 10")

    return {
        'passes': len(violations) == 0,
        'violations': violations,
        'values': {
            'MW': props['MW'],
            'LogP': props['LogP'],
            'HBD': props['HBD'],
            'HBA': props['HBA']
        }
    }


def veber_filter(smiles: str) -> Dict[str, Any]:
    """
    Apply Veber rules for oral bioavailability.

    Veber criteria:
    - Topological polar surface area (TPSA) ≤ 140 Ų
    - Number of rotatable bonds ≤ 10

    Args:
        smiles: SMILES representation of the molecule

    Returns:
        Dictionary with:
        - passes (bool): True if all criteria pass
        - violations (List[str]): List of violated criteria
        - values (Dict[str, float]): Calculated descriptor values

    Raises:
        MoleculeError: If SMILES is invalid

    Example:
        >>> result = veber_filter("CCO")  # Ethanol
        >>> result['passes']
        True
        >>> result['values']['TPSA']
        20.23
        >>> result['values']['RotatableBonds']
        0
    """
    mol = smiles_to_mol(smiles)

    # Calculate Veber descriptors
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)

    # Check criteria
    violations = []
    if tpsa > 140:
        violations.append("TPSA > 140")
    if rot_bonds > 10:
        violations.append("RotatableBonds > 10")

    return {
        'passes': len(violations) == 0,
        'violations': violations,
        'values': {
            'TPSA': tpsa,
            'RotatableBonds': rot_bonds
        }
    }


def pains_filter(smiles: str) -> Dict[str, Any]:
    """
    Detect PAINS (Pan-Assay Interference Compounds) using RDKit FilterCatalog.

    PAINS are compounds that often show false positives in biological assays
    due to their reactive or promiscuous nature.

    Args:
        smiles: SMILES representation of the molecule

    Returns:
        Dictionary with:
        - passes (bool): True if no PAINS alerts detected
        - violations (List[str]): List of PAINS alert descriptions
        - values (Dict[str, int]): Number of PAINS alerts detected

    Raises:
        MoleculeError: If SMILES is invalid

    Example:
        >>> result = pains_filter("CCO")  # Ethanol - clean compound
        >>> result['passes']
        True
        >>> result['values']['num_pains_alerts']
        0
    """
    mol = smiles_to_mol(smiles)

    # Initialize PAINS filter catalog
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)

    # Check for PAINS alerts
    matches = catalog.GetMatches(mol)

    violations = []
    for match in matches:
        violations.append(match.GetDescription())

    return {
        'passes': len(violations) == 0,
        'violations': violations,
        'values': {
            'num_pains_alerts': len(violations)
        }
    }


def filter_molecules(
    smiles_list: List[str],
    rules: List[str] = ["lipinski", "veber", "pains"]
) -> List[Dict[str, Any]]:
    """
    Apply multiple filters to a list of molecules.

    Args:
        smiles_list: List of SMILES strings
        rules: List of filter names to apply. Valid values:
               "lipinski", "veber", "pains", or "all" for all filters

    Returns:
        List of dictionaries, one per molecule, containing:
        - smiles (str): Input SMILES string
        - valid (bool): Whether SMILES was valid
        - error (str, optional): Error message if SMILES invalid
        - filters (Dict[str, Dict]): Results for each filter applied
        - overall_pass (bool): True if all filters pass (only for valid molecules)

    Example:
        >>> results = filter_molecules(
        ...     ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "INVALID"],
        ...     rules=["lipinski", "veber"]
        ... )
        >>> len(results)
        3
        >>> results[0]['valid']
        True
        >>> results[0]['overall_pass']
        True
        >>> results[2]['valid']
        False
    """
    # Normalize rules
    if "all" in rules:
        rules = ["lipinski", "veber", "pains"]

    # Map rule names to functions
    filter_map = {
        "lipinski": lipinski_filter,
        "veber": veber_filter,
        "pains": pains_filter
    }

    results = []

    for smiles in smiles_list:
        result = {
            'smiles': smiles,
            'valid': True,
            'filters': {},
            'overall_pass': True
        }

        try:
            # Apply each filter
            for rule in rules:
                if rule not in filter_map:
                    raise ValueError(f"Unknown filter rule: {rule}")

                filter_result = filter_map[rule](smiles)
                result['filters'][rule] = filter_result

                # Update overall pass status
                if not filter_result['passes']:
                    result['overall_pass'] = False

        except MoleculeError as e:
            # Invalid SMILES
            result['valid'] = False
            result['error'] = str(e)
            result['overall_pass'] = False

        except Exception as e:
            # Other errors
            result['valid'] = False
            result['error'] = f"Error processing molecule: {str(e)}"
            result['overall_pass'] = False

        results.append(result)

    return results
