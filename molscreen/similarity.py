"""
Molecular similarity search using RDKit fingerprints.

This module provides:
- Morgan fingerprint computation
- Tanimoto similarity calculation
- Similarity-based ranking
- Diversity subset selection (MaxMin picking)
"""

from typing import List, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from molscreen.properties import MoleculeError, smiles_to_mol


def compute_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """
    Compute Morgan fingerprint for a molecule.

    Morgan fingerprints (also known as circular fingerprints) are a common
    representation used for molecular similarity calculations.

    Args:
        smiles: SMILES representation of the molecule
        radius: Radius of the Morgan fingerprint (default: 2)
        n_bits: Number of bits in the fingerprint (default: 2048)

    Returns:
        RDKit ExplicitBitVect fingerprint object

    Raises:
        MoleculeError: If SMILES is invalid

    Example:
        >>> fp = compute_fingerprint("CCO")  # Ethanol
        >>> fp.GetNumBits()
        2048
    """
    mol = smiles_to_mol(smiles)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return fingerprint


def tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """
    Calculate Tanimoto similarity between two molecules.

    The Tanimoto coefficient measures the similarity between two molecules
    based on their fingerprints. Returns a value between 0.0 (completely
    different) and 1.0 (identical).

    Args:
        smiles1: SMILES representation of the first molecule
        smiles2: SMILES representation of the second molecule

    Returns:
        Tanimoto similarity coefficient (0.0-1.0)

    Raises:
        MoleculeError: If either SMILES is invalid

    Example:
        >>> similarity = tanimoto_similarity("CCO", "CCO")  # Same molecule
        >>> similarity
        1.0
        >>> similarity = tanimoto_similarity("CCO", "CCCO")  # Similar
        >>> 0.5 < similarity < 1.0
        True
    """
    fp1 = compute_fingerprint(smiles1)
    fp2 = compute_fingerprint(smiles2)
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return float(similarity)


def rank_by_similarity(
    query_smiles: str,
    candidates: List[str],
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Rank candidate molecules by similarity to a query molecule.

    Computes the Tanimoto similarity between the query molecule and each
    candidate, then returns the top_k most similar candidates sorted by
    similarity score in descending order.

    Args:
        query_smiles: SMILES representation of the query molecule
        candidates: List of candidate SMILES strings
        top_k: Number of top results to return (default: 10)

    Returns:
        List of (smiles, similarity_score) tuples, sorted by score descending.
        If top_k exceeds the number of candidates, returns all candidates.

    Raises:
        MoleculeError: If query_smiles is invalid

    Example:
        >>> query = "CCO"
        >>> candidates = ["CCCO", "CC", "c1ccccc1"]
        >>> results = rank_by_similarity(query, candidates, top_k=2)
        >>> len(results) <= 2
        True
        >>> results[0][1] >= results[1][1]  # Sorted descending
        True
    """
    if not candidates:
        return []

    # Compute query fingerprint
    query_fp = compute_fingerprint(query_smiles)

    # Compute similarities for all candidates
    similarities = []
    for candidate in candidates:
        try:
            candidate_fp = compute_fingerprint(candidate)
            similarity = DataStructs.TanimotoSimilarity(query_fp, candidate_fp)
            similarities.append((candidate, float(similarity)))
        except MoleculeError:
            # Skip invalid SMILES in candidates
            continue

    # Sort by similarity (descending) and take top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def find_diverse_subset(
    smiles_list: List[str],
    n: int = 10,
    threshold: float = 0.4
) -> List[str]:
    """
    Select a diverse subset of molecules using MaxMin diversity picking.

    The MaxMin algorithm iteratively selects molecules that are maximally
    different from those already selected. This ensures good coverage of
    chemical space.

    Args:
        smiles_list: List of SMILES strings to select from
        n: Maximum number of molecules to select (default: 10)
        threshold: Minimum similarity threshold - molecules with similarity
                  below this value are considered diverse (default: 0.4)

    Returns:
        List of SMILES strings representing the diverse subset.
        May return fewer than n molecules if the input is smaller or if
        all remaining molecules are too similar to those already selected.

    Raises:
        MoleculeError: If any SMILES in the input is invalid

    Example:
        >>> molecules = ["CCO", "CCCO", "c1ccccc1", "CC(C)C"]
        >>> diverse = find_diverse_subset(molecules, n=2)
        >>> len(diverse) <= 2
        True
    """
    if not smiles_list:
        return []

    if len(smiles_list) <= n:
        # If we have fewer molecules than requested, return all
        # (after validating they're all valid SMILES)
        for smiles in smiles_list:
            smiles_to_mol(smiles)  # Validate
        return smiles_list[:]

    # Compute all fingerprints
    fingerprints = []
    valid_smiles = []
    for smiles in smiles_list:
        try:
            fp = compute_fingerprint(smiles)
            fingerprints.append(fp)
            valid_smiles.append(smiles)
        except MoleculeError:
            # Skip invalid SMILES
            continue

    if not fingerprints:
        return []

    # MaxMin diversity picking
    selected_indices = []
    remaining_indices = list(range(len(fingerprints)))

    # Start with first molecule (arbitrary choice)
    selected_indices.append(remaining_indices.pop(0))

    # Iteratively add most diverse molecules
    while len(selected_indices) < n and remaining_indices:
        max_min_similarity = -1
        best_idx_in_remaining = -1

        # For each remaining molecule, find minimum similarity to selected set
        for i, rem_idx in enumerate(remaining_indices):
            min_similarity_to_selected = min(
                DataStructs.TanimotoSimilarity(
                    fingerprints[rem_idx],
                    fingerprints[sel_idx]
                )
                for sel_idx in selected_indices
            )

            # Track the molecule with maximum minimum similarity
            if min_similarity_to_selected > max_min_similarity:
                max_min_similarity = min_similarity_to_selected
                best_idx_in_remaining = i

        # Check if the most diverse remaining molecule meets threshold
        if max_min_similarity >= threshold:
            # All remaining molecules are too similar
            break

        # Add the most diverse molecule
        if best_idx_in_remaining >= 0:
            selected_idx = remaining_indices.pop(best_idx_in_remaining)
            selected_indices.append(selected_idx)

    # Return selected SMILES
    return [valid_smiles[i] for i in selected_indices]
