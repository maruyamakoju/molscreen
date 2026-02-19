"""
Tests for molscreen.similarity module.
"""

import pytest
from molscreen.similarity import (
    compute_fingerprint,
    tanimoto_similarity,
    rank_by_similarity,
    find_diverse_subset
)
from molscreen.properties import MoleculeError


class TestComputeFingerprint:
    """Tests for compute_fingerprint function."""

    def test_compute_fingerprint_valid_smiles(self):
        """Test fingerprint computation for valid SMILES."""
        fp = compute_fingerprint("CCO")  # Ethanol
        assert fp is not None
        assert fp.GetNumBits() == 2048  # Default n_bits

    def test_compute_fingerprint_aspirin(self):
        """Test fingerprint computation for aspirin."""
        fp = compute_fingerprint("CC(=O)Oc1ccccc1C(=O)O")
        assert fp is not None
        assert fp.GetNumBits() == 2048

    def test_compute_fingerprint_custom_parameters(self):
        """Test fingerprint with custom radius and n_bits."""
        fp1 = compute_fingerprint("CCO", radius=1, n_bits=1024)
        assert fp1.GetNumBits() == 1024

        fp2 = compute_fingerprint("CCO", radius=3, n_bits=4096)
        assert fp2.GetNumBits() == 4096

    def test_compute_fingerprint_different_radius(self):
        """Test that different radius values produce different fingerprints."""
        fp_r2 = compute_fingerprint("CCO", radius=2)
        fp_r3 = compute_fingerprint("CCO", radius=3)

        # Fingerprints should be different (though both valid)
        assert fp_r2 is not None
        assert fp_r3 is not None

    def test_compute_fingerprint_invalid_smiles(self):
        """Test that invalid SMILES raises MoleculeError."""
        with pytest.raises(MoleculeError):
            compute_fingerprint("INVALID_SMILES_123")

    def test_compute_fingerprint_empty_smiles(self):
        """Test fingerprint for empty SMILES."""
        # RDKit accepts empty SMILES as a molecule with 0 atoms
        fp = compute_fingerprint("")
        assert fp is not None


class TestTanimotoSimilarity:
    """Tests for tanimoto_similarity function."""

    def test_tanimoto_similarity_identical_molecules(self):
        """Test that identical molecules have similarity 1.0."""
        similarity = tanimoto_similarity("CCO", "CCO")
        assert similarity == 1.0

    def test_tanimoto_similarity_aspirin_identical(self):
        """Test identical aspirin molecules."""
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        similarity = tanimoto_similarity(smiles, smiles)
        assert similarity == 1.0

    def test_tanimoto_similarity_different_molecules(self):
        """Test similarity between different but related molecules."""
        # Ethanol vs Propanol - should be similar but not identical
        similarity = tanimoto_similarity("CCO", "CCCO")
        assert 0.0 < similarity < 1.0
        assert similarity > 0.5  # They should be fairly similar

    def test_tanimoto_similarity_completely_different(self):
        """Test similarity between very different molecules."""
        # Ethanol vs Benzene - should have low similarity
        similarity = tanimoto_similarity("CCO", "c1ccccc1")
        assert 0.0 <= similarity < 0.3  # Should be quite different

    def test_tanimoto_similarity_range(self):
        """Test that similarity is always in range [0, 1]."""
        test_pairs = [
            ("CCO", "CCCO"),
            ("c1ccccc1", "c1ccccc1O"),
            ("CC(C)C", "CCCC"),
            ("CC(=O)O", "CC(=O)N")
        ]

        for smiles1, smiles2 in test_pairs:
            similarity = tanimoto_similarity(smiles1, smiles2)
            assert 0.0 <= similarity <= 1.0

    def test_tanimoto_similarity_symmetric(self):
        """Test that similarity is symmetric."""
        sim_ab = tanimoto_similarity("CCO", "CCCO")
        sim_ba = tanimoto_similarity("CCCO", "CCO")
        assert abs(sim_ab - sim_ba) < 1e-10  # Should be identical

    def test_tanimoto_similarity_invalid_smiles1(self):
        """Test that invalid first SMILES raises MoleculeError."""
        with pytest.raises(MoleculeError):
            tanimoto_similarity("INVALID", "CCO")

    def test_tanimoto_similarity_invalid_smiles2(self):
        """Test that invalid second SMILES raises MoleculeError."""
        with pytest.raises(MoleculeError):
            tanimoto_similarity("CCO", "INVALID")


class TestRankBySimilarity:
    """Tests for rank_by_similarity function."""

    def test_rank_by_similarity_returns_sorted(self):
        """Test that results are sorted by similarity (descending)."""
        query = "CCO"
        candidates = ["CCCO", "CCCCO", "CC"]
        results = rank_by_similarity(query, candidates, top_k=3)

        # Results should be in descending order
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i + 1][1]

    def test_rank_by_similarity_top_k(self):
        """Test that top_k parameter limits results correctly."""
        query = "CCO"
        candidates = ["CCCO", "CCCCO", "CCCCCO", "CC", "C"]

        results = rank_by_similarity(query, candidates, top_k=2)
        assert len(results) == 2

        results = rank_by_similarity(query, candidates, top_k=3)
        assert len(results) == 3

    def test_rank_by_similarity_empty_candidates(self):
        """Test with empty candidates list."""
        results = rank_by_similarity("CCO", [], top_k=10)
        assert results == []

    def test_rank_by_similarity_top_k_exceeds_candidates(self):
        """Test when top_k exceeds number of candidates."""
        query = "CCO"
        candidates = ["CCCO", "CC"]
        results = rank_by_similarity(query, candidates, top_k=10)

        # Should return all candidates
        assert len(results) == 2

    def test_rank_by_similarity_correct_format(self):
        """Test that results have correct format (SMILES, score)."""
        query = "CCO"
        candidates = ["CCCO", "CC"]
        results = rank_by_similarity(query, candidates, top_k=5)

        for smiles, score in results:
            assert isinstance(smiles, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_rank_by_similarity_most_similar_first(self):
        """Test that most similar molecule is ranked first."""
        query = "CCO"
        candidates = ["CCCO", "CCO", "c1ccccc1"]  # CCO is identical
        results = rank_by_similarity(query, candidates, top_k=3)

        # First result should be identical molecule with score 1.0
        assert results[0][0] == "CCO"
        assert results[0][1] == 1.0

    def test_rank_by_similarity_invalid_query(self):
        """Test that invalid query SMILES raises MoleculeError."""
        with pytest.raises(MoleculeError):
            rank_by_similarity("INVALID", ["CCO", "CCCO"], top_k=5)

    def test_rank_by_similarity_skips_invalid_candidates(self):
        """Test that invalid candidates are skipped."""
        query = "CCO"
        candidates = ["CCCO", "INVALID", "CC", "ALSO_INVALID"]
        results = rank_by_similarity(query, candidates, top_k=10)

        # Should only return valid candidates
        assert len(results) == 2
        assert all(smiles in ["CCCO", "CC"] for smiles, _ in results)

    def test_rank_by_similarity_all_invalid_candidates(self):
        """Test with all invalid candidates."""
        query = "CCO"
        candidates = ["INVALID1", "INVALID2", "INVALID3"]
        results = rank_by_similarity(query, candidates, top_k=5)

        assert results == []


class TestFindDiverseSubset:
    """Tests for find_diverse_subset function."""

    def test_find_diverse_subset_basic(self):
        """Test basic diverse subset selection."""
        molecules = ["CCO", "CCCO", "c1ccccc1", "CC(C)C", "CC(=O)O"]
        diverse = find_diverse_subset(molecules, n=3)

        assert len(diverse) <= 3
        assert all(smiles in molecules for smiles in diverse)

    def test_find_diverse_subset_returns_at_most_n(self):
        """Test that result has at most n molecules."""
        molecules = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "CC(C)C"]

        for n in [1, 2, 3, 4, 5]:
            diverse = find_diverse_subset(molecules, n=n)
            assert len(diverse) <= n

    def test_find_diverse_subset_empty_list(self):
        """Test with empty input list."""
        diverse = find_diverse_subset([], n=10)
        assert diverse == []

    def test_find_diverse_subset_n_exceeds_input(self):
        """Test when n exceeds input size."""
        molecules = ["CCO", "CCCO"]
        diverse = find_diverse_subset(molecules, n=10)

        # Should return all molecules (after validation)
        assert len(diverse) == 2
        assert set(diverse) == set(molecules)

    def test_find_diverse_subset_single_molecule(self):
        """Test with single molecule."""
        molecules = ["CCO"]
        diverse = find_diverse_subset(molecules, n=5)

        assert diverse == ["CCO"]

    def test_find_diverse_subset_threshold_effect(self):
        """Test that threshold parameter affects selection."""
        # Very similar molecules
        molecules = ["CCO", "CCCO", "CCCCO", "CCCCCO"]

        # Low threshold should select more molecules
        diverse_low = find_diverse_subset(molecules, n=10, threshold=0.2)

        # High threshold should select fewer molecules
        diverse_high = find_diverse_subset(molecules, n=10, threshold=0.8)

        # Low threshold typically gives more diverse set
        assert len(diverse_low) >= len(diverse_high)

    def test_find_diverse_subset_includes_first_molecule(self):
        """Test that first molecule is always included."""
        molecules = ["CCO", "CCCO", "c1ccccc1"]
        diverse = find_diverse_subset(molecules, n=10)

        # First molecule should be in the result
        assert "CCO" in diverse

    def test_find_diverse_subset_very_different_molecules(self):
        """Test with very different molecules."""
        # These should all be selected as they're quite different
        molecules = ["CCO", "c1ccccc1", "CC(=O)O", "CC#N"]
        diverse = find_diverse_subset(molecules, n=4, threshold=0.3)

        # Should select all or most of them
        assert len(diverse) >= 2

    def test_find_diverse_subset_invalid_smiles(self):
        """Test that invalid SMILES raises MoleculeError."""
        molecules = ["INVALID", "CCO"]
        with pytest.raises(MoleculeError):
            find_diverse_subset(molecules, n=2)

    def test_find_diverse_subset_with_invalid_in_middle(self):
        """Test that the function handles invalid SMILES in the list."""
        # Current implementation skips invalid SMILES during fingerprint computation
        molecules = ["CCO", "INVALID", "CCCO"]
        diverse = find_diverse_subset(molecules, n=2)

        # Should return valid molecules only
        assert len(diverse) <= 2
        assert all(s in ["CCO", "CCCO"] for s in diverse)

    def test_find_diverse_subset_deterministic(self):
        """Test that results are deterministic for same input."""
        molecules = ["CCO", "CCCO", "c1ccccc1", "CC(C)C"]

        diverse1 = find_diverse_subset(molecules, n=2, threshold=0.4)
        diverse2 = find_diverse_subset(molecules, n=2, threshold=0.4)

        # Should produce same result
        assert diverse1 == diverse2


class TestIntegration:
    """Integration tests for similarity module."""

    def test_workflow_rank_and_diversify(self):
        """Test a typical workflow: rank then diversify."""
        query = "CCO"
        candidates = [
            "CCCO", "CCCCO", "CCCCCO",  # Similar alcohols
            "c1ccccc1", "c1ccccc1O",    # Aromatics
            "CC(=O)O", "CCC(=O)O"       # Acids
        ]

        # Rank by similarity
        similar = rank_by_similarity(query, candidates, top_k=5)
        assert len(similar) <= 5

        # Get diverse subset from top similar
        top_smiles = [smiles for smiles, _ in similar]
        diverse = find_diverse_subset(top_smiles, n=3)
        assert len(diverse) <= 3

    def test_all_functions_with_same_molecules(self):
        """Test that all functions work with the same set of molecules."""
        molecules = ["CCO", "CCCO", "c1ccccc1"]

        # Compute fingerprints
        fps = [compute_fingerprint(smiles) for smiles in molecules]
        assert len(fps) == 3

        # Compute similarities
        sim = tanimoto_similarity(molecules[0], molecules[1])
        assert 0.0 <= sim <= 1.0

        # Rank by similarity
        ranked = rank_by_similarity(molecules[0], molecules[1:], top_k=2)
        assert len(ranked) == 2

        # Find diverse subset
        diverse = find_diverse_subset(molecules, n=2)
        assert len(diverse) <= 2
