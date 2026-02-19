"""
Tests for molscreen.scaffold module.
"""

import pytest
from molscreen.scaffold import (
    get_murcko_scaffold,
    get_generic_scaffold,
    group_by_scaffold,
    scaffold_diversity_score,
)


class TestMurckoScaffold:
    """Tests for Bemis-Murcko scaffold extraction."""

    def test_benzene_scaffold(self):
        """Test that benzene returns itself as scaffold."""
        scaffold = get_murcko_scaffold("c1ccccc1")
        assert scaffold == "c1ccccc1"

    def test_toluene_scaffold(self):
        """Test that toluene returns benzene as scaffold."""
        scaffold = get_murcko_scaffold("Cc1ccccc1")
        assert scaffold == "c1ccccc1"

    def test_aspirin_scaffold(self):
        """Test scaffold extraction from aspirin."""
        scaffold = get_murcko_scaffold("CC(=O)Oc1ccccc1C(=O)O")
        # Aspirin should have benzene ring as scaffold
        assert scaffold is not None
        assert "c1ccccc1" in scaffold or "C1=CC=CC=C1" in scaffold

    def test_acyclic_molecule_no_scaffold(self):
        """Test that acyclic molecules return None."""
        # Ethanol - acyclic, no rings
        scaffold = get_murcko_scaffold("CCO")
        assert scaffold is None

        # Propane - acyclic
        scaffold = get_murcko_scaffold("CCC")
        assert scaffold is None

    def test_invalid_smiles_returns_none(self):
        """Test that invalid SMILES returns None."""
        scaffold = get_murcko_scaffold("INVALID_SMILES_123")
        assert scaffold is None

    def test_empty_smiles_returns_none(self):
        """Test that empty SMILES returns None."""
        scaffold = get_murcko_scaffold("")
        assert scaffold is None

    def test_pyridine_scaffold(self):
        """Test scaffold extraction from pyridine."""
        scaffold = get_murcko_scaffold("c1ncccc1")
        # Pyridine should return itself as scaffold
        assert scaffold is not None
        assert len(scaffold) > 0


class TestGenericScaffold:
    """Tests for generic scaffold generation."""

    def test_benzene_generic_scaffold(self):
        """Test generic scaffold for benzene."""
        generic = get_generic_scaffold("c1ccccc1")
        # Generic benzene should be all carbons
        assert generic == "C1CCCCC1"

    def test_pyridine_generic_scaffold(self):
        """Test that pyridine and benzene have same generic scaffold."""
        benzene_generic = get_generic_scaffold("c1ccccc1")
        pyridine_generic = get_generic_scaffold("c1ncccc1")
        # Both should be C1CCCCC1 (all carbons)
        assert benzene_generic == pyridine_generic
        assert benzene_generic == "C1CCCCC1"

    def test_toluene_generic_scaffold(self):
        """Test generic scaffold for toluene."""
        generic = get_generic_scaffold("Cc1ccccc1")
        # Should be generic benzene (side chain removed)
        assert generic == "C1CCCCC1"

    def test_acyclic_molecule_no_generic_scaffold(self):
        """Test that acyclic molecules return None for generic scaffold."""
        generic = get_generic_scaffold("CCO")
        assert generic is None

    def test_invalid_smiles_returns_none(self):
        """Test that invalid SMILES returns None for generic scaffold."""
        generic = get_generic_scaffold("INVALID")
        assert generic is None


class TestGroupByScaffold:
    """Tests for grouping molecules by scaffold."""

    def test_group_benzene_derivatives(self):
        """Test grouping benzene derivatives together."""
        molecules = ["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1"]
        groups = group_by_scaffold(molecules)

        # All should have benzene scaffold
        assert "c1ccccc1" in groups
        assert len(groups["c1ccccc1"]) == 3

    def test_group_mixed_molecules(self):
        """Test grouping molecules with different scaffolds."""
        molecules = [
            "c1ccccc1",      # Benzene
            "Cc1ccccc1",     # Toluene (benzene scaffold)
            "CCO",           # Ethanol (no scaffold)
            "C1CCCCC1",      # Cyclohexane
        ]
        groups = group_by_scaffold(molecules)

        # Should have at least 2 groups: benzene derivatives and no_scaffold
        assert len(groups) >= 2
        assert "no_scaffold" in groups
        assert "CCO" in groups["no_scaffold"]

    def test_group_acyclic_molecules(self):
        """Test that acyclic molecules are grouped under 'no_scaffold'."""
        molecules = ["CCO", "CCC", "CCCC"]
        groups = group_by_scaffold(molecules)

        assert "no_scaffold" in groups
        assert len(groups["no_scaffold"]) == 3

    def test_group_skips_invalid_smiles(self):
        """Test that invalid SMILES are skipped."""
        molecules = ["c1ccccc1", "INVALID", "Cc1ccccc1"]
        groups = group_by_scaffold(molecules)

        # Should only have benzene group
        assert "c1ccccc1" in groups
        assert len(groups["c1ccccc1"]) == 2

    def test_group_empty_list(self):
        """Test grouping empty list returns empty dict."""
        groups = group_by_scaffold([])
        assert groups == {}

    def test_group_all_invalid(self):
        """Test grouping all invalid SMILES returns empty dict."""
        molecules = ["INVALID1", "INVALID2", "INVALID3"]
        groups = group_by_scaffold(molecules)
        assert groups == {}


class TestScaffoldDiversityScore:
    """Tests for scaffold diversity score calculation."""

    def test_diversity_all_same_scaffold(self):
        """Test diversity score when all molecules have same scaffold."""
        molecules = ["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1"]
        score = scaffold_diversity_score(molecules)
        # All have benzene scaffold, so unique/total = 1/3
        assert 0.3 < score < 0.4

    def test_diversity_all_different_scaffolds(self):
        """Test diversity score when all molecules have different scaffolds."""
        molecules = [
            "c1ccccc1",      # Benzene
            "C1CCCCC1",      # Cyclohexane
            "c1ncccc1",      # Pyridine
        ]
        score = scaffold_diversity_score(molecules)
        # All different scaffolds, so unique/total = 3/3 = 1.0
        assert score == 1.0

    def test_diversity_score_range(self):
        """Test that diversity score is between 0 and 1."""
        molecules = ["c1ccccc1", "Cc1ccccc1", "C1CCCCC1", "CCO"]
        score = scaffold_diversity_score(molecules)
        assert 0.0 <= score <= 1.0

    def test_diversity_single_molecule(self):
        """Test diversity score for single molecule is 1.0."""
        molecules = ["c1ccccc1"]
        score = scaffold_diversity_score(molecules)
        # One unique scaffold out of one molecule = 1.0
        assert score == 1.0

    def test_diversity_empty_list(self):
        """Test diversity score for empty list is 0.0."""
        score = scaffold_diversity_score([])
        assert score == 0.0

    def test_diversity_all_invalid(self):
        """Test diversity score when all SMILES are invalid."""
        molecules = ["INVALID1", "INVALID2"]
        score = scaffold_diversity_score(molecules)
        # No valid molecules, should return 0.0
        assert score == 0.0

    def test_diversity_with_invalid_smiles(self):
        """Test diversity score skips invalid SMILES."""
        molecules = ["c1ccccc1", "INVALID", "Cc1ccccc1"]
        score = scaffold_diversity_score(molecules)
        # Only 2 valid molecules, both with benzene scaffold
        # unique/total = 1/2 = 0.5
        assert score == 0.5

    def test_diversity_includes_acyclic(self):
        """Test that acyclic molecules are counted in diversity."""
        molecules = [
            "c1ccccc1",  # Benzene scaffold
            "CCO",       # No scaffold (acyclic)
        ]
        score = scaffold_diversity_score(molecules)
        # 2 different "scaffold types" (benzene and no_scaffold)
        # unique/total = 2/2 = 1.0
        assert score == 1.0


class TestScaffoldIntegration:
    """Integration tests for scaffold module."""

    def test_scaffold_functions_consistent(self):
        """Test that scaffold functions work together consistently."""
        smiles = "Cc1ccccc1"  # Toluene

        murcko = get_murcko_scaffold(smiles)
        generic = get_generic_scaffold(smiles)

        assert murcko is not None
        assert generic is not None
        assert murcko == "c1ccccc1"  # Benzene
        assert generic == "C1CCCCC1"  # Generic benzene

    def test_import_from_package(self):
        """Test that scaffold functions can be imported from package."""
        from molscreen import (
            get_murcko_scaffold,
            get_generic_scaffold,
            group_by_scaffold,
            scaffold_diversity_score,
        )

        # Just verify they're callable
        assert callable(get_murcko_scaffold)
        assert callable(get_generic_scaffold)
        assert callable(group_by_scaffold)
        assert callable(scaffold_diversity_score)
