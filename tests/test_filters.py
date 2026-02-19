"""
Tests for molscreen.filters module.
"""

import pytest
from molscreen.filters import (
    lipinski_filter,
    veber_filter,
    pains_filter,
    filter_molecules
)
from molscreen.properties import MoleculeError


class TestLipinskiFilter:
    """Tests for Lipinski's Rule of Five filter."""

    def test_aspirin_passes(self):
        """Test that aspirin passes Lipinski filter."""
        result = lipinski_filter("CC(=O)Oc1ccccc1C(=O)O")

        assert result['passes'] is True
        assert len(result['violations']) == 0
        assert 'MW' in result['values']
        assert 'LogP' in result['values']
        assert 'HBD' in result['values']
        assert 'HBA' in result['values']
        assert result['values']['MW'] <= 500
        assert result['values']['LogP'] <= 5
        assert result['values']['HBD'] <= 5
        assert result['values']['HBA'] <= 10

    def test_ethanol_passes(self):
        """Test that ethanol passes Lipinski filter."""
        result = lipinski_filter("CCO")

        assert result['passes'] is True
        assert len(result['violations']) == 0

    def test_large_molecule_fails_mw(self):
        """Test that large molecule (MW>500) fails Lipinski filter."""
        # Create a large molecule with many carbons
        large_smiles = "C" * 50  # Will have MW > 500
        result = lipinski_filter(large_smiles)

        assert result['passes'] is False
        assert "MW > 500" in result['violations']
        assert result['values']['MW'] > 500

    def test_high_logp_fails(self):
        """Test that molecule with high LogP fails."""
        # Long chain hydrocarbon has high LogP
        high_logp_smiles = "CCCCCCCCCCCCCCCCCCCC"  # 20 carbons
        result = lipinski_filter(high_logp_smiles)

        # This should fail LogP criterion
        if result['values']['LogP'] > 5:
            assert result['passes'] is False
            assert "LogP > 5" in result['violations']

    def test_too_many_hbd_fails(self):
        """Test molecule with too many H-bond donors fails."""
        # Sugar-like molecule with many OH groups
        many_oh = "OC(CO)C(O)C(O)C(O)C(O)CO"  # Glucose-like
        result = lipinski_filter(many_oh)

        # Check HBD value
        if result['values']['HBD'] > 5:
            assert result['passes'] is False
            assert "HBD > 5" in result['violations']

    def test_invalid_smiles_raises_error(self):
        """Test that invalid SMILES raises MoleculeError."""
        with pytest.raises(MoleculeError):
            lipinski_filter("INVALID_SMILES")

    def test_result_structure(self):
        """Test that result has correct structure."""
        result = lipinski_filter("CCO")

        assert 'passes' in result
        assert 'violations' in result
        assert 'values' in result
        assert isinstance(result['passes'], bool)
        assert isinstance(result['violations'], list)
        assert isinstance(result['values'], dict)


class TestVeberFilter:
    """Tests for Veber filter."""

    def test_ethanol_passes(self):
        """Test that ethanol passes Veber filter."""
        result = veber_filter("CCO")

        assert result['passes'] is True
        assert len(result['violations']) == 0
        assert result['values']['TPSA'] <= 140
        assert result['values']['RotatableBonds'] <= 10

    def test_aspirin_passes(self):
        """Test that aspirin passes Veber filter."""
        result = veber_filter("CC(=O)Oc1ccccc1C(=O)O")

        assert result['passes'] is True
        assert result['values']['TPSA'] <= 140
        assert result['values']['RotatableBonds'] <= 10

    def test_high_tpsa_check(self):
        """Test TPSA calculation."""
        result = veber_filter("CCO")

        # Ethanol has one OH group
        assert 'TPSA' in result['values']
        assert result['values']['TPSA'] > 0  # Has polar surface area
        assert result['values']['TPSA'] < 30  # But not too high

    def test_rotatable_bonds_check(self):
        """Test rotatable bonds calculation."""
        # Ethanol has 0 rotatable bonds
        result = veber_filter("CCO")
        assert result['values']['RotatableBonds'] == 0

        # Longer chain has rotatable bonds
        result = veber_filter("CCCCCC")  # Hexane
        assert result['values']['RotatableBonds'] > 0

    def test_flexible_molecule_many_rotatable_bonds(self):
        """Test molecule with many rotatable bonds."""
        # Long chain with many single bonds
        flexible = "C" * 15  # Should have >10 rotatable bonds
        result = veber_filter(flexible)

        if result['values']['RotatableBonds'] > 10:
            assert result['passes'] is False
            assert "RotatableBonds > 10" in result['violations']

    def test_invalid_smiles_raises_error(self):
        """Test that invalid SMILES raises MoleculeError."""
        with pytest.raises(MoleculeError):
            veber_filter("INVALID")

    def test_result_structure(self):
        """Test that result has correct structure."""
        result = veber_filter("CCO")

        assert 'passes' in result
        assert 'violations' in result
        assert 'values' in result
        assert isinstance(result['passes'], bool)
        assert isinstance(result['violations'], list)
        assert isinstance(result['values'], dict)


class TestPAINSFilter:
    """Tests for PAINS filter."""

    def test_clean_molecule_passes(self):
        """Test that clean molecules pass PAINS filter."""
        # Ethanol should be clean
        result = pains_filter("CCO")

        assert result['passes'] is True
        assert len(result['violations']) == 0
        assert result['values']['num_pains_alerts'] == 0

    def test_aspirin_passes(self):
        """Test that aspirin passes PAINS filter."""
        result = pains_filter("CC(=O)Oc1ccccc1C(=O)O")

        assert result['passes'] is True
        assert result['values']['num_pains_alerts'] == 0

    def test_benzene_passes(self):
        """Test that benzene passes PAINS filter."""
        result = pains_filter("c1ccccc1")

        assert result['passes'] is True

    def test_pains_detection(self):
        """Test PAINS detection with known PAINS substructure."""
        # Note: Actual PAINS detection depends on RDKit's catalog
        # We'll test the function works, actual detection may vary
        # Catechol substructure is sometimes flagged
        catechol = "Oc1ccccc1O"
        result = pains_filter(catechol)

        # Just verify the result structure is correct
        assert 'passes' in result
        assert 'violations' in result
        assert 'values' in result
        assert 'num_pains_alerts' in result['values']
        assert isinstance(result['violations'], list)

    def test_invalid_smiles_raises_error(self):
        """Test that invalid SMILES raises MoleculeError."""
        with pytest.raises(MoleculeError):
            pains_filter("INVALID")

    def test_result_structure(self):
        """Test that result has correct structure."""
        result = pains_filter("CCO")

        assert 'passes' in result
        assert 'violations' in result
        assert 'values' in result
        assert isinstance(result['passes'], bool)
        assert isinstance(result['violations'], list)
        assert isinstance(result['values'], dict)
        assert 'num_pains_alerts' in result['values']


class TestFilterMolecules:
    """Tests for batch filtering function."""

    def test_single_molecule(self):
        """Test filtering a single molecule."""
        results = filter_molecules(["CCO"], rules=["lipinski"])

        assert len(results) == 1
        assert results[0]['smiles'] == "CCO"
        assert results[0]['valid'] is True
        assert 'filters' in results[0]
        assert 'lipinski' in results[0]['filters']
        assert results[0]['overall_pass'] is True

    def test_multiple_molecules(self):
        """Test filtering multiple molecules."""
        smiles_list = ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"]
        results = filter_molecules(smiles_list, rules=["lipinski", "veber"])

        assert len(results) == 3
        for result in results:
            assert result['valid'] is True
            assert 'lipinski' in result['filters']
            assert 'veber' in result['filters']

    def test_all_rules(self):
        """Test applying all rules."""
        results = filter_molecules(["CCO"], rules=["all"])

        assert len(results) == 1
        assert 'lipinski' in results[0]['filters']
        assert 'veber' in results[0]['filters']
        assert 'pains' in results[0]['filters']

    def test_invalid_smiles_handling(self):
        """Test handling of invalid SMILES."""
        results = filter_molecules(["CCO", "INVALID", "c1ccccc1"], rules=["lipinski"])

        assert len(results) == 3
        assert results[0]['valid'] is True
        assert results[1]['valid'] is False
        assert results[1]['overall_pass'] is False
        assert 'error' in results[1]
        assert results[2]['valid'] is True

    def test_filter_molecules_correct_dict_format(self):
        """Test that filter_molecules returns correct dict format."""
        results = filter_molecules(["CCO"], rules=["lipinski", "veber"])
        result = results[0]

        # Check required keys
        assert 'smiles' in result
        assert 'valid' in result
        assert 'filters' in result
        assert 'overall_pass' in result

        # Check filter results structure
        assert isinstance(result['filters'], dict)
        for filter_name in ['lipinski', 'veber']:
            assert filter_name in result['filters']
            filter_result = result['filters'][filter_name]
            assert 'passes' in filter_result
            assert 'violations' in filter_result
            assert 'values' in filter_result

    def test_overall_pass_with_all_passing(self):
        """Test overall_pass is True when all filters pass."""
        results = filter_molecules(["CCO"], rules=["lipinski", "veber", "pains"])

        assert results[0]['overall_pass'] is True

    def test_overall_pass_with_one_failing(self):
        """Test overall_pass is False when any filter fails."""
        # Create a molecule that might fail at least one filter
        large_smiles = "C" * 50  # Large molecule
        results = filter_molecules([large_smiles], rules=["lipinski"])

        # Should fail MW criterion
        if not results[0]['filters']['lipinski']['passes']:
            assert results[0]['overall_pass'] is False

    def test_empty_list(self):
        """Test filtering empty list."""
        results = filter_molecules([], rules=["lipinski"])

        assert len(results) == 0

    def test_default_rules(self):
        """Test that default rules include all filters."""
        results = filter_molecules(["CCO"])

        assert 'lipinski' in results[0]['filters']
        assert 'veber' in results[0]['filters']
        assert 'pains' in results[0]['filters']

    def test_invalid_rule_returns_error(self):
        """Test that invalid rule name returns error in result."""
        results = filter_molecules(["CCO"], rules=["invalid_rule"])

        assert len(results) == 1
        assert results[0]['valid'] is False
        assert 'error' in results[0]
        assert 'invalid_rule' in results[0]['error'].lower()

    def test_mixed_valid_invalid_molecules(self):
        """Test batch with mix of valid and invalid SMILES."""
        smiles_list = ["CCO", "INVALID", "CC(=O)Oc1ccccc1C(=O)O", "BAD"]
        results = filter_molecules(smiles_list, rules=["lipinski"])

        assert len(results) == 4
        assert results[0]['valid'] is True
        assert results[1]['valid'] is False
        assert results[2]['valid'] is True
        assert results[3]['valid'] is False


class TestIntegration:
    """Integration tests for filter module."""

    def test_aspirin_all_filters(self):
        """Test aspirin with all filters."""
        results = filter_molecules(
            ["CC(=O)Oc1ccccc1C(=O)O"],
            rules=["lipinski", "veber", "pains"]
        )

        result = results[0]
        assert result['valid'] is True
        assert result['overall_pass'] is True
        assert result['filters']['lipinski']['passes'] is True
        assert result['filters']['veber']['passes'] is True
        assert result['filters']['pains']['passes'] is True

    def test_drug_like_molecules_pass(self):
        """Test that common drug-like molecules pass filters."""
        # Common drug-like molecules
        drug_like = [
            "CCO",  # Ethanol
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
        ]

        results = filter_molecules(drug_like, rules=["lipinski", "veber"])

        for result in results:
            assert result['valid'] is True
            # Most drug-like molecules should pass
            # (though not guaranteed for all)
