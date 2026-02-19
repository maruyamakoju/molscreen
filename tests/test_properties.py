"""
Tests for molscreen.properties module.
"""

import pytest
from molscreen.properties import (
    smiles_to_mol,
    calculate_properties,
    check_lipinski,
    get_molecule_summary,
    MoleculeError
)


class TestSmilesConversion:
    """Tests for SMILES to molecule conversion."""

    def test_valid_smiles(self):
        """Test conversion of valid SMILES."""
        mol = smiles_to_mol("CCO")  # Ethanol
        assert mol is not None

    def test_aspirin_smiles(self):
        """Test conversion of aspirin SMILES."""
        mol = smiles_to_mol("CC(=O)Oc1ccccc1C(=O)O")
        assert mol is not None

    def test_invalid_smiles(self):
        """Test that invalid SMILES raises MoleculeError."""
        with pytest.raises(MoleculeError):
            smiles_to_mol("INVALID_SMILES_123")

    def test_empty_smiles(self):
        """Test that empty SMILES returns a molecule (RDKit accepts it)."""
        # RDKit actually accepts empty SMILES as a molecule with 0 atoms
        mol = smiles_to_mol("")
        assert mol is not None


class TestPropertyCalculations:
    """Tests for molecular property calculations."""

    def test_aspirin_properties(self):
        """Test property calculations for aspirin."""
        props = calculate_properties("CC(=O)Oc1ccccc1C(=O)O")

        # Aspirin: C9H8O4
        assert abs(props['MW'] - 180.16) < 0.1
        assert abs(props['LogP'] - 1.31) < 0.5  # LogP can vary slightly
        assert props['HBD'] == 1  # One -OH group
        assert props['HBA'] == 3  # Three oxygen acceptors (not counting -OH)

    def test_ethanol_properties(self):
        """Test property calculations for ethanol."""
        props = calculate_properties("CCO")

        # Ethanol: C2H6O
        assert abs(props['MW'] - 46.07) < 0.1
        assert props['HBD'] == 1  # One -OH group
        assert props['HBA'] == 1  # One oxygen acceptor

    def test_benzene_properties(self):
        """Test property calculations for benzene."""
        props = calculate_properties("c1ccccc1")

        # Benzene: C6H6
        assert abs(props['MW'] - 78.11) < 0.1
        assert props['HBD'] == 0  # No H-bond donors
        assert props['HBA'] == 0  # No H-bond acceptors

    def test_properties_have_correct_keys(self):
        """Test that all expected property keys are present."""
        props = calculate_properties("CCO")
        expected_keys = {'MW', 'LogP', 'HBD', 'HBA'}
        assert set(props.keys()) == expected_keys

    def test_invalid_smiles_raises_error(self):
        """Test that invalid SMILES raises MoleculeError."""
        with pytest.raises(MoleculeError):
            calculate_properties("INVALID")


class TestLipinskiRules:
    """Tests for Lipinski's Rule of Five compliance checking."""

    def test_aspirin_passes_lipinski(self):
        """Test that aspirin passes Lipinski's rules."""
        result = check_lipinski("CC(=O)Oc1ccccc1C(=O)O")

        assert result['MW_ok'] is True
        assert result['LogP_ok'] is True
        assert result['HBD_ok'] is True
        assert result['HBA_ok'] is True
        assert result['passes_lipinski'] is True

    def test_small_molecule_passes(self):
        """Test that small drug-like molecules pass."""
        result = check_lipinski("CCO")  # Ethanol
        assert result['passes_lipinski'] is True

    def test_with_precomputed_properties(self):
        """Test Lipinski check with pre-computed properties."""
        properties = {
            'MW': 200.0,
            'LogP': 2.0,
            'HBD': 2,
            'HBA': 3
        }
        result = check_lipinski(properties=properties)
        assert result['passes_lipinski'] is True

    def test_high_molecular_weight_fails(self):
        """Test that high MW fails Lipinski."""
        properties = {
            'MW': 600.0,  # > 500
            'LogP': 2.0,
            'HBD': 2,
            'HBA': 3
        }
        result = check_lipinski(properties=properties)
        assert result['MW_ok'] is False
        assert result['passes_lipinski'] is False

    def test_high_logp_fails(self):
        """Test that high LogP fails Lipinski."""
        properties = {
            'MW': 200.0,
            'LogP': 6.0,  # > 5
            'HBD': 2,
            'HBA': 3
        }
        result = check_lipinski(properties=properties)
        assert result['LogP_ok'] is False
        assert result['passes_lipinski'] is False

    def test_too_many_hbd_fails(self):
        """Test that too many H-bond donors fails Lipinski."""
        properties = {
            'MW': 200.0,
            'LogP': 2.0,
            'HBD': 6,  # > 5
            'HBA': 3
        }
        result = check_lipinski(properties=properties)
        assert result['HBD_ok'] is False
        assert result['passes_lipinski'] is False

    def test_too_many_hba_fails(self):
        """Test that too many H-bond acceptors fails Lipinski."""
        properties = {
            'MW': 200.0,
            'LogP': 2.0,
            'HBD': 2,
            'HBA': 11  # > 10
        }
        result = check_lipinski(properties=properties)
        assert result['HBA_ok'] is False
        assert result['passes_lipinski'] is False

    def test_lipinski_without_arguments_raises_error(self):
        """Test that calling without arguments raises ValueError."""
        with pytest.raises(ValueError):
            check_lipinski()

    def test_lipinski_result_keys(self):
        """Test that all expected keys are in the result."""
        result = check_lipinski("CCO")
        expected_keys = {'MW_ok', 'LogP_ok', 'HBD_ok', 'HBA_ok', 'passes_lipinski'}
        assert set(result.keys()) == expected_keys


class TestMoleculeSummary:
    """Tests for comprehensive molecule summary."""

    def test_summary_structure(self):
        """Test that summary has correct structure."""
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        summary = get_molecule_summary(smiles)

        assert 'smiles' in summary
        assert 'properties' in summary
        assert 'lipinski' in summary
        assert summary['smiles'] == smiles

    def test_summary_properties(self):
        """Test that summary contains correct properties."""
        summary = get_molecule_summary("CCO")

        assert 'MW' in summary['properties']
        assert 'LogP' in summary['properties']
        assert 'HBD' in summary['properties']
        assert 'HBA' in summary['properties']

    def test_summary_lipinski(self):
        """Test that summary contains Lipinski results."""
        summary = get_molecule_summary("CCO")

        assert 'passes_lipinski' in summary['lipinski']
        assert 'MW_ok' in summary['lipinski']
        assert 'LogP_ok' in summary['lipinski']
        assert 'HBD_ok' in summary['lipinski']
        assert 'HBA_ok' in summary['lipinski']

    def test_summary_invalid_smiles(self):
        """Test that summary raises error for invalid SMILES."""
        with pytest.raises(MoleculeError):
            get_molecule_summary("INVALID")

    def test_aspirin_summary(self):
        """Test complete summary for aspirin."""
        summary = get_molecule_summary("CC(=O)Oc1ccccc1C(=O)O")

        assert summary['smiles'] == "CC(=O)Oc1ccccc1C(=O)O"
        assert abs(summary['properties']['MW'] - 180.16) < 0.1
        assert summary['lipinski']['passes_lipinski'] is True
