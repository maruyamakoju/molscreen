"""
Tests for molscreen.io module.
"""

import os
import tempfile
import warnings
import pytest
import pandas as pd
from molscreen.io import read_molecules, write_molecules, detect_smiles_column


class TestDetectSmilesColumn:
    """Tests for SMILES column detection."""

    def test_detect_lowercase_smiles(self):
        """Test detection of 'smiles' column."""
        df = pd.DataFrame({'smiles': ['CCO'], 'name': ['ethanol']})
        assert detect_smiles_column(df) == 'smiles'

    def test_detect_uppercase_smiles(self):
        """Test detection of 'SMILES' column."""
        df = pd.DataFrame({'SMILES': ['CCO'], 'name': ['ethanol']})
        assert detect_smiles_column(df) == 'SMILES'

    def test_detect_canonical_smiles(self):
        """Test detection of 'canonical_smiles' column."""
        df = pd.DataFrame({'canonical_smiles': ['CCO'], 'name': ['ethanol']})
        assert detect_smiles_column(df) == 'canonical_smiles'

    def test_detect_titlecase_smiles(self):
        """Test detection of 'Smiles' column."""
        df = pd.DataFrame({'Smiles': ['CCO'], 'name': ['ethanol']})
        assert detect_smiles_column(df) == 'Smiles'

    def test_no_smiles_column(self):
        """Test that None is returned when no SMILES column exists."""
        df = pd.DataFrame({'compound': ['CCO'], 'name': ['ethanol']})
        assert detect_smiles_column(df) is None

    def test_priority_order(self):
        """Test that 'smiles' is preferred over other variants."""
        df = pd.DataFrame({
            'smiles': ['CCO'],
            'SMILES': ['c1ccccc1'],
            'name': ['test']
        })
        # Should return the first match in priority order
        assert detect_smiles_column(df) == 'smiles'


class TestReadMoleculesSMI:
    """Tests for reading SMI files."""

    def test_read_smi_with_names(self):
        """Test reading SMI file with molecule names."""
        fixture_path = 'tests/fixtures/sample.smi'
        molecules = read_molecules(fixture_path)

        assert len(molecules) == 5
        assert molecules[0] == ('CCO', 'ethanol')
        assert molecules[1] == ('CC(=O)Oc1ccccc1C(=O)O', 'aspirin')
        assert molecules[2] == ('c1ccccc1', 'benzene')

    def test_read_smi_handles_invalid_smiles(self):
        """Test that invalid SMILES are skipped with warnings."""
        fixture_path = 'tests/fixtures/invalid.smi'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            molecules = read_molecules(fixture_path)

            # Should have warnings for invalid SMILES
            assert len(w) == 2  # Two invalid SMILES in the file
            assert 'invalid' in str(w[0].message).lower()

        # Should only return valid molecules
        assert len(molecules) == 3
        smiles_list = [s for s, n in molecules]
        assert 'CCO' in smiles_list
        assert 'CC(=O)Oc1ccccc1C(=O)O' in smiles_list
        assert 'c1ccccc1' in smiles_list


class TestReadMoleculesCSV:
    """Tests for reading CSV files."""

    def test_read_csv_with_smiles_column(self):
        """Test reading CSV file with SMILES column."""
        fixture_path = 'tests/fixtures/sample.csv'
        molecules = read_molecules(fixture_path)

        assert len(molecules) == 5
        assert molecules[0] == ('CCO', 'ethanol')
        assert molecules[1] == ('CC(=O)Oc1ccccc1C(=O)O', 'aspirin')
        assert molecules[2] == ('c1ccccc1', 'benzene')

    def test_read_csv_without_name_column(self):
        """Test reading CSV without name column."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write('SMILES\n')
            f.write('CCO\n')
            f.write('c1ccccc1\n')
            csv_path = f.name

        try:
            molecules = read_molecules(csv_path)
            assert len(molecules) == 2
            assert molecules[0] == ('CCO', None)
            assert molecules[1] == ('c1ccccc1', None)
        finally:
            os.remove(csv_path)

    def test_read_csv_no_smiles_column_raises_error(self):
        """Test that CSV without SMILES column raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write('compound,name\n')
            f.write('CCO,ethanol\n')
            csv_path = f.name

        try:
            with pytest.raises(ValueError, match="Could not find SMILES column"):
                read_molecules(csv_path)
        finally:
            os.remove(csv_path)


class TestReadMoleculesSDF:
    """Tests for reading SDF files."""

    def test_read_sdf_file(self):
        """Test reading SDF file."""
        fixture_path = 'tests/fixtures/sample.sdf'
        molecules = read_molecules(fixture_path)

        assert len(molecules) == 4

        # Check names
        names = [n for s, n in molecules]
        assert 'ethanol' in names
        assert 'aspirin' in names
        assert 'benzene' in names
        assert 'ibuprofen' in names

        # Check that SMILES are valid (converted from SDF)
        smiles_list = [s for s, n in molecules]
        assert all(s for s in smiles_list)  # All should be non-empty


class TestReadMoleculesErrors:
    """Tests for error handling in read_molecules."""

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            read_molecules('nonexistent_file.smi')

    def test_unsupported_format(self):
        """Test that ValueError is raised for unsupported format."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.xyz') as f:
            f.write('invalid')
            xyz_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                read_molecules(xyz_path)
        finally:
            os.remove(xyz_path)


class TestWriteMoleculesCSV:
    """Tests for writing molecules to CSV format."""

    def test_write_and_read_csv(self):
        """Test CSV write and read round-trip."""
        molecules = [
            ('CCO', 'ethanol'),
            ('c1ccccc1', 'benzene'),
            ('CC(=O)O', 'acetic_acid')
        ]

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            csv_path = f.name

        try:
            write_molecules(molecules, csv_path, format='csv')
            assert os.path.exists(csv_path)

            # Read back
            read_mols = read_molecules(csv_path)
            assert len(read_mols) == 3
            assert read_mols[0] == ('CCO', 'ethanol')
            assert read_mols[1] == ('c1ccccc1', 'benzene')
            assert read_mols[2] == ('CC(=O)O', 'acetic_acid')
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_write_csv_without_names(self):
        """Test writing CSV with molecules without names."""
        molecules = [
            ('CCO', None),
            ('c1ccccc1', None)
        ]

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            csv_path = f.name

        try:
            write_molecules(molecules, csv_path, format='csv')

            # Verify file content
            df = pd.read_csv(csv_path)
            assert 'smiles' in df.columns
            assert 'name' in df.columns
            assert len(df) == 2
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)


class TestWriteMoleculesSDF:
    """Tests for writing molecules to SDF format."""

    def test_write_and_read_sdf(self):
        """Test SDF write and read round-trip."""
        molecules = [
            ('CCO', 'ethanol'),
            ('c1ccccc1', 'benzene'),
            ('CC(=O)O', 'acetic_acid')
        ]

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sdf') as f:
            sdf_path = f.name

        try:
            write_molecules(molecules, sdf_path, format='sdf')
            assert os.path.exists(sdf_path)

            # Read back
            read_mols = read_molecules(sdf_path)
            assert len(read_mols) == 3

            # Check names are preserved
            names = [n for s, n in read_mols]
            assert 'ethanol' in names
            assert 'benzene' in names
            assert 'acetic_acid' in names
        finally:
            if os.path.exists(sdf_path):
                os.remove(sdf_path)

    def test_write_sdf_invalid_smiles(self):
        """Test that writing invalid SMILES raises ValueError."""
        molecules = [
            ('INVALID_SMILES', 'bad_mol')
        ]

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sdf') as f:
            sdf_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid SMILES"):
                write_molecules(molecules, sdf_path, format='sdf')
        finally:
            if os.path.exists(sdf_path):
                os.remove(sdf_path)


class TestWriteMoleculesSMI:
    """Tests for writing molecules to SMI format."""

    def test_write_and_read_smi(self):
        """Test SMI write and read round-trip."""
        molecules = [
            ('CCO', 'ethanol'),
            ('c1ccccc1', 'benzene'),
            ('CC(=O)O', 'acetic_acid')
        ]

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.smi') as f:
            smi_path = f.name

        try:
            write_molecules(molecules, smi_path, format='smi')
            assert os.path.exists(smi_path)

            # Read back
            read_mols = read_molecules(smi_path)
            assert len(read_mols) == 3
            assert read_mols[0] == ('CCO', 'ethanol')
            assert read_mols[1] == ('c1ccccc1', 'benzene')
            assert read_mols[2] == ('CC(=O)O', 'acetic_acid')
        finally:
            if os.path.exists(smi_path):
                os.remove(smi_path)

    def test_write_smi_without_names(self):
        """Test writing SMI with molecules without names."""
        molecules = [
            ('CCO', None),
            ('c1ccccc1', None)
        ]

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.smi') as f:
            smi_path = f.name

        try:
            write_molecules(molecules, smi_path, format='smi')

            # Read file content
            with open(smi_path, 'r') as f:
                lines = f.readlines()

            assert len(lines) == 2
            assert lines[0].strip() == 'CCO'
            assert lines[1].strip() == 'c1ccccc1'
        finally:
            if os.path.exists(smi_path):
                os.remove(smi_path)

    def test_write_smi_invalid_smiles(self):
        """Test that writing invalid SMILES raises ValueError."""
        molecules = [
            ('INVALID_SMILES', 'bad_mol')
        ]

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.smi') as f:
            smi_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid SMILES"):
                write_molecules(molecules, smi_path, format='smi')
        finally:
            if os.path.exists(smi_path):
                os.remove(smi_path)


class TestWriteMoleculesErrors:
    """Tests for error handling in write_molecules."""

    def test_unsupported_format(self):
        """Test that ValueError is raised for unsupported format."""
        molecules = [('CCO', 'ethanol')]

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            output_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                write_molecules(molecules, output_path, format='xyz')
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


class TestIntegration:
    """Integration tests for I/O operations."""

    def test_format_conversion_smi_to_csv(self):
        """Test converting SMI to CSV."""
        # Read SMI
        molecules = read_molecules('tests/fixtures/sample.smi')

        # Write CSV
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            csv_path = f.name

        try:
            write_molecules(molecules, csv_path, format='csv')

            # Read CSV back
            csv_mols = read_molecules(csv_path)

            # Compare
            assert len(csv_mols) == len(molecules)
            for orig, read in zip(molecules, csv_mols):
                assert orig == read
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_format_conversion_csv_to_sdf(self):
        """Test converting CSV to SDF."""
        # Read CSV
        molecules = read_molecules('tests/fixtures/sample.csv')

        # Write SDF
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sdf') as f:
            sdf_path = f.name

        try:
            write_molecules(molecules, sdf_path, format='sdf')

            # Read SDF back
            sdf_mols = read_molecules(sdf_path)

            # Compare names
            orig_names = {n for s, n in molecules}
            read_names = {n for s, n in sdf_mols}
            assert orig_names == read_names
        finally:
            if os.path.exists(sdf_path):
                os.remove(sdf_path)
