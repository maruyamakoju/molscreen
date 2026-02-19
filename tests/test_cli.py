"""
Tests for molscreen.cli module.
"""

import os
import tempfile
import json
import pandas as pd
from click.testing import CliRunner
import pytest
from molscreen.cli import main, predict, properties, lipinski, solubility, admet


class TestCLIMain:
    """Tests for main CLI entry point."""

    def test_main_help(self):
        """Test that main command shows help."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'molscreen' in result.output
        assert 'SMILES-based drug candidate screening tool' in result.output

    def test_main_version(self):
        """Test that version flag works."""
        runner = CliRunner()
        result = runner.invoke(main, ['--version'])
        assert result.exit_code == 0
        assert 'molscreen' in result.output or 'version' in result.output.lower()

    def test_main_no_command(self):
        """Test that main without command shows usage."""
        runner = CliRunner()
        result = runner.invoke(main, [])
        # Click returns exit code 2 when no command is provided
        assert result.exit_code in [0, 2]  # Either is acceptable
        assert 'Usage:' in result.output or 'Commands:' in result.output


class TestPredictCommand:
    """Tests for predict command."""

    def test_predict_aspirin(self):
        """Test predict command with aspirin."""
        runner = CliRunner()
        result = runner.invoke(predict, ['CC(=O)Oc1ccccc1C(=O)O'])
        assert result.exit_code == 0
        assert 'MOLECULAR SCREENING REPORT' in result.output
        assert 'Lipinski' in result.output
        assert 'Solubility' in result.output

    def test_predict_ethanol(self):
        """Test predict command with ethanol."""
        runner = CliRunner()
        result = runner.invoke(predict, ['CCO'])
        assert result.exit_code == 0
        assert 'SMILES: CCO' in result.output
        assert 'PASSES' in result.output  # Should pass Lipinski

    def test_predict_invalid_smiles(self):
        """Test predict with invalid SMILES."""
        runner = CliRunner()
        result = runner.invoke(predict, ['INVALID_SMILES'])
        assert result.exit_code == 1
        assert 'Error' in result.output

    def test_predict_json_output(self):
        """Test predict with JSON output."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json_path = f.name

        try:
            result = runner.invoke(predict, ['CCO', '--json', json_path])
            assert result.exit_code == 0
            assert os.path.exists(json_path)

            # Verify JSON content
            with open(json_path, 'r') as f:
                data = json.load(f)
            assert 'smiles' in data
            assert 'properties' in data
            assert 'lipinski' in data
            assert data['smiles'] == 'CCO'
        finally:
            if os.path.exists(json_path):
                os.remove(json_path)

    def test_predict_html_output(self):
        """Test predict with HTML output."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as f:
            html_path = f.name

        try:
            result = runner.invoke(predict, ['CCO', '--html', html_path])
            assert result.exit_code == 0
            assert os.path.exists(html_path)

            # Verify HTML content
            with open(html_path, 'r') as f:
                content = f.read()
            assert '<html' in content
            assert 'CCO' in content
        finally:
            if os.path.exists(html_path):
                os.remove(html_path)

    def test_predict_both_outputs(self):
        """Test predict with both JSON and HTML outputs."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, 'test.json')
            html_path = os.path.join(tmpdir, 'test.html')

            result = runner.invoke(predict, [
                'CCO',
                '--json', json_path,
                '--html', html_path
            ])
            assert result.exit_code == 0
            assert os.path.exists(json_path)
            assert os.path.exists(html_path)

    def test_predict_no_solubility(self):
        """Test predict with solubility disabled."""
        runner = CliRunner()
        result = runner.invoke(predict, ['CCO', '--no-solubility'])
        assert result.exit_code == 0
        assert 'MOLECULAR SCREENING REPORT' in result.output
        # Should not have solubility section
        assert 'Solubility Prediction' not in result.output

    def test_predict_quiet_mode(self):
        """Test predict in quiet mode."""
        runner = CliRunner()
        result = runner.invoke(predict, ['CCO', '--quiet'])
        assert result.exit_code == 0
        # Output should be minimal (no report)
        assert 'MOLECULAR SCREENING REPORT' not in result.output

    def test_predict_quiet_with_output(self):
        """Test predict quiet mode with file output."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json_path = f.name

        try:
            result = runner.invoke(predict, ['CCO', '--json', json_path, '--quiet'])
            assert result.exit_code == 0
            # Should still create the file
            assert os.path.exists(json_path)
        finally:
            if os.path.exists(json_path):
                os.remove(json_path)


class TestPropertiesCommand:
    """Tests for properties command."""

    def test_properties_ethanol(self):
        """Test properties command with ethanol."""
        runner = CliRunner()
        result = runner.invoke(properties, ['CCO'])
        assert result.exit_code == 0
        assert 'Molecular Properties' in result.output
        assert 'SMILES:' in result.output
        assert 'CCO' in result.output
        assert 'Molecular Weight:' in result.output
        assert 'LogP:' in result.output

    def test_properties_aspirin(self):
        """Test properties command with aspirin."""
        runner = CliRunner()
        result = runner.invoke(properties, ['CC(=O)Oc1ccccc1C(=O)O'])
        assert result.exit_code == 0
        assert '180.16' in result.output  # MW of aspirin

    def test_properties_invalid_smiles(self):
        """Test properties with invalid SMILES."""
        runner = CliRunner()
        result = runner.invoke(properties, ['INVALID'])
        assert result.exit_code == 1
        assert 'Error' in result.output


class TestLipinskiCommand:
    """Tests for lipinski command."""

    def test_lipinski_pass(self):
        """Test lipinski command with passing molecule."""
        runner = CliRunner()
        result = runner.invoke(lipinski, ['CCO'])  # Ethanol passes
        assert result.exit_code == 0
        assert 'Lipinski' in result.output
        assert 'PASSES' in result.output

    def test_lipinski_aspirin(self):
        """Test lipinski command with aspirin."""
        runner = CliRunner()
        result = runner.invoke(lipinski, ['CC(=O)Oc1ccccc1C(=O)O'])
        assert result.exit_code == 0
        assert 'PASSES' in result.output

    def test_lipinski_shows_criteria(self):
        """Test that lipinski shows all criteria."""
        runner = CliRunner()
        result = runner.invoke(lipinski, ['CCO'])
        assert result.exit_code == 0
        assert 'MW' in result.output
        assert 'LogP' in result.output
        assert 'HBD' in result.output
        assert 'HBA' in result.output

    def test_lipinski_invalid_smiles(self):
        """Test lipinski with invalid SMILES."""
        runner = CliRunner()
        result = runner.invoke(lipinski, ['INVALID'])
        assert result.exit_code == 2
        assert 'Error' in result.output


class TestSolubilityCommand:
    """Tests for solubility command."""

    def test_solubility_ethanol(self):
        """Test solubility command with ethanol."""
        runner = CliRunner()
        result = runner.invoke(solubility, ['CCO'])
        assert result.exit_code == 0
        assert 'Solubility Prediction' in result.output
        assert 'LogS:' in result.output
        assert 'Interpretation:' in result.output

    def test_solubility_aspirin(self):
        """Test solubility command with aspirin."""
        runner = CliRunner()
        result = runner.invoke(solubility, ['CC(=O)Oc1ccccc1C(=O)O'])
        assert result.exit_code == 0
        assert 'SMILES:' in result.output
        assert 'mg/mL' in result.output

    def test_solubility_invalid_smiles(self):
        """Test solubility with invalid SMILES."""
        runner = CliRunner()
        result = runner.invoke(solubility, ['INVALID'])
        assert result.exit_code == 1
        assert 'Error' in result.output


class TestADMETCommand:
    """Tests for admet command."""

    def test_admet_command_with_smiles(self):
        """Test admet command with single SMILES."""
        runner = CliRunner()
        result = runner.invoke(admet, ['--smiles', 'CCO'])
        assert result.exit_code == 0
        assert 'ADMET PREDICTION REPORT' in result.output
        assert 'Absorption' in result.output
        assert 'Distribution' in result.output
        assert 'Metabolism' in result.output
        assert 'Excretion' in result.output
        assert 'Toxicity' in result.output
        assert 'Overall Score' in result.output

    def test_admet_command_aspirin(self):
        """Test admet command with aspirin."""
        runner = CliRunner()
        result = runner.invoke(admet, ['--smiles', 'CC(=O)Oc1ccccc1C(=O)O'])
        assert result.exit_code == 0
        assert 'CC(=O)Oc1ccccc1C(=O)O' in result.output
        assert 'Overall Score' in result.output

    def test_admet_invalid_smiles(self):
        """Test admet with invalid SMILES."""
        runner = CliRunner()
        result = runner.invoke(admet, ['--smiles', 'INVALID'])
        assert result.exit_code == 1
        assert 'Error' in result.output

    def test_admet_help(self):
        """Test admet command help text."""
        runner = CliRunner()
        result = runner.invoke(admet, ['--help'])
        assert result.exit_code == 0
        assert 'ADMET' in result.output
        assert 'Absorption' in result.output or 'absorption' in result.output.lower()

    def test_admet_no_options(self):
        """Test admet without options shows error."""
        runner = CliRunner()
        result = runner.invoke(admet, [])
        assert result.exit_code == 1
        assert 'Error' in result.output

    def test_admet_both_options_error(self):
        """Test that using both --smiles and --input is an error."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write('SMILES\nCCO\n')
            csv_path = f.name

        try:
            result = runner.invoke(admet, ['--smiles', 'CCO', '--input', csv_path])
            assert result.exit_code == 1
            assert 'Error' in result.output
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_admet_batch_mode(self):
        """Test admet batch processing mode."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create input CSV
            input_csv = os.path.join(tmpdir, 'input.csv')
            output_csv = os.path.join(tmpdir, 'output.csv')

            df = pd.DataFrame({
                'SMILES': ['CCO', 'CC(=O)Oc1ccccc1C(=O)O', 'c1ccccc1'],
                'Name': ['Ethanol', 'Aspirin', 'Benzene']
            })
            df.to_csv(input_csv, index=False)

            # Run batch processing
            result = runner.invoke(admet, ['--input', input_csv, '--output', output_csv])
            assert result.exit_code == 0
            assert os.path.exists(output_csv)

            # Verify output CSV
            result_df = pd.read_csv(output_csv)
            assert len(result_df) == 3
            assert 'SMILES' in result_df.columns
            assert 'overall_score' in result_df.columns
            assert 'caco2_class' in result_df.columns

    def test_admet_batch_no_output_error(self):
        """Test that batch mode requires --output."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write('SMILES\nCCO\n')
            csv_path = f.name

        try:
            result = runner.invoke(admet, ['--input', csv_path])
            assert result.exit_code == 1
            assert 'Error' in result.output
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_admet_batch_invalid_csv(self):
        """Test batch mode with invalid CSV (missing SMILES column)."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv = os.path.join(tmpdir, 'input.csv')
            output_csv = os.path.join(tmpdir, 'output.csv')

            # Create CSV without SMILES column
            df = pd.DataFrame({
                'Name': ['Ethanol', 'Aspirin'],
                'Formula': ['C2H6O', 'C9H8O4']
            })
            df.to_csv(input_csv, index=False)

            result = runner.invoke(admet, ['--input', input_csv, '--output', output_csv])
            assert result.exit_code == 1
            assert 'Error' in result.output

    def test_admet_batch_with_invalid_smiles(self):
        """Test batch mode handles invalid SMILES gracefully."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv = os.path.join(tmpdir, 'input.csv')
            output_csv = os.path.join(tmpdir, 'output.csv')

            df = pd.DataFrame({
                'SMILES': ['CCO', 'INVALID_SMILES', 'c1ccccc1'],
            })
            df.to_csv(input_csv, index=False)

            result = runner.invoke(admet, ['--input', input_csv, '--output', output_csv])
            assert result.exit_code == 0  # Should complete despite error
            assert os.path.exists(output_csv)

            # Check that output has all rows (including the error one)
            result_df = pd.read_csv(output_csv)
            assert len(result_df) == 3


class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_all_commands_available(self):
        """Test that all commands are registered."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'predict' in result.output
        assert 'properties' in result.output
        assert 'lipinski' in result.output
        assert 'solubility' in result.output
        assert 'admet' in result.output

    def test_predict_help(self):
        """Test that predict command has help."""
        runner = CliRunner()
        result = runner.invoke(predict, ['--help'])
        assert result.exit_code == 0
        assert 'SMILES' in result.output

    def test_complete_workflow(self):
        """Test complete workflow with file outputs."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, 'molecule.json')
            html_path = os.path.join(tmpdir, 'molecule.html')

            # Run predict with outputs
            result = runner.invoke(predict, [
                'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
                '--json', json_path,
                '--html', html_path
            ])

            # Verify success
            assert result.exit_code == 0

            # Verify files exist
            assert os.path.exists(json_path)
            assert os.path.exists(html_path)

            # Verify JSON content
            with open(json_path, 'r') as f:
                data = json.load(f)
            assert data['lipinski']['passes_lipinski'] is True
            assert 'solubility' in data

            # Verify HTML content
            with open(html_path, 'r') as f:
                html = f.read()
            assert 'PASSES Lipinski' in html
            assert 'Solubility' in html
