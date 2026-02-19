"""
Tests for molscreen.report module.
"""

import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock

from molscreen.report import generate_interactive_html_report


class TestInteractiveHTMLReport:
    """Tests for interactive HTML report generation with Plotly."""

    @pytest.fixture
    def sample_results(self):
        """Sample molecular results for testing."""
        return [
            {
                "smiles": "CCO",
                "mw": 46.07,
                "logp": -0.18,
                "hbd": 1,
                "hba": 1,
                "tpsa": 20.23,
                "qsar_score": -0.5
            },
            {
                "smiles": "c1ccccc1",
                "mw": 78.11,
                "logp": 1.69,
                "hbd": 0,
                "hba": 0,
                "tpsa": 0.0,
                "qsar_score": -2.1
            },
            {
                "smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "mw": 180.16,
                "logp": 1.31,
                "hbd": 1,
                "hba": 3,
                "tpsa": 63.60,
                "qsar_score": -1.5
            }
        ]

    @pytest.fixture
    def sample_results_uppercase(self):
        """Sample results with uppercase property keys."""
        return [
            {
                "smiles": "CCO",
                "MW": 46.07,
                "LogP": -0.18,
                "HBD": 1,
                "HBA": 1,
                "TPSA": 20.23
            }
        ]

    @pytest.fixture
    def sample_results_no_qsar(self):
        """Sample results without QSAR scores."""
        return [
            {
                "smiles": "CCO",
                "mw": 46.07,
                "logp": -0.18,
                "hbd": 1,
                "hba": 1,
                "tpsa": 20.23
            },
            {
                "smiles": "c1ccccc1",
                "mw": 78.11,
                "logp": 1.69,
                "hbd": 0,
                "hba": 0,
                "tpsa": 0.0
            }
        ]

    def test_generate_interactive_html_report_creates_file(self, sample_results):
        """Test that generate_interactive_html_report creates an HTML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name

        try:
            result = generate_interactive_html_report(
                sample_results,
                output_path,
                title="Test Report"
            )

            assert result == output_path
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_interactive_report_contains_plotly(self, sample_results):
        """Test that generated HTML contains plotly references."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name

        try:
            generate_interactive_html_report(
                sample_results,
                output_path
            )

            with open(output_path, 'r') as f:
                content = f.read()

            # Check for plotly CDN reference
            assert 'plotly' in content.lower()
            assert 'cdn.plot.ly' in content or 'plotly.min.js' in content

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_interactive_report_contains_data(self, sample_results):
        """Test that generated HTML contains molecular data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name

        try:
            generate_interactive_html_report(
                sample_results,
                output_path
            )

            with open(output_path, 'r') as f:
                content = f.read()

            # Check that data is present in HTML
            # The data should be in the plotly JSON
            assert 'CCO' in content or '46.07' in content
            # Check for visualization references
            assert 'Molecular Weight' in content or 'MW' in content
            assert 'LogP' in content or 'logp' in content
            assert 'TPSA' in content or 'tpsa' in content

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_interactive_report_empty_results(self):
        """Test that function handles empty results gracefully."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name

        try:
            result = generate_interactive_html_report(
                [],
                output_path,
                title="Empty Report"
            )

            assert result == output_path
            assert os.path.exists(output_path)

            with open(output_path, 'r') as f:
                content = f.read()

            # Should contain warning about no data
            assert 'No data' in content or 'no data' in content or 'empty' in content.lower()

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_interactive_report_valid_html(self, sample_results):
        """Test that generated output is valid HTML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name

        try:
            generate_interactive_html_report(
                sample_results,
                output_path
            )

            with open(output_path, 'r') as f:
                content = f.read()

            # Check for basic HTML structure
            assert '<html' in content.lower()
            assert '<body' in content.lower()
            assert '</body>' in content.lower()
            assert '</html>' in content.lower()
            assert '<head' in content.lower()
            assert '</head>' in content.lower()

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_interactive_report_with_qsar_scores(self, sample_results):
        """Test that QSAR scores are visualized when present."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name

        try:
            generate_interactive_html_report(
                sample_results,
                output_path
            )

            with open(output_path, 'r') as f:
                content = f.read()

            # Check for QSAR-related content
            assert 'qsar' in content.lower() or 'solubility' in content.lower()

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_interactive_report_without_qsar_scores(self, sample_results_no_qsar):
        """Test graceful handling when QSAR scores are missing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name

        try:
            result = generate_interactive_html_report(
                sample_results_no_qsar,
                output_path
            )

            assert result == output_path
            assert os.path.exists(output_path)

            with open(output_path, 'r') as f:
                content = f.read()

            # Should still have other visualizations
            assert 'Molecular Weight' in content or 'MW' in content
            assert 'LogP' in content or 'logp' in content

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_interactive_report_uppercase_keys(self, sample_results_uppercase):
        """Test that function handles uppercase property keys."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name

        try:
            result = generate_interactive_html_report(
                sample_results_uppercase,
                output_path
            )

            assert result == output_path
            assert os.path.exists(output_path)

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_plotly_import_error_handling(self, sample_results):
        """Test that ImportError is raised when plotly is not available."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name

        try:
            # Mock the import to raise ImportError
            with patch('builtins.__import__', side_effect=ImportError("No module named 'plotly'")):
                with pytest.raises(ImportError) as exc_info:
                    generate_interactive_html_report(
                        sample_results,
                        output_path
                    )

                # Check that an ImportError was raised (message may vary based on how it's caught)
                assert "plotly" in str(exc_info.value).lower()

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_invalid_results_type(self):
        """Test that ValueError is raised for invalid results type."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                generate_interactive_html_report(
                    "invalid",  # Should be a list
                    output_path
                )

            assert "must be a list" in str(exc_info.value).lower()

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_custom_title(self, sample_results):
        """Test that custom title is used in the report."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name

        custom_title = "My Custom Molscreen Report"

        try:
            generate_interactive_html_report(
                sample_results,
                output_path,
                title=custom_title
            )

            with open(output_path, 'r') as f:
                content = f.read()

            assert custom_title in content

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_lipinski_compliance_visualization(self, sample_results):
        """Test that Lipinski compliance is visualized."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name

        try:
            generate_interactive_html_report(
                sample_results,
                output_path
            )

            with open(output_path, 'r') as f:
                content = f.read()

            # Check for Lipinski-related content
            assert 'lipinski' in content.lower() or 'rule' in content.lower()
            # Check for boundary indicators
            assert '500' in content  # MW limit
            assert '5' in content  # LogP limit

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
