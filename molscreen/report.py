"""
Report generation module for molecular screening results.

This module provides:
- JSON report generation
- HTML report generation using Jinja2 templates
- Formatting and presentation of molecular analysis results
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape

from molscreen.properties import calculate_properties, check_lipinski
from molscreen.models import predict_solubility


def generate_json_report(smiles: str,
                        properties: Optional[Dict] = None,
                        lipinski: Optional[Dict] = None,
                        solubility: Optional[Dict] = None,
                        include_metadata: bool = True) -> str:
    """
    Generate a JSON report for molecular screening results.

    Args:
        smiles: SMILES representation of the molecule
        properties: Molecular properties (if None, will calculate)
        lipinski: Lipinski compliance results (if None, will calculate)
        solubility: Solubility prediction results (optional)
        include_metadata: Whether to include metadata (timestamp, version)

    Returns:
        JSON string containing the report

    Example:
        >>> json_report = generate_json_report("CCO")
        >>> data = json.loads(json_report)
        >>> 'properties' in data
        True
    """
    # Calculate properties if not provided
    if properties is None:
        properties = calculate_properties(smiles)

    if lipinski is None:
        lipinski = check_lipinski(properties=properties)

    # Build report data
    report_data = {
        'smiles': smiles,
        'properties': properties,
        'lipinski': lipinski
    }

    # Add solubility if provided
    if solubility is not None:
        report_data['solubility'] = solubility

    # Add metadata if requested
    if include_metadata:
        from molscreen import __version__
        report_data['metadata'] = {
            'version': __version__,
            'timestamp': datetime.now().isoformat(),
            'generator': 'molscreen'
        }

    return json.dumps(report_data, indent=2)


def save_json_report(smiles: str,
                     output_path: str,
                     properties: Optional[Dict] = None,
                     lipinski: Optional[Dict] = None,
                     solubility: Optional[Dict] = None) -> None:
    """
    Generate and save a JSON report to a file.

    Args:
        smiles: SMILES representation of the molecule
        output_path: Path to save the JSON file
        properties: Molecular properties (if None, will calculate)
        lipinski: Lipinski compliance results (if None, will calculate)
        solubility: Solubility prediction results (optional)

    Example:
        >>> save_json_report("CCO", "output.json")
    """
    json_content = generate_json_report(smiles, properties, lipinski, solubility)

    with open(output_path, 'w') as f:
        f.write(json_content)


def generate_html_report(smiles: str,
                        properties: Optional[Dict] = None,
                        lipinski: Optional[Dict] = None,
                        solubility: Optional[Dict] = None) -> str:
    """
    Generate an HTML report for molecular screening results.

    Args:
        smiles: SMILES representation of the molecule
        properties: Molecular properties (if None, will calculate)
        lipinski: Lipinski compliance results (if None, will calculate)
        solubility: Solubility prediction results (optional)

    Returns:
        HTML string containing the formatted report

    Example:
        >>> html = generate_html_report("CCO")
        >>> '<html' in html
        True
    """
    # Calculate properties if not provided
    if properties is None:
        properties = calculate_properties(smiles)

    if lipinski is None:
        lipinski = check_lipinski(properties=properties)

    # Set up Jinja2 environment
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(['html', 'xml'])
    )

    # Load template
    template = env.get_template('report.html.j2')

    # Get version and timestamp
    from molscreen import __version__
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Render template
    html_content = template.render(
        smiles=smiles,
        properties=properties,
        lipinski=lipinski,
        solubility=solubility,
        version=__version__,
        timestamp=timestamp
    )

    return html_content


def save_html_report(smiles: str,
                     output_path: str,
                     properties: Optional[Dict] = None,
                     lipinski: Optional[Dict] = None,
                     solubility: Optional[Dict] = None) -> None:
    """
    Generate and save an HTML report to a file.

    Args:
        smiles: SMILES representation of the molecule
        output_path: Path to save the HTML file
        properties: Molecular properties (if None, will calculate)
        lipinski: Lipinski compliance results (if None, will calculate)
        solubility: Solubility prediction results (optional)

    Example:
        >>> save_html_report("CCO", "output.html")
    """
    html_content = generate_html_report(smiles, properties, lipinski, solubility)

    with open(output_path, 'w') as f:
        f.write(html_content)


def generate_full_report(smiles: str,
                        include_solubility: bool = True,
                        json_path: Optional[str] = None,
                        html_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a complete molecular screening report with all analyses.

    This is a high-level convenience function that:
    1. Calculates all molecular properties
    2. Checks Lipinski compliance
    3. Predicts solubility (if requested)
    4. Generates reports in requested formats

    Args:
        smiles: SMILES representation of the molecule
        include_solubility: Whether to include solubility prediction
        json_path: If provided, save JSON report to this path
        html_path: If provided, save HTML report to this path

    Returns:
        Dictionary containing all analysis results

    Example:
        >>> results = generate_full_report("CC(=O)Oc1ccccc1C(=O)O",
        ...                                json_path="aspirin.json",
        ...                                html_path="aspirin.html")
        >>> results['lipinski']['passes_lipinski']
        True
    """
    # Calculate all properties
    properties = calculate_properties(smiles)
    lipinski = check_lipinski(properties=properties)

    # Predict solubility if requested
    solubility = None
    if include_solubility:
        solubility = predict_solubility(smiles)

    # Generate reports if paths provided
    if json_path:
        save_json_report(smiles, json_path, properties, lipinski, solubility)

    if html_path:
        save_html_report(smiles, html_path, properties, lipinski, solubility)

    # Return all results
    results = {
        'smiles': smiles,
        'properties': properties,
        'lipinski': lipinski
    }

    if solubility:
        results['solubility'] = solubility

    return results


def format_console_output(smiles: str,
                         properties: Dict,
                         lipinski: Dict,
                         solubility: Optional[Dict] = None) -> str:
    """
    Format screening results for console output.

    Args:
        smiles: SMILES representation of the molecule
        properties: Molecular properties
        lipinski: Lipinski compliance results
        solubility: Solubility prediction results (optional)

    Returns:
        Formatted string for console display

    Example:
        >>> props = calculate_properties("CCO")
        >>> lip = check_lipinski(properties=props)
        >>> output = format_console_output("CCO", props, lip)
        >>> "SMILES" in output
        True
    """
    lines = []
    lines.append("=" * 60)
    lines.append("MOLECULAR SCREENING REPORT")
    lines.append("=" * 60)
    lines.append(f"\nSMILES: {smiles}")

    lines.append("\n--- Molecular Properties ---")
    lines.append(f"  Molecular Weight:  {properties['MW']:.2f} g/mol")
    lines.append(f"  LogP:              {properties['LogP']:.2f}")
    lines.append(f"  H-Bond Donors:     {properties['HBD']}")
    lines.append(f"  H-Bond Acceptors:  {properties['HBA']}")

    lines.append("\n--- Lipinski's Rule of Five ---")
    lines.append(f"  MW ≤ 500:     {'✓ Pass' if lipinski['MW_ok'] else '✗ Fail'} ({properties['MW']:.2f})")
    lines.append(f"  LogP ≤ 5:     {'✓ Pass' if lipinski['LogP_ok'] else '✗ Fail'} ({properties['LogP']:.2f})")
    lines.append(f"  HBD ≤ 5:      {'✓ Pass' if lipinski['HBD_ok'] else '✗ Fail'} ({properties['HBD']})")
    lines.append(f"  HBA ≤ 10:     {'✓ Pass' if lipinski['HBA_ok'] else '✗ Fail'} ({properties['HBA']})")
    lines.append(f"\n  Overall: {'✓ PASSES' if lipinski['passes_lipinski'] else '✗ FAILS'} Lipinski's Rule of Five")

    if solubility:
        lines.append("\n--- Solubility Prediction ---")
        lines.append(f"  LogS:             {solubility['logS']:.2f}")
        lines.append(f"  Solubility:       {solubility['solubility_mg_per_mL']:.2f} mg/mL")
        lines.append(f"  Interpretation:   {solubility['interpretation']}")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)
