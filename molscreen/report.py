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


def generate_interactive_html_report(
    results: list,
    output_path: str,
    title: str = "Molscreen Results"
) -> str:
    """
    Generate an interactive HTML report with Plotly visualizations.

    This function creates a self-contained HTML file with interactive charts
    showing molecular property distributions and Lipinski chemical space.

    Args:
        results: List of dictionaries containing molecular properties.
                 Each dictionary should have keys: smiles, mw/MW, logp/LogP,
                 hbd/HBD, hba/HBA, tpsa/TPSA, and optionally qsar_score.
                 Example: [{"smiles": "CCO", "mw": 46.07, "logp": -0.18,
                          "hbd": 1, "hba": 1, "tpsa": 20.23, "qsar_score": -0.5}]
        output_path: Path where the HTML file will be saved
        title: Title for the report (default: "Molscreen Results")

    Returns:
        str: Path to the generated HTML file

    Raises:
        ImportError: If plotly is not installed
        ValueError: If results list is invalid

    Example:
        >>> results = [
        ...     {"smiles": "CCO", "mw": 46.07, "logp": -0.18,
        ...      "hbd": 1, "hba": 1, "tpsa": 20.23},
        ...     {"smiles": "c1ccccc1", "mw": 78.11, "logp": 1.69,
        ...      "hbd": 0, "hba": 0, "tpsa": 0.0}
        ... ]
        >>> generate_interactive_html_report(results, "report.html")
        'report.html'

    Visualizations included:
        1. Molecular Weight (MW) histogram
        2. LogP vs MW scatter plot (Lipinski space with Ro5 boundaries)
        3. TPSA distribution
        4. QSAR score distribution (if qsar_score present in results)
    """
    # Import version info
    from molscreen import __version__

    # Check for plotly dependency
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError(
            "plotly is required for interactive reports. "
            "Install it with: pip install plotly"
        )

    # Validate input
    if not isinstance(results, list):
        raise ValueError("results must be a list of dictionaries")

    # Handle empty results
    if len(results) == 0:
        # Create minimal HTML for empty results
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="warning">
            <strong>No data available:</strong> No molecular results provided for visualization.
        </div>
    </div>
</body>
</html>"""
        with open(output_path, 'w') as f:
            f.write(html_content)
        return output_path

    # Normalize property keys (handle both lowercase and uppercase)
    def get_prop(result, key):
        """Get property value, trying both lowercase and uppercase keys."""
        return result.get(key.lower(), result.get(key.upper(), result.get(key)))

    # Extract data from results
    smiles_list = [get_prop(r, 'smiles') for r in results]
    mw_list = [get_prop(r, 'mw') for r in results]
    logp_list = [get_prop(r, 'logp') for r in results]
    hbd_list = [get_prop(r, 'hbd') for r in results]
    hba_list = [get_prop(r, 'hba') for r in results]
    tpsa_list = [get_prop(r, 'tpsa') for r in results]

    # Check if QSAR scores are present
    has_qsar = any('qsar_score' in r for r in results)
    qsar_list = [get_prop(r, 'qsar_score') for r in results] if has_qsar else None

    # Determine subplot layout
    n_plots = 4 if has_qsar else 3

    # Create subplots
    if has_qsar:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Molecular Weight Distribution',
                'Lipinski Chemical Space (LogP vs MW)',
                'TPSA Distribution',
                'QSAR Solubility Score Distribution'
            ),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "histogram"}]]
        )
    else:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Molecular Weight Distribution',
                'Lipinski Chemical Space (LogP vs MW)',
                'TPSA Distribution',
                ''
            ),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "histogram"}, None]]
        )

    # 1. Molecular Weight Histogram
    fig.add_trace(
        go.Histogram(
            x=mw_list,
            nbinsx=20,
            name='MW',
            marker_color='#3498db',
            hovertemplate='MW: %{x:.2f} g/mol<br>Count: %{y}<extra></extra>'
        ),
        row=1, col=1
    )

    # 2. LogP vs MW Scatter (Lipinski Space)
    # Color by Lipinski compliance
    colors = []
    for i in range(len(results)):
        mw_ok = mw_list[i] <= 500 if mw_list[i] is not None else False
        logp_ok = logp_list[i] <= 5 if logp_list[i] is not None else False
        hbd_ok = hbd_list[i] <= 5 if hbd_list[i] is not None else False
        hba_ok = hba_list[i] <= 10 if hba_list[i] is not None else False
        passes = all([mw_ok, logp_ok, hbd_ok, hba_ok])
        colors.append('#27ae60' if passes else '#e74c3c')

    fig.add_trace(
        go.Scatter(
            x=mw_list,
            y=logp_list,
            mode='markers',
            name='Molecules',
            marker=dict(
                size=8,
                color=colors,
                line=dict(width=1, color='white')
            ),
            text=smiles_list,
            hovertemplate='<b>%{text}</b><br>MW: %{x:.2f} g/mol<br>LogP: %{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )

    # Add Lipinski Rule of Five boundary lines
    # Vertical line at MW = 500
    valid_mw = [m for m in mw_list if m is not None]
    max_mw = max(valid_mw) if valid_mw else 500
    valid_logp = [lp for lp in logp_list if lp is not None]
    max_logp = max(valid_logp) if valid_logp else 5

    fig.add_trace(
        go.Scatter(
            x=[500, 500],
            y=[-5, max(max_logp, 5) + 1],
            mode='lines',
            name='MW ≤ 500',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=True,
            hovertemplate='Lipinski MW limit<extra></extra>'
        ),
        row=1, col=2
    )

    # Horizontal line at LogP = 5
    fig.add_trace(
        go.Scatter(
            x=[0, max(max_mw, 500) + 50],
            y=[5, 5],
            mode='lines',
            name='LogP ≤ 5',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=True,
            hovertemplate='Lipinski LogP limit<extra></extra>'
        ),
        row=1, col=2
    )

    # 3. TPSA Distribution
    fig.add_trace(
        go.Histogram(
            x=tpsa_list,
            nbinsx=20,
            name='TPSA',
            marker_color='#9b59b6',
            hovertemplate='TPSA: %{x:.2f} Ų<br>Count: %{y}<extra></extra>'
        ),
        row=2, col=1
    )

    # 4. QSAR Score Distribution (if available)
    if has_qsar:
        fig.add_trace(
            go.Histogram(
                x=qsar_list,
                nbinsx=20,
                name='QSAR Score',
                marker_color='#e67e22',
                hovertemplate='QSAR Score: %{x:.2f}<br>Count: %{y}<extra></extra>'
            ),
            row=2, col=2
        )

    # Update axes labels
    fig.update_xaxes(title_text="Molecular Weight (g/mol)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)

    fig.update_xaxes(title_text="Molecular Weight (g/mol)", row=1, col=2)
    fig.update_yaxes(title_text="LogP", row=1, col=2)

    fig.update_xaxes(title_text="TPSA (Ų)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    if has_qsar:
        fig.update_xaxes(title_text="QSAR Score (LogS)", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)

    # Update layout
    fig.update_layout(
        title_text=title,
        title_font_size=24,
        showlegend=True,
        height=800,
        template='plotly_white'
    )

    # Generate HTML with CDN-based plotly.js
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .summary {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .summary p {{
            margin: 5px 0;
        }}
        .legend {{
            margin: 20px 0;
            padding: 15px;
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            border-radius: 4px;
        }}
        .legend p {{
            margin: 5px 0;
        }}
        .footer {{
            text-align: center;
            color: #7f8c8d;
            font-size: 12px;
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid #ecf0f1;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="summary">
            <p><strong>Total Molecules:</strong> {len(results)}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="legend">
            <p><strong>Lipinski's Rule of Five:</strong></p>
            <p>🟢 Green points: Pass all Lipinski criteria (MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10)</p>
            <p>🔴 Red points: Fail one or more Lipinski criteria</p>
            <p>Red dashed lines: Lipinski boundary limits</p>
        </div>

        <div id="plotly-div"></div>

        <div class="footer">
            Generated by molscreen v{__version__} with Plotly {go.__version__ if hasattr(go, '__version__') else '6.5+'}
        </div>
    </div>

    <script>
        var plotlyData = {fig.to_json()};
        Plotly.newPlot('plotly-div', plotlyData.data, plotlyData.layout, {{responsive: true}});
    </script>
</body>
</html>"""

    # Write to file
    with open(output_path, 'w') as f:
        f.write(html_content)

    return output_path
