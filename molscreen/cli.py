"""
Command-line interface for molscreen.

This module provides the CLI commands for molecular screening.
"""

import sys
import click
from molscreen.properties import calculate_properties, check_lipinski, MoleculeError
from molscreen.models import predict_solubility
from molscreen.report import (
    generate_full_report,
    format_console_output,
    save_json_report,
    save_html_report
)
from molscreen import __version__


@click.group()
@click.version_option(version=__version__, prog_name='molscreen')
def main():
    """
    molscreen: SMILES-based drug candidate screening tool

    A molecular screening tool that combines RDKit property calculations
    with machine learning-based solubility prediction.
    """
    pass


@main.command()
@click.argument('smiles')
@click.option('--json', '-j', 'json_output', type=click.Path(),
              help='Save JSON report to file')
@click.option('--html', '-h', 'html_output', type=click.Path(),
              help='Save HTML report to file')
@click.option('--no-solubility', is_flag=True,
              help='Skip solubility prediction')
@click.option('--quiet', '-q', is_flag=True,
              help='Suppress console output')
def predict(smiles, json_output, html_output, no_solubility, quiet):
    """
    Predict molecular properties and drug-likeness for a SMILES string.

    This command analyzes a molecule specified by its SMILES representation
    and provides:
    - Molecular properties (MW, LogP, HBD, HBA)
    - Lipinski's Rule of Five compliance
    - Aqueous solubility prediction (optional)

    Examples:

        molscreen predict "CCO"

        molscreen predict "CC(=O)Oc1ccccc1C(=O)O" --json aspirin.json

        molscreen predict "c1ccccc1" --html benzene.html --no-solubility
    """
    try:
        # Calculate properties
        properties = calculate_properties(smiles)
        lipinski = check_lipinski(properties=properties)

        # Predict solubility unless disabled
        solubility = None
        if not no_solubility:
            try:
                solubility = predict_solubility(smiles)
            except Exception as e:
                click.echo(f"Warning: Solubility prediction failed: {e}", err=True)

        # Save reports if requested
        if json_output:
            save_json_report(smiles, json_output, properties, lipinski, solubility)
            if not quiet:
                click.echo(f"JSON report saved to: {json_output}")

        if html_output:
            save_html_report(smiles, html_output, properties, lipinski, solubility)
            if not quiet:
                click.echo(f"HTML report saved to: {html_output}")

        # Print to console unless quiet
        if not quiet:
            console_output = format_console_output(smiles, properties, lipinski, solubility)
            click.echo(console_output)

        # Exit with success
        sys.exit(0)

    except MoleculeError as e:
        click.echo(f"Error: Invalid SMILES string: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('smiles')
def properties(smiles):
    """
    Calculate basic molecular properties for a SMILES string.

    This command calculates and displays:
    - Molecular weight (g/mol)
    - LogP (octanol-water partition coefficient)
    - Number of hydrogen bond donors
    - Number of hydrogen bond acceptors

    Example:

        molscreen properties "CCO"
    """
    try:
        props = calculate_properties(smiles)

        click.echo("Molecular Properties:")
        click.echo(f"  SMILES:           {smiles}")
        click.echo(f"  Molecular Weight: {props['MW']:.2f} g/mol")
        click.echo(f"  LogP:             {props['LogP']:.2f}")
        click.echo(f"  H-Bond Donors:    {props['HBD']}")
        click.echo(f"  H-Bond Acceptors: {props['HBA']}")

        sys.exit(0)

    except MoleculeError as e:
        click.echo(f"Error: Invalid SMILES string: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('smiles')
def lipinski(smiles):
    """
    Check Lipinski's Rule of Five compliance for a SMILES string.

    Lipinski's Rule of Five is used to evaluate drug-likeness.
    A compound is considered drug-like if it satisfies:
    - Molecular weight ≤ 500 Da
    - LogP ≤ 5
    - Hydrogen bond donors ≤ 5
    - Hydrogen bond acceptors ≤ 10

    Example:

        molscreen lipinski "CC(=O)Oc1ccccc1C(=O)O"
    """
    try:
        results = check_lipinski(smiles)
        props = calculate_properties(smiles)

        click.echo("Lipinski's Rule of Five:")
        click.echo(f"  SMILES: {smiles}")
        click.echo(f"\n  Criteria:")
        click.echo(f"    MW ≤ 500:     {'✓ Pass' if results['MW_ok'] else '✗ Fail'} ({props['MW']:.2f})")
        click.echo(f"    LogP ≤ 5:     {'✓ Pass' if results['LogP_ok'] else '✗ Fail'} ({props['LogP']:.2f})")
        click.echo(f"    HBD ≤ 5:      {'✓ Pass' if results['HBD_ok'] else '✗ Fail'} ({props['HBD']})")
        click.echo(f"    HBA ≤ 10:     {'✓ Pass' if results['HBA_ok'] else '✗ Fail'} ({props['HBA']})")
        click.echo(f"\n  Overall: {'✓ PASSES' if results['passes_lipinski'] else '✗ FAILS'}")

        # Exit with 0 if passes, 1 if fails (useful for scripting)
        sys.exit(0 if results['passes_lipinski'] else 1)

    except MoleculeError as e:
        click.echo(f"Error: Invalid SMILES string: {e}", err=True)
        sys.exit(2)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)


@main.command()
@click.argument('smiles')
def solubility(smiles):
    """
    Predict aqueous solubility for a SMILES string.

    Uses a QSAR model trained on the Delaney (ESOL) dataset
    to predict log solubility (logS).

    Example:

        molscreen solubility "CCO"
    """
    try:
        result = predict_solubility(smiles)

        click.echo("Solubility Prediction:")
        click.echo(f"  SMILES:           {smiles}")
        click.echo(f"  LogS:             {result['logS']:.2f}")
        click.echo(f"  Solubility:       {result['solubility_mg_per_mL']:.2f} mg/mL")
        click.echo(f"  Interpretation:   {result['interpretation']}")

        sys.exit(0)

    except MoleculeError as e:
        click.echo(f"Error: Invalid SMILES string: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
