"""
Command-line interface for molscreen.

This module provides the CLI commands for molecular screening.
"""

import sys
import click
import pandas as pd
from molscreen.properties import calculate_properties, check_lipinski, MoleculeError
from molscreen.models import predict_solubility
from molscreen.admet import predict_admet
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


@main.command()
@click.option('--smiles', '-s', help='SMILES string to analyze')
@click.option('--input', '-i', 'input_file', type=click.Path(exists=True),
              help='Input CSV file with SMILES column')
@click.option('--output', '-o', 'output_file', type=click.Path(),
              help='Output CSV file for batch processing')
def admet(smiles, input_file, output_file):
    """
    Predict ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties.

    This command provides comprehensive ADMET profiling using rule-based
    classification with RDKit descriptors.

    Single molecule mode:

        molscreen admet --smiles "CC(=O)Oc1ccccc1C(=O)O"

    Batch processing mode:

        molscreen admet --input molecules.csv --output results.csv

    For batch mode, the input CSV must have a 'SMILES' column.
    """
    # Validate input options
    if smiles and input_file:
        click.echo("Error: Cannot use both --smiles and --input options", err=True)
        sys.exit(1)

    if not smiles and not input_file:
        click.echo("Error: Must provide either --smiles or --input option", err=True)
        click.echo("Use 'molscreen admet --help' for usage information", err=True)
        sys.exit(1)

    # Single molecule mode
    if smiles:
        try:
            result = predict_admet(smiles)

            click.echo("=" * 60)
            click.echo("ADMET PREDICTION REPORT")
            click.echo("=" * 60)
            click.echo(f"\nSMILES: {smiles}")

            # Absorption
            click.echo("\n--- Absorption ---")
            click.echo(f"  Caco-2 Permeability:  {result['absorption']['caco2_class'].upper()}")
            ro5_symbol = "✓" if result['absorption']['bioavailability_ro5'] else "✗"
            click.echo(f"  Lipinski Ro5:         {ro5_symbol} {'PASS' if result['absorption']['bioavailability_ro5'] else 'FAIL'}")

            # Distribution
            click.echo("\n--- Distribution ---")
            bbb_symbol = "✓" if result['distribution']['bbb_penetrant'] else "✗"
            click.echo(f"  BBB Penetrant:        {bbb_symbol} {'YES' if result['distribution']['bbb_penetrant'] else 'NO'}")
            click.echo(f"  Volume Distribution:  {result['distribution']['vd_class'].upper()}")

            # Metabolism
            click.echo("\n--- Metabolism ---")
            cyp_symbol = "✓" if not result['metabolism']['cyp_alert'] else "✗"
            click.echo(f"  CYP Alert:            {cyp_symbol} {'NO ALERT' if not result['metabolism']['cyp_alert'] else 'ALERT'}")

            # Excretion
            click.echo("\n--- Excretion ---")
            renal_symbol = "✓" if result['excretion']['renal_clearance'] == 'likely' else "✗"
            click.echo(f"  Renal Clearance:      {renal_symbol} {result['excretion']['renal_clearance'].upper()}")

            # Toxicity
            click.echo("\n--- Toxicity ---")
            herg_symbol = "✓" if not result['toxicity']['herg_alert'] else "✗"
            ames_symbol = "✓" if not result['toxicity']['ames_alert'] else "✗"
            hepato_symbol = "✓" if not result['toxicity']['hepatotox_alert'] else "✗"
            click.echo(f"  hERG Alert:           {herg_symbol} {'NO ALERT' if not result['toxicity']['herg_alert'] else 'ALERT'}")
            click.echo(f"  Ames Alert:           {ames_symbol} {'NO ALERT' if not result['toxicity']['ames_alert'] else 'ALERT'}")
            click.echo(f"  Hepatotox Alert:      {hepato_symbol} {'NO ALERT' if not result['toxicity']['hepatotox_alert'] else 'ALERT'}")

            # Overall Score
            click.echo("\n--- Overall Assessment ---")
            score_percent = result['overall_score'] * 100
            click.echo(f"  Overall Score:        {score_percent:.1f}% (higher is better)")

            # Risk interpretation
            if result['overall_score'] >= 0.7:
                interpretation = "LOW RISK - Good drug-like properties"
            elif result['overall_score'] >= 0.5:
                interpretation = "MODERATE RISK - Some concerns identified"
            else:
                interpretation = "HIGH RISK - Multiple alerts detected"
            click.echo(f"  Risk Assessment:      {interpretation}")

            click.echo("\n" + "=" * 60)
            sys.exit(0)

        except MoleculeError as e:
            click.echo(f"Error: Invalid SMILES string: {e}", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

    # Batch processing mode
    elif input_file:
        if not output_file:
            click.echo("Error: --output option required for batch processing", err=True)
            sys.exit(1)

        try:
            # Read input CSV
            df = pd.read_csv(input_file)

            if 'SMILES' not in df.columns:
                click.echo("Error: Input CSV must have a 'SMILES' column", err=True)
                sys.exit(1)

            click.echo(f"Processing {len(df)} molecules from {input_file}...")

            # Initialize result columns
            results = []
            for idx, row in df.iterrows():
                smiles_str = row['SMILES']
                try:
                    result = predict_admet(smiles_str)

                    # Flatten ADMET result for CSV
                    flat_result = {
                        'SMILES': smiles_str,
                        'caco2_class': result['absorption']['caco2_class'],
                        'bioavailability_ro5': result['absorption']['bioavailability_ro5'],
                        'bbb_penetrant': result['distribution']['bbb_penetrant'],
                        'vd_class': result['distribution']['vd_class'],
                        'cyp_alert': result['metabolism']['cyp_alert'],
                        'renal_clearance': result['excretion']['renal_clearance'],
                        'herg_alert': result['toxicity']['herg_alert'],
                        'ames_alert': result['toxicity']['ames_alert'],
                        'hepatotox_alert': result['toxicity']['hepatotox_alert'],
                        'overall_score': result['overall_score'],
                    }
                    results.append(flat_result)

                except Exception as e:
                    click.echo(f"Warning: Failed to process SMILES '{smiles_str}': {e}", err=True)
                    # Add row with error marker
                    results.append({
                        'SMILES': smiles_str,
                        'caco2_class': 'ERROR',
                        'bioavailability_ro5': False,
                        'bbb_penetrant': False,
                        'vd_class': 'ERROR',
                        'cyp_alert': True,
                        'renal_clearance': 'ERROR',
                        'herg_alert': True,
                        'ames_alert': True,
                        'hepatotox_alert': True,
                        'overall_score': 0.0,
                    })

            # Create output DataFrame
            result_df = pd.DataFrame(results)

            # Save to CSV
            result_df.to_csv(output_file, index=False)
            click.echo(f"Results saved to {output_file}")
            click.echo(f"Successfully processed {len(results)} molecules")
            sys.exit(0)

        except FileNotFoundError:
            click.echo(f"Error: Input file '{input_file}' not found", err=True)
            sys.exit(1)
        except pd.errors.EmptyDataError:
            click.echo(f"Error: Input file '{input_file}' is empty", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


if __name__ == '__main__':
    main()
