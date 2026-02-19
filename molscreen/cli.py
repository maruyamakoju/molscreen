"""
Command-line interface for molscreen.

This module provides the CLI commands for molecular screening.
"""

import sys
import click
import pandas as pd
from molscreen.properties import calculate_properties, check_lipinski, MoleculeError
from molscreen.models import predict_solubility
from molscreen.report import (
    generate_full_report,
    format_console_output,
    save_json_report,
    save_html_report
)
from molscreen.filters import (
    lipinski_filter,
    veber_filter,
    pains_filter,
    filter_molecules
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
@click.option('--smiles', '-s', type=str,
              help='SMILES string to filter')
@click.option('--input', '-i', 'input_file', type=click.Path(exists=True),
              help='Input CSV file with SMILES column')
@click.option('--output', '-o', 'output_file', type=click.Path(),
              help='Output CSV file for results')
@click.option('--rules', '-r', type=str, default='all',
              help='Comma-separated filter rules (lipinski,veber,pains) or "all"')
@click.option('--quiet', '-q', is_flag=True,
              help='Suppress console output')
def filter(smiles, input_file, output_file, rules, quiet):
    """
    Apply pharmacokinetic filters to molecules.

    This command filters molecules using drug-likeness criteria:
    - Lipinski's Rule of Five (oral bioavailability)
    - Veber rules (TPSA and rotatable bonds)
    - PAINS detection (assay interference)

    You can either filter a single SMILES string or batch process a CSV file.

    Examples:

        molscreen filter --smiles "CCO" --rules lipinski,veber

        molscreen filter --smiles "CC(=O)Oc1ccccc1C(=O)O" --rules all

        molscreen filter --input molecules.csv --output filtered.csv --rules all
    """
    try:
        # Parse rules
        if rules.lower() == 'all':
            rules_list = ['lipinski', 'veber', 'pains']
        else:
            rules_list = [r.strip().lower() for r in rules.split(',')]

        # Validate rules
        valid_rules = {'lipinski', 'veber', 'pains'}
        for rule in rules_list:
            if rule not in valid_rules:
                click.echo(f"Error: Invalid rule '{rule}'. Valid rules: {', '.join(valid_rules)}", err=True)
                sys.exit(1)

        # Single SMILES mode
        if smiles:
            if input_file:
                click.echo("Error: Cannot specify both --smiles and --input", err=True)
                sys.exit(1)

            results = filter_molecules([smiles], rules=rules_list)
            result = results[0]

            if not result['valid']:
                click.echo(f"Error: {result.get('error', 'Invalid SMILES')}", err=True)
                sys.exit(1)

            # Display results
            if not quiet:
                click.echo("=" * 60)
                click.echo("MOLECULAR FILTER RESULTS")
                click.echo("=" * 60)
                click.echo(f"\nSMILES: {smiles}")
                click.echo(f"\nOverall: {'✓ PASS' if result['overall_pass'] else '✗ FAIL'}")

                for rule_name, rule_result in result['filters'].items():
                    click.echo(f"\n--- {rule_name.upper()} Filter ---")
                    click.echo(f"  Status: {'✓ Pass' if rule_result['passes'] else '✗ Fail'}")

                    if rule_result['violations']:
                        click.echo("  Violations:")
                        for violation in rule_result['violations']:
                            click.echo(f"    - {violation}")

                    if rule_result['values']:
                        click.echo("  Values:")
                        for key, value in rule_result['values'].items():
                            if isinstance(value, float):
                                click.echo(f"    {key}: {value:.2f}")
                            else:
                                click.echo(f"    {key}: {value}")

                click.echo("\n" + "=" * 60)

            # Save to CSV if requested
            if output_file:
                df = pd.DataFrame([{
                    'SMILES': result['smiles'],
                    'Valid': result['valid'],
                    'Overall_Pass': result['overall_pass'],
                    **{f'{rule}_Pass': result['filters'][rule]['passes'] for rule in rules_list},
                    **{f'{rule}_Violations': ', '.join(result['filters'][rule]['violations']) for rule in rules_list}
                }])
                df.to_csv(output_file, index=False)
                if not quiet:
                    click.echo(f"Results saved to: {output_file}")

            sys.exit(0 if result['overall_pass'] else 1)

        # Batch CSV mode
        elif input_file:
            # Read input CSV
            try:
                df = pd.read_csv(input_file)
            except Exception as e:
                click.echo(f"Error reading CSV file: {e}", err=True)
                sys.exit(1)

            if 'SMILES' not in df.columns:
                click.echo("Error: Input CSV must have a 'SMILES' column", err=True)
                sys.exit(1)

            smiles_list = df['SMILES'].tolist()

            if not quiet:
                click.echo(f"Processing {len(smiles_list)} molecules...")

            # Apply filters
            results = filter_molecules(smiles_list, rules=rules_list)

            # Prepare output data
            output_data = []
            for result in results:
                row = {
                    'SMILES': result['smiles'],
                    'Valid': result['valid'],
                    'Overall_Pass': result['overall_pass']
                }

                if result['valid']:
                    for rule in rules_list:
                        rule_result = result['filters'][rule]
                        row[f'{rule}_Pass'] = rule_result['passes']
                        row[f'{rule}_Violations'] = ', '.join(rule_result['violations']) if rule_result['violations'] else ''

                        # Add values
                        for key, value in rule_result['values'].items():
                            row[f'{rule}_{key}'] = value
                else:
                    row['Error'] = result.get('error', 'Unknown error')

                output_data.append(row)

            output_df = pd.DataFrame(output_data)

            # Save or display
            if output_file:
                output_df.to_csv(output_file, index=False)
                if not quiet:
                    click.echo(f"Results saved to: {output_file}")
            elif not quiet:
                # Display summary
                valid_count = sum(1 for r in results if r['valid'])
                pass_count = sum(1 for r in results if r.get('overall_pass', False))

                click.echo(f"\nProcessed: {len(results)} molecules")
                click.echo(f"Valid: {valid_count}")
                click.echo(f"Passed all filters: {pass_count}")
                click.echo(f"\nFirst 10 results:")
                click.echo(output_df.head(10).to_string())

            sys.exit(0)

        else:
            click.echo("Error: Must specify either --smiles or --input", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
