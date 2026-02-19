"""
Command-line interface for molscreen.

This module provides the CLI commands for molecular screening.
"""

import sys
import os
import click
from molscreen.properties import calculate_properties, check_lipinski, MoleculeError
from molscreen.models import predict_solubility
from molscreen.report import (
    generate_full_report,
    format_console_output,
    save_json_report,
    save_html_report
)
from molscreen.benchmark import (
    run_qsar_benchmark,
    run_solubility_benchmark,
    run_bbbp_benchmark
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
@click.option('--dataset', '-d', default='delaney',
              help='Dataset to use: "delaney", "bbbp", or path to CSV file')
@click.option('--target', '-t', type=str,
              help='Target column name (required for custom datasets)')
@click.option('--smiles', '-s', type=str, default='smiles',
              help='SMILES column name (default: "smiles")')
@click.option('--cv', type=int, default=5,
              help='Number of cross-validation folds (default: 5)')
def benchmark(dataset, target, smiles, cv):
    """
    Run QSAR model benchmarks on datasets.

    This command evaluates QSAR models using cross-validation on benchmark
    datasets. Built-in datasets include:

    - delaney: Delaney solubility dataset (regression, 30 molecules)
    - bbbp: Blood-brain barrier permeability dataset (classification, 50 molecules)

    You can also specify a custom CSV file with --dataset and provide
    --target and --smiles column names.

    Examples:

        molscreen benchmark --dataset delaney

        molscreen benchmark --dataset bbbp

        molscreen benchmark --dataset /path/to/data.csv --target activity --smiles smiles
    """
    try:
        # Handle built-in datasets
        if dataset == 'delaney':
            click.echo("Running benchmark on Delaney solubility dataset...")
            click.echo("=" * 60)
            results = run_solubility_benchmark(cv=cv)
        elif dataset == 'bbbp':
            click.echo("Running benchmark on BBBP dataset...")
            click.echo("=" * 60)
            results = run_bbbp_benchmark(cv=cv)
        else:
            # Custom dataset
            if not os.path.exists(dataset):
                click.echo(f"Error: Dataset file not found: {dataset}", err=True)
                sys.exit(1)

            if target is None:
                click.echo("Error: --target option is required for custom datasets", err=True)
                sys.exit(1)

            click.echo(f"Running benchmark on custom dataset: {dataset}")
            click.echo("=" * 60)
            results = run_qsar_benchmark(
                dataset_path=dataset,
                target_col=target,
                smiles_col=smiles,
                cv=cv
            )

        # Display results
        click.echo(f"\nBenchmark Results:")
        click.echo("-" * 60)
        click.echo(f"  Dataset:        {results.get('dataset_path', dataset)}")
        click.echo(f"  Task Type:      {results['task_type']}")
        click.echo(f"  Samples:        {results['n_samples']}")
        click.echo(f"  Features:       {results['n_features']}")
        click.echo(f"  CV Folds:       {results['cv_folds']}")
        click.echo()

        if results['task_type'] == 'regression':
            click.echo("  Metrics (Regression):")
            click.echo(f"    R² Score:     {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
            click.echo(f"    RMSE:         {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
        else:
            click.echo("  Metrics (Classification):")
            click.echo(f"    Accuracy:     {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
            if results.get('auc_mean') is not None:
                click.echo(f"    ROC-AUC:      {results['auc_mean']:.4f} ± {results['auc_std']:.4f}")

        click.echo("=" * 60)
        sys.exit(0)

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
