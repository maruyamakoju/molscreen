"""
Command-line interface for molscreen using Click.
"""

import click


@click.group()
@click.version_option()
def main():
    """molscreen - Molecular screening tool for drug candidates"""
    pass


@main.command()
@click.argument('smiles')
@click.option('--format', '-f', type=click.Choice(['text', 'json', 'html']),
              default='text', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def predict(smiles, format, output):
    """Predict properties and screening results for a SMILES string"""
    # Implementation will be added later
    click.echo(f"Analyzing SMILES: {smiles}")
    click.echo(f"Output format: {format}")
    if output:
        click.echo(f"Output file: {output}")


if __name__ == '__main__':
    main()
