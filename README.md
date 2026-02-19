# molscreen

SMILES-based drug candidate screening tool with RDKit + ML

## Features

- **Molecular Property Calculations**: Calculate molecular weight, LogP, hydrogen bond donors/acceptors using RDKit
- **Lipinski's Rule of Five**: Automatic drug-likeness assessment
- **Solubility Prediction**: QSAR model trained on Delaney/ESOL dataset using scikit-learn
- **Multiple Output Formats**: Console output, JSON, and HTML reports
- **CLI and Python API**: Use from command line or integrate into Python code

## Installation

### Requirements

- Python 3.10 or higher
- pip

### Install from source

```bash
# Clone the repository
git clone <repository-url>
cd molscreen

# Create virtual environment
python3 -m venv .venv

# Install package with development dependencies
.venv/bin/pip install -e .[dev]
```

## Publishing to PyPI

This project uses GitHub Actions to automatically publish to PyPI when a new version tag is pushed.

### Setup PyPI Token

Before publishing, you need to configure the PyPI API token as a GitHub secret:

1. Generate a PyPI API token:
   - Go to https://pypi.org/manage/account/token/
   - Create a new API token with upload permissions
   - Copy the token (it starts with `pypi-`)

2. Add the token to GitHub Secrets:
   - Go to your GitHub repository settings
   - Navigate to **Settings** → **Secrets and variables** → **Actions**
   - Click **New repository secret**
   - Name: `PYPI_TOKEN`
   - Value: Paste your PyPI API token
   - Click **Add secret**

### Publishing a New Release

To publish a new version to PyPI:

```bash
# Update version in pyproject.toml if needed
# Commit your changes
git add .
git commit -m "chore: bump version to X.Y.Z"

# Create and push a version tag
git tag vX.Y.Z
git push origin vX.Y.Z
```

The GitHub Actions workflow will automatically:
1. Build the package
2. Upload it to PyPI

You can monitor the workflow progress in the **Actions** tab of your GitHub repository.

## Quick Start

### Command Line Interface

```bash
# Analyze a molecule (aspirin)
molscreen predict "CC(=O)Oc1ccccc1C(=O)O"

# Save reports to files
molscreen predict "CC(=O)Oc1ccccc1C(=O)O" --json aspirin.json --html aspirin.html

# Get only molecular properties
molscreen properties "CCO"

# Check Lipinski's Rule of Five
molscreen lipinski "CC(=O)Oc1ccccc1C(=O)O"

# Predict solubility
molscreen solubility "c1ccccc1"

# Skip solubility prediction for faster results
molscreen predict "CCO" --no-solubility

# Quiet mode (only save to files)
molscreen predict "CCO" --json output.json --quiet
```

### Python API

```python
from molscreen.properties import calculate_properties, check_lipinski
from molscreen.models import predict_solubility
from molscreen.report import generate_full_report

# Calculate molecular properties
smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
properties = calculate_properties(smiles)
print(f"Molecular Weight: {properties['MW']:.2f} g/mol")
print(f"LogP: {properties['LogP']:.2f}")

# Check Lipinski's Rule of Five
lipinski = check_lipinski(smiles)
print(f"Passes Lipinski: {lipinski['passes_lipinski']}")

# Predict solubility
solubility = predict_solubility(smiles)
print(f"LogS: {solubility['logS']:.2f}")
print(f"Interpretation: {solubility['interpretation']}")

# Generate complete report
results = generate_full_report(
    smiles,
    json_path="report.json",
    html_path="report.html"
)
```

## Output Example

```
============================================================
MOLECULAR SCREENING REPORT
============================================================

SMILES: CC(=O)Oc1ccccc1C(=O)O

--- Molecular Properties ---
  Molecular Weight:  180.16 g/mol
  LogP:              1.31
  H-Bond Donors:     1
  H-Bond Acceptors:  3

--- Lipinski's Rule of Five ---
  MW ≤ 500:     ✓ Pass (180.16)
  LogP ≤ 5:     ✓ Pass (1.31)
  HBD ≤ 5:      ✓ Pass (1)
  HBA ≤ 10:     ✓ Pass (3)

  Overall: ✓ PASSES Lipinski's Rule of Five

--- Solubility Prediction ---
  LogS:             -1.51
  Solubility:       5.57 mg/mL
  Interpretation:   Soluble

============================================================
```

## Technical Details

### Molecular Properties

The following properties are calculated using RDKit:

- **Molecular Weight (MW)**: Molecular weight in g/mol
- **LogP**: Octanol-water partition coefficient (lipophilicity)
- **HBD**: Number of hydrogen bond donors
- **HBA**: Number of hydrogen bond acceptors

### Lipinski's Rule of Five

A molecule is considered drug-like if it satisfies:

- Molecular weight ≤ 500 Da
- LogP ≤ 5
- Hydrogen bond donors ≤ 5
- Hydrogen bond acceptors ≤ 10

### QSAR Solubility Model

- **Algorithm**: Random Forest Regression
- **Training Data**: Delaney (ESOL) dataset (96 compounds)
- **Features**: 9 molecular descriptors (MW, LogP, TPSA, rotatable bonds, etc.)
- **Performance**: R² = 0.90 on test set, RMSE = 0.59
- **Output**: LogS (log10 of solubility in mol/L)

## Development

### Running Tests

```bash
# Run all tests
.venv/bin/pytest tests/ -v

# Run specific test module
.venv/bin/pytest tests/test_properties.py -v

# Run with coverage
.venv/bin/pytest tests/ --cov=molscreen --cov-report=html
```

### Project Structure

```
molscreen/
├── __init__.py          # Package initialization
├── cli.py               # Command-line interface
├── properties.py        # Molecular property calculations
├── models.py            # QSAR solubility prediction
├── report.py            # Report generation (JSON/HTML)
├── data/
│   └── delaney.csv      # Training dataset
└── templates/
    └── report.html.j2   # HTML report template

tests/
├── test_properties.py   # Property calculation tests
├── test_models.py       # QSAR model tests
└── test_cli.py          # CLI tests
```

## License

MIT License

## Citation

If you use the Delaney dataset, please cite:

Delaney, J. S. (2004). ESOL: Estimating Aqueous Solubility Directly from Molecular Structure. *Journal of Chemical Information and Computer Sciences*, 44(3), 1000-1005.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
