"""
File I/O module for reading and writing molecules in various formats.

This module provides functions to:
- Read molecules from SDF, CSV, SMI, and TXT files
- Write molecules to CSV, SDF, and SMI files
- Automatically detect SMILES column in CSV files
- Handle invalid SMILES gracefully with warnings
"""

import os
import warnings
from typing import List, Optional, Tuple
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol


def detect_smiles_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect SMILES column name in a DataFrame.

    Checks for common SMILES column name variants:
    - smiles
    - SMILES
    - canonical_smiles
    - Smiles

    Args:
        df: pandas DataFrame to search

    Returns:
        Column name containing SMILES, or None if not found

    Example:
        >>> df = pd.DataFrame({'SMILES': ['CCO'], 'name': ['ethanol']})
        >>> detect_smiles_column(df)
        'SMILES'
    """
    # Common SMILES column name variants
    smiles_columns = ['smiles', 'SMILES', 'canonical_smiles', 'Smiles']

    for col in smiles_columns:
        if col in df.columns:
            return col

    return None


def read_molecules(file_path: str) -> List[Tuple[str, Optional[str]]]:
    """
    Read molecules from a file and return as (SMILES, name) tuples.

    Supported formats:
    - .sdf: SDF format using RDKit SDMolSupplier
    - .csv: CSV with automatic SMILES column detection
    - .smi: SMI format (one SMILES per line, optional space-separated name)
    - .txt: Plain text (one SMILES per line, optional space-separated name)

    Invalid SMILES are skipped with warnings.

    Args:
        file_path: Path to the input file

    Returns:
        List of (SMILES, name) tuples. Name may be None if not provided.

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported

    Example:
        >>> molecules = read_molecules('compounds.smi')
        >>> for smiles, name in molecules:
        ...     print(f"{name}: {smiles}")
    """
    file_path = str(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get file extension
    ext = Path(file_path).suffix.lower()

    molecules = []

    if ext == '.sdf':
        # Read SDF file
        supplier = Chem.SDMolSupplier(file_path)
        for mol in supplier:
            if mol is None:
                warnings.warn(f"Skipping invalid molecule in SDF file: {file_path}")
                continue

            # Convert to SMILES
            smiles = Chem.MolToSmiles(mol)

            # Get molecule name
            name = mol.GetProp('_Name') if mol.HasProp('_Name') else None
            if name == '':
                name = None

            molecules.append((smiles, name))

    elif ext == '.csv':
        # Read CSV file
        df = pd.read_csv(file_path)

        # Detect SMILES column
        smiles_col = detect_smiles_column(df)
        if smiles_col is None:
            raise ValueError(
                f"Could not find SMILES column in CSV file. "
                f"Expected one of: smiles, SMILES, canonical_smiles, Smiles"
            )

        # Detect name column (if exists)
        name_col = 'name' if 'name' in df.columns else None

        # Read molecules
        for idx, row in df.iterrows():
            smiles = str(row[smiles_col]).strip()

            # Validate SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                warnings.warn(f"Skipping invalid SMILES at row {idx}: {smiles}")
                continue

            # Get name if available
            name = None
            if name_col:
                name = str(row[name_col]).strip()
                if name == '' or name == 'nan':
                    name = None

            molecules.append((smiles, name))

    elif ext in ['.smi', '.txt']:
        # Read SMI/TXT file (one SMILES per line)
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Parse line (SMILES and optional name separated by whitespace)
                parts = line.split(None, 1)  # Split on first whitespace
                smiles = parts[0]
                name = parts[1] if len(parts) > 1 else None

                # Validate SMILES
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    warnings.warn(f"Skipping invalid SMILES at line {line_num}: {smiles}")
                    continue

                molecules.append((smiles, name))

    else:
        raise ValueError(
            f"Unsupported file format: {ext}. "
            f"Supported formats: .sdf, .csv, .smi, .txt"
        )

    return molecules


def write_molecules(
    molecules: List[Tuple[str, Optional[str]]],
    file_path: str,
    format: str = 'csv'
) -> None:
    """
    Write molecules to a file in the specified format.

    Supported formats:
    - 'csv': CSV file with smiles and name columns
    - 'sdf': SDF file with molecular structures
    - 'smi': SMI file (SMILES and optional name per line)

    Args:
        molecules: List of (SMILES, name) tuples
        file_path: Output file path
        format: Output format ('csv', 'sdf', or 'smi')

    Raises:
        ValueError: If format is not supported or SMILES is invalid

    Example:
        >>> molecules = [('CCO', 'ethanol'), ('c1ccccc1', 'benzene')]
        >>> write_molecules(molecules, 'output.csv', format='csv')
    """
    format = format.lower()

    if format not in ['csv', 'sdf', 'smi']:
        raise ValueError(
            f"Unsupported format: {format}. "
            f"Supported formats: csv, sdf, smi"
        )

    if format == 'csv':
        # Write CSV file
        data = []
        for smiles, name in molecules:
            data.append({
                'smiles': smiles,
                'name': name if name is not None else ''
            })

        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

    elif format == 'sdf':
        # Write SDF file
        writer = Chem.SDWriter(file_path)

        for smiles, name in molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")

            # Set molecule name
            if name is not None:
                mol.SetProp('_Name', name)

            writer.write(mol)

        writer.close()

    elif format == 'smi':
        # Write SMI file
        with open(file_path, 'w') as f:
            for smiles, name in molecules:
                # Validate SMILES
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError(f"Invalid SMILES: {smiles}")

                # Write SMILES and optional name
                if name is not None:
                    f.write(f"{smiles} {name}\n")
                else:
                    f.write(f"{smiles}\n")
