"""
QSAR model for solubility prediction using scikit-learn.

This module provides:
- Training a Random Forest model on Delaney/ESOL dataset
- Predicting aqueous solubility (logS) from SMILES
- Feature extraction using RDKit molecular descriptors
"""

import os
from typing import List, Optional, Tuple
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import Chem
from rdkit.Chem import Descriptors

from molscreen.properties import smiles_to_mol


class SolubilityModel:
    """QSAR model for predicting aqueous solubility."""

    def __init__(self):
        """Initialize the solubility model."""
        self.model: Optional[RandomForestRegressor] = None
        self.feature_names: List[str] = [
            'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors',
            'TPSA', 'NumRotatableBonds', 'NumAromaticRings',
            'NumAliphaticRings', 'NumSaturatedRings'
        ]
        self.is_trained = False

    def _calculate_descriptors(self, smiles: str) -> np.ndarray:
        """
        Calculate molecular descriptors from SMILES.

        Args:
            smiles: SMILES representation of the molecule

        Returns:
            Array of molecular descriptors

        Raises:
            ValueError: If SMILES is invalid
        """
        mol = smiles_to_mol(smiles)

        descriptors = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.NumSaturatedRings(mol)
        ]

        return np.array(descriptors).reshape(1, -1)

    def _calculate_descriptors_batch(self, smiles_list: List[str]) -> np.ndarray:
        """
        Calculate molecular descriptors for multiple SMILES.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            2D array of molecular descriptors
        """
        descriptors_list = []
        for smiles in smiles_list:
            try:
                desc = self._calculate_descriptors(smiles)
                descriptors_list.append(desc[0])
            except Exception as e:
                # Skip invalid molecules
                print(f"Warning: Skipping invalid SMILES '{smiles}': {e}")
                continue

        return np.array(descriptors_list)

    def train(self, data_path: Optional[str] = None, test_size: float = 0.2,
              random_state: int = 42) -> dict:
        """
        Train the solubility prediction model.

        Args:
            data_path: Path to CSV file with 'SMILES' and 'logS' columns
                      (default: package's built-in Delaney dataset)
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with training metrics (R2, RMSE)

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data is invalid
        """
        # Use built-in dataset if no path provided
        if data_path is None:
            package_dir = os.path.dirname(__file__)
            data_path = os.path.join(package_dir, 'data', 'delaney.csv')

        # Load data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_csv(data_path)
        if 'SMILES' not in df.columns or 'logS' not in df.columns:
            raise ValueError("Data must contain 'SMILES' and 'logS' columns")

        # Calculate descriptors
        X = self._calculate_descriptors_batch(df['SMILES'].tolist())
        y = df['logS'].values[:len(X)]  # Match length in case some SMILES were skipped

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        metrics = {
            'train_r2': r2_score(y_train, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_r2': r2_score(y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }

        return metrics

    def predict(self, smiles: str) -> float:
        """
        Predict aqueous solubility (logS) for a molecule.

        Args:
            smiles: SMILES representation of the molecule

        Returns:
            Predicted logS value (log10 of solubility in mol/L)

        Raises:
            RuntimeError: If model is not trained
            ValueError: If SMILES is invalid

        Example:
            >>> model = SolubilityModel()
            >>> model.train()
            >>> logS = model.predict("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        X = self._calculate_descriptors(smiles)
        prediction = self.model.predict(X)[0]

        return float(prediction)

    def predict_batch(self, smiles_list: List[str]) -> List[float]:
        """
        Predict solubility for multiple molecules.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            List of predicted logS values

        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        X = self._calculate_descriptors_batch(smiles_list)
        predictions = self.model.predict(X)

        return predictions.tolist()

    def save(self, path: str):
        """
        Save the trained model to disk.

        Args:
            path: File path to save the model

        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before saving")

        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }, f)

    def load(self, path: str):
        """
        Load a trained model from disk.

        Args:
            path: File path to load the model from

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']
            self.is_trained = data['is_trained']


def get_pretrained_model() -> SolubilityModel:
    """
    Get a pre-trained solubility model.

    Returns:
        Trained SolubilityModel instance

    Example:
        >>> model = get_pretrained_model()
        >>> logS = model.predict("CCO")  # Ethanol
    """
    model = SolubilityModel()
    model.train()
    return model


def predict_solubility(smiles: str, model: Optional[SolubilityModel] = None) -> dict:
    """
    Predict solubility and provide interpretation.

    Args:
        smiles: SMILES representation of the molecule
        model: Pre-trained model (if None, will train a new one)

    Returns:
        Dictionary with:
        - logS: Predicted log solubility
        - solubility_mol_per_L: Solubility in mol/L
        - solubility_mg_per_mL: Solubility in mg/mL (requires MW calculation)
        - interpretation: Human-readable interpretation

    Example:
        >>> result = predict_solubility("CC(=O)Oc1ccccc1C(=O)O")
        >>> print(result['interpretation'])
    """
    if model is None:
        model = get_pretrained_model()

    logS = model.predict(smiles)
    solubility_mol_per_L = 10 ** logS

    # Get molecular weight for mg/mL calculation
    from molscreen.properties import calculate_properties
    props = calculate_properties(smiles)
    mw = props['MW']
    solubility_mg_per_mL = solubility_mol_per_L * mw

    # Interpretation
    if logS >= -1:
        interpretation = "Highly soluble"
    elif logS >= -2:
        interpretation = "Soluble"
    elif logS >= -3:
        interpretation = "Moderately soluble"
    elif logS >= -4:
        interpretation = "Slightly soluble"
    else:
        interpretation = "Poorly soluble"

    return {
        'logS': logS,
        'solubility_mol_per_L': solubility_mol_per_L,
        'solubility_mg_per_mL': solubility_mg_per_mL,
        'interpretation': interpretation
    }
