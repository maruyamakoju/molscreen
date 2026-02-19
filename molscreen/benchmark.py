"""
Benchmark evaluation module for QSAR models.

This module provides:
- QSARModel: Flexible class for regression and classification tasks
- run_qsar_benchmark: Generic benchmark function for any dataset
- run_solubility_benchmark: Benchmark on Delaney solubility dataset
- run_bbbp_benchmark: Benchmark on BBBP permeability dataset
"""

import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score
from rdkit import Chem
from rdkit.Chem import Descriptors

from molscreen.properties import smiles_to_mol


class QSARModel:
    """
    Flexible QSAR model supporting both regression and classification tasks.

    Automatically detects task type from target values and uses appropriate
    metrics and models.
    """

    def __init__(self, task_type: Optional[str] = None, random_state: int = 42):
        """
        Initialize QSAR model.

        Args:
            task_type: 'regression' or 'classification' (auto-detected if None)
            random_state: Random seed for reproducibility
        """
        self.task_type = task_type
        self.random_state = random_state
        self.model = None
        self.feature_names: List[str] = [
            'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors',
            'TPSA', 'NumRotatableBonds', 'NumAromaticRings',
            'NumAliphaticRings', 'NumSaturatedRings'
        ]

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
                print(f"Warning: Skipping invalid SMILES '{smiles}': {e}")
                continue

        return np.array(descriptors_list)

    def _detect_task_type(self, y: np.ndarray) -> str:
        """
        Detect whether task is regression or classification.

        Args:
            y: Target values

        Returns:
            'regression' or 'classification'
        """
        unique_values = np.unique(y)

        # If only 2-10 unique integer values, likely classification
        if len(unique_values) <= 10 and np.all(y == y.astype(int)):
            return 'classification'
        else:
            return 'regression'

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the QSAR model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
        """
        # Auto-detect task type if not specified
        if self.task_type is None:
            self.task_type = self._detect_task_type(y)

        # Create appropriate model
        if self.task_type == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )

        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted values
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        return self.model.predict(X)

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation and return metrics.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            cv: Number of cross-validation folds

        Returns:
            Dictionary with metrics (mean and std)
        """
        # Auto-detect task type if not specified
        if self.task_type is None:
            self.task_type = self._detect_task_type(y)

        # Create appropriate model
        if self.task_type == 'classification':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

            # Calculate metrics
            accuracy_scores = cross_val_score(model, X, y, cv=cv_splitter, scoring='accuracy')

            # For AUC, we need probability predictions
            auc_scores = []
            for train_idx, test_idx in cv_splitter.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model_copy = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                model_copy.fit(X_train, y_train)

                # Get probability predictions for positive class
                if len(np.unique(y)) == 2:
                    y_pred_proba = model_copy.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_pred_proba)
                    auc_scores.append(auc)

            return {
                'task_type': 'classification',
                'accuracy_mean': float(np.mean(accuracy_scores)),
                'accuracy_std': float(np.std(accuracy_scores)),
                'auc_mean': float(np.mean(auc_scores)) if auc_scores else None,
                'auc_std': float(np.std(auc_scores)) if auc_scores else None,
                'n_samples': len(y),
                'n_features': X.shape[1],
                'cv_folds': cv
            }
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

            # Calculate R² scores
            r2_scores = cross_val_score(model, X, y, cv=cv_splitter, scoring='r2')

            # Calculate RMSE scores
            rmse_scores = []
            for train_idx, test_idx in cv_splitter.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model_copy = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                model_copy.fit(X_train, y_train)
                y_pred = model_copy.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                rmse_scores.append(rmse)

            return {
                'task_type': 'regression',
                'r2_mean': float(np.mean(r2_scores)),
                'r2_std': float(np.std(r2_scores)),
                'rmse_mean': float(np.mean(rmse_scores)),
                'rmse_std': float(np.std(rmse_scores)),
                'n_samples': len(y),
                'n_features': X.shape[1],
                'cv_folds': cv
            }


def run_qsar_benchmark(
    dataset_path: str,
    target_col: str,
    smiles_col: str = 'smiles',
    cv: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Run QSAR benchmark on a dataset.

    Automatically detects task type (regression/classification) and returns
    appropriate metrics from cross-validation.

    Args:
        dataset_path: Path to CSV file
        target_col: Name of target column
        smiles_col: Name of SMILES column (default: 'smiles')
        cv: Number of cross-validation folds (default: 5)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with metrics:
        - For regression: r2_mean, r2_std, rmse_mean, rmse_std, n_samples, ...
        - For classification: auc_mean, auc_std, accuracy_mean, accuracy_std, n_samples, ...

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If required columns are missing
    """
    # Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    # Validate columns
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in dataset")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    # Create model
    model = QSARModel(random_state=random_state)

    # Calculate descriptors
    X = model._calculate_descriptors_batch(df[smiles_col].tolist())
    y = df[target_col].values[:len(X)]  # Match length in case some SMILES were skipped

    # Run cross-validation
    results = model.cross_validate(X, y, cv=cv)

    # Add dataset info
    results['dataset_path'] = dataset_path
    results['target_col'] = target_col
    results['smiles_col'] = smiles_col

    return results


def run_solubility_benchmark(cv: int = 5, random_state: int = 42) -> Dict[str, Any]:
    """
    Run benchmark on Delaney solubility dataset.

    Uses the built-in delaney_solubility.csv dataset for solubility prediction
    (regression task).

    Args:
        cv: Number of cross-validation folds (default: 5)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with regression metrics:
        - r2_mean: Mean R² score across folds
        - r2_std: Standard deviation of R² scores
        - rmse_mean: Mean RMSE across folds
        - rmse_std: Standard deviation of RMSE
        - n_samples: Number of samples
        - cv_folds: Number of CV folds used

    Example:
        >>> results = run_solubility_benchmark()
        >>> print(f"R² = {results['r2_mean']:.3f} ± {results['r2_std']:.3f}")
    """
    # Get package root directory (go up from molscreen/ to project root)
    package_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_path = os.path.join(package_dir, 'data', 'delaney_solubility.csv')

    return run_qsar_benchmark(
        dataset_path=dataset_path,
        target_col='measured_log_solubility_mol_l',
        smiles_col='smiles',
        cv=cv,
        random_state=random_state
    )


def run_bbbp_benchmark(cv: int = 5, random_state: int = 42) -> Dict[str, Any]:
    """
    Run benchmark on BBBP (Blood-Brain Barrier Permeability) dataset.

    Uses the built-in bbbp_subset.csv dataset for BBB permeability classification
    (binary classification task).

    Args:
        cv: Number of cross-validation folds (default: 5)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with classification metrics:
        - auc_mean: Mean ROC-AUC score across folds
        - auc_std: Standard deviation of ROC-AUC scores
        - accuracy_mean: Mean accuracy across folds
        - accuracy_std: Standard deviation of accuracy
        - n_samples: Number of samples
        - cv_folds: Number of CV folds used

    Example:
        >>> results = run_bbbp_benchmark()
        >>> print(f"AUC = {results['auc_mean']:.3f} ± {results['auc_std']:.3f}")
    """
    # Get package root directory (go up from molscreen/ to project root)
    package_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_path = os.path.join(package_dir, 'data', 'bbbp_subset.csv')

    return run_qsar_benchmark(
        dataset_path=dataset_path,
        target_col='p_np',
        smiles_col='smiles',
        cv=cv,
        random_state=random_state
    )
