"""
Tests for molscreen.benchmark module.
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from molscreen.benchmark import (
    QSARModel,
    run_qsar_benchmark,
    run_solubility_benchmark,
    run_bbbp_benchmark
)


class TestDatasets:
    """Tests for benchmark datasets."""

    def test_delaney_solubility_exists(self):
        """Test that delaney_solubility.csv exists."""
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(__file__))
        dataset_path = os.path.join(project_root, 'data', 'delaney_solubility.csv')
        assert os.path.exists(dataset_path), "delaney_solubility.csv not found"

    def test_delaney_solubility_format(self):
        """Test that delaney_solubility.csv has correct format."""
        project_root = os.path.dirname(os.path.dirname(__file__))
        dataset_path = os.path.join(project_root, 'data', 'delaney_solubility.csv')

        df = pd.read_csv(dataset_path)

        # Check columns
        assert 'smiles' in df.columns
        assert 'compound' in df.columns
        assert 'measured_log_solubility_mol_l' in df.columns

        # Check number of rows (should be 30 molecules)
        assert len(df) == 30

        # Check that SMILES are non-empty strings
        assert all(isinstance(s, str) and len(s) > 0 for s in df['smiles'])

        # Check that solubility values are numeric
        assert df['measured_log_solubility_mol_l'].dtype in [np.float64, np.int64, float, int]

    def test_bbbp_subset_exists(self):
        """Test that bbbp_subset.csv exists."""
        project_root = os.path.dirname(os.path.dirname(__file__))
        dataset_path = os.path.join(project_root, 'data', 'bbbp_subset.csv')
        assert os.path.exists(dataset_path), "bbbp_subset.csv not found"

    def test_bbbp_subset_format(self):
        """Test that bbbp_subset.csv has correct format."""
        project_root = os.path.dirname(os.path.dirname(__file__))
        dataset_path = os.path.join(project_root, 'data', 'bbbp_subset.csv')

        df = pd.read_csv(dataset_path)

        # Check columns
        assert 'smiles' in df.columns
        assert 'name' in df.columns
        assert 'p_np' in df.columns

        # Check number of rows (should be around 50 molecules)
        assert len(df) >= 45  # Allow some flexibility

        # Check that SMILES are non-empty strings
        assert all(isinstance(s, str) and len(s) > 0 for s in df['smiles'])

        # Check that p_np values are binary (0 or 1)
        assert set(df['p_np'].unique()).issubset({0, 1})


class TestQSARModel:
    """Tests for QSARModel class."""

    def test_model_initialization(self):
        """Test that QSARModel can be initialized."""
        model = QSARModel()
        assert model is not None
        assert model.task_type is None
        assert model.model is None
        assert len(model.feature_names) == 9

    def test_model_with_task_type(self):
        """Test initialization with specific task type."""
        model_reg = QSARModel(task_type='regression')
        assert model_reg.task_type == 'regression'

        model_clf = QSARModel(task_type='classification')
        assert model_clf.task_type == 'classification'

    def test_calculate_descriptors(self):
        """Test descriptor calculation for a single molecule."""
        model = QSARModel()
        descriptors = model._calculate_descriptors("CCO")  # Ethanol

        assert descriptors.shape == (1, 9)
        assert np.all(np.isfinite(descriptors))

    def test_calculate_descriptors_batch(self):
        """Test batch descriptor calculation."""
        model = QSARModel()
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
        descriptors = model._calculate_descriptors_batch(smiles_list)

        assert descriptors.shape[0] == 3
        assert descriptors.shape[1] == 9
        assert np.all(np.isfinite(descriptors))

    def test_detect_task_type_regression(self):
        """Test task type detection for regression."""
        model = QSARModel()
        y_reg = np.array([1.5, 2.3, -0.5, 3.7, 1.2])
        task_type = model._detect_task_type(y_reg)
        assert task_type == 'regression'

    def test_detect_task_type_classification(self):
        """Test task type detection for classification."""
        model = QSARModel()
        y_clf = np.array([0, 1, 1, 0, 1, 0])
        task_type = model._detect_task_type(y_clf)
        assert task_type == 'classification'

    def test_fit_regression(self):
        """Test fitting a regression model."""
        model = QSARModel(task_type='regression')
        X = np.random.randn(20, 9)
        y = np.random.randn(20)

        model.fit(X, y)
        assert model.model is not None

    def test_fit_classification(self):
        """Test fitting a classification model."""
        model = QSARModel(task_type='classification')
        X = np.random.randn(20, 9)
        y = np.random.randint(0, 2, 20)

        model.fit(X, y)
        assert model.model is not None

    def test_predict_without_fit_fails(self):
        """Test that prediction fails if model not fitted."""
        model = QSARModel()
        X = np.random.randn(5, 9)

        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_cross_validate_regression(self):
        """Test cross-validation for regression."""
        model = QSARModel(task_type='regression')
        X = np.random.randn(30, 9)
        y = np.random.randn(30)

        results = model.cross_validate(X, y, cv=5)

        assert 'task_type' in results
        assert results['task_type'] == 'regression'
        assert 'r2_mean' in results
        assert 'r2_std' in results
        assert 'rmse_mean' in results
        assert 'rmse_std' in results
        assert 'n_samples' in results
        assert results['n_samples'] == 30

    def test_cross_validate_classification(self):
        """Test cross-validation for classification."""
        model = QSARModel(task_type='classification')
        X = np.random.randn(50, 9)
        y = np.random.randint(0, 2, 50)

        results = model.cross_validate(X, y, cv=5)

        assert 'task_type' in results
        assert results['task_type'] == 'classification'
        assert 'accuracy_mean' in results
        assert 'accuracy_std' in results
        assert 'auc_mean' in results
        assert 'auc_std' in results
        assert 'n_samples' in results
        assert results['n_samples'] == 50


class TestRunQSARBenchmark:
    """Tests for run_qsar_benchmark function."""

    def test_run_qsar_benchmark_with_custom_dataset(self):
        """Test run_qsar_benchmark with a custom dataset."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("smiles,target\n")
            f.write("CCO,-0.5\n")
            f.write("c1ccccc1,-2.0\n")
            f.write("CC(=O)O,-1.0\n")
            f.write("CCCC,-2.5\n")
            f.write("c1ccc(cc1)O,-1.5\n")
            f.write("CCC,-1.8\n")
            f.write("CC,-1.1\n")
            f.write("CCCCO,-0.6\n")
            f.write("c1ccc(cc1)Cl,-3.0\n")
            f.write("CCc1ccccc1,-3.2\n")
            temp_path = f.name

        try:
            results = run_qsar_benchmark(
                dataset_path=temp_path,
                target_col='target',
                smiles_col='smiles',
                cv=3
            )

            assert 'task_type' in results
            assert results['task_type'] == 'regression'
            assert 'r2_mean' in results
            assert 'rmse_mean' in results
            assert results['n_samples'] == 10
            assert results['cv_folds'] == 3

        finally:
            os.remove(temp_path)

    def test_run_qsar_benchmark_nonexistent_file(self):
        """Test that nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            run_qsar_benchmark(
                dataset_path='/nonexistent/file.csv',
                target_col='target'
            )

    def test_run_qsar_benchmark_missing_column(self):
        """Test that missing column raises error."""
        # Create temporary CSV without target column
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("smiles,other\n")
            f.write("CCO,1.0\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Target column"):
                run_qsar_benchmark(
                    dataset_path=temp_path,
                    target_col='target'
                )
        finally:
            os.remove(temp_path)


class TestRunSolubilityBenchmark:
    """Tests for run_solubility_benchmark function."""

    def test_run_solubility_benchmark_returns_dict(self):
        """Test that run_solubility_benchmark returns a dictionary."""
        results = run_solubility_benchmark()
        assert isinstance(results, dict)

    def test_run_solubility_benchmark_has_required_keys(self):
        """Test that results contain required keys."""
        results = run_solubility_benchmark()

        required_keys = ['task_type', 'r2_mean', 'r2_std', 'rmse_mean', 'rmse_std', 'n_samples']
        for key in required_keys:
            assert key in results, f"Missing key: {key}"

    def test_run_solubility_benchmark_is_regression(self):
        """Test that solubility benchmark is regression task."""
        results = run_solubility_benchmark()
        assert results['task_type'] == 'regression'

    def test_run_solubility_benchmark_r2_positive(self):
        """Test that R² score is >= 0.0 (model is learning)."""
        results = run_solubility_benchmark()
        assert results['r2_mean'] >= 0.0, "R² should be >= 0.0 (model should learn something)"

    def test_run_solubility_benchmark_n_samples(self):
        """Test that benchmark uses 30 samples."""
        results = run_solubility_benchmark()
        assert results['n_samples'] == 30

    def test_run_solubility_benchmark_metrics_are_numeric(self):
        """Test that all metrics are numeric."""
        results = run_solubility_benchmark()

        assert isinstance(results['r2_mean'], (int, float))
        assert isinstance(results['r2_std'], (int, float))
        assert isinstance(results['rmse_mean'], (int, float))
        assert isinstance(results['rmse_std'], (int, float))

    def test_run_solubility_benchmark_std_non_negative(self):
        """Test that standard deviations are non-negative."""
        results = run_solubility_benchmark()

        assert results['r2_std'] >= 0
        assert results['rmse_std'] >= 0


class TestRunBBBPBenchmark:
    """Tests for run_bbbp_benchmark function."""

    def test_run_bbbp_benchmark_returns_dict(self):
        """Test that run_bbbp_benchmark returns a dictionary."""
        results = run_bbbp_benchmark()
        assert isinstance(results, dict)

    def test_run_bbbp_benchmark_has_required_keys(self):
        """Test that results contain required keys."""
        results = run_bbbp_benchmark()

        required_keys = ['task_type', 'accuracy_mean', 'accuracy_std', 'auc_mean', 'auc_std', 'n_samples']
        for key in required_keys:
            assert key in results, f"Missing key: {key}"

    def test_run_bbbp_benchmark_is_classification(self):
        """Test that BBBP benchmark is classification task."""
        results = run_bbbp_benchmark()
        assert results['task_type'] == 'classification'

    def test_run_bbbp_benchmark_auc_better_than_random(self):
        """Test that AUC is >= 0.5 (better than random)."""
        results = run_bbbp_benchmark()
        assert results['auc_mean'] >= 0.5, "AUC should be >= 0.5 (better than random guessing)"

    def test_run_bbbp_benchmark_accuracy_reasonable(self):
        """Test that accuracy is in valid range."""
        results = run_bbbp_benchmark()
        assert 0.0 <= results['accuracy_mean'] <= 1.0

    def test_run_bbbp_benchmark_n_samples(self):
        """Test that benchmark uses approximately 50 samples."""
        results = run_bbbp_benchmark()
        # Allow some flexibility (some SMILES may be invalid)
        assert 45 <= results['n_samples'] <= 50

    def test_run_bbbp_benchmark_metrics_are_numeric(self):
        """Test that all metrics are numeric."""
        results = run_bbbp_benchmark()

        assert isinstance(results['accuracy_mean'], (int, float))
        assert isinstance(results['accuracy_std'], (int, float))
        assert isinstance(results['auc_mean'], (int, float))
        assert isinstance(results['auc_std'], (int, float))

    def test_run_bbbp_benchmark_std_non_negative(self):
        """Test that standard deviations are non-negative."""
        results = run_bbbp_benchmark()

        assert results['accuracy_std'] >= 0
        assert results['auc_std'] >= 0


class TestBenchmarkIntegration:
    """Integration tests for benchmark module."""

    def test_both_benchmarks_complete_successfully(self):
        """Test that both benchmarks can run without errors."""
        sol_results = run_solubility_benchmark()
        bbbp_results = run_bbbp_benchmark()

        assert sol_results is not None
        assert bbbp_results is not None

    def test_solubility_benchmark_reproducible(self):
        """Test that benchmark with same random seed gives same results."""
        results1 = run_solubility_benchmark(random_state=42)
        results2 = run_solubility_benchmark(random_state=42)

        assert abs(results1['r2_mean'] - results2['r2_mean']) < 0.01
        assert abs(results1['rmse_mean'] - results2['rmse_mean']) < 0.01

    def test_bbbp_benchmark_reproducible(self):
        """Test that benchmark with same random seed gives same results."""
        results1 = run_bbbp_benchmark(random_state=42)
        results2 = run_bbbp_benchmark(random_state=42)

        assert abs(results1['accuracy_mean'] - results2['accuracy_mean']) < 0.01
        assert abs(results1['auc_mean'] - results2['auc_mean']) < 0.01

    def test_custom_cv_folds(self):
        """Test that custom CV folds work."""
        results = run_solubility_benchmark(cv=3)
        assert results['cv_folds'] == 3

        results = run_bbbp_benchmark(cv=10)
        assert results['cv_folds'] == 10
