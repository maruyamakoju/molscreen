"""
Tests for molscreen.models module.
"""

import os
import tempfile
import pytest
import numpy as np
from molscreen.models import (
    SolubilityModel,
    get_pretrained_model,
    predict_solubility
)


class TestSolubilityModel:
    """Tests for SolubilityModel class."""

    def test_model_initialization(self):
        """Test model can be initialized."""
        model = SolubilityModel()
        assert model is not None
        assert model.is_trained is False
        assert model.model is None
        assert len(model.feature_names) == 9

    def test_model_training(self):
        """Test model can be trained."""
        model = SolubilityModel()
        metrics = model.train()

        assert model.is_trained is True
        assert model.model is not None
        assert 'train_r2' in metrics
        assert 'test_r2' in metrics
        assert 'train_rmse' in metrics
        assert 'test_rmse' in metrics
        assert metrics['train_r2'] > 0.5  # Reasonable performance
        assert metrics['test_r2'] > 0.5

    def test_model_training_reproducibility(self):
        """Test that training with same random seed gives same results."""
        model1 = SolubilityModel()
        metrics1 = model1.train(random_state=42)

        model2 = SolubilityModel()
        metrics2 = model2.train(random_state=42)

        assert abs(metrics1['test_r2'] - metrics2['test_r2']) < 0.01

    def test_calculate_descriptors(self):
        """Test descriptor calculation for a single molecule."""
        model = SolubilityModel()
        descriptors = model._calculate_descriptors("CCO")  # Ethanol

        assert descriptors.shape == (1, 9)
        assert np.all(np.isfinite(descriptors))  # No NaN or inf values

    def test_calculate_descriptors_batch(self):
        """Test descriptor calculation for multiple molecules."""
        model = SolubilityModel()
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
        descriptors = model._calculate_descriptors_batch(smiles_list)

        assert descriptors.shape[0] == 3
        assert descriptors.shape[1] == 9
        assert np.all(np.isfinite(descriptors))

    def test_calculate_descriptors_invalid_smiles(self):
        """Test that invalid SMILES raises error."""
        model = SolubilityModel()
        with pytest.raises(Exception):
            model._calculate_descriptors("INVALID_SMILES")

    def test_prediction_requires_training(self):
        """Test that prediction fails if model not trained."""
        model = SolubilityModel()
        with pytest.raises(RuntimeError):
            model.predict("CCO")

    def test_prediction_single_molecule(self):
        """Test prediction for a single molecule."""
        model = SolubilityModel()
        model.train()

        logS = model.predict("CCO")  # Ethanol
        assert isinstance(logS, float)
        assert -2 < logS < 2  # Reasonable range for ethanol

    def test_prediction_aspirin(self):
        """Test prediction for aspirin."""
        model = SolubilityModel()
        model.train()

        logS = model.predict("CC(=O)Oc1ccccc1C(=O)O")
        assert isinstance(logS, float)
        # Aspirin is moderately soluble
        assert -3 < logS < 0

    def test_prediction_batch(self):
        """Test batch prediction."""
        model = SolubilityModel()
        model.train()

        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
        predictions = model.predict_batch(smiles_list)

        assert len(predictions) == 3
        assert all(isinstance(p, float) for p in predictions)

    def test_save_untrained_model_fails(self):
        """Test that saving untrained model raises error."""
        model = SolubilityModel()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(RuntimeError):
                model.save(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_save_and_load_model(self):
        """Test saving and loading a trained model."""
        # Train and save
        model1 = SolubilityModel()
        model1.train(random_state=42)
        pred1 = model1.predict("CCO")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name

        try:
            model1.save(temp_path)

            # Load and predict
            model2 = SolubilityModel()
            model2.load(temp_path)
            pred2 = model2.predict("CCO")

            assert model2.is_trained is True
            assert abs(pred1 - pred2) < 0.01  # Same predictions
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_nonexistent_file_fails(self):
        """Test that loading nonexistent file raises error."""
        model = SolubilityModel()
        with pytest.raises(FileNotFoundError):
            model.load("/nonexistent/path/model.pkl")


class TestPretrainedModel:
    """Tests for pretrained model helper function."""

    def test_get_pretrained_model(self):
        """Test getting a pretrained model."""
        model = get_pretrained_model()

        assert model is not None
        assert model.is_trained is True
        assert model.model is not None

    def test_pretrained_model_can_predict(self):
        """Test that pretrained model can make predictions."""
        model = get_pretrained_model()
        logS = model.predict("CCO")

        assert isinstance(logS, float)
        assert np.isfinite(logS)


class TestPredictSolubility:
    """Tests for predict_solubility helper function."""

    def test_predict_solubility_structure(self):
        """Test that predict_solubility returns correct structure."""
        result = predict_solubility("CCO")

        assert 'logS' in result
        assert 'solubility_mol_per_L' in result
        assert 'solubility_mg_per_mL' in result
        assert 'interpretation' in result

    def test_predict_solubility_types(self):
        """Test that return values have correct types."""
        result = predict_solubility("CCO")

        assert isinstance(result['logS'], float)
        assert isinstance(result['solubility_mol_per_L'], float)
        assert isinstance(result['solubility_mg_per_mL'], float)
        assert isinstance(result['interpretation'], str)

    def test_predict_solubility_values_consistent(self):
        """Test that solubility values are mathematically consistent."""
        result = predict_solubility("CCO")

        # solubility_mol_per_L should equal 10^logS
        expected_mol_per_L = 10 ** result['logS']
        assert abs(result['solubility_mol_per_L'] - expected_mol_per_L) < 0.01

    def test_predict_solubility_aspirin(self):
        """Test prediction for aspirin."""
        result = predict_solubility("CC(=O)Oc1ccccc1C(=O)O")

        assert result['logS'] < 0  # Aspirin is not highly soluble
        assert result['interpretation'] in [
            "Highly soluble", "Soluble", "Moderately soluble",
            "Slightly soluble", "Poorly soluble"
        ]

    def test_predict_solubility_ethanol(self):
        """Test prediction for ethanol (highly soluble)."""
        result = predict_solubility("CCO")

        # Ethanol should be highly or very soluble
        assert result['logS'] > -2
        assert result['interpretation'] in ["Highly soluble", "Soluble"]

    def test_predict_solubility_benzene(self):
        """Test prediction for benzene (poorly soluble)."""
        result = predict_solubility("c1ccccc1")

        # Benzene should be poorly soluble in water
        assert result['logS'] < -1

    def test_predict_solubility_with_custom_model(self):
        """Test using a custom model for prediction."""
        model = SolubilityModel()
        model.train(random_state=42)

        result = predict_solubility("CCO", model=model)
        assert 'logS' in result

    def test_interpretation_categories(self):
        """Test that different solubilities get appropriate interpretations."""
        # Test multiple molecules to cover interpretation range
        molecules = {
            "CCO": "highly_or_soluble",  # Ethanol - very soluble
            "c1ccccc1": "poor_or_slight",  # Benzene - poorly soluble
            "CC(=O)O": "soluble_range"  # Acetic acid - moderate
        }

        for smiles, expected_category in molecules.items():
            result = predict_solubility(smiles)
            interpretation = result['interpretation']

            if expected_category == "highly_or_soluble":
                assert interpretation in ["Highly soluble", "Soluble", "Moderately soluble"]
            elif expected_category == "poor_or_slight":
                assert interpretation in ["Slightly soluble", "Poorly soluble", "Moderately soluble"]
            elif expected_category == "soluble_range":
                assert interpretation in [
                    "Highly soluble", "Soluble", "Moderately soluble",
                    "Slightly soluble", "Poorly soluble"
                ]


class TestModelMetrics:
    """Tests for model training metrics."""

    def test_metrics_have_all_keys(self):
        """Test that training returns all expected metrics."""
        model = SolubilityModel()
        metrics = model.train()

        expected_keys = ['train_r2', 'train_rmse', 'test_r2', 'test_rmse', 'n_train', 'n_test']
        for key in expected_keys:
            assert key in metrics

    def test_metrics_are_reasonable(self):
        """Test that metric values are in reasonable ranges."""
        model = SolubilityModel()
        metrics = model.train()

        # R² should be between 0 and 1 (can be negative for bad models, but ours should be good)
        assert 0 <= metrics['train_r2'] <= 1
        assert 0 <= metrics['test_r2'] <= 1

        # RMSE should be positive
        assert metrics['train_rmse'] > 0
        assert metrics['test_rmse'] > 0

        # Sample counts should be positive
        assert metrics['n_train'] > 0
        assert metrics['n_test'] > 0

    def test_train_performance_better_than_test(self):
        """Test that training performance is typically better than test."""
        model = SolubilityModel()
        metrics = model.train()

        # Train R² should generally be >= test R² (model sees training data)
        assert metrics['train_r2'] >= metrics['test_r2'] - 0.1  # Allow small margin
