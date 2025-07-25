"""
File: test_memorization_calculator.py
Directory: memorization_reproduction/tests/

Comprehensive tests for memorization_calculator.py module.
Validates core memorization calculations and mathematical correctness.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from unittest.mock import Mock, patch

# Import the modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from memorization_calculator import (
    calculate_compression_rate,
    calculate_joint_compression_rate,
    calculate_unintended_memorization,
    calculate_total_memorization,
    calculate_memorization_per_sequence,
    calculate_baseline_entropy,
    validate_memorization_calculation,
    create_synthetic_reference_model,
    calculate_mutual_information_approximation,
    batch_calculate_memorization
)
from model_trainer import ModelConfig, GPTModel, create_gpt_model


class TestCompressionRateCalculation:
    """Test basic compression rate calculations."""
    
    def setup_method(self):
        """Set up test models and data."""
        self.device = "cpu"
        self.vocab_size = 10
        self.seq_length = 5
        
        # Create simple test model
        self.config = ModelConfig(
            n_layers=1,
            d_model=16,
            n_heads=2,
            vocab_size=self.vocab_size,
            max_seq_length=20
        )
        self.model = create_gpt_model(self.config)
        
        # Create test sequence
        self.test_sequence = torch.tensor([1, 2, 3, 4, 5])
    
    def test_basic_compression_rate(self):
        """Test basic compression rate calculation."""
        compression_rate = calculate_compression_rate(
            self.model, self.test_sequence, self.device
        )
        
        assert compression_rate > 0  # Should be positive (negative log-likelihood)
        assert isinstance(compression_rate, float)
    
    def test_compression_rate_deterministic(self):
        """Test that compression rate is deterministic."""
        rate1 = calculate_compression_rate(self.model, self.test_sequence, self.device)
        rate2 = calculate_compression_rate(self.model, self.test_sequence, self.device)
        
        assert abs(rate1 - rate2) < 1e-6  # Should be identical
    
    def test_compression_rate_manual_calculation(self):
        """Test compression rate against manual calculation."""
        sequence = torch.tensor([1, 2])  # Simple 2-token sequence
        
        # Calculate using our function
        calculated_rate = calculate_compression_rate(self.model, sequence, self.device)
        
        # Manual calculation
        self.model.eval()
        with torch.no_grad():
            logits = self.model(sequence.unsqueeze(0))
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = sequence[1:].unsqueeze(0)
            
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_prob = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1))
            manual_nll = -token_log_prob.sum().item()
            manual_rate = manual_nll / math.log(2)  # Convert to bits
        
        assert abs(calculated_rate - manual_rate) < 1e-6
    
    def test_different_sequence_lengths(self):
        """Test compression rate with different sequence lengths."""
        short_seq = torch.tensor([1, 2])
        long_seq = torch.tensor([1, 2, 3, 4, 5, 6])
        
        short_rate = calculate_compression_rate(self.model, short_seq, self.device)
        long_rate = calculate_compression_rate(self.model, long_seq, self.device)
        
        # Longer sequences should generally have higher total compression rates
        assert long_rate > short_rate
        assert short_rate > 0
        assert long_rate > 0
    
    def test_single_token_sequence(self):
        """Test compression rate with single token (edge case)."""
        single_token = torch.tensor([3])
        
        # Should handle gracefully but return 0 (no prediction needed)
        rate = calculate_compression_rate(self.model, single_token, self.device)
        assert rate == 0.0  # No tokens to predict


class TestJointCompressionRate:
    """Test joint compression rate calculations."""
    
    def setup_method(self):
        """Set up test models and data."""
        self.device = "cpu"
        self.vocab_size = 10
        
        self.config = ModelConfig(
            n_layers=1,
            d_model=16,
            n_heads=2,
            vocab_size=self.vocab_size,
            max_seq_length=20
        )
        
        # Create two different models
        self.model1 = create_gpt_model(self.config)
        self.model2 = create_gpt_model(self.config)
        
        # Initialize with different weights
        with torch.no_grad():
            for p1, p2 in zip(self.model1.parameters(), self.model2.parameters()):
                p2.data = p1.data + 0.1 * torch.randn_like(p1.data)
        
        self.test_sequence = torch.tensor([1, 2, 3, 4])
    
    def test_joint_compression_basic(self):
        """Test basic joint compression calculation."""
        joint_rate = calculate_joint_compression_rate(
            self.model1, self.model2, self.test_sequence, self.device
        )
        
        assert joint_rate > 0
        assert isinstance(joint_rate, float)
    
    def test_joint_compression_vs_individual(self):
        """Test that joint compression is better than or equal to individual."""
        model1_rate = calculate_compression_rate(self.model1, self.test_sequence, self.device)
        model2_rate = calculate_compression_rate(self.model2, self.test_sequence, self.device)
        joint_rate = calculate_joint_compression_rate(
            self.model1, self.model2, self.test_sequence, self.device
        )
        
        # Joint should be better (lower) than both individual rates
        assert joint_rate <= model1_rate + 1e-6  # Small tolerance for numerical errors
        assert joint_rate <= model2_rate + 1e-6
    
    def test_joint_compression_identical_models(self):
        """Test joint compression with identical models."""
        joint_rate = calculate_joint_compression_rate(
            self.model1, self.model1, self.test_sequence, self.device
        )
        individual_rate = calculate_compression_rate(
            self.model1, self.test_sequence, self.device
        )
        
        # Should be identical when models are the same
        assert abs(joint_rate - individual_rate) < 1e-6
    
    def test_joint_compression_manual_verification(self):
        """Test joint compression against manual max calculation."""
        sequence = torch.tensor([1, 2])
        
        calculated_joint = calculate_joint_compression_rate(
            self.model1, self.model2, sequence, self.device
        )
        
        # Manual calculation
        self.model1.eval()
        self.model2.eval()
        with torch.no_grad():
            logits1 = self.model1(sequence.unsqueeze(0))
            logits2 = self.model2(sequence.unsqueeze(0))
            
            shift_logits1 = logits1[..., :-1, :].contiguous()
            shift_logits2 = logits2[..., :-1, :].contiguous()
            shift_labels = sequence[1:].unsqueeze(0)
            
            log_probs1 = F.log_softmax(shift_logits1, dim=-1)
            log_probs2 = F.log_softmax(shift_logits2, dim=-1)
            
            token_log_probs1 = log_probs1.gather(dim=-1, index=shift_labels.unsqueeze(-1))
            token_log_probs2 = log_probs2.gather(dim=-1, index=shift_labels.unsqueeze(-1))
            
            max_log_probs = torch.maximum(token_log_probs1.squeeze(-1), token_log_probs2.squeeze(-1))
            manual_joint_nll = -max_log_probs.sum().item()
            manual_joint_rate = manual_joint_nll / math.log(2)
        
        assert abs(calculated_joint - manual_joint_rate) < 1e-6


class TestUnintendedMemorization:
    """Test unintended memorization calculations."""
    
    def setup_method(self):
        """Set up test models and data."""
        self.device = "cpu"
        self.vocab_size = 10
        
        self.config = ModelConfig(
            n_layers=1,
            d_model=16,
            n_heads=2,
            vocab_size=self.vocab_size,
            max_seq_length=20
        )
        
        self.target_model = create_gpt_model(self.config)
        self.reference_model = create_gpt_model(self.config)
        
        # Make reference model different (and potentially worse)
        with torch.no_grad():
            for param in self.reference_model.parameters():
                param.data = torch.randn_like(param.data) * 0.1
        
        self.test_sequence = torch.tensor([1, 2, 3, 4])
    
    def test_unintended_memorization_basic(self):
        """Test basic unintended memorization calculation."""
        memorization = calculate_unintended_memorization(
            self.target_model, self.reference_model, self.test_sequence, self.device
        )
        
        assert memorization >= 0  # Should be non-negative
        assert isinstance(memorization, float)
    
    def test_unintended_memorization_formula(self):
        """Test unintended memorization formula manually."""
        ref_rate = calculate_compression_rate(
            self.reference_model, self.test_sequence, self.device
        )
        joint_rate = calculate_joint_compression_rate(
            self.target_model, self.reference_model, self.test_sequence, self.device
        )
        
        expected_memorization = max(0.0, ref_rate - joint_rate)
        
        calculated_memorization = calculate_unintended_memorization(
            self.target_model, self.reference_model, self.test_sequence, self.device
        )
        
        assert abs(calculated_memorization - expected_memorization) < 1e-6
    
    def test_memorization_identical_models(self):
        """Test memorization when target and reference models are identical."""
        memorization = calculate_unintended_memorization(
            self.target_model, self.target_model, self.test_sequence, self.device
        )
        
        # Should be 0 when models are identical
        assert memorization < 1e-6
    
    def test_memorization_non_negative(self):
        """Test that memorization is always non-negative."""
        # Create a scenario where target model might be worse than reference
        sequences = [torch.randint(0, self.vocab_size, (6,)) for _ in range(10)]
        
        for sequence in sequences:
            memorization = calculate_unintended_memorization(
                self.target_model, self.reference_model, sequence, self.device
            )
            assert memorization >= 0


class TestTotalMemorization:
    """Test total memorization calculations."""
    
    def setup_method(self):
        """Set up test models and data."""
        self.device = "cpu"
        self.vocab_size = 10
        
        self.config = ModelConfig(
            n_layers=1,
            d_model=16,
            n_heads=2,
            vocab_size=self.vocab_size,
            max_seq_length=20
        )
        
        self.target_model = create_gpt_model(self.config)
        self.reference_model = create_gpt_model(self.config)
        
        self.test_dataset = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5, 6]),
            torch.tensor([7, 8, 9])
        ]
    
    def test_total_memorization_basic(self):
        """Test basic total memorization calculation."""
        total_mem = calculate_total_memorization(
            self.target_model, self.reference_model, self.test_dataset, self.device
        )
        
        assert total_mem >= 0
        assert isinstance(total_mem, float)
    
    def test_total_memorization_vs_individual_sum(self):
        """Test that total equals sum of individual memorizations."""
        total_mem = calculate_total_memorization(
            self.target_model, self.reference_model, self.test_dataset, self.device
        )
        
        individual_sum = 0.0
        for sequence in self.test_dataset:
            individual_mem = calculate_unintended_memorization(
                self.target_model, self.reference_model, sequence, self.device
            )
            individual_sum += individual_mem
        
        assert abs(total_mem - individual_sum) < 1e-6
    
    def test_total_memorization_empty_dataset(self):
        """Test total memorization with empty dataset."""
        total_mem = calculate_total_memorization(
            self.target_model, self.reference_model, [], self.device
        )
        
        assert total_mem == 0.0
    
    def test_memorization_per_sequence(self):
        """Test per-sequence memorization calculation."""
        per_seq_mem = calculate_memorization_per_sequence(
            self.target_model, self.reference_model, self.test_dataset, self.device
        )
        
        assert len(per_seq_mem) == len(self.test_dataset)
        assert all(mem >= 0 for mem in per_seq_mem)
        assert isinstance(per_seq_mem, np.ndarray)
    
    def test_memorization_per_sequence_empty(self):
        """Test per-sequence memorization with empty dataset."""
        per_seq_mem = calculate_memorization_per_sequence(
            self.target_model, self.reference_model, [], self.device
        )
        
        assert len(per_seq_mem) == 0
        assert isinstance(per_seq_mem, np.ndarray)


class TestBaselineEntropy:
    """Test baseline entropy calculations."""
    
    def test_baseline_entropy_uniform(self):
        """Test baseline entropy for uniform data."""
        vocab_size = 8
        sequences = [
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([5, 6, 7, 0])
        ]
        
        baseline = calculate_baseline_entropy(sequences, vocab_size)
        
        total_tokens = 8  # 4 + 4
        expected_baseline = total_tokens * math.log2(vocab_size)
        
        assert abs(baseline - expected_baseline) < 1e-6
    
    def test_baseline_entropy_empty(self):
        """Test baseline entropy with empty data."""
        baseline = calculate_baseline_entropy([], vocab_size=10)
        assert baseline == 0.0
    
    def test_baseline_entropy_single_token(self):
        """Test baseline entropy with single token sequences."""
        sequences = [torch.tensor([5])]
        baseline = calculate_baseline_entropy(sequences, vocab_size=10)
        
        expected = math.log2(10)  # One token, log2(vocab_size) bits
        assert abs(baseline - expected) < 1e-6


class TestValidationFunctions:
    """Test validation and sanity check functions."""
    
    def setup_method(self):
        """Set up test models and data."""
        self.device = "cpu"
        self.vocab_size = 8
        
        self.config = ModelConfig(
            n_layers=1,
            d_model=16,
            n_heads=2,
            vocab_size=self.vocab_size,
            max_seq_length=20
        )
        
        self.target_model = create_gpt_model(self.config)
        self.reference_model = create_gpt_model(self.config)
        
        self.test_sequences = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5, 6])
        ]
    
    def test_validate_memorization_calculation(self):
        """Test memorization validation function."""
        validation = validate_memorization_calculation(
            self.target_model, self.reference_model, 
            self.test_sequences, self.vocab_size, self.device
        )
        
        required_keys = [
            "total_memorization", "baseline_entropy", "memorization_ratio",
            "mean_memorization", "max_memorization", "min_memorization", "num_sequences"
        ]
        
        for key in required_keys:
            assert key in validation
        
        assert validation["total_memorization"] >= 0
        assert validation["baseline_entropy"] > 0
        assert validation["memorization_ratio"] >= 0
        assert validation["num_sequences"] == len(self.test_sequences)
    
    def test_validate_memorization_empty(self):
        """Test validation with empty data."""
        validation = validate_memorization_calculation(
            self.target_model, self.reference_model, [], self.vocab_size, self.device
        )
        
        assert validation["total_memorization"] == 0.0
        assert validation["baseline_entropy"] == 0.0
        assert validation["memorization_ratio"] == 0.0
        assert validation["num_sequences"] == 0
    
    def test_memorization_ratio_bounds(self):
        """Test that memorization ratio is within reasonable bounds."""
        validation = validate_memorization_calculation(
            self.target_model, self.reference_model,
            self.test_sequences, self.vocab_size, self.device
        )
        
        # Memorization ratio should be between 0 and 1 for reasonable models
        assert 0 <= validation["memorization_ratio"] <= 1.0


class TestSyntheticReferenceModel:
    """Test synthetic reference model creation."""
    
    def test_synthetic_reference_creation(self):
        """Test creation of synthetic uniform reference model."""
        config = ModelConfig(
            n_layers=1, d_model=16, n_heads=2, vocab_size=10, max_seq_length=20
        )
        target_model = create_gpt_model(config)
        
        ref_model = create_synthetic_reference_model(target_model, vocab_size=10)
        
        assert ref_model is not None
        assert hasattr(ref_model, 'forward')
    
    def test_synthetic_reference_uniform_output(self):
        """Test that synthetic reference model produces uniform probabilities."""
        ref_model = create_synthetic_reference_model(None, vocab_size=5)
        
        test_input = torch.tensor([[1, 2, 3]])
        logits = ref_model(test_input)
        
        # Should produce uniform logits (all zeros)
        assert torch.allclose(logits, torch.zeros_like(logits))
        
        # After softmax, should be uniform probabilities
        probs = F.softmax(logits, dim=-1)
        expected_prob = 1.0 / 5  # Uniform over vocab_size=5
        
        assert torch.allclose(probs, torch.full_like(probs, expected_prob), atol=1e-6)


class TestBatchProcessing:
    """Test batch processing functionality."""
    
    def setup_method(self):
        """Set up test models and data."""
        self.device = "cpu"
        self.vocab_size = 8
        
        self.config = ModelConfig(
            n_layers=1,
            d_model=16,
            n_heads=2,
            vocab_size=self.vocab_size,
            max_seq_length=20
        )
        
        self.target_model = create_gpt_model(self.config)
        self.reference_model = create_gpt_model(self.config)
        
        # Create larger dataset for batch testing
        self.large_dataset = [
            torch.randint(0, self.vocab_size, (4,)) for _ in range(20)
        ]
    
    def test_batch_calculate_memorization(self):
        """Test batch memorization calculation."""
        total_mem, per_seq_mem = batch_calculate_memorization(
            self.target_model, self.reference_model, 
            self.large_dataset, batch_size=5, device=self.device
        )
        
        assert total_mem >= 0
        assert len(per_seq_mem) == len(self.large_dataset)
        assert all(mem >= 0 for mem in per_seq_mem)
    
    def test_batch_vs_individual_calculation(self):
        """Test that batch calculation matches individual calculation."""
        # Small dataset for exact comparison
        small_dataset = self.large_dataset[:5]
        
        # Batch calculation
        batch_total, batch_per_seq = batch_calculate_memorization(
            self.target_model, self.reference_model,
            small_dataset, batch_size=2, device=self.device
        )
        
        # Individual calculation
        individual_total = calculate_total_memorization(
            self.target_model, self.reference_model, small_dataset, self.device
        )
        individual_per_seq = calculate_memorization_per_sequence(
            self.target_model, self.reference_model, small_dataset, self.device
        )
        
        assert abs(batch_total - individual_total) < 1e-6
        assert np.allclose(batch_per_seq, individual_per_seq, atol=1e-6)
    
    def test_batch_empty_dataset(self):
        """Test batch calculation with empty dataset."""
        total_mem, per_seq_mem = batch_calculate_memorization(
            self.target_model, self.reference_model, [], batch_size=5, device=self.device
        )
        
        assert total_mem == 0.0
        assert len(per_seq_mem) == 0


class TestMutualInformationApproximation:
    """Test mutual information approximation."""
    
    def setup_method(self):
        """Set up test model and data."""
        self.device = "cpu"
        self.vocab_size = 8
        
        self.config = ModelConfig(
            n_layers=1,
            d_model=16,
            n_heads=2,
            vocab_size=self.vocab_size,
            max_seq_length=20
        )
        
        self.model = create_gpt_model(self.config)
        self.test_sequences = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5, 6])
        ]
    
    def test_mutual_information_basic(self):
        """Test basic mutual information approximation."""
        mi_approx = calculate_mutual_information_approximation(
            self.model, self.test_sequences, self.device
        )
        
        assert mi_approx >= 0
        assert isinstance(mi_approx, float)
    
    def test_mutual_information_empty(self):
        """Test mutual information with empty dataset."""
        mi_approx = calculate_mutual_information_approximation(
            self.model, [], self.device
        )
        
        assert mi_approx == 0.0


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def setup_method(self):
        """Set up comprehensive test environment."""
        self.device = "cpu"
        self.vocab_size = 16
        
        self.config = ModelConfig(
            n_layers=1,
            d_model=32,
            n_heads=4,
            vocab_size=self.vocab_size,
            max_seq_length=20
        )
        
        self.target_model = create_gpt_model(self.config)
        self.reference_model = create_gpt_model(self.config)
        
        # Create diverse test dataset
        self.test_dataset = [
            torch.randint(0, self.vocab_size, (6,)) for _ in range(10)
        ]
    
    def test_full_memorization_pipeline(self):
        """Test complete memorization calculation pipeline."""
        # Calculate all memorization metrics
        total_mem = calculate_total_memorization(
            self.target_model, self.reference_model, self.test_dataset, self.device
        )
        
        per_seq_mem = calculate_memorization_per_sequence(
            self.target_model, self.reference_model, self.test_dataset, self.device
        )
        
        baseline = calculate_baseline_entropy(self.test_dataset, self.vocab_size)
        
        validation = validate_memorization_calculation(
            self.target_model, self.reference_model,
            self.test_dataset, self.vocab_size, self.device
        )
        
        # Verify consistency
        assert abs(total_mem - np.sum(per_seq_mem)) < 1e-6
        assert abs(total_mem - validation["total_memorization"]) < 1e-6
        assert abs(baseline - validation["baseline_entropy"]) < 1e-6
        
        # Verify all values are reasonable
        assert total_mem >= 0
        assert baseline > 0
        assert validation["memorization_ratio"] >= 0
    
    def test_memorization_with_synthetic_reference(self):
        """Test memorization calculation with synthetic reference model."""
        synthetic_ref = create_synthetic_reference_model(
            self.target_model, self.vocab_size
        )
        
        # Calculate memorization using synthetic reference
        total_mem = calculate_total_memorization(
            self.target_model, synthetic_ref, self.test_dataset, self.device
        )
        
        # Should be reasonable (synthetic reference should be worse than trained model)
        assert total_mem >= 0
        
        # Validation should work
        validation = validate_memorization_calculation(
            self.target_model, synthetic_ref,
            self.test_dataset, self.vocab_size, self.device
        )
        
        assert validation["total_memorization"] == total_mem
    
    def test_memorization_bounds_check(self):
        """Test that memorization respects theoretical bounds."""
        baseline = calculate_baseline_entropy(self.test_dataset, self.vocab_size)
        
        total_mem = calculate_total_memorization(
            self.target_model, self.reference_model, self.test_dataset, self.device
        )
        
        # Total memorization should not exceed baseline entropy
        # (though in practice it might be close due to model approximations)
        assert total_mem <= baseline * 1.1  # Allow 10% tolerance for numerical issues


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
