"""
File: test_data_generator.py
Directory: memorization_reproduction/tests/

Comprehensive tests for data_generator.py module.
Validates all data generation functionality before advancing to model training.
"""

import pytest
import torch
import numpy as np
import random
import hashlib
from unittest.mock import Mock, patch
from typing import List

# Import the module to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_generator import (
    generate_uniform_bitstrings,
    prepare_text_dataset,
    create_train_test_split,
    calculate_dataset_entropy,
    verify_data_properties,
    create_dataset_size_series
)


class TestGenerateUniformBitstrings:
    """Test uniform bitstring generation functionality."""
    
    def test_basic_generation(self):
        """Test basic bitstring generation with standard parameters."""
        n_samples = 100
        seq_length = 64
        vocab_size = 2048
        
        sequences = generate_uniform_bitstrings(n_samples, seq_length, vocab_size)
        
        # Check basic properties
        assert len(sequences) == n_samples
        assert all(len(seq) == seq_length for seq in sequences)
        assert all(isinstance(seq, torch.Tensor) for seq in sequences)
        assert all(seq.dtype == torch.long for seq in sequences)
    
    def test_vocabulary_range(self):
        """Test that generated tokens are within vocabulary range."""
        n_samples = 50
        seq_length = 32
        vocab_size = 1000
        
        sequences = generate_uniform_bitstrings(n_samples, seq_length, vocab_size)
        
        # Check vocabulary range
        all_tokens = torch.cat(sequences, dim=0)
        min_token = all_tokens.min().item()
        max_token = all_tokens.max().item()
        
        assert min_token >= 0
        assert max_token < vocab_size
    
    def test_entropy_approximates_uniform(self):
        """Test that entropy approximates theoretical uniform entropy."""
        n_samples = 1000  # Large sample for good entropy estimate
        seq_length = 64
        vocab_size = 256
        
        sequences = generate_uniform_bitstrings(n_samples, seq_length, vocab_size, seed=42)
        
        entropy = calculate_dataset_entropy(sequences, vocab_size)
        expected_entropy = np.log2(vocab_size)
        
        # Allow 5% deviation for finite sampling
        assert abs(entropy - expected_entropy) / expected_entropy < 0.05
    
    def test_reproducibility_with_seed(self):
        """Test that identical seeds produce identical sequences."""
        n_samples = 10
        seq_length = 16
        vocab_size = 100
        seed = 12345
        
        sequences1 = generate_uniform_bitstrings(n_samples, seq_length, vocab_size, seed=seed)
        sequences2 = generate_uniform_bitstrings(n_samples, seq_length, vocab_size, seed=seed)
        
        # Check sequences are identical
        for seq1, seq2 in zip(sequences1, sequences2):
            assert torch.equal(seq1, seq2)
    
    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different sequences."""
        n_samples = 10
        seq_length = 16
        vocab_size = 100
        
        sequences1 = generate_uniform_bitstrings(n_samples, seq_length, vocab_size, seed=1)
        sequences2 = generate_uniform_bitstrings(n_samples, seq_length, vocab_size, seed=2)
        
        # At least some sequences should be different
        differences = sum(not torch.equal(s1, s2) for s1, s2 in zip(sequences1, sequences2))
        assert differences > 0
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Single sample
        sequences = generate_uniform_bitstrings(1, 1, 2)
        assert len(sequences) == 1
        assert len(sequences[0]) == 1
        
        # Minimum vocabulary
        sequences = generate_uniform_bitstrings(5, 10, 1)
        all_tokens = torch.cat(sequences, dim=0)
        assert all(token == 0 for token in all_tokens)
    
    def test_large_vocabulary(self):
        """Test with large vocabulary size."""
        n_samples = 100
        seq_length = 64
        vocab_size = 50000
        
        sequences = generate_uniform_bitstrings(n_samples, seq_length, vocab_size)
        
        all_tokens = torch.cat(sequences, dim=0)
        assert all_tokens.min() >= 0
        assert all_tokens.max() < vocab_size


class TestPrepareTextDataset:
    """Test text dataset preparation functionality."""
    
    def setup_method(self):
        """Set up mock tokenizer for testing."""
        self.mock_tokenizer = Mock()
        # Create test text that encodes to 1000 tokens (0-999)
        self.mock_tokenizer.encode.return_value = list(range(1000))
    
    def test_basic_text_preparation(self):
        """Test basic text dataset preparation."""
        text_data = "This is test text for tokenization."
        seq_length = 64
        n_samples = 10
        
        sequences = prepare_text_dataset(
            text_data, seq_length, n_samples, self.mock_tokenizer, deduplicate=False
        )
        
        assert len(sequences) <= n_samples  # May be fewer due to text length
        assert all(len(seq) == seq_length for seq in sequences)
        assert all(isinstance(seq, torch.Tensor) for seq in sequences)
    
    def test_deduplication(self):
        """Test that deduplication removes duplicate sequences."""
        text_data = "test text"
        seq_length = 10
        n_samples = 20
        
        # Mock tokenizer to return repeated pattern
        self.mock_tokenizer.encode.return_value = ([1, 2, 3] * 100)  # Repeated pattern
        
        sequences_with_dedup = prepare_text_dataset(
            text_data, seq_length, n_samples, self.mock_tokenizer, deduplicate=True
        )
        sequences_without_dedup = prepare_text_dataset(
            text_data, seq_length, n_samples, self.mock_tokenizer, deduplicate=False
        )
        
        # With deduplication should have fewer (or equal) sequences
        assert len(sequences_with_dedup) <= len(sequences_without_dedup)
    
    def test_insufficient_text_length(self):
        """Test handling of text shorter than required sequence length."""
        text_data = "short"
        seq_length = 100
        n_samples = 10
        
        # Mock very short tokenized text
        self.mock_tokenizer.encode.return_value = [1, 2, 3]
        
        with pytest.raises(ValueError, match="Text too short"):
            prepare_text_dataset(text_data, seq_length, n_samples, self.mock_tokenizer)
    
    def test_sampling_with_replacement(self):
        """Test sampling with replacement when n_samples > possible sequences."""
        text_data = "test"
        seq_length = 5
        n_samples = 200  # More than possible unique sequences
        
        # Mock tokenizer with limited tokens
        self.mock_tokenizer.encode.return_value = list(range(20))  # Only 16 possible sequences
        
        sequences = prepare_text_dataset(
            text_data, seq_length, n_samples, self.mock_tokenizer, deduplicate=False
        )
        
        assert len(sequences) <= n_samples


class TestCreateTrainTestSplit:
    """Test train/test split functionality."""
    
    def setup_method(self):
        """Create test data for splitting."""
        self.test_data = [torch.randint(0, 100, (10,)) for _ in range(100)]
    
    def test_basic_split(self):
        """Test basic train/test split functionality."""
        test_fraction = 0.2
        
        train_data, test_data = create_train_test_split(
            self.test_data, test_fraction=test_fraction, seed=42
        )
        
        expected_test_size = int(len(self.test_data) * test_fraction)
        expected_train_size = len(self.test_data) - expected_test_size
        
        assert len(train_data) == expected_train_size
        assert len(test_data) == expected_test_size
    
    def test_no_data_leakage(self):
        """Test that train and test sets don't overlap."""
        train_data, test_data = create_train_test_split(self.test_data, seed=42)
        
        # Convert to hashable format for set operations
        train_hashes = {hashlib.md5(seq.numpy().tobytes()).hexdigest() for seq in train_data}
        test_hashes = {hashlib.md5(seq.numpy().tobytes()).hexdigest() for seq in test_data}
        
        # No overlap between train and test
        assert len(train_hashes.intersection(test_hashes)) == 0
    
    def test_all_data_preserved(self):
        """Test that all original data is preserved in split."""
        train_data, test_data = create_train_test_split(self.test_data, seed=42)
        
        total_split_size = len(train_data) + len(test_data)
        assert total_split_size == len(self.test_data)
    
    def test_reproducibility(self):
        """Test that identical seeds produce identical splits."""
        seed = 12345
        
        train1, test1 = create_train_test_split(self.test_data, seed=seed)
        train2, test2 = create_train_test_split(self.test_data, seed=seed)
        
        # Check train sets are identical
        for seq1, seq2 in zip(train1, train2):
            assert torch.equal(seq1, seq2)
        
        # Check test sets are identical
        for seq1, seq2 in zip(test1, test2):
            assert torch.equal(seq1, seq2)
    
    def test_edge_cases(self):
        """Test edge cases for train/test split."""
        # Single sample
        single_data = [torch.tensor([1, 2, 3])]
        train, test = create_train_test_split(single_data, test_fraction=0.1)
        assert len(train) + len(test) == 1
        
        # All data to test
        train, test = create_train_test_split(self.test_data, test_fraction=1.0)
        assert len(train) == 0
        assert len(test) == len(self.test_data)


class TestCalculateDatasetEntropy:
    """Test entropy calculation functionality."""
    
    def test_uniform_distribution_entropy(self):
        """Test entropy calculation for uniform distribution."""
        vocab_size = 256
        # Create perfectly uniform distribution
        sequences = []
        for token_id in range(vocab_size):
            sequences.append(torch.tensor([token_id] * 10))  # Each token appears equally
        
        entropy = calculate_dataset_entropy(sequences, vocab_size)
        expected_entropy = np.log2(vocab_size)
        
        assert abs(entropy - expected_entropy) < 1e-10
    
    def test_single_token_entropy(self):
        """Test entropy for single token (should be 0)."""
        vocab_size = 100
        sequences = [torch.tensor([5] * 20)]  # Only token 5 appears
        
        entropy = calculate_dataset_entropy(sequences, vocab_size)
        assert entropy == 0.0
    
    def test_empty_data_entropy(self):
        """Test entropy calculation with empty data."""
        entropy = calculate_dataset_entropy([], vocab_size=100)
        assert entropy == 0.0
    
    def test_partial_vocabulary_entropy(self):
        """Test entropy when only subset of vocabulary is used."""
        vocab_size = 100
        # Use only tokens 0, 1, 2 equally
        sequences = [
            torch.tensor([0] * 10),
            torch.tensor([1] * 10), 
            torch.tensor([2] * 10)
        ]
        
        entropy = calculate_dataset_entropy(sequences, vocab_size)
        expected_entropy = np.log2(3)  # Only 3 tokens used
        
        assert abs(entropy - expected_entropy) < 1e-10


class TestVerifyDataProperties:
    """Test data property verification functionality."""
    
    def test_valid_data_verification(self):
        """Test verification of valid data."""
        vocab_size = 1000
        seq_length = 64
        sequences = generate_uniform_bitstrings(100, seq_length, vocab_size, seed=42)
        
        results = verify_data_properties(sequences, vocab_size, seq_length)
        
        assert results["length_consistent"] is True
        assert results["vocab_in_range"] is True
        assert results["actual_lengths"] == [seq_length]
        assert results["entropy_ratio"] > 0.95  # Should be close to 1 for uniform data
        assert results["duplicate_count"] >= 0
    
    def test_invalid_sequence_lengths(self):
        """Test verification with inconsistent sequence lengths."""
        sequences = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5]),  # Different length
            torch.tensor([6, 7, 8])
        ]
        
        results = verify_data_properties(sequences, vocab_size=10, expected_length=3)
        
        assert results["length_consistent"] is False
        assert 2 in results["actual_lengths"]
        assert 3 in results["actual_lengths"]
    
    def test_vocabulary_out_of_range(self):
        """Test verification with tokens outside vocabulary range."""
        sequences = [torch.tensor([1, 2, 15])]  # Token 15 outside range
        
        results = verify_data_properties(sequences, vocab_size=10, expected_length=3)
        
        assert results["vocab_in_range"] is False
        assert results["actual_vocab_range"][1] >= 10  # Max token >= vocab_size
    
    def test_empty_data_verification(self):
        """Test verification with empty data."""
        results = verify_data_properties([], vocab_size=100, expected_length=10)
        
        assert "error" in results
    
    def test_duplicate_detection(self):
        """Test duplicate sequence detection."""
        # Create sequences with known duplicates
        seq1 = torch.tensor([1, 2, 3])
        seq2 = torch.tensor([4, 5, 6])
        seq3 = torch.tensor([1, 2, 3])  # Duplicate of seq1
        
        sequences = [seq1, seq2, seq3]
        
        results = verify_data_properties(sequences, vocab_size=10, expected_length=3)
        
        assert results["duplicate_count"] == 1
        assert results["unique_sequences"] == 2


class TestCreateDatasetSizeSeries:
    """Test dataset size series creation functionality."""
    
    def setup_method(self):
        """Create base dataset for testing."""
        self.base_data = [torch.randint(0, 100, (10,)) for _ in range(50)]
    
    def test_basic_size_series(self):
        """Test basic dataset size series creation."""
        target_sizes = [10, 20, 30]
        
        datasets = create_dataset_size_series(self.base_data, target_sizes, seed=42)
        
        assert len(datasets) == len(target_sizes)
        for size in target_sizes:
            assert size in datasets
            assert len(datasets[size]) == size
    
    def test_sampling_without_replacement(self):
        """Test sampling without replacement for sizes <= base size."""
        target_sizes = [10, 25]  # Both <= 50 (base size)
        
        datasets = create_dataset_size_series(self.base_data, target_sizes, seed=42)
        
        # Check no duplicates in smaller datasets
        for size in target_sizes:
            dataset = datasets[size]
            hashes = set()
            duplicates = 0
            
            for seq in dataset:
                seq_hash = hashlib.md5(seq.numpy().tobytes()).hexdigest()
                if seq_hash in hashes:
                    duplicates += 1
                hashes.add(seq_hash)
            
            assert duplicates == 0  # No duplicates when sampling without replacement
    
    def test_sampling_with_replacement(self):
        """Test sampling with replacement for sizes > base size."""
        target_sizes = [100]  # Larger than base size (50)
        
        datasets = create_dataset_size_series(self.base_data, target_sizes, seed=42)
        
        assert len(datasets[100]) == 100
    
    def test_reproducibility(self):
        """Test reproducibility of dataset size series."""
        target_sizes = [10, 20]
        seed = 12345
        
        datasets1 = create_dataset_size_series(self.base_data, target_sizes, seed=seed)
        datasets2 = create_dataset_size_series(self.base_data, target_sizes, seed=seed)
        
        for size in target_sizes:
            data1 = datasets1[size]
            data2 = datasets2[size]
            
            for seq1, seq2 in zip(data1, data2):
                assert torch.equal(seq1, seq2)
    
    def test_empty_target_sizes(self):
        """Test with empty target sizes list."""
        datasets = create_dataset_size_series(self.base_data, [], seed=42)
        assert len(datasets) == 0
    
    def test_zero_size_dataset(self):
        """Test creation of zero-size dataset."""
        datasets = create_dataset_size_series(self.base_data, [0], seed=42)
        assert len(datasets[0]) == 0


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_synthetic_pipeline(self):
        """Test complete synthetic data generation pipeline."""
        # Generate data
        n_samples = 100
        seq_length = 32
        vocab_size = 256
        
        sequences = generate_uniform_bitstrings(n_samples, seq_length, vocab_size, seed=42)
        
        # Verify properties
        results = verify_data_properties(sequences, vocab_size, seq_length)
        assert results["length_consistent"]
        assert results["vocab_in_range"]
        
        # Split data
        train_data, test_data = create_train_test_split(sequences, test_fraction=0.2, seed=42)
        
        # Create size series
        target_sizes = [10, 25, 50]
        datasets = create_dataset_size_series(train_data, target_sizes, seed=42)
        
        # Verify final datasets
        for size in target_sizes:
            dataset = datasets[size]
            assert len(dataset) == size
            assert all(len(seq) == seq_length for seq in dataset)
    
    def test_text_pipeline_with_mock(self):
        """Test text processing pipeline with mocked tokenizer."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = list(range(200))
        
        # Prepare text dataset
        sequences = prepare_text_dataset(
            "mock text", seq_length=20, n_samples=30, 
            tokenizer=mock_tokenizer, deduplicate=True
        )
        
        # Verify and split
        results = verify_data_properties(sequences, vocab_size=200, expected_length=20)
        assert results["length_consistent"]
        
        train_data, test_data = create_train_test_split(sequences, seed=42)
        assert len(train_data) + len(test_data) == len(sequences)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
