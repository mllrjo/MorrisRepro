"""
File: data_generator.py
Directory: memorization_reproduction/src/

Data generation module for language model memorization experiments.
Generates uniform random bitstrings and prepares text datasets following Morris et al.
"""

from typing import List, Tuple, Optional, Any, Set
import torch
import numpy as np
import random
from collections import defaultdict
import hashlib


def generate_uniform_bitstrings(
    n_samples: int, 
    seq_length: int, 
    vocab_size: int,
    seed: Optional[int] = None
) -> List[torch.Tensor]:
    """
    Generate uniform random bitstring sequences for synthetic experiments.
    
    Each token is uniformly sampled from vocabulary, independent of previous tokens.
    This creates data with no generalizable patterns, pure memorization target.
    
    Args:
        n_samples: Number of sequences to generate
        seq_length: Length of each sequence in tokens
        vocab_size: Size of vocabulary (e.g., 2048 as in Morris et al.)
        seed: Random seed for reproducibility
        
    Returns:
        List of tokenized sequences as tensors
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    sequences = []
    for _ in range(n_samples):
        # Sample each token independently from uniform distribution
        sequence = torch.randint(0, vocab_size, (seq_length,), dtype=torch.long)
        sequences.append(sequence)
    
    return sequences


def prepare_text_dataset(
    text_data: str,
    seq_length: int,
    n_samples: int,
    tokenizer: Any,
    deduplicate: bool = True
) -> List[torch.Tensor]:
    """
    Prepare real text data for experiments with deduplication.
    
    Following Morris et al., careful deduplication is critical for faithful
    measurement of memorization vs generalization.
    
    Args:
        text_data: Raw text string
        seq_length: Target sequence length
        n_samples: Number of samples to extract
        tokenizer: Tokenizer to use (e.g., GPT-2 tokenizer)
        deduplicate: Whether to remove duplicate sequences
        
    Returns:
        List of tokenized text sequences
    """
    # Tokenize the entire text
    tokens = tokenizer.encode(text_data)
    
    if len(tokens) < seq_length:
        raise ValueError(f"Text too short: {len(tokens)} tokens < {seq_length} required")
    
    # Extract sequences of target length
    sequences = []
    sequence_hashes = set() if deduplicate else None
    
    max_start_idx = len(tokens) - seq_length
    
    # If we need more samples than possible, sample with replacement
    if n_samples > max_start_idx:
        # Sample with replacement
        start_indices = np.random.choice(max_start_idx, n_samples, replace=True)
    else:
        # Sample without replacement initially
        start_indices = np.random.choice(max_start_idx, n_samples * 2, replace=False)
    
    for start_idx in start_indices:
        if len(sequences) >= n_samples:
            break
            
        sequence_tokens = tokens[start_idx:start_idx + seq_length]
        sequence_tensor = torch.tensor(sequence_tokens, dtype=torch.long)
        
        if deduplicate:
            # Create hash of sequence for deduplication
            sequence_hash = hashlib.md5(sequence_tensor.numpy().tobytes()).hexdigest()
            
            if sequence_hash in sequence_hashes:
                continue  # Skip duplicate
                
            sequence_hashes.add(sequence_hash)
        
        sequences.append(sequence_tensor)
    
    if len(sequences) < n_samples:
        print(f"Warning: Only found {len(sequences)} unique sequences, requested {n_samples}")
    
    return sequences[:n_samples]


def create_train_test_split(
    data: List[torch.Tensor],
    test_fraction: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Split data into train and test sets.
    
    Args:
        data: List of tokenized sequences
        test_fraction: Fraction of data for testing
        seed: Random seed
        
    Returns:
        Tuple of (train_data, test_data)
    """
    if seed is not None:
        random.seed(seed)
    
    # Shuffle data
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    # Calculate split point
    test_size = int(len(data_copy) * test_fraction)
    train_size = len(data_copy) - test_size
    
    train_data = data_copy[:train_size]
    test_data = data_copy[train_size:]
    
    return train_data, test_data


def calculate_dataset_entropy(
    sequences: List[torch.Tensor],
    vocab_size: int
) -> float:
    """
    Calculate Shannon entropy of dataset for validation.
    
    For uniform random data, entropy should be log2(vocab_size) per token.
    Useful for verifying synthetic data generation.
    
    Args:
        sequences: List of tokenized sequences
        vocab_size: Vocabulary size
        
    Returns:
        Entropy in bits per token
    """
    if not sequences:
        return 0.0
    
    # Count token frequencies
    token_counts = defaultdict(int)
    total_tokens = 0
    
    for sequence in sequences:
        for token in sequence:
            token_counts[token.item()] += 1
            total_tokens += 1
    
    # Calculate empirical probabilities
    probabilities = []
    for token_id in range(vocab_size):
        count = token_counts.get(token_id, 0)
        prob = count / total_tokens if total_tokens > 0 else 0
        if prob > 0:
            probabilities.append(prob)
    
    # Calculate entropy
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    
    return entropy


def verify_data_properties(
    sequences: List[torch.Tensor],
    vocab_size: int,
    expected_length: int
) -> dict:
    """
    Verify generated data has expected properties.
    
    Args:
        sequences: Generated sequences
        vocab_size: Expected vocabulary size
        expected_length: Expected sequence length
        
    Returns:
        Dictionary of verification results
    """
    if not sequences:
        return {"error": "No sequences provided"}
    
    results = {}
    
    # Check sequence lengths
    lengths = [len(seq) for seq in sequences]
    results["length_consistent"] = all(l == expected_length for l in lengths)
    results["actual_lengths"] = list(set(lengths))
    
    # Check vocabulary range
    all_tokens = torch.cat(sequences, dim=0)
    min_token = all_tokens.min().item()
    max_token = all_tokens.max().item()
    
    results["vocab_in_range"] = (min_token >= 0) and (max_token < vocab_size)
    results["actual_vocab_range"] = (min_token, max_token)
    results["unique_tokens"] = len(torch.unique(all_tokens))
    
    # Calculate entropy for uniform data validation
    entropy = calculate_dataset_entropy(sequences, vocab_size)
    expected_entropy = np.log2(vocab_size)
    
    results["entropy"] = entropy
    results["expected_entropy"] = expected_entropy
    results["entropy_ratio"] = entropy / expected_entropy if expected_entropy > 0 else 0
    
    # Check for duplicates
    sequence_hashes = set()
    duplicates = 0
    
    for seq in sequences:
        seq_hash = hashlib.md5(seq.numpy().tobytes()).hexdigest()
        if seq_hash in sequence_hashes:
            duplicates += 1
        sequence_hashes.add(seq_hash)
    
    results["duplicate_count"] = duplicates
    results["unique_sequences"] = len(sequence_hashes)
    
    return results


def create_dataset_size_series(
    base_data: List[torch.Tensor],
    target_sizes: List[int],
    seed: Optional[int] = None
) -> dict:
    """
    Create datasets of different sizes from base data.
    
    Used for capacity estimation experiments where we need multiple
    dataset sizes to find memorization plateaus.
    
    Args:
        base_data: Base dataset to sample from
        target_sizes: List of desired dataset sizes
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping size -> dataset
    """
    if seed is not None:
        random.seed(seed)
    
    datasets = {}
    base_size = len(base_data)
    
    for size in target_sizes:
        if size <= base_size:
            # Sample without replacement
            sampled_data = random.sample(base_data, size)
        else:
            # Sample with replacement
            sampled_data = random.choices(base_data, k=size)
        
        datasets[size] = sampled_data
    
    return datasets
