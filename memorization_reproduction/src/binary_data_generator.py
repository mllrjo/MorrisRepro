
"""
Binary Sequence Data Generator for Morris Memorization Reproduction
Generates random binary sequences for cleaner memorization experiments.

File: src/binary_data_generator.py
"""

import torch
import numpy as np
from typing import List, Optional, Tuple
import random


def generate_random_binary_sequences(
    n_samples: int,
    seq_length: int,
    seed: Optional[int] = None,
    ensure_diversity: bool = True
) -> List[torch.Tensor]:
    """
    Generate random binary sequences for memorization experiments.
    
    Args:
        n_samples: Number of sequences to generate
        seq_length: Length of each sequence (including BOS/EOS if used)
        seed: Random seed for reproducibility
        ensure_diversity: Ensure no duplicate sequences when possible
        
    Returns:
        List of binary sequences as torch.LongTensor (values in {0, 1})
    """
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    # Maximum possible unique sequences for given length
    max_unique = 2 ** seq_length
    
    if ensure_diversity and n_samples > max_unique:
        print(f"Warning: Requested {n_samples} sequences but only {max_unique} unique "
              f"sequences possible with length {seq_length}. Some duplicates will occur.")
    
    sequences = []
    seen_sequences = set() if ensure_diversity else None
    
    attempts = 0
    max_attempts = n_samples * 10  # Prevent infinite loops
    
    while len(sequences) < n_samples and attempts < max_attempts:
        # Generate random binary sequence
        sequence = torch.randint(0, 2, (seq_length,), dtype=torch.long)
        
        # Check for duplicates if diversity is required
        if ensure_diversity:
            sequence_tuple = tuple(sequence.tolist())
            if sequence_tuple in seen_sequences:
                attempts += 1
                continue
            seen_sequences.add(sequence_tuple)
        
        sequences.append(sequence)
        attempts += 1
    
    if len(sequences) < n_samples:
        print(f"Warning: Only generated {len(sequences)}/{n_samples} unique sequences")
    
    return sequences


def create_binary_dataset_size_series(
    seq_length: int,
    max_samples: int = 1000000,
    min_samples: int = 10
) -> List[int]:
    """
    Create appropriate dataset size series for binary sequence experiments.
    
    Considers the constraint that max unique sequences = 2^seq_length.
    """
    
    max_unique = 2 ** seq_length
    
    # Standard geometric progression
    base_sizes = []
    current = min_samples
    while current <= max_samples:
        base_sizes.append(current)
        current = int(current * 2.5)  # Geometric progression
    
    # Filter based on uniqueness constraints
    filtered_sizes = []
    for size in base_sizes:
        if size <= max_unique * 0.8:  # Keep below 80% of max unique to avoid too many duplicates
            filtered_sizes.append(size)
        elif len(filtered_sizes) == 0 or size <= max_unique:
            # Always include at least one size, even if it approaches max unique
            filtered_sizes.append(min(size, max_unique))
    
    # Ensure we have a reasonable range
    if len(filtered_sizes) < 3:
        # Add more intermediate points
        if max_unique >= 100:
            extra_sizes = [max_unique // 4, max_unique // 2, int(max_unique * 0.8)]
            filtered_sizes.extend([s for s in extra_sizes if s not in filtered_sizes and s >= min_samples])
    
    return sorted(list(set(filtered_sizes)))


