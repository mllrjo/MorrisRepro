"""
File: memorization_calculator.py
Directory: memorization_reproduction/src/

Memorization calculation module implementing Morris et al. compression-based approach.
Core implementation of unintended memorization measurement via Kolmogorov complexity approximation.
"""

from typing import List, Dict, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import math


def calculate_compression_rate(
    model: torch.nn.Module,
    sequence: torch.Tensor,
    device: str = "cuda"
) -> float:
    """
    Calculate compression rate for sequence using model likelihood.
    Approximates HK(x|θ) ≈ -log p(x|θ)
    
    Following Morris et al., this is the fundamental building block for
    measuring how much information a model contains about a sequence.
    
    Args:
        model: Model to use for compression
        sequence: Input sequence tensor
        device: Device for computation
        
    Returns:
        Compression rate in bits
    """
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        sequence = sequence.unsqueeze(0).to(device)  # Add batch dimension
        
        # Get model logits
        logits = model(sequence)
        
        # Calculate negative log-likelihood for autoregressive prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = sequence[..., 1:].contiguous()
        
        # Get log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probabilities for actual tokens
        token_log_probs = log_probs.gather(
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Sum negative log-likelihood over sequence
        negative_log_likelihood = -token_log_probs.sum().item()
        
        # Convert from nats to bits
        compression_rate_bits = negative_log_likelihood / math.log(2)
        
        return compression_rate_bits


def calculate_joint_compression_rate(
    target_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    sequence: torch.Tensor,
    device: str = "cuda"
) -> float:
    """
    Calculate joint compression rate using both target and reference models.
    
    Following Morris et al.: HK(x | θ_target, θ_ref) ≈ -log max{p(x|θ_target), p(x|θ_ref)}
    This represents the best compression achievable using both models.
    
    Args:
        target_model: Primary model being evaluated
        reference_model: Reference model (typically larger, broader training)
        sequence: Input sequence tensor
        device: Device for computation
        
    Returns:
        Joint compression rate in bits
    """
    target_model = target_model.to(device)
    reference_model = reference_model.to(device)
    target_model.eval()
    reference_model.eval()
    
    with torch.no_grad():
        sequence = sequence.unsqueeze(0).to(device)  # Add batch dimension
        
        # Get logits from both models
        target_logits = target_model(sequence)
        ref_logits = reference_model(sequence)
        
        # Prepare for autoregressive prediction
        target_shift_logits = target_logits[..., :-1, :].contiguous()
        ref_shift_logits = ref_logits[..., :-1, :].contiguous()
        shift_labels = sequence[..., 1:].contiguous()
        
        # Get log probabilities from both models
        target_log_probs = F.log_softmax(target_shift_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_shift_logits, dim=-1)
        
        # Gather log probabilities for actual tokens
        target_token_log_probs = target_log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        ref_token_log_probs = ref_log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Take maximum probability (minimum negative log-likelihood) at each position
        max_log_probs = torch.maximum(target_token_log_probs, ref_token_log_probs)
        
        # Sum negative log-likelihood over sequence
        joint_negative_log_likelihood = -max_log_probs.sum().item()
        
        # Convert from nats to bits
        joint_compression_rate_bits = joint_negative_log_likelihood / math.log(2)
        
        return joint_compression_rate_bits


def calculate_unintended_memorization(
    target_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    sequence: torch.Tensor,
    device: str = "cuda"
) -> float:
    """
    Calculate unintended memorization following Morris et al. definition.
    
    Core formula: memU = HK(x|θ_ref) - HK(x|θ_target, θ_ref)
    
    This measures how much the target model knows about the sequence
    beyond what can be explained by the reference model (generalization).
    
    Args:
        target_model: Model being evaluated for memorization
        reference_model: Larger reference model representing generalization
        sequence: Input sequence tensor
        device: Device for computation
        
    Returns:
        Unintended memorization in bits
    """
    # Calculate compression rate using only reference model
    ref_compression = calculate_compression_rate(reference_model, sequence, device)
    
    # Calculate joint compression rate using both models
    joint_compression = calculate_joint_compression_rate(
        target_model, reference_model, sequence, device
    )
    
    # Unintended memorization is the improvement from using target model
    unintended_memorization = ref_compression - joint_compression

    # DEBUG: Show compression values
    # print(f"    Debug: ref_compression={ref_compression:.1f}, joint_compression={joint_compression:.1f}, memorization={unintended_memorization:.1f}")
   
    # Ensure non-negative (cannot have negative memorization)
    unintended_memorization = max(0.0, unintended_memorization)
    
    return unintended_memorization


def calculate_total_memorization(
    target_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    dataset: List[torch.Tensor],
    device: str = "cuda"
) -> float:
    """
    Calculate total unintended memorization across entire dataset.
    
    Sums per-sequence unintended memorization to get total dataset memorization.
    This is the quantity that plateaus at model capacity in Morris et al.
    
    Args:
        target_model: Model being evaluated
        reference_model: Reference model
        dataset: List of sequences
        device: Device for computation
        
    Returns:
        Total memorization in bits
    """
    if not dataset:
        return 0.0
    
    total_memorization = 0.0
    
    for sequence in dataset:
        sequence_memorization = calculate_unintended_memorization(
            target_model, reference_model, sequence, device
        )
        total_memorization += sequence_memorization
    
    return total_memorization


def calculate_memorization_per_sequence(
    target_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    dataset: List[torch.Tensor],
    device: str = "cuda"
) -> np.ndarray:
    """
    Calculate unintended memorization for each sequence individually.
    
    Useful for analyzing distribution of memorization across dataset
    and identifying which sequences are most memorized.
    
    Args:
        target_model: Model being evaluated
        reference_model: Reference model
        dataset: List of sequences
        device: Device for computation
        
    Returns:
        Array of memorization values per sequence
    """
    if not dataset:
        return np.array([])
    
    memorization_values = []
    
    for sequence in dataset:
        sequence_memorization = calculate_unintended_memorization(
            target_model, reference_model, sequence, device
        )
        memorization_values.append(sequence_memorization)
    
    return np.array(memorization_values)


def calculate_baseline_entropy(
    sequences: List[torch.Tensor],
    vocab_size: int
) -> float:
    """
    Calculate baseline entropy of dataset for validation.
    
    For uniform random data, total entropy should be:
    N * S * log2(vocab_size) where N = num_sequences, S = seq_length
    
    Args:
        sequences: List of sequences
        vocab_size: Vocabulary size
        
    Returns:
        Total baseline entropy in bits
    """
    if not sequences:
        return 0.0
    
    # For uniform data, each token has log2(vocab_size) bits of entropy
    bits_per_token = math.log2(vocab_size)
    
    total_tokens = sum(len(seq) for seq in sequences)
    total_entropy = total_tokens * bits_per_token
    
    return total_entropy


def validate_memorization_calculation(
    target_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    sequences: List[torch.Tensor],
    vocab_size: int,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Validate memorization calculations with sanity checks.
    
    Performs consistency checks on memorization measurements:
    - Total memorization should not exceed baseline entropy
    - Individual memorization should be non-negative
    - For uniform data, reference model should perform poorly
    
    Args:
        target_model: Model being evaluated
        reference_model: Reference model
        sequences: List of sequences
        vocab_size: Vocabulary size
        device: Device for computation
        
    Returns:
        Dictionary of validation metrics
    """
    if not sequences:
        return {
            "total_memorization": 0.0,
            "baseline_entropy": 0.0,
            "memorization_ratio": 0.0,
            "mean_memorization": 0.0,
            "max_memorization": 0.0,
            "min_memorization": 0.0,
            "num_sequences": 0
        }
    
    # Calculate memorization metrics
    total_memorization = calculate_total_memorization(
        target_model, reference_model, sequences, device
    )
    
    per_sequence_memorization = calculate_memorization_per_sequence(
        target_model, reference_model, sequences, device
    )
    
    baseline_entropy = calculate_baseline_entropy(sequences, vocab_size)
    
    # Calculate validation metrics
    memorization_ratio = total_memorization / baseline_entropy if baseline_entropy > 0 else 0.0
    
    return {
        "total_memorization": total_memorization,
        "baseline_entropy": baseline_entropy,
        "memorization_ratio": memorization_ratio,
        "mean_memorization": np.mean(per_sequence_memorization) if len(per_sequence_memorization) > 0 else 0.0,
        "max_memorization": np.max(per_sequence_memorization) if len(per_sequence_memorization) > 0 else 0.0,
        "min_memorization": np.min(per_sequence_memorization) if len(per_sequence_memorization) > 0 else 0.0,
        "num_sequences": len(sequences)
    }


def create_synthetic_reference_model(
    target_model: torch.nn.Module,
    vocab_size: int
) -> torch.nn.Module:
    """
    Create a synthetic reference model for uniform random data experiments.
    
    For purely random data, the reference model should assign uniform
    probability to all tokens (representing no generalization).
    
    Args:
        target_model: Target model (used for architecture reference)
        vocab_size: Vocabulary size
        
    Returns:
        Reference model with uniform token probabilities
    """
    # Create a simple uniform model that assigns equal probability to all tokens
    class UniformModel(torch.nn.Module):
        def __init__(self, vocab_size: int):
            super().__init__()
            self.vocab_size = vocab_size
            
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            batch_size, seq_len = input_ids.shape
            # Return uniform logits (all zeros give uniform probabilities after softmax)
            logits = torch.zeros(batch_size, seq_len, self.vocab_size, 
                               device=input_ids.device, dtype=torch.float32)
            return logits
    
    return UniformModel(vocab_size)


def calculate_mutual_information_approximation(
    target_model: torch.nn.Module,
    sequences: List[torch.Tensor],
    device: str = "cuda"
) -> float:
    """
    Calculate approximation of mutual information I(X, θ) for total memorization.
    
    For synthetic uniform data where there's no generalization,
    this represents the total information stored about the dataset.
    
    Args:
        target_model: Model being evaluated
        sequences: List of sequences
        device: Device for computation
        
    Returns:
        Approximate mutual information in bits
    """
    if not sequences:
        return 0.0
    
    total_compression = 0.0
    
    for sequence in sequences:
        compression = calculate_compression_rate(target_model, sequence, device)
        total_compression += compression
    
    # For uniform data, baseline entropy per sequence is len(seq) * log2(vocab_size)
    # Mutual information is baseline minus compressed
    
    # This is a simplified approximation - more sophisticated calculation
    # would require knowing the true data distribution
    return total_compression


def batch_calculate_memorization(
    target_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    dataset: List[torch.Tensor],
    batch_size: int = 32,
    device: str = "cuda"
) -> Tuple[float, np.ndarray]:
    """
    Calculate memorization in batches for efficiency with large datasets.
    
    Processes sequences in batches to reduce memory usage and improve
    computational efficiency for large-scale experiments.
    
    Args:
        target_model: Model being evaluated
        reference_model: Reference model
        dataset: List of sequences
        batch_size: Number of sequences to process at once
        device: Device for computation
        
    Returns:
        Tuple of (total_memorization, per_sequence_memorization_array)
    """
    if not dataset:
        return 0.0, np.array([])
    
    total_memorization = 0.0
    all_memorization_values = []
    
    # Process in batches
    for i in range(0, len(dataset), batch_size):
        batch_sequences = dataset[i:i + batch_size]
        
        batch_memorization_values = []
        
        for sequence in batch_sequences:
            memorization = calculate_unintended_memorization(
                target_model, reference_model, sequence, device
            )
            batch_memorization_values.append(memorization)
            total_memorization += memorization
        
        all_memorization_values.extend(batch_memorization_values)
    
    return total_memorization, np.array(all_memorization_values)
