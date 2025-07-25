"""
File: model_trainer.py
Directory: memorization_reproduction/src/

Model training module for GPT-style transformers following Morris et al.
Implements small-scale transformers for memorization capacity experiments.
"""

from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for transformer model architecture."""
    n_layers: int
    d_model: int
    n_heads: int
    vocab_size: int
    max_seq_length: int
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int
    learning_rate: float
    max_steps: int
    warmup_steps: int
    weight_decay: float = 0.01


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Compute queries, keys, values
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.w_o(output)


class FeedForward(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, 4 * d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x


class GPTModel(nn.Module):
    """GPT-style transformer model following Morris et al. architecture."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Output layer
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights following GPT-2 initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive generation."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position indices
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        x = token_embeds + position_embeds
        
        # Create causal mask
        mask = self._create_causal_mask(seq_len, device)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Final normalization and output projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits


class SequenceDataset(Dataset):
    """Dataset wrapper for sequence data."""
    
    def __init__(self, sequences: List[torch.Tensor]):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]


def create_gpt_model(config: ModelConfig) -> torch.nn.Module:
    """
    Create GPT-style transformer model following Morris et al. architecture.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized transformer model
    """
    return GPTModel(config)


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_dataloader(
    sequences: List[torch.Tensor],
    batch_size: int,
    shuffle: bool = True
) -> DataLoader:
    """Create DataLoader from sequence list."""
    dataset = SequenceDataset(sequences)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_linear_warmup_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int
) -> optim.lr_scheduler.LambdaLR:
    """Create linear warmup + cosine decay scheduler."""
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_model(
    model: torch.nn.Module,
    train_data: List[torch.Tensor],
    config: TrainingConfig,
    device: str = "cuda"
) -> Dict[str, List[float]]:
    """
    Train model until convergence with proper termination criteria.
    
    Termination conditions:
    1. Loss < memorization_threshold (memorization achieved)
    2. No improvement for patience steps (plateau)
    3. max_steps reached (safety fallback)
    """
    model = model.to(device)
    model.train()
    
    if not train_data:
        print(f"    WARNING: Empty training data")
        return {"train_loss": [], "learning_rate": [], "step": []}
    
    dataset_size = len(train_data)
    
    # CONVERGENCE CRITERIA for memorization
    memorization_threshold = 0.15  # Near-perfect memorization
    patience = 2000  # Steps to wait for improvement
    min_steps = 1000  # Minimum training before early stopping
    max_steps = config.max_steps  # Safety fallback
    
    print(f"    Training: {dataset_size} sequences until convergence (loss < {memorization_threshold}, patience={patience})")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Create scheduler (will adjust based on actual steps)
    scheduler = get_linear_warmup_scheduler(
        optimizer, config.warmup_steps, max_steps
    )
    
    # Create dataloader
    dataloader = create_dataloader(train_data, config.batch_size, shuffle=True)
    
    # Training state
    metrics = {"train_loss": [], "learning_rate": [], "step": []}
    step = 0
    initial_loss = None
    best_loss = float('inf')
    steps_without_improvement = 0
    
    # Training loop with convergence checking
    while step < max_steps:
        epoch_loss = 0.0
        epoch_steps = 0
        
        for batch in dataloader:
            if step >= max_steps:
                break
                
            batch = batch.to(device)
            
            # Forward pass
            logits = model(batch)
            
            # Calculate loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='mean'
            )
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Track loss
            current_loss = loss.item()
            epoch_loss += current_loss
            epoch_steps += 1
            
            # Log periodically
            if step % 100 == 0:
                metrics["train_loss"].append(current_loss)
                metrics["learning_rate"].append(scheduler.get_last_lr()[0])
                metrics["step"].append(step)
            
            step += 1
            
            # CHECK CONVERGENCE every 100 steps (after minimum training)
            if step % 100 == 0 and step >= min_steps:
                avg_recent_loss = epoch_loss / epoch_steps if epoch_steps > 0 else current_loss
                
                # CONDITION 1: Memorization achieved
                if avg_recent_loss < memorization_threshold:
                    print(f"    CONVERGED: Memorization achieved (loss {avg_recent_loss:.3f} < {memorization_threshold})")
                    break
                
                # CONDITION 2: Check for improvement
                if avg_recent_loss < best_loss - 0.01:  # Significant improvement
                    best_loss = avg_recent_loss
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 100
                
                # Early stopping if no improvement
                if steps_without_improvement >= patience:
                    print(f"    CONVERGED: No improvement for {patience} steps (loss plateau at {avg_recent_loss:.3f})")
                    break
                
                # Reset for next epoch tracking
                epoch_loss = 0.0
                epoch_steps = 0
    
    # Final metrics
    final_loss = metrics["train_loss"][-1] if metrics["train_loss"] else float('nan')
    
    # Determine termination reason
    if step >= max_steps:
        termination_reason = f"MAX_STEPS ({max_steps})"
    elif final_loss < memorization_threshold:
        termination_reason = "MEMORIZATION_ACHIEVED"
    else:
        termination_reason = "LOSS_PLATEAU"
    
    print(f"    Training completed: {step} steps, {initial_loss:.3f} → {final_loss:.3f} loss ({termination_reason})")
    
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    data: List[torch.Tensor],
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate model and return likelihood-based metrics.
    
    Args:
        model: Trained model
        data: Evaluation sequences
        device: Device for evaluation
        
    Returns:
        Dictionary with loss, perplexity, likelihood metrics
    """
    model = model.to(device)
    model.eval()
    
    # Handle empty data gracefully
    if not data:
        return {
            "loss": float('nan'),
            "perplexity": float('nan'),
            "total_tokens": 0,
            "total_loss": 0.0
        }
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for sequence in data:
            sequence = sequence.unsqueeze(0).to(device)  # Add batch dimension
            
            logits = model(sequence)
            
            # Calculate loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = sequence[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
    
    # Handle case where no tokens were processed
    if total_tokens == 0:
        return {
            "loss": float('nan'),
            "perplexity": float('nan'),
            "total_tokens": 0,
            "total_loss": total_loss
        }
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
        "total_loss": total_loss
    }


def get_sequence_likelihoods(
    model: torch.nn.Module,
    sequences: List[torch.Tensor],
    device: str = "cuda"
) -> np.ndarray:
    """
    Get per-sequence negative log-likelihoods from model.
    
    Critical for memorization calculation: HK(x|θ) ≈ -log p(x|θ)
    
    Args:
        model: Trained model
        sequences: Input sequences
        device: Device for computation
        
    Returns:
        Array of negative log-likelihoods per sequence (in bits)
    """
    model = model.to(device)
    model.eval()
    
    sequence_nlls = []
    
    with torch.no_grad():
        for sequence in sequences:
            sequence = sequence.unsqueeze(0).to(device)  # Add batch dimension
            
            logits = model(sequence)
            
            # Calculate negative log-likelihood
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = sequence[..., 1:].contiguous()
            
            # Get log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Gather log probabilities for actual tokens
            token_log_probs = log_probs.gather(
                dim=-1, 
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Sum over sequence (negative log-likelihood)
            sequence_nll = -token_log_probs.sum().item()
            
            # Convert to bits (from nats)
            sequence_nll_bits = sequence_nll / math.log(2)
            
            sequence_nlls.append(sequence_nll_bits)
    
    return np.array(sequence_nlls)


def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    step: int,
    loss: float,
    filepath: str
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'loss': loss
    }
    torch.save(checkpoint, filepath)


def load_model_checkpoint(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    filepath: str,
    device: str = "cuda"
) -> Tuple[int, float]:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['step'], checkpoint['loss']


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Calculate model size in megabytes."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return size_mb
