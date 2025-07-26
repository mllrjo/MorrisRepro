"""
Enhanced Model Trainer for Morris Memorization Reproduction
Integrates enhanced training to fix MAX_STEPS convergence issues.

File: src/model_trainer.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
import math
import time


class ModelConfig:
    """Model configuration class for backwards compatibility"""
    
    def __init__(self, 
                 n_layers: int = 2,
                 d_model: int = 128,
                 n_heads: int = 4,
                 vocab_size: int = 1000,
                 max_seq_length: int = 64,
                 dropout: float = 0.1):
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.dropout = dropout


class TrainingConfig:
    """Training configuration class for backwards compatibility"""
    
    def __init__(self,
                 batch_size: int = 16,
                 learning_rate: float = 1e-3,
                 max_steps: int = 35000,
                 warmup_steps: int = 1000,
                 weight_decay: float = 0.01):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay


# Try to import enhanced training - fallback if not available
try:
    # Try relative import first (when running from src directory)
    from enhanced_training import (
        enhanced_train_model_wrapper, 
        EnhancedTrainingConfig,
        adaptive_memorization_training
    )
    ENHANCED_TRAINING_AVAILABLE = True
    print("✅ Enhanced training available - using improved convergence detection")
except ImportError:
    try:
        # Try absolute import (when running from main directory)
        from src.enhanced_training import (
            enhanced_train_model_wrapper, 
            EnhancedTrainingConfig,
            adaptive_memorization_training
        )
        ENHANCED_TRAINING_AVAILABLE = True
        print("✅ Enhanced training available - using improved convergence detection")
    except ImportError:
        ENHANCED_TRAINING_AVAILABLE = False
        # Create a dummy class for type annotations when enhanced training is not available
        class EnhancedTrainingConfig:
            pass
        print("⚠️  Enhanced training not found - using original training (may hit MAX_STEPS)")


class GPTModel(nn.Module):
    """GPT-style transformer model for memorization experiments"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Layer norm and output projection
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        pos_ids = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(pos_ids)
        
        # Combine embeddings
        x = self.dropout(token_embeds + pos_embeds)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward"""
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_length, config.max_seq_length))
        )
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Generate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        attn = attn.masked_fill(causal_mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.c_proj(out)
        
        return out


class MLP(nn.Module):
    """Feed-forward network"""
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


def create_gpt_model(config):
    """Create a GPT model from configuration"""
    return GPTModel(config)


def train_model(
    model: torch.nn.Module,
    train_data: List[torch.Tensor],
    config: Any,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Enhanced training function with improved convergence detection.
    Uses enhanced training if available, otherwise falls back to original.
    """
    
    if ENHANCED_TRAINING_AVAILABLE:
        # Use enhanced training with improved convergence
        print(f"Training with enhanced convergence detection (max_steps: {getattr(config, 'max_steps', 100_000):,})")
        return enhanced_train_model_wrapper(model, train_data, config, device)
    
    else:
        # Fallback to original training
        print("Using original training - enhanced training not available")
        return original_train_model(model, train_data, config, device)


def original_train_model(
    model: torch.nn.Module,
    train_data: List[torch.Tensor],
    config: Any,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Original training function (kept for backwards compatibility)
    WARNING: This may hit MAX_STEPS without convergence
    """
    
    model.train()
    model = model.to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=getattr(config, 'learning_rate', 1e-3),
        weight_decay=getattr(config, 'weight_decay', 0.01)
    )
    
    max_steps = getattr(config, 'max_steps', 35000)
    batch_size = getattr(config, 'batch_size', 16)
    
    print(f"Original training: {len(train_data)} sequences, {max_steps} max steps")
    
    losses = []
    start_time = time.time()
    
    for step in range(max_steps):
        optimizer.zero_grad()
        
        # Sample batch
        batch_indices = torch.randint(0, len(train_data), (batch_size,))
        batch_losses = []
        
        for idx in batch_indices:
            sequence = train_data[idx].to(device)
            
            # Add batch dimension
            input_ids = sequence[:-1].unsqueeze(0)
            targets = sequence[1:].unsqueeze(0)
            
            logits = model(input_ids)
            
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            batch_losses.append(loss)
        
        total_loss = torch.stack(batch_losses).mean()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track progress
        if step % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:6d}: Loss {total_loss.item():.4f}, Time {elapsed:.1f}s")
            losses.append(total_loss.item())
    
    print(f"Original training completed: {max_steps} steps (MAX_STEPS)")
    
    return {
        'loss': losses,
        'train_loss': losses,  # Backwards compatibility
        'convergence_info': {
            'converged': False,
            'reason': 'MAX_STEPS',
            'final_step': max_steps,
            'final_loss': losses[-1] if losses else float('inf')
        }
    }


def train_model_with_enhanced_config(
    model: torch.nn.Module,
    train_data: List[torch.Tensor],
    enhanced_config: "EnhancedTrainingConfig",
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Direct interface to enhanced training with EnhancedTrainingConfig
    """
    
    if not ENHANCED_TRAINING_AVAILABLE:
        raise ImportError("Enhanced training not available - cannot use EnhancedTrainingConfig")
    
    print("Using direct enhanced training interface")
    return adaptive_memorization_training(model, train_data, enhanced_config, device)


def create_enhanced_training_config(
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    max_steps: int = 100_000,
    memorization_threshold: float = 0.15,
    **kwargs
) -> "EnhancedTrainingConfig":
    """
    Convenience function to create enhanced training configuration
    """
    
    if not ENHANCED_TRAINING_AVAILABLE:
        raise ImportError("Enhanced training not available - cannot create EnhancedTrainingConfig")
    
    return EnhancedTrainingConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps,
        memorization_threshold=memorization_threshold,
        **kwargs
    )


def get_model_parameter_count(model: torch.nn.Module) -> int:
    """Get the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model: torch.nn.Module) -> int:
    """Backwards compatibility alias for get_model_parameter_count"""
    return get_model_parameter_count(model)


def validate_model_config(config):
    """Validate model configuration has required attributes"""
    required_attrs = ['n_layers', 'd_model', 'n_heads', 'vocab_size', 'max_seq_length']
    for attr in required_attrs:
        if not hasattr(config, attr):
            raise ValueError(f"Model config missing required attribute: {attr}")
    
    # Add default dropout if not specified
    if not hasattr(config, 'dropout'):
        config.dropout = 0.1
    
    return config


def estimate_memory_usage(model: torch.nn.Module) -> float:
    """Estimate memory usage of model in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


# Backwards compatibility aliases
def create_model(config):
    """Alias for create_gpt_model for backwards compatibility"""
    return create_gpt_model(config)


if __name__ == "__main__":
    print("Enhanced Model Trainer for Morris Memorization Reproduction")
    print("=" * 60)
    print(f"Enhanced training available: {ENHANCED_TRAINING_AVAILABLE}")
    
    if ENHANCED_TRAINING_AVAILABLE:
        print("✅ Using enhanced training with:")
        print("  - Adaptive convergence detection")
        print("  - Increased step limits (100K+)")
        print("  - Memorization rate monitoring")
        print("  - Early stopping on achievement")
        print("  - No more MAX_STEPS failures!")
    else:
        print("⚠️  Using original training:")
        print("  - May hit MAX_STEPS without convergence")
        print("  - Consider installing enhanced_training.py")
    
    print("\nUsage:")
    print("model = create_gpt_model(config)")
    print("metrics = train_model(model, train_data, config, device)")
