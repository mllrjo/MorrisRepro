"""
Binary Morris Configuration and Integration
Configures models and training for binary sequence memorization experiments.

File: src/binary_morris_config.py
"""

# Import existing components
from model_trainer import ModelConfig, TrainingConfig, create_gpt_model
from binary_data_generator import generate_random_binary_sequences


def create_binary_model_config(
    n_layers: int = 2,
    d_model: int = 128,
    n_heads: int = 4,
    max_seq_length: int = 32,  # Longer sequences for binary data
    dropout: float = 0.05      # Lower dropout for memorization
) -> ModelConfig:
    """
    Create model configuration optimized for binary sequence memorization.
    
    Args:
        n_layers: Number of transformer layers
        d_model: Model dimension
        n_heads: Number of attention heads
        max_seq_length: Maximum sequence length (recommend 16-64 for binary)
        dropout: Dropout rate (lower for memorization tasks)
        
    Returns:
        ModelConfig configured for binary sequences (vocab_size=2)
    """
    
    return ModelConfig(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        vocab_size=2,  # Binary vocabulary
        max_seq_length=max_seq_length,
        dropout=dropout
    )


def create_binary_training_config(
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    max_steps: int = 100000,    # Enhanced training will likely converge much earlier
    warmup_steps: int = 500,    # Shorter warmup for binary
    weight_decay: float = 0.01
) -> TrainingConfig:
    """
    Create training configuration optimized for binary sequence memorization.
    """
    
    return TrainingConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay
    )


def count_model_parameters(config: ModelConfig) -> int:
    """
    Estimate parameter count for a model configuration without instantiating.
    
    Faster than creating the model for configuration optimization.
    """
    
    # Token embedding: vocab_size * d_model
    token_embed_params = config.vocab_size * config.d_model
    
    # Position embedding: max_seq_length * d_model  
    pos_embed_params = config.max_seq_length * config.d_model
    
    # Per transformer layer:
    # - Multi-head attention: 4 * d_model * d_model (Q, K, V, projection)
    # - Layer norms: 2 * d_model (attention and MLP)
    # - MLP: d_model * (4 * d_model) + (4 * d_model) * d_model = 8 * d_model²
    layer_params = (
        4 * config.d_model * config.d_model +  # Attention
        2 * config.d_model +                   # Layer norms
        8 * config.d_model * config.d_model    # MLP
    )
    
    transformer_params = config.n_layers * layer_params
    
    # Final layer norm: d_model
    final_ln_params = config.d_model
    
    # Output projection: d_model * vocab_size (no bias)
    output_params = config.d_model * config.vocab_size
    
    total_params = (
        token_embed_params + pos_embed_params + 
        transformer_params + final_ln_params + output_params
    )
    
    return total_params


def create_quick_binary_test() -> dict:
    """
    Quick test of binary sequence training for development/debugging.
    """
    
    print("🧪 Binary Morris Quick Test")
    
    # Small configuration for quick testing
    config = create_binary_model_config(
        n_layers=2,
        d_model=64,
        max_seq_length=16
    )
    
    training_config = create_binary_training_config(
        batch_size=8,
        max_steps=10000
    )
    
    # Generate small dataset
    sequences = generate_random_binary_sequences(
        n_samples=1000,
        seq_length=16,
        seed=42
    )
    
    # Test model creation
    model = create_gpt_model(config)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model: {param_count:,} parameters")
    print(f"✅ Data: {len(sequences)} sequences")
    print(f"✅ Config: {config.n_layers}L×{config.d_model}D, vocab={config.vocab_size}")
    
    return {
        'model_config': config,
        'training_config': training_config,
        'sequences': sequences,
        'model_params': param_count,
        'ready_for_training': True
    }


