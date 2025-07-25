"""
File: test_model_trainer.py
Directory: memorization_reproduction/tests/

Comprehensive tests for model_trainer.py module.
Validates GPT model architecture, training, and likelihood calculations.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import tempfile
import os
from unittest.mock import Mock, patch

# Import the module to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_trainer import (
    ModelConfig,
    TrainingConfig,
    GPTModel,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
    SequenceDataset,
    create_gpt_model,
    count_parameters,
    train_model,
    evaluate_model,
    get_sequence_likelihoods,
    create_dataloader,
    get_linear_warmup_scheduler,
    save_model_checkpoint,
    load_model_checkpoint,
    get_model_size_mb
)


class TestModelConfig:
    """Test ModelConfig dataclass."""
    
    def test_basic_config_creation(self):
        """Test basic model configuration creation."""
        config = ModelConfig(
            n_layers=2,
            d_model=64,
            n_heads=4,
            vocab_size=1000,
            max_seq_length=128
        )
        
        assert config.n_layers == 2
        assert config.d_model == 64
        assert config.n_heads == 4
        assert config.vocab_size == 1000
        assert config.max_seq_length == 128
        assert config.dropout == 0.1  # Default value
    
    def test_config_with_custom_dropout(self):
        """Test configuration with custom dropout."""
        config = ModelConfig(
            n_layers=1,
            d_model=32,
            n_heads=2,
            vocab_size=100,
            max_seq_length=64,
            dropout=0.2
        )
        
        assert config.dropout == 0.2


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""
    
    def test_basic_training_config(self):
        """Test basic training configuration."""
        config = TrainingConfig(
            batch_size=32,
            learning_rate=1e-3,
            max_steps=1000,
            warmup_steps=100
        )
        
        assert config.batch_size == 32
        assert config.learning_rate == 1e-3
        assert config.max_steps == 1000
        assert config.warmup_steps == 100
        assert config.weight_decay == 0.01  # Default value


class TestMultiHeadAttention:
    """Test MultiHeadAttention module."""
    
    def setup_method(self):
        """Set up test parameters."""
        self.d_model = 64
        self.n_heads = 4
        self.seq_len = 16
        self.batch_size = 2
    
    def test_attention_creation(self):
        """Test attention module creation."""
        attention = MultiHeadAttention(self.d_model, self.n_heads)
        
        assert attention.d_model == self.d_model
        assert attention.n_heads == self.n_heads
        assert attention.d_k == self.d_model // self.n_heads
    
    def test_attention_forward_pass(self):
        """Test attention forward pass."""
        attention = MultiHeadAttention(self.d_model, self.n_heads)
        
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = attention(x)
        
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)
    
    def test_attention_with_mask(self):
        """Test attention with causal mask."""
        attention = MultiHeadAttention(self.d_model, self.n_heads)
        
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        mask = torch.tril(torch.ones(self.seq_len, self.seq_len)).unsqueeze(0).unsqueeze(0)
        
        output = attention(x, mask)
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)
    
    def test_attention_invalid_dimensions(self):
        """Test attention with invalid d_model/n_heads combination."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=63, n_heads=4)  # 63 not divisible by 4


class TestFeedForward:
    """Test FeedForward module."""
    
    def test_feedforward_creation(self):
        """Test feed-forward module creation."""
        d_model = 64
        d_ff = 256
        ff = FeedForward(d_model, d_ff)
        
        assert ff.linear1.in_features == d_model
        assert ff.linear1.out_features == d_ff
        assert ff.linear2.in_features == d_ff
        assert ff.linear2.out_features == d_model
    
    def test_feedforward_forward_pass(self):
        """Test feed-forward forward pass."""
        d_model = 64
        d_ff = 256
        batch_size = 2
        seq_len = 16
        
        ff = FeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = ff(x)
        assert output.shape == (batch_size, seq_len, d_model)


class TestTransformerBlock:
    """Test TransformerBlock module."""
    
    def setup_method(self):
        """Set up test parameters."""
        self.d_model = 64
        self.n_heads = 4
        self.batch_size = 2
        self.seq_len = 16
    
    def test_transformer_block_creation(self):
        """Test transformer block creation."""
        block = TransformerBlock(self.d_model, self.n_heads)
        
        assert isinstance(block.attention, MultiHeadAttention)
        assert isinstance(block.feed_forward, FeedForward)
        assert isinstance(block.norm1, nn.LayerNorm)
        assert isinstance(block.norm2, nn.LayerNorm)
    
    def test_transformer_block_forward(self):
        """Test transformer block forward pass."""
        block = TransformerBlock(self.d_model, self.n_heads)
        
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = block(x)
        
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)
    
    def test_transformer_block_with_mask(self):
        """Test transformer block with causal mask."""
        block = TransformerBlock(self.d_model, self.n_heads)
        
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        mask = torch.tril(torch.ones(self.seq_len, self.seq_len)).unsqueeze(0).unsqueeze(0)
        
        output = block(x, mask)
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)


class TestGPTModel:
    """Test GPTModel implementation."""
    
    def setup_method(self):
        """Set up test model configuration."""
        self.config = ModelConfig(
            n_layers=2,
            d_model=64,
            n_heads=4,
            vocab_size=1000,
            max_seq_length=128
        )
    
    def test_gpt_model_creation(self):
        """Test GPT model creation."""
        model = GPTModel(self.config)
        
        assert model.config == self.config
        assert len(model.transformer_blocks) == self.config.n_layers
        assert model.token_embedding.num_embeddings == self.config.vocab_size
        assert model.token_embedding.embedding_dim == self.config.d_model
    
    def test_gpt_model_forward_pass(self):
        """Test GPT model forward pass."""
        model = GPTModel(self.config)
        
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        logits = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, self.config.vocab_size)
    
    def test_causal_mask_creation(self):
        """Test causal mask creation."""
        model = GPTModel(self.config)
        seq_len = 5
        device = torch.device('cpu')
        
        mask = model._create_causal_mask(seq_len, device)
        
        assert mask.shape == (1, 1, seq_len, seq_len)
        
        # Check that mask is lower triangular
        expected_mask = torch.tril(torch.ones(seq_len, seq_len))
        assert torch.equal(mask.squeeze(), expected_mask)
    
    def test_model_weight_initialization(self):
        """Test that model weights are properly initialized."""
        model = GPTModel(self.config)
        
        # Check that embeddings are initialized with reasonable values
        token_embed_std = model.token_embedding.weight.std().item()
        assert 0.01 < token_embed_std < 0.03  # Should be around 0.02
        
        # Check layer norm initialization
        for block in model.transformer_blocks:
            assert torch.allclose(block.norm1.weight, torch.ones_like(block.norm1.weight))
            assert torch.allclose(block.norm1.bias, torch.zeros_like(block.norm1.bias))


class TestSequenceDataset:
    """Test SequenceDataset implementation."""
    
    def test_dataset_creation(self):
        """Test dataset creation and basic functionality."""
        sequences = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        dataset = SequenceDataset(sequences)
        
        assert len(dataset) == 2
        assert torch.equal(dataset[0], sequences[0])
        assert torch.equal(dataset[1], sequences[1])
    
    def test_empty_dataset(self):
        """Test empty dataset."""
        dataset = SequenceDataset([])
        assert len(dataset) == 0


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_gpt_model(self):
        """Test GPT model creation function."""
        config = ModelConfig(
            n_layers=1,
            d_model=32,
            n_heads=2,
            vocab_size=100,
            max_seq_length=64
        )
        
        model = create_gpt_model(config)
        
        assert isinstance(model, GPTModel)
        assert model.config == config
    
    def test_count_parameters(self):
        """Test parameter counting."""
        config = ModelConfig(
            n_layers=1,
            d_model=32,
            n_heads=2,
            vocab_size=100,
            max_seq_length=64
        )
        
        model = create_gpt_model(config)
        param_count = count_parameters(model)
        
        # Verify parameter count is reasonable
        assert param_count > 0
        
        # Manual calculation for verification
        expected_params = (
            100 * 32 +  # token embedding
            64 * 32 +   # position embedding
            # Transformer block parameters (approximate)
            32 * 32 * 3 +  # attention weights (q, k, v)
            32 * 32 +      # attention output
            32 * 128 +     # feed forward 1
            128 * 32 +     # feed forward 2
            32 * 2 +       # layer norms
            100 * 32       # output head
        )
        
        # Should be in the right ballpark
        assert abs(param_count - expected_params) / expected_params < 0.5
    
    def test_create_dataloader(self):
        """Test dataloader creation."""
        sequences = [torch.randint(0, 100, (10,)) for _ in range(20)]
        batch_size = 4
        
        dataloader = create_dataloader(sequences, batch_size, shuffle=False)
        
        assert len(dataloader.dataset) == 20
        assert dataloader.batch_size == batch_size
    
    def test_linear_warmup_scheduler(self):
        """Test linear warmup scheduler."""
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        warmup_steps = 100
        total_steps = 1000
        
        scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)
        
        # Test warmup phase
        assert scheduler.get_last_lr()[0] == 0.0  # Should start at 0
        
        # Step through warmup
        for step in range(50):
            optimizer.step()
            scheduler.step()
        
        current_lr = scheduler.get_last_lr()[0]
        expected_lr = (50 / warmup_steps) * 1e-3
        assert abs(current_lr - expected_lr) < 1e-6
    
    def test_get_model_size_mb(self):
        """Test model size calculation."""
        config = ModelConfig(
            n_layers=1,
            d_model=32,
            n_heads=2,
            vocab_size=100,
            max_seq_length=64
        )
        
        model = create_gpt_model(config)
        size_mb = get_model_size_mb(model)
        
        assert size_mb > 0
        assert size_mb < 10  # Should be small for this tiny model


class TestModelTraining:
    """Test model training functionality."""
    
    def setup_method(self):
        """Set up training test environment."""
        self.device = "cpu"  # Use CPU for testing
        
        self.model_config = ModelConfig(
            n_layers=1,
            d_model=32,
            n_heads=2,
            vocab_size=100,
            max_seq_length=32
        )
        
        self.training_config = TrainingConfig(
            batch_size=4,
            learning_rate=1e-3,
            max_steps=10,  # Very short for testing
            warmup_steps=2
        )
        
        # Create simple training data
        self.train_data = [
            torch.randint(0, 100, (16,)) for _ in range(20)
        ]
    
    def test_basic_training(self):
        """Test basic model training."""
        model = create_gpt_model(self.model_config)
        
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        metrics = train_model(model, self.train_data, self.training_config, self.device)
        
        # Check that training metrics are returned
        assert "train_loss" in metrics
        assert "learning_rate" in metrics
        assert "step" in metrics
        
        # Check that model parameters have changed
        params_changed = False
        for name, param in model.named_parameters():
            if not torch.equal(param, initial_params[name]):
                params_changed = True
                break
        
        assert params_changed, "Model parameters should change during training"
    
    def test_training_with_empty_data(self):
        """Test training behavior with empty data."""
        model = create_gpt_model(self.model_config)
        
        # Should handle empty data gracefully
        metrics = train_model(model, [], self.training_config, self.device)
        
        assert "train_loss" in metrics
        assert len(metrics["train_loss"]) >= 0  # May be empty but shouldn't crash


class TestModelEvaluation:
    """Test model evaluation functionality."""
    
    def setup_method(self):
        """Set up evaluation test environment."""
        self.device = "cpu"
        
        self.config = ModelConfig(
            n_layers=1,
            d_model=32,
            n_heads=2,
            vocab_size=100,
            max_seq_length=32
        )
        
        self.model = create_gpt_model(self.config)
        
        self.test_data = [
            torch.randint(0, 100, (16,)) for _ in range(10)
        ]
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        results = evaluate_model(self.model, self.test_data, self.device)
        
        assert "loss" in results
        assert "perplexity" in results
        assert "total_tokens" in results
        assert "total_loss" in results
        
        assert results["loss"] > 0
        assert results["perplexity"] > 1  # exp(loss) should be > 1
        assert results["total_tokens"] > 0
    
    def test_evaluation_with_empty_data(self):
        """Test evaluation with empty data."""
        results = evaluate_model(self.model, [], self.device)
        
        assert results["total_tokens"] == 0
        assert math.isnan(results["loss"])
        assert math.isnan(results["perplexity"])
        assert results["total_loss"] == 0.0
    
    def test_sequence_likelihoods(self):
        """Test sequence likelihood calculation."""
        sequences = [torch.randint(0, 100, (10,)) for _ in range(5)]
        
        likelihoods = get_sequence_likelihoods(self.model, sequences, self.device)
        
        assert len(likelihoods) == len(sequences)
        assert all(likelihood > 0 for likelihood in likelihoods)  # Should be positive (negative log-likelihood)
        assert isinstance(likelihoods, np.ndarray)
    
    def test_likelihood_calculation_accuracy(self):
        """Test that likelihood calculation matches expected values."""
        # Create a simple sequence
        sequence = torch.tensor([1, 2, 3, 4, 5])
        
        likelihoods = get_sequence_likelihoods(self.model, [sequence], self.device)
        
        # Manual calculation for verification
        self.model.eval()
        with torch.no_grad():
            logits = self.model(sequence.unsqueeze(0))
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = sequence[1:].unsqueeze(0)
            
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1))
            manual_nll = -token_log_probs.sum().item() / math.log(2)  # Convert to bits
        
        assert abs(likelihoods[0] - manual_nll) < 1e-6


class TestModelCheckpointing:
    """Test model checkpointing functionality."""
    
    def setup_method(self):
        """Set up checkpointing test environment."""
        self.config = ModelConfig(
            n_layers=1,
            d_model=32,
            n_heads=2,
            vocab_size=100,
            max_seq_length=32
        )
        
        self.model = create_gpt_model(self.config)
        self.optimizer = optim.Adam(self.model.parameters())
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading model checkpoints."""
        step = 100
        loss = 2.5
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            filepath = tmp_file.name
        
        try:
            # Save checkpoint
            save_model_checkpoint(self.model, self.optimizer, step, loss, filepath)
            
            assert os.path.exists(filepath)
            
            # Create new model and optimizer
            new_model = create_gpt_model(self.config)
            new_optimizer = optim.Adam(new_model.parameters())
            
            # Load checkpoint
            loaded_step, loaded_loss = load_model_checkpoint(
                new_model, new_optimizer, filepath, device="cpu"
            )
            
            assert loaded_step == step
            assert loaded_loss == loss
            
            # Check that model parameters match
            for (name1, param1), (name2, param2) in zip(
                self.model.named_parameters(), new_model.named_parameters()
            ):
                assert name1 == name2
                assert torch.equal(param1, param2)
        
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_training_pipeline(self):
        """Test complete training and evaluation pipeline."""
        device = "cpu"
        
        # Create model configuration
        model_config = ModelConfig(
            n_layers=1,
            d_model=32,
            n_heads=2,
            vocab_size=50,
            max_seq_length=16
        )
        
        training_config = TrainingConfig(
            batch_size=4,
            learning_rate=1e-3,
            max_steps=5,
            warmup_steps=1
        )
        
        # Generate training data
        train_data = [torch.randint(0, 50, (12,)) for _ in range(16)]
        test_data = [torch.randint(0, 50, (12,)) for _ in range(4)]
        
        # Create and train model
        model = create_gpt_model(model_config)
        initial_param_count = count_parameters(model)
        
        metrics = train_model(model, train_data, training_config, device)
        
        # Evaluate model
        eval_results = evaluate_model(model, test_data, device)
        
        # Get sequence likelihoods
        likelihoods = get_sequence_likelihoods(model, test_data[:2], device)
        
        # Verify all components work together
        assert initial_param_count > 0
        assert len(metrics["train_loss"]) > 0
        assert eval_results["loss"] > 0
        assert len(likelihoods) == 2
        assert all(likelihood > 0 for likelihood in likelihoods)
    
    def test_model_configurations(self):
        """Test various model configurations."""
        configurations = [
            ModelConfig(n_layers=1, d_model=16, n_heads=2, vocab_size=32, max_seq_length=8),
            ModelConfig(n_layers=2, d_model=32, n_heads=4, vocab_size=64, max_seq_length=16),
            ModelConfig(n_layers=3, d_model=64, n_heads=8, vocab_size=128, max_seq_length=32),
        ]
        
        for config in configurations:
            model = create_gpt_model(config)
            
            # Test forward pass
            batch_size = 2
            seq_len = min(config.max_seq_length, 8)  # Keep small for testing
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            
            logits = model(input_ids)
            assert logits.shape == (batch_size, seq_len, config.vocab_size)
            
            # Test parameter counting
            param_count = count_parameters(model)
            assert param_count > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
