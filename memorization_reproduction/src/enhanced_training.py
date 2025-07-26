"""
Enhanced Training Module for Morris Memorization Reproduction
Fixes MAX_STEPS convergence issues with adaptive training and better convergence detection.

File: src/enhanced_training.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Callable, Tuple
import math
import numpy as np
from collections import deque
import time

class EnhancedTrainingConfig:
    """Enhanced training configuration addressing MAX_STEPS issues"""
    
    def __init__(self, 
                 batch_size: int = 16,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 0.01,
                 max_steps: int = 100_000,  # Increased from 35,000
                 warmup_steps: int = 1000,
                 memorization_threshold: float = 0.15,
                 patience: int = 8000,
                 min_improvement: float = 0.001,
                 convergence_window: int = 1000,
                 use_adaptive_lr: bool = True,
                 lr_decay_factor: float = 0.7,
                 lr_patience: int = 3000,
                 min_lr: float = 1e-6,
                 memorization_check_interval: int = 500,
                 target_memorization_rate: float = 0.95,
                 early_stop_on_memorization: bool = True,
                 memorization_patience: int = 2000):
        
        # Basic training parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        
        # Enhanced convergence detection
        self.memorization_threshold = memorization_threshold
        self.patience = patience
        self.min_improvement = min_improvement
        self.convergence_window = convergence_window
        
        # Adaptive learning rate
        self.use_adaptive_lr = use_adaptive_lr
        self.lr_decay_factor = lr_decay_factor
        self.lr_patience = lr_patience
        self.min_lr = min_lr
        
        # Memorization rate monitoring
        self.memorization_check_interval = memorization_check_interval
        self.target_memorization_rate = target_memorization_rate
        
        # Early stopping based on memorization achievement
        self.early_stop_on_memorization = early_stop_on_memorization
        self.memorization_patience = memorization_patience


def adaptive_memorization_training(
    model: torch.nn.Module,
    train_data: List[torch.Tensor],
    config: EnhancedTrainingConfig,
    device: str = "cuda",
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Enhanced training with adaptive convergence detection.
    Fixes MAX_STEPS termination by implementing sophisticated early stopping.
    """
    
    model.train()
    model = model.to(device)
    
    # Setup optimizer with warmup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler (removed verbose parameter for compatibility)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config.lr_decay_factor,
        patience=config.lr_patience,
        min_lr=config.min_lr
    )
    
    # Training state tracking
    training_state = {
        'step': 0,
        'best_loss': float('inf'),
        'steps_without_improvement': 0,
        'loss_history': deque(maxlen=config.convergence_window),
        'memorization_history': deque(maxlen=20),
        'current_lr': config.learning_rate,
        'memorization_achieved': False,
        'convergence_reason': 'MAX_STEPS'
    }
    
    # Metrics tracking
    metrics = {
        'loss': [],
        'learning_rate': [],
        'memorization_rate': [],
        'steps': [],
        'convergence_info': {}
    }
    
    print(f"Starting enhanced training with {len(train_data)} sequences")
    print(f"Target: loss < {config.memorization_threshold}, max steps: {config.max_steps}")
    
    start_time = time.time()
    
    for step in range(config.max_steps):
        training_state['step'] = step
        
        # Learning rate warmup
        if step < config.warmup_steps:
            lr = config.learning_rate * (step + 1) / config.warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            training_state['current_lr'] = lr
        
        # Training step
        optimizer.zero_grad()
        
        # Sample batch
        batch_indices = torch.randint(0, len(train_data), (config.batch_size,))
        batch_losses = []
        
        for idx in batch_indices:
            sequence = train_data[idx].to(device)
            
            # Add batch dimension for model input
            input_sequence = sequence[:-1].unsqueeze(0)  # Shape: [1, seq_len-1]
            target_sequence = sequence[1:].unsqueeze(0)  # Shape: [1, seq_len-1]
            
            # Forward pass
            logits = model(input_sequence)  # Shape: [1, seq_len-1, vocab_size]
            
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),  # [seq_len-1, vocab_size]
                target_sequence.view(-1)           # [seq_len-1]
            )
            batch_losses.append(loss)
        
        # Aggregate batch loss
        total_loss = torch.stack(batch_losses).mean()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update training state
        current_loss = total_loss.item()
        training_state['loss_history'].append(current_loss)
        
        # Track metrics
        if step % 100 == 0 or step == 0:  # Always track step 0
            metrics['loss'].append(current_loss)
            metrics['learning_rate'].append(training_state['current_lr'])
            metrics['steps'].append(step)
            
            if step % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"Step {step:6d}: Loss {current_loss:.4f}, LR {training_state['current_lr']:.2e}, Time {elapsed:.1f}s")
        
        # Check memorization rate periodically
        if step % config.memorization_check_interval == 0:
            memorization_rate = calculate_memorization_rate(
                model, train_data, config.memorization_threshold, device
            )
            training_state['memorization_history'].append(memorization_rate)
            metrics['memorization_rate'].append(memorization_rate)
            
            if step % 1000 == 0:
                print(f"         Memorization rate: {memorization_rate:.3f}")
        
        # Convergence detection
        converged, reason, convergence_info = detect_memorization_convergence(
            training_state, config
        )
        
        if converged:
            print(f"CONVERGED: {reason} at step {step}")
            print(f"Final loss: {current_loss:.4f}")
            if training_state['memorization_history']:
                final_mem_rate = training_state['memorization_history'][-1]
                print(f"Final memorization rate: {final_mem_rate:.3f}")
            
            metrics['convergence_info'] = {
                'converged': True,
                'reason': reason,
                'final_step': step,
                'final_loss': current_loss,
                'details': convergence_info
            }
            break
        
        # Learning rate scheduling
        if config.use_adaptive_lr and step > config.warmup_steps:
            if step % config.lr_patience == 0:
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(current_loss)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != old_lr:
                    print(f"Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
                    training_state['current_lr'] = new_lr
        
        # Progress callback
        if progress_callback and step % 1000 == 0:
            progress_callback(step, current_loss, training_state)
    
    else:
        # Hit max steps without convergence
        print(f"Reached max steps ({config.max_steps}) without convergence")
        print(f"Final loss: {current_loss:.4f}")
        
        metrics['convergence_info'] = {
            'converged': False,
            'reason': 'MAX_STEPS',
            'final_step': config.max_steps,
            'final_loss': current_loss,
            'details': {'max_steps_reached': True}
        }
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f} seconds")
    
    # Ensure we have at least some loss metrics
    if not metrics['loss']:
        # Add final loss if no metrics were collected
        metrics['loss'].append(current_loss)
        metrics['steps'].append(step)
        metrics['learning_rate'].append(training_state['current_lr'])
    
    # Return metrics with backwards compatibility keys
    final_metrics = {
        'loss': metrics['loss'],
        'train_loss': metrics['loss'],  # Backwards compatibility
        'learning_rate': metrics['learning_rate'],
        'memorization_rate': metrics['memorization_rate'],
        'steps': metrics['steps'],
        'convergence_info': metrics['convergence_info']
    }
    
    return final_metrics


def detect_memorization_convergence(
    training_state: Dict[str, Any],
    config: EnhancedTrainingConfig
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Sophisticated convergence detection using multiple criteria.
    """
    
    current_loss = training_state['loss_history'][-1] if training_state['loss_history'] else float('inf')
    step = training_state['step']
    
    # Criterion 1: Memorization threshold achieved
    if current_loss < config.memorization_threshold:
        if not training_state['memorization_achieved']:
            training_state['memorization_achieved'] = True
            training_state['memorization_achieved_step'] = step
        
        # Wait for stability after achieving memorization
        steps_since_memorization = step - training_state.get('memorization_achieved_step', step)
        if steps_since_memorization >= config.memorization_patience:
            return True, "MEMORIZATION_ACHIEVED", {
                'threshold': config.memorization_threshold,
                'final_loss': current_loss,
                'steps_since_achievement': steps_since_memorization
            }
    
    # Criterion 2: High memorization rate achieved
    if training_state['memorization_history']:
        recent_mem_rate = training_state['memorization_history'][-1]
        if recent_mem_rate >= config.target_memorization_rate:
            return True, "HIGH_MEMORIZATION_RATE", {
                'memorization_rate': recent_mem_rate,
                'target': config.target_memorization_rate,
                'final_loss': current_loss
            }
    
    # Criterion 3: Loss plateau detection (improved)
    if len(training_state['loss_history']) >= config.convergence_window:
        recent_losses = list(training_state['loss_history'])
        
        # Check for plateau: very small improvement over window
        loss_improvement = recent_losses[0] - recent_losses[-1]
        relative_improvement = loss_improvement / max(recent_losses[0], 1e-8)
        
        if relative_improvement < config.min_improvement:
            training_state['steps_without_improvement'] += 1
        else:
            training_state['steps_without_improvement'] = 0
            
        # Update best loss
        if current_loss < training_state['best_loss']:
            training_state['best_loss'] = current_loss
            training_state['steps_without_improvement'] = 0
        
        # Converge if no improvement for patience steps
        if training_state['steps_without_improvement'] >= config.patience:
            return True, "LOSS_PLATEAU", {
                'plateau_loss': current_loss,
                'steps_without_improvement': training_state['steps_without_improvement'],
                'patience': config.patience
            }
    
    # No convergence yet
    return False, "", {}


def calculate_memorization_rate(
    model: torch.nn.Module,
    sequences: List[torch.Tensor],
    threshold: float = 0.15,
    device: str = "cuda"
) -> float:
    """
    Calculate fraction of sequences memorized by model.
    """
    model.eval()
    memorized_count = 0
    
    with torch.no_grad():
        for sequence in sequences:
            try:
                sequence = sequence.to(device)
                
                # Ensure sequence has enough tokens
                if len(sequence) < 2:
                    continue
                
                # Add batch dimension for model input
                input_sequence = sequence[:-1].unsqueeze(0)  # Shape: [1, seq_len-1]
                target_sequence = sequence[1:].unsqueeze(0)  # Shape: [1, seq_len-1]
                
                # Calculate loss for this sequence
                logits = model(input_sequence)
                
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_sequence.view(-1),
                    reduction='mean'
                )
                
                if loss.item() < threshold:
                    memorized_count += 1
                    
            except Exception as e:
                # Skip problematic sequences
                print(f"Warning: Skipping sequence due to error: {e}")
                continue
    
    model.train()
    return memorized_count / max(len(sequences), 1)


def enhanced_train_model_wrapper(
    model: torch.nn.Module,
    train_data: List[torch.Tensor],
    original_config: Any,
    device: str = "cuda",
    enable_enhanced_training: bool = True
) -> Dict[str, Any]:
    """
    Drop-in replacement for existing train_model function.
    Maintains backward compatibility while fixing convergence issues.
    """
    
    if enable_enhanced_training:
        # Create enhanced config from original
        enhanced_config = EnhancedTrainingConfig(
            batch_size=getattr(original_config, 'batch_size', 16),
            learning_rate=getattr(original_config, 'learning_rate', 1e-3),
            weight_decay=getattr(original_config, 'weight_decay', 0.01),
            max_steps=max(getattr(original_config, 'max_steps', 35000), 100000),  # Ensure minimum 100k
            warmup_steps=getattr(original_config, 'warmup_steps', 1000),
            memorization_threshold=0.15,
            patience=8000
        )
        
        print("Using enhanced training with improved convergence detection")
        metrics = adaptive_memorization_training(model, train_data, enhanced_config, device)
        
        # Ensure backwards compatibility with expected metric keys
        if 'loss' in metrics and metrics['loss']:
            metrics['train_loss'] = metrics['loss']
        elif 'loss' not in metrics or not metrics['loss']:
            # If no loss metrics, create minimal compatible format
            final_loss = metrics['convergence_info'].get('final_loss', float('nan'))
            metrics['loss'] = [final_loss]
            metrics['train_loss'] = [final_loss]
        
        return metrics
    
    else:
        # Fallback to original training (not implemented here)
        print("Enhanced training disabled - using original training")
        raise NotImplementedError("Original training fallback not implemented - enhanced training required")


def create_enhanced_config_from_original(
    original_config: Any,
    memorization_threshold: float = 0.15,
    max_steps: int = 100_000,
    **enhanced_kwargs
) -> EnhancedTrainingConfig:
    """
    Create EnhancedTrainingConfig from existing TrainingConfig.
    """
    return EnhancedTrainingConfig(
        batch_size=getattr(original_config, 'batch_size', 16),
        learning_rate=getattr(original_config, 'learning_rate', 1e-3),
        weight_decay=getattr(original_config, 'weight_decay', 0.01),
        max_steps=max(getattr(original_config, 'max_steps', 35000), max_steps),
        warmup_steps=getattr(original_config, 'warmup_steps', 1000),
        memorization_threshold=memorization_threshold,
        **enhanced_kwargs
    )


if __name__ == "__main__":
    print("Enhanced Training Module for Morris Memorization Reproduction")
    print("=" * 60)
    print("Fixes for MAX_STEPS convergence issues:")
    print("✅ Increased step limit: 35K → 100K+")
    print("✅ Adaptive learning rate scheduling") 
    print("✅ Better convergence detection")
    print("✅ Memorization rate monitoring")
    print("✅ Early stopping on achievement")
    print("✅ Proper batch dimension handling")
    print("✅ PyTorch compatibility fixes")
    print()
    print("Usage:")
    print("from enhanced_training import adaptive_memorization_training, EnhancedTrainingConfig")
    print("config = EnhancedTrainingConfig(max_steps=100_000)")
    print("metrics = adaptive_memorization_training(model, data, config, device)")
