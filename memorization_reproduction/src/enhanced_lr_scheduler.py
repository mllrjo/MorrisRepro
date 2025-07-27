"""
Enhanced Learning Rate Scheduler for Morris Memorization Experiments
Implements warmup + decay schedules with model-size-dependent scaling

File: enhanced_lr_scheduler.py
Directory: memorization_reproduction/src/
"""

import math
import torch
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass


@dataclass
class EnhancedTrainingConfig:
    """Enhanced training configuration with sophisticated LR scheduling."""
    batch_size: int
    max_steps: int
    base_learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    decay_steps: int
    weight_decay: float
    scheduler_type: str  # 'cosine', 'linear', 'exponential', 'polynomial'
    warmup_type: str     # 'linear', 'exponential'
    gradient_clip_norm: float
    scheduler_params: Dict[str, Any]


class ModelSizeAwareLRScaler:
    """Scale learning rate based on model size following established practices."""
    
    @staticmethod
    def get_base_lr_for_model_size(model_params: int, base_lr: float = 1e-3) -> float:
        """
        Scale base learning rate based on model parameter count.
        Follows GPT scaling practices: larger models need smaller learning rates.
        """
        # Scaling law based on GPT experiments
        # Roughly: lr ∝ 1/sqrt(model_size) for stability
        if model_params < 100_000:        # Micro models
            return base_lr * 10.0         # 1e-2 for tiny models
        elif model_params < 1_000_000:    # Small models  
            return base_lr * 3.0          # 3e-3 for small models
        elif model_params < 10_000_000:   # Medium models
            return base_lr * 1.0          # 1e-3 for medium models  
        elif model_params < 100_000_000:  # Large models
            return base_lr * 0.3          # 3e-4 for large models
        else:                             # Very large models
            return base_lr * 0.1          # 1e-4 for very large models
    
    @staticmethod
    def get_warmup_steps_for_model_size(model_params: int, max_steps: int) -> int:
        """
        Scale warmup period based on model size.
        Larger models need longer warmup for stability.
        """
        if model_params < 100_000:        # Micro: 2% warmup
            return max(100, int(0.02 * max_steps))
        elif model_params < 1_000_000:    # Small: 5% warmup
            return max(500, int(0.05 * max_steps))
        elif model_params < 10_000_000:   # Medium: 8% warmup
            return max(1000, int(0.08 * max_steps))
        elif model_params < 100_000_000:  # Large: 12% warmup
            return max(2000, int(0.12 * max_steps))
        else:                             # Very large: 15% warmup
            return max(5000, int(0.15 * max_steps))


class AdvancedLRScheduler:
    """Advanced learning rate scheduler with multiple schedule types."""
    
    def __init__(self, config: EnhancedTrainingConfig, optimizer: torch.optim.Optimizer):
        self.config = config
        self.optimizer = optimizer
        self.step_count = 0
        
        # Validate configuration
        assert config.warmup_steps < config.max_steps, "Warmup steps must be less than max steps"
        assert config.min_learning_rate < config.base_learning_rate, "Min LR must be less than base LR"
        
    def get_lr_multiplier(self, step: int) -> float:
        """Get learning rate multiplier for current step."""
        if step < self.config.warmup_steps:
            return self._get_warmup_multiplier(step)
        else:
            return self._get_decay_multiplier(step)
    
    def _get_warmup_multiplier(self, step: int) -> float:
        """Calculate warmup multiplier."""
        progress = step / self.config.warmup_steps
        
        if self.config.warmup_type == 'linear':
            return progress
        elif self.config.warmup_type == 'exponential':
            # Exponential warmup: slower start, faster finish
            return progress ** 2
        else:
            raise ValueError(f"Unknown warmup type: {self.config.warmup_type}")
    
    def _get_decay_multiplier(self, step: int) -> float:
        """Calculate decay multiplier after warmup."""
        # Steps in decay phase
        decay_step = step - self.config.warmup_steps
        total_decay_steps = self.config.decay_steps or (self.config.max_steps - self.config.warmup_steps)
        
        if decay_step >= total_decay_steps:
            # Use minimum learning rate
            return self.config.min_learning_rate / self.config.base_learning_rate
        
        progress = decay_step / total_decay_steps
        min_multiplier = self.config.min_learning_rate / self.config.base_learning_rate
        
        if self.config.scheduler_type == 'cosine':
            # Cosine annealing
            multiplier = min_multiplier + (1 - min_multiplier) * 0.5 * (1 + math.cos(math.pi * progress))
            
        elif self.config.scheduler_type == 'linear':
            # Linear decay
            multiplier = 1 - progress * (1 - min_multiplier)
            
        elif self.config.scheduler_type == 'exponential':
            # Exponential decay
            decay_rate = self.config.scheduler_params.get('decay_rate', 0.96)
            steps_per_decay = self.config.scheduler_params.get('steps_per_decay', 1000)
            multiplier = decay_rate ** (decay_step / steps_per_decay)
            multiplier = max(multiplier, min_multiplier)
            
        elif self.config.scheduler_type == 'polynomial':
            # Polynomial decay
            power = self.config.scheduler_params.get('power', 2.0)
            multiplier = (1 - progress) ** power
            multiplier = min_multiplier + (1 - min_multiplier) * multiplier
            
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")
        
        return multiplier
    
    def step(self) -> float:
        """Step the scheduler and return current learning rate."""
        self.step_count += 1
        multiplier = self.get_lr_multiplier(self.step_count)
        current_lr = self.config.base_learning_rate * multiplier
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
            
        return current_lr
    
    def get_current_lr(self) -> float:
        """Get current learning rate without stepping."""
        multiplier = self.get_lr_multiplier(self.step_count)
        return self.config.base_learning_rate * multiplier


def create_enhanced_training_config(
    model_size: str, 
    model_params: int,
    device: str,
    max_steps: Optional[int] = None
) -> EnhancedTrainingConfig:
    """
    Create enhanced training configuration with model-size-aware LR scheduling.
    
    Args:
        model_size: Model size identifier
        model_params: Actual parameter count
        device: Training device
        max_steps: Override max steps
        
    Returns:
        Enhanced training configuration
    """
    # Base configuration map - more conservative than original
    config_map = {
        "Debug-Micro": {"batch_size": 64, "max_steps": 50000},
        "Debug-Mini": {"batch_size": 64, "max_steps": 75000}, 
        "Debug-Small": {"batch_size": 64, "max_steps": 100000},
        "Debug-Medium": {"batch_size": 64, "max_steps": 150000},
        "Debug-Large": {"batch_size": 32, "max_steps": 200000},
        "Morris-Small": {"batch_size": 32, "max_steps": 200000},
        "Morris-Medium": {"batch_size": 16, "max_steps": 300000},
        "Morris-Wide": {"batch_size": 24, "max_steps": 250000},
    }
    
    base_config = config_map.get(model_size, config_map["Debug-Small"])
    if max_steps:
        base_config["max_steps"] = max_steps
    
    # Get model-size-aware learning rate
    base_lr = ModelSizeAwareLRScaler.get_base_lr_for_model_size(model_params)
    min_lr = base_lr * 0.01  # Minimum LR is 1% of base LR
    
    # Get model-size-aware warmup steps
    warmup_steps = ModelSizeAwareLRScaler.get_warmup_steps_for_model_size(
        model_params, base_config["max_steps"]
    )
    
    # Choose scheduler based on model size
    if model_params < 1_000_000:
        scheduler_type = 'cosine'
        scheduler_params = {}
    else:
        # Larger models benefit from polynomial decay
        scheduler_type = 'polynomial'
        scheduler_params = {'power': 1.5}
    
    # Adjust for device constraints
    if device == "cpu":
        base_config["batch_size"] = max(16, base_config["batch_size"] // 2)
        base_config["max_steps"] = max(25000, base_config["max_steps"] // 2)
        warmup_steps = warmup_steps // 2
    
    return EnhancedTrainingConfig(
        batch_size=base_config["batch_size"],
        max_steps=base_config["max_steps"],
        base_learning_rate=base_lr,
        min_learning_rate=min_lr,
        warmup_steps=warmup_steps,
        decay_steps=base_config["max_steps"] - warmup_steps,
        weight_decay=0.0001,  # Minimal regularization for memorization
        scheduler_type=scheduler_type,
        warmup_type='linear',
        gradient_clip_norm=1.0,
        scheduler_params=scheduler_params
    )


def create_optimizer_with_scheduler(
    model: torch.nn.Module,
    config: EnhancedTrainingConfig
) -> tuple[torch.optim.Optimizer, AdvancedLRScheduler]:
    """
    Create optimizer and scheduler pair.
    
    Args:
        model: PyTorch model
        config: Enhanced training configuration
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Use AdamW optimizer with weight decay separation
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.base_learning_rate,  # Will be overridden by scheduler
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),  # GPT-style betas
        eps=1e-8
    )
    
    scheduler = AdvancedLRScheduler(config, optimizer)
    
    return optimizer, scheduler


def print_lr_schedule_preview(config: EnhancedTrainingConfig, num_preview_steps: int = 20):
    """Print preview of learning rate schedule."""
    print(f"\nLearning Rate Schedule Preview:")
    print(f"Base LR: {config.base_learning_rate:.2e}")
    print(f"Min LR: {config.min_learning_rate:.2e}")
    print(f"Warmup steps: {config.warmup_steps:,}")
    print(f"Total steps: {config.max_steps:,}")
    print(f"Scheduler: {config.scheduler_type} with {config.warmup_type} warmup")
    
    # Create dummy optimizer for preview
    dummy_param = torch.nn.Parameter(torch.tensor(1.0))
    dummy_optimizer = torch.optim.AdamW([dummy_param], lr=config.base_learning_rate)
    scheduler = AdvancedLRScheduler(config, dummy_optimizer)
    
    print(f"\nStep      LR        Phase")
    print(f"----      --        -----")
    
    preview_steps = [
        0, config.warmup_steps // 4, config.warmup_steps // 2, 
        config.warmup_steps - 1, config.warmup_steps,
        config.warmup_steps + 1000, config.max_steps // 2,
        config.max_steps - 1000, config.max_steps - 1
    ]
    
    for step in preview_steps:
        if step < config.max_steps:
            lr = config.base_learning_rate * scheduler.get_lr_multiplier(step)
            phase = "Warmup" if step < config.warmup_steps else "Decay"
            print(f"{step:8,} {lr:8.2e} {phase}")


# Example usage and testing
if __name__ == "__main__":
    # Example: Create config for different model sizes
    model_configs = [
        ("Debug-Micro", 30_000),
        ("Debug-Mini", 117_000), 
        ("Debug-Small", 167_000),
        ("Debug-Medium", 666_000),
        ("Morris-Small", 12_000_000),
        ("Morris-Medium", 15_000_000)
    ]
    
    for model_name, param_count in model_configs:
        print(f"\n{'='*60}")
        print(f"Configuration for {model_name} ({param_count:,} parameters)")
        print(f"{'='*60}")
        
        config = create_enhanced_training_config(model_name, param_count, "cuda")
        print_lr_schedule_preview(config, num_preview_steps=10)
