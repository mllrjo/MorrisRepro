
#!/usr/bin/env python3
"""
Run Binary Morris Memorization Experiment
Main script to execute binary sequence memorization experiments.

Usage:
    python run_binary_morris.py --single --seq-length 20 --model-params 100000

File: run_binary_morris.py
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import torch

# Import our binary Morris system
from src.binary_morris_config import (
    create_binary_model_config,
    create_binary_training_config,
    count_model_parameters,
    create_quick_binary_test
)
from src.binary_data_generator import generate_random_binary_sequences
from src.model_trainer import create_gpt_model

# Import enhanced training components
try:
    from src.enhanced_model_trainer import enhanced_train_model_wrapper
    from src.model_trainer import train_model
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False


def run_single_binary_experiment(
    seq_length: int = 20,
    model_params: int = 100000,
    dataset_size: int = 10000,
    device: str = "auto"
) -> dict:
    """
    Run a single binary memorization experiment for testing.
    """
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🧪 Single Binary Experiment")
    print(f"   Sequence length: {seq_length} bits")
    print(f"   Target model size: {model_params:,} parameters")
    print(f"   Dataset size: {dataset_size:,} sequences")
    print(f"   Device: {device}")
    
    # Create configuration targeting specific parameter count
    model_config = create_binary_model_config(
        n_layers=2,
        d_model=64,
        max_seq_length=seq_length
    )
    
    # Adjust d_model to hit target parameters approximately
    actual_params = count_model_parameters(model_config)
    if actual_params < model_params:
        scale_factor = (model_params / actual_params) ** 0.5
        new_d_model = int(model_config.d_model * scale_factor)
        # Ensure divisible by n_heads
        new_d_model = (new_d_model // model_config.n_heads) * model_config.n_heads
        model_config.d_model = max(32, new_d_model)
    
    actual_params = count_model_parameters(model_config)
    
    # Create training configuration
    training_config = create_binary_training_config(
        batch_size=16,
        learning_rate=1e-3,
        max_steps=25000
    )
    
    # Generate binary data
    print(f"   Generating {dataset_size:,} binary sequences...")
    sequences = generate_random_binary_sequences(
        n_samples=dataset_size,
        seq_length=seq_length,
        seed=42
    )
    
    # Create and train model
    print(f"   Creating model with {actual_params:,} parameters...")
    model = create_gpt_model(model_config)
    
    start_time = time.time()
    
    if ENHANCED_AVAILABLE:
        print(f"   Training with enhanced method...")
        results = enhanced_train_model_wrapper(
            model=model,
            train_data=sequences,
            original_config=training_config,
            device=device,
            enable_enhanced_training=True
        )
    else:
        print(f"   Training with original method...")
        results = train_model(model, sequences, training_config, device)
    
    training_time = time.time() - start_time
    
    # Calculate memorization
    total_bits = dataset_size * seq_length
    bits_per_param = total_bits / actual_params
    
    # Compile results
    experiment_results = {
        'config': {
            'seq_length': seq_length,
            'target_params': model_params,
            'actual_params': actual_params,
            'dataset_size': dataset_size,
        },
        'results': {
            'total_bits': total_bits,
            'bits_per_param': bits_per_param,
            'training_time': training_time,
            'final_loss': results.get('final_loss', float('inf')),
            'convergence_achieved': results.get('convergence_achieved', False),
            'final_status': results.get('final_status', 'unknown'),
            'morris_progress': min(bits_per_param / 3.6, 1.0)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Display results
    print(f"\n📊 Results:")
    print(f"   Bits per parameter: {bits_per_param:.3f}")
    print(f"   Morris progress: {experiment_results['results']['morris_progress']:.1%}")
    print(f"   Training time: {training_time:.1f} seconds")
    print(f"   Final loss: {results.get('final_loss', 'N/A')}")
    print(f"   Converged: {'✓' if results.get('convergence_achieved') else '✗'}")
    
    return experiment_results


def main():
    parser = argparse.ArgumentParser(description="Run Binary Morris Memorization Experiments")
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto',
                       help='Computing device')
    parser.add_argument('--single', action='store_true',
                       help='Run single experiment')
    parser.add_argument('--seq-length', type=int, default=20,
                       help='Sequence length')
    parser.add_argument('--model-params', type=int, default=100000,
                       help='Target model parameters')
    parser.add_argument('--dataset-size', type=int, default=25000,
                       help='Dataset size')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick system test')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🔬 Binary Morris Memorization Experiments")
    print(f"Device: {device}")
    print(f"Enhanced training available: {ENHANCED_AVAILABLE}")
    
    try:
        if args.quick_test:
            # Quick system test
            test_results = create_quick_binary_test()
            print(f"✅ Quick test completed")
            
        elif args.single:
            # Run single experiment
            results = run_single_binary_experiment(
                seq_length=args.seq_length,
                model_params=args.model_params,
                dataset_size=args.dataset_size,
                device=device
            )
            
            bits_per_param = results['results']['bits_per_param']
            morris_progress = results['results']['morris_progress']
            
            print(f"\n🎯 Final Summary:")
            print(f"   Bits per parameter: {bits_per_param:.3f}")
            print(f"   Morris target progress: {morris_progress:.1%}")
            
        else:
            print("Use --single or --quick-test")
            
    except Exception as e:
        print(f"\n❌ Experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()


