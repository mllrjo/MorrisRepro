"""
Binary Memorization Scaling Exploration
Test memorization performance across 1-100 binary sequences to understand scaling.

File: binary_scaling_exploration.py
"""

import torch
import torch.nn.functional as F
import time
import json
from datetime import datetime
from src.model_trainer import ModelConfig, TrainingConfig, create_gpt_model
from src.binary_data_generator import generate_random_binary_sequences

# Enhanced training import
try:
    from src.enhanced_model_trainer import enhanced_train_model_wrapper
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False


def create_binary_test_config():
    """Create consistent model/training config for all tests."""
    
    model_config = ModelConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        vocab_size=2,
        max_seq_length=16,
        dropout=0.0  # No dropout for memorization
    )
    
    training_config = TrainingConfig(
        batch_size=8,
        learning_rate=1e-2,    # Optimal from our tests
        max_steps=2000,        # Should be plenty for small datasets
        warmup_steps=50,
        weight_decay=0.0       # No weight decay for memorization
    )
    
    return model_config, training_config


def test_single_dataset_size(
    n_sequences: int,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    device: str,
    timeout_seconds: int = 120
) -> dict:
    """Test memorization for a specific number of sequences."""
    
    # Generate binary sequences
    sequences = generate_random_binary_sequences(
        n_samples=n_sequences,
        seq_length=model_config.max_seq_length,
        seed=42 + n_sequences  # Unique seed for each size
    )
    
    # Create fresh model
    model = create_gpt_model(model_config)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_bits = n_sequences * model_config.max_seq_length
    bits_per_param = total_bits / param_count
    
    print(f"  Testing {n_sequences:3d} sequences: {total_bits:4d} bits, {bits_per_param:.4f} bits/param")
    
    start_time = time.time()
    
    try:
        if ENHANCED_AVAILABLE:
            # Use enhanced training
            results = enhanced_train_model_wrapper(
                model=model,
                train_data=sequences,
                original_config=training_config,
                device=device,
                enable_enhanced_training=True
            )
            
            training_time = time.time() - start_time
            final_loss = results.get('final_loss', float('inf'))
            converged = results.get('convergence_achieved', False)
            final_status = results.get('final_status', 'unknown')
            
        else:
            # Fallback to direct training
            results = train_directly(model, sequences, training_config, device, timeout_seconds)
            training_time = results['training_time']
            final_loss = results['final_loss']
            converged = results['converged']
            final_status = results['final_status']
    
    except Exception as e:
        print(f"    ❌ Training failed: {str(e)}")
        return {
            'n_sequences': n_sequences,
            'total_bits': total_bits,
            'bits_per_param': bits_per_param,
            'param_count': param_count,
            'final_loss': float('inf'),
            'converged': False,
            'training_time': 0,
            'final_status': 'error',
            'memorization_quality': 'failed'
        }
    
    # Determine memorization quality
    if final_loss < 0.01:
        memorization_quality = 'perfect'
    elif final_loss < 0.05:
        memorization_quality = 'excellent'
    elif final_loss < 0.1:
        memorization_quality = 'good'
    elif final_loss < 0.2:
        memorization_quality = 'partial'
    else:
        memorization_quality = 'failed'
    
    # Status indicator
    if memorization_quality in ['perfect', 'excellent']:
        status = "✅"
    elif memorization_quality == 'good':
        status = "🟡"
    else:
        status = "❌"
    
    print(f"    {status} Loss: {final_loss:.4f}, Time: {training_time:.1f}s, Quality: {memorization_quality}")
    
    return {
        'n_sequences': n_sequences,
        'total_bits': total_bits,
        'bits_per_param': bits_per_param,
        'param_count': param_count,
        'final_loss': final_loss,
        'converged': converged,
        'training_time': training_time,
        'final_status': final_status,
        'memorization_quality': memorization_quality
    }


def train_directly(model, sequences, config, device, timeout_seconds):
    """Direct training fallback when enhanced training not available."""
    
    model.train()
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    start_time = time.time()
    
    for step in range(config.max_steps):
        # Check timeout
        if time.time() - start_time > timeout_seconds:
            break
            
        optimizer.zero_grad()
        
        # Sample a random sequence from the dataset
        seq_idx = torch.randint(0, len(sequences), (1,)).item()
        sequence = sequences[seq_idx].to(device)
        
        input_ids = sequence[:-1].unsqueeze(0)
        targets = sequence[1:].unsqueeze(0)
        
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        loss.backward()
        optimizer.step()
        
        # Check convergence every 50 steps
        if step % 50 == 0 and step > 0:
            if loss.item() < 0.01:
                converged = True
                break
    else:
        converged = loss.item() < 0.1
    
    training_time = time.time() - start_time
    
    return {
        'final_loss': loss.item(),
        'converged': converged,
        'training_time': training_time,
        'final_status': 'converged' if converged else 'max_steps'
    }


def explore_binary_scaling(
    sequence_counts: list = None,
    device: str = "auto"
) -> dict:
    """
    Explore binary memorization scaling across different dataset sizes.
    
    Args:
        sequence_counts: List of sequence counts to test (default: 1-100)
        device: Computing device
        
    Returns:
        Complete results dictionary
    """
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if sequence_counts is None:
        # Test key points from 1 to 100
        sequence_counts = (
            list(range(1, 11)) +           # 1-10: should be trivial
            list(range(15, 51, 5)) +       # 15,20,25,30,35,40,45,50: scaling up
            list(range(60, 101, 10))       # 60,70,80,90,100: larger tests
        )
    
    print(f"🔬 BINARY MEMORIZATION SCALING EXPLORATION")
    print(f"Testing {len(sequence_counts)} dataset sizes from {min(sequence_counts)} to {max(sequence_counts)} sequences")
    print(f"Device: {device}")
    print(f"Enhanced training: {'✅' if ENHANCED_AVAILABLE else '❌'}")
    print("=" * 70)
    
    # Create consistent configs
    model_config, training_config = create_binary_test_config()
    
    # Get model parameter count for analysis
    temp_model = create_gpt_model(model_config)
    param_count = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
    
    print(f"Model: {param_count:,} parameters")
    print(f"Config: {model_config.n_layers}L×{model_config.d_model}D, seq_len={model_config.max_seq_length}")
    print(f"Training: LR={training_config.learning_rate}, weight_decay={training_config.weight_decay}")
    print("")
    
    results = []
    start_time = time.time()
    
    for i, n_seq in enumerate(sequence_counts):
        print(f"[{i+1:2d}/{len(sequence_counts)}]", end=" ")
        
        result = test_single_dataset_size(
            n_sequences=n_seq,
            model_config=model_config,
            training_config=training_config,
            device=device
        )
        
        results.append(result)
    
    total_time = time.time() - start_time
    
    # Analysis
    print(f"\n📊 SCALING ANALYSIS")
    print("=" * 70)
    
    # Categorize results
    perfect = [r for r in results if r['memorization_quality'] == 'perfect']
    excellent = [r for r in results if r['memorization_quality'] == 'excellent']
    good = [r for r in results if r['memorization_quality'] == 'good']
    partial = [r for r in results if r['memorization_quality'] == 'partial']
    failed = [r for r in results if r['memorization_quality'] == 'failed']
    
    print(f"Results breakdown:")
    print(f"  Perfect (loss < 0.01):   {len(perfect):2d} / {len(results)} tests")
    print(f"  Excellent (loss < 0.05): {len(excellent):2d} / {len(results)} tests")
    print(f"  Good (loss < 0.1):       {len(good):2d} / {len(results)} tests")
    print(f"  Partial (loss < 0.2):    {len(partial):2d} / {len(results)} tests")
    print(f"  Failed (loss ≥ 0.2):     {len(failed):2d} / {len(results)} tests")
    
    # Find capacity transition points
    perfect_max = max([r['n_sequences'] for r in perfect]) if perfect else 0
    excellent_max = max([r['n_sequences'] for r in (perfect + excellent)]) if (perfect + excellent) else 0
    good_max = max([r['n_sequences'] for r in (perfect + excellent + good)]) if (perfect + excellent + good) else 0
    
    print(f"\nCapacity transition points:")
    print(f"  Perfect memorization up to:   {perfect_max:3d} sequences")
    print(f"  Excellent memorization up to: {excellent_max:3d} sequences")
    print(f"  Good memorization up to:      {good_max:3d} sequences")
    
    # Bits per parameter analysis
    perfect_bits_per_param = [r['bits_per_param'] for r in perfect]
    excellent_bits_per_param = [r['bits_per_param'] for r in (perfect + excellent)]
    
    if perfect_bits_per_param:
        max_perfect_bpp = max(perfect_bits_per_param)
        print(f"  Maximum perfect bits/param:   {max_perfect_bpp:.4f}")
    
    if excellent_bits_per_param:
        max_excellent_bpp = max(excellent_bits_per_param)
        print(f"  Maximum excellent bits/param: {max_excellent_bpp:.4f}")
    
    # Training time analysis
    avg_time = sum(r['training_time'] for r in results) / len(results)
    max_time = max(r['training_time'] for r in results)
    
    print(f"\nTraining time analysis:")
    print(f"  Average training time: {avg_time:.1f} seconds")
    print(f"  Maximum training time: {max_time:.1f} seconds")
    print(f"  Total experiment time: {total_time/60:.1f} minutes")
    
    # Morris progress analysis
    morris_target = 3.6
    best_result = max(results, key=lambda x: x['bits_per_param'])
    best_bpp = best_result['bits_per_param']
    morris_progress = best_bpp / morris_target
    
    print(f"\nMorris target analysis:")
    print(f"  Best bits/param achieved: {best_bpp:.4f}")
    print(f"  Morris target: {morris_target}")
    print(f"  Progress toward Morris: {morris_progress:.1%}")
    
    # Summary
    success_rate = len([r for r in results if r['memorization_quality'] in ['perfect', 'excellent', 'good']]) / len(results)
    
    print(f"\n🎯 EXPLORATION SUMMARY:")
    print(f"Overall success rate: {success_rate:.1%}")
    if success_rate > 0.8:
        print(f"✅ Binary memorization working excellently!")
    elif success_rate > 0.5:
        print(f"🟡 Binary memorization working partially")
    else:
        print(f"❌ Binary memorization needs fixing")
    
    return {
        'results': results,
        'model_config': model_config.__dict__,
        'training_config': training_config.__dict__,
        'param_count': param_count,
        'sequence_counts': sequence_counts,
        'summary': {
            'perfect_count': len(perfect),
            'excellent_count': len(excellent),
            'good_count': len(good),
            'partial_count': len(partial),
            'failed_count': len(failed),
            'success_rate': success_rate,
            'perfect_max_sequences': perfect_max,
            'excellent_max_sequences': excellent_max,
            'good_max_sequences': good_max,
            'max_perfect_bits_per_param': max(perfect_bits_per_param) if perfect_bits_per_param else 0,
            'max_excellent_bits_per_param': max(excellent_bits_per_param) if excellent_bits_per_param else 0,
            'best_bits_per_param': best_bpp,
            'morris_progress': morris_progress,
            'avg_training_time': avg_time,
            'total_experiment_time': total_time
        },
        'timestamp': datetime.now().isoformat()
    }


def save_scaling_results(results: dict, filename: str = None):
    """Save scaling exploration results to JSON file."""
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"binary_scaling_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Results saved to: {filename}")
    return filename


def print_detailed_results(results: dict):
    """Print detailed results table."""
    
    print(f"\n📋 DETAILED RESULTS TABLE")
    print("=" * 80)
    print(f"{'Seq':<4} {'Bits':<5} {'B/P':<7} {'Loss':<8} {'Time':<6} {'Quality':<10} {'Status'}")
    print("-" * 80)
    
    for r in results['results']:
        print(f"{r['n_sequences']:<4d} "
              f"{r['total_bits']:<5d} "
              f"{r['bits_per_param']:<7.4f} "
              f"{r['final_loss']:<8.4f} "
              f"{r['training_time']:<6.1f} "
              f"{r['memorization_quality']:<10} "
              f"{r['final_status']}")


if __name__ == "__main__":
    print("Binary Memorization Scaling Exploration")
    print("Testing 1-100 sequences to understand scaling behavior")
    print("=" * 60)
    
    # Run the exploration
    results = explore_binary_scaling()
    
    # Save results
    filename = save_scaling_results(results)
    
    # Print detailed table
    print_detailed_results(results)
    
    print(f"\n🎯 EXPLORATION COMPLETE!")
    print(f"Results saved to: {filename}")
