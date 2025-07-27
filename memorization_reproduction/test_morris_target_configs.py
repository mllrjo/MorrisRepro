"""
Test Script for morris_target_configs.py
Validates Morris-target model configurations and functionality

File: test_morris_target_configs.py
Directory: memorization_reproduction/src/
"""

import sys
import torch
import traceback
from typing import List, Dict, Any, Tuple

# Import the module to test
try:
    from morris_target_configs import (
        MorrisTargetModelConfig,
        create_morris_target_configs,
        get_morris_target_dataset_sizes,
        create_morris_enhanced_training_config,
        create_morris_simple_config,
        estimate_morris_execution_time,
        validate_morris_target_feasibility,
        create_morris_reproduction_plan,
        preview_morris_targets
    )
    print("✅ Morris target configs imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class TestMorrisTargetConfigs:
    """Test suite for Morris target configurations."""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
    
    def assert_test(self, condition: bool, test_name: str, expected: Any = None, actual: Any = None):
        """Assert test condition and track results."""
        if condition:
            print(f"✅ {test_name}")
            self.tests_passed += 1
        else:
            error_msg = f"❌ {test_name}"
            if expected is not None and actual is not None:
                error_msg += f" (expected: {expected}, got: {actual})"
            print(error_msg)
            self.tests_failed += 1
            self.failures.append(test_name)
    
    def test_morris_target_model_config(self):
        """Test MorrisTargetModelConfig dataclass."""
        print("\n🧪 Testing MorrisTargetModelConfig...")
        
        config = MorrisTargetModelConfig(
            name="Test-Model",
            n_layers=12,
            d_model=768,
            n_heads=12,
            vocab_size=2048,
            max_seq_length=64,
            dropout=0.1
        )
        
        # Test basic properties
        self.assert_test(config.name == "Test-Model", "Config name")
        self.assert_test(config.n_layers == 12, "Config n_layers")
        self.assert_test(config.d_model == 768, "Config d_model")
        self.assert_test(config.n_heads == 12, "Config n_heads")
        self.assert_test(config.vocab_size == 2048, "Config vocab_size")
        self.assert_test(config.max_seq_length == 64, "Config max_seq_length")
        
        # Test parameter estimation
        estimated_params = config.estimate_parameters()
        self.assert_test(estimated_params > 0, "Parameter estimation positive")
        self.assert_test(estimated_params > 1_000_000, "Parameter estimation reasonable size")
        
        # Test that d_model is divisible by n_heads (required for attention)
        self.assert_test(config.d_model % config.n_heads == 0, "d_model divisible by n_heads")
    
    def test_create_morris_target_configs(self):
        """Test create_morris_target_configs function."""
        print("\n🧪 Testing create_morris_target_configs...")
        
        configs = create_morris_target_configs()
        
        # Should return list of tuples
        self.assert_test(isinstance(configs, list), "Returns list")
        self.assert_test(len(configs) > 0, "Non-empty config list")
        
        expected_models = ["Morris-Small", "Morris-Medium", "Morris-Wide"]
        found_models = []
        
        for config, name, params in configs:
            # Test tuple structure
            self.assert_test(isinstance(config, MorrisTargetModelConfig), f"{name} config is MorrisTargetModelConfig")
            self.assert_test(isinstance(name, str), f"{name} name is string")
            self.assert_test(isinstance(params, int), f"{name} params is int")
            
            # Test parameter ranges (should be Morris-scale: 12-15M)
            self.assert_test(10_000_000 <= params <= 20_000_000, f"{name} parameter count in Morris range", 
                           "10M-20M", f"{params:,}")
            
            # Test model properties
            self.assert_test(config.vocab_size == 2048, f"{name} vocab size is Morris standard")
            self.assert_test(config.max_seq_length == 64, f"{name} seq length is Morris standard")
            self.assert_test(config.n_layers >= 8, f"{name} has reasonable depth")
            self.assert_test(config.d_model >= 512, f"{name} has reasonable width")
            
            found_models.append(name)
        
        # Test that we have all expected models
        for expected in expected_models:
            self.assert_test(expected in found_models, f"Found {expected} model")
    
    def test_parameter_estimation_accuracy(self):
        """Test parameter estimation accuracy."""
        print("\n🧪 Testing Parameter Estimation Accuracy...")
        
        configs = create_morris_target_configs()
        
        for config, name, estimated_params in configs:
            # Create a minimal model to check actual parameters
            try:
                # Simple parameter calculation for validation
                embed_params = config.vocab_size * config.d_model
                pos_embed_params = config.max_seq_length * config.d_model
                layer_params = config.n_layers * (12 * config.d_model ** 2 + 4 * config.d_model)
                output_params = config.d_model * config.vocab_size
                
                manual_estimate = embed_params + pos_embed_params + layer_params + output_params
                
                # Should be within 5% of manual calculation
                error_ratio = abs(estimated_params - manual_estimate) / manual_estimate
                self.assert_test(error_ratio < 0.05, f"{name} parameter estimation accurate", 
                               f"<5% error", f"{error_ratio*100:.1f}% error")
                
            except Exception as e:
                self.assert_test(False, f"{name} parameter estimation failed: {e}")
    
    def test_get_morris_target_dataset_sizes(self):
        """Test get_morris_target_dataset_sizes function."""
        print("\n🧪 Testing get_morris_target_dataset_sizes...")
        
        model_names = ["Morris-Small", "Morris-Medium", "Morris-Wide"]
        
        for model_name in model_names:
            dataset_sizes = get_morris_target_dataset_sizes(model_name)
            
            # Test return type and structure
            self.assert_test(isinstance(dataset_sizes, list), f"{model_name} returns list")
            self.assert_test(len(dataset_sizes) > 0, f"{model_name} non-empty dataset sizes")
            
            # Test dataset size properties
            self.assert_test(all(isinstance(size, int) for size in dataset_sizes), 
                           f"{model_name} all sizes are integers")
            self.assert_test(all(size > 0 for size in dataset_sizes), 
                           f"{model_name} all sizes positive")
            self.assert_test(dataset_sizes == sorted(dataset_sizes), 
                           f"{model_name} sizes in ascending order")
            
            # Test size ranges (should be scaled up from breakthrough: 500-32000)
            min_size, max_size = min(dataset_sizes), max(dataset_sizes)
            self.assert_test(min_size >= 100, f"{model_name} minimum size reasonable")
            self.assert_test(max_size <= 50_000, f"{model_name} maximum size reasonable")
            self.assert_test(max_size >= 10 * min_size, f"{model_name} good size range")
        
        # Test unknown model name
        unknown_sizes = get_morris_target_dataset_sizes("Unknown-Model")
        self.assert_test(isinstance(unknown_sizes, list), "Unknown model returns default list")
    
    def test_create_morris_enhanced_training_config(self):
        """Test create_morris_enhanced_training_config function."""
        print("\n🧪 Testing create_morris_enhanced_training_config...")
        
        model_configs = [
            ("Morris-Small", 12_000_000),
            ("Morris-Medium", 15_000_000),
            ("Morris-Wide", 14_000_000)
        ]
        
        for model_name, param_count in model_configs:
            try:
                config = create_morris_enhanced_training_config(model_name, param_count, "cuda")
                
                # Test config properties
                self.assert_test(hasattr(config, 'batch_size'), f"{model_name} has batch_size")
                self.assert_test(hasattr(config, 'max_steps'), f"{model_name} has max_steps")
                
                # Test that larger models get more conservative settings
                if hasattr(config, 'base_learning_rate'):
                    # Enhanced config
                    self.assert_test(config.base_learning_rate > 0, f"{model_name} positive base LR")
                    self.assert_test(config.max_steps >= 100_000, f"{model_name} long training for Morris scale")
                    
                    if "Medium" in model_name:
                        # Largest model should have longest training
                        self.assert_test(config.max_steps >= 250_000, f"{model_name} extra long training")
                
                elif hasattr(config, 'learning_rate'):
                    # Simple config fallback
                    self.assert_test(config.learning_rate > 0, f"{model_name} positive LR")
                    self.assert_test(config.max_steps >= 100_000, f"{model_name} long training")
                
                self.assert_test(True, f"{model_name} config creation successful")
                
            except Exception as e:
                self.assert_test(False, f"{model_name} config creation failed: {e}")
    
    def test_create_morris_simple_config(self):
        """Test create_morris_simple_config function."""
        print("\n🧪 Testing create_morris_simple_config...")
        
        model_names = ["Morris-Small", "Morris-Medium", "Morris-Wide"]
        
        for model_name in model_names:
            try:
                # Test CUDA config
                config_cuda = create_morris_simple_config(model_name, "cuda")
                self.assert_test(config_cuda.batch_size > 0, f"{model_name} CUDA positive batch size")
                self.assert_test(config_cuda.learning_rate > 0, f"{model_name} CUDA positive LR")
                self.assert_test(config_cuda.max_steps >= 100_000, f"{model_name} CUDA long training")
                
                # Test CPU config (should have adjusted settings)
                config_cpu = create_morris_simple_config(model_name, "cpu")
                self.assert_test(config_cpu.batch_size <= config_cuda.batch_size, 
                               f"{model_name} CPU batch size <= CUDA")
                self.assert_test(config_cpu.max_steps <= config_cuda.max_steps, 
                               f"{model_name} CPU max steps <= CUDA")
                
                self.assert_test(True, f"{model_name} simple config creation successful")
                
            except Exception as e:
                self.assert_test(False, f"{model_name} simple config creation failed: {e}")
    
    def test_estimate_morris_execution_time(self):
        """Test estimate_morris_execution_time function."""
        print("\n🧪 Testing estimate_morris_execution_time...")
        
        configs = create_morris_target_configs()
        
        for config, name, params in configs:
            try:
                # Create dummy training config
                training_config = type('Config', (), {
                    'max_steps': 200_000,
                    'batch_size': 32
                })()
                
                dataset_sizes = [1000, 2000, 4000]
                
                # Test CUDA time estimation
                time_cuda = estimate_morris_execution_time(config, training_config, dataset_sizes, "cuda")
                self.assert_test(time_cuda > 0, f"{name} CUDA time positive")
                self.assert_test(time_cuda < 1e9, f"{name} CUDA time reasonable")  # Less than crazy large
                
                # Test CPU time estimation
                time_cpu = estimate_morris_execution_time(config, training_config, dataset_sizes, "cpu")
                self.assert_test(time_cpu > 0, f"{name} CPU time positive")
                self.assert_test(time_cpu >= time_cuda, f"{name} CPU time >= CUDA time")
                
                # Test scaling with dataset size
                larger_datasets = [1000, 2000, 4000, 8000, 16000]
                time_larger = estimate_morris_execution_time(config, training_config, larger_datasets, "cuda")
                self.assert_test(time_larger > time_cuda, f"{name} time scales with dataset size")
                
                self.assert_test(True, f"{name} execution time estimation successful")
                
            except Exception as e:
                self.assert_test(False, f"{name} execution time estimation failed: {e}")
    
    def test_validate_morris_target_feasibility(self):
        """Test validate_morris_target_feasibility function."""
        print("\n🧪 Testing validate_morris_target_feasibility...")
        
        # Test with high memory (should allow all models)
        feasible_high = validate_morris_target_feasibility("cuda", 16.0)
        self.assert_test(isinstance(feasible_high, list), "High memory returns list")
        self.assert_test(len(feasible_high) > 0, "High memory allows some models")
        
        # Test with low memory (should restrict models)
        feasible_low = validate_morris_target_feasibility("cuda", 4.0)
        self.assert_test(isinstance(feasible_low, list), "Low memory returns list")
        self.assert_test(len(feasible_low) <= len(feasible_high), "Low memory <= high memory feasible models")
        
        # Test with very low memory (should allow no models)
        feasible_none = validate_morris_target_feasibility("cuda", 1.0)
        self.assert_test(isinstance(feasible_none, list), "Very low memory returns list")
        self.assert_test(len(feasible_none) == 0, "Very low memory allows no models")
        
        # Test device types
        feasible_cpu = validate_morris_target_feasibility("cpu", 8.0)
        self.assert_test(isinstance(feasible_cpu, list), "CPU device returns list")
    
    def test_create_morris_reproduction_plan(self):
        """Test create_morris_reproduction_plan function."""
        print("\n🧪 Testing create_morris_reproduction_plan...")
        
        # Test with feasible constraints
        constraints_good = {"gpu_memory_gb": 16.0, "cpu_memory_available_gb": 12.0}
        plan_good = create_morris_reproduction_plan("cuda", constraints_good)
        
        self.assert_test(isinstance(plan_good, dict), "Returns dictionary")
        self.assert_test("objective" in plan_good, "Has objective")
        self.assert_test("breakthrough_baseline" in plan_good, "Has breakthrough baseline")
        self.assert_test("scaling_factor_needed" in plan_good, "Has scaling factor")
        self.assert_test("proven_methodology" in plan_good, "Has proven methodology")
        self.assert_test("execution_feasible" in plan_good, "Has feasibility assessment")
        
        # With good constraints, should be feasible
        self.assert_test(plan_good["execution_feasible"] == True, "Good constraints are feasible")
        self.assert_test("recommended_model" in plan_good, "Has recommended model")
        
        # Test with poor constraints
        constraints_poor = {"gpu_memory_gb": 2.0, "cpu_memory_available_gb": 4.0}
        plan_poor = create_morris_reproduction_plan("cpu", constraints_poor)
        
        self.assert_test(isinstance(plan_poor, dict), "Poor constraints return dict")
        # Should be less feasible
        self.assert_test(plan_poor["execution_feasible"] == False, "Poor constraints not feasible")
        self.assert_test("recommendation" in plan_poor, "Poor constraints have recommendation")
    
    def test_preview_morris_targets(self):
        """Test preview_morris_targets function."""
        print("\n🧪 Testing preview_morris_targets...")
        
        try:
            print("--- Morris Targets Preview Output ---")
            preview_morris_targets()
            print("--- End Preview Output ---")
            self.assert_test(True, "Morris targets preview works")
            
        except Exception as e:
            self.assert_test(False, f"Morris targets preview failed: {e}")
    
    def test_morris_scale_validation(self):
        """Test that Morris-scale models are actually Morris-scale."""
        print("\n🧪 Testing Morris Scale Validation...")
        
        configs = create_morris_target_configs()
        
        # Test against known Morris requirements
        for config, name, params in configs:
            # Morris used models in the 12-15M parameter range
            self.assert_test(10_000_000 <= params <= 20_000_000, 
                           f"{name} in Morris parameter range")
            
            # Morris used specific experimental setup
            self.assert_test(config.vocab_size == 2048, f"{name} uses Morris vocab size")
            self.assert_test(config.max_seq_length == 64, f"{name} uses Morris sequence length")
            
            # Architecture should be reasonable transformer
            self.assert_test(config.n_layers >= 8, f"{name} has sufficient depth")
            self.assert_test(config.d_model >= 512, f"{name} has sufficient width")
            self.assert_test(config.n_heads >= 8, f"{name} has sufficient attention heads")
            
            # Model should be trainable
            self.assert_test(config.d_model % config.n_heads == 0, f"{name} valid attention configuration")
    
    def test_integration_with_enhanced_lr(self):
        """Test integration with enhanced LR scheduler."""
        print("\n🧪 Testing Integration with Enhanced LR...")
        
        try:
            # Test that Morris configs work with enhanced LR
            configs = create_morris_target_configs()
            
            for config, name, params in configs[:1]:  # Test first config only
                try:
                    training_config = create_morris_enhanced_training_config(name, params, "cuda")
                    
                    # Should work with enhanced LR if available
                    if hasattr(training_config, 'scheduler_type'):
                        self.assert_test(True, f"{name} works with enhanced LR")
                    else:
                        self.assert_test(True, f"{name} works with simple LR fallback")
                        
                except Exception as e:
                    self.assert_test(False, f"{name} integration failed: {e}")
            
        except Exception as e:
            self.assert_test(False, f"Enhanced LR integration test failed: {e}")
    
    def run_all_tests(self):
        """Run all tests and report results."""
        print("🧪 MORRIS TARGET CONFIGS TEST SUITE")
        print("=" * 50)
        
        try:
            self.test_morris_target_model_config()
            self.test_create_morris_target_configs()
            self.test_parameter_estimation_accuracy()
            self.test_get_morris_target_dataset_sizes()
            self.test_create_morris_enhanced_training_config()
            self.test_create_morris_simple_config()
            self.test_estimate_morris_execution_time()
            self.test_validate_morris_target_feasibility()
            self.test_create_morris_reproduction_plan()
            self.test_preview_morris_targets()
            self.test_morris_scale_validation()
            self.test_integration_with_enhanced_lr()
        except Exception as e:
            print(f"💥 Test suite crashed: {e}")
            traceback.print_exc()
        
        # Report results
        print("\n" + "=" * 50)
        print("🧪 TEST RESULTS SUMMARY")
        print("=" * 50)
        print(f"✅ Tests Passed: {self.tests_passed}")
        print(f"❌ Tests Failed: {self.tests_failed}")
        print(f"📊 Success Rate: {self.tests_passed/(self.tests_passed + self.tests_failed)*100:.1f}%")
        
        if self.tests_failed > 0:
            print(f"\n❌ Failed Tests:")
            for failure in self.failures:
                print(f"  - {failure}")
        
        if self.tests_failed == 0:
            print("\n🎉 ALL TESTS PASSED! Morris target configs are ready for reproduction.")
            return True
        else:
            print(f"\n⚠ {self.tests_failed} tests failed. Review and fix issues before proceeding.")
            return False


def main():
    """Main test execution."""
    tester = TestMorrisTargetConfigs()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ Morris Target Configs validation complete - ready for Morris reproduction!")
    else:
        print("\n❌ Morris Target Configs have issues - fix before proceeding!")
        sys.exit(1)


if __name__ == "__main__":
    main()
