"""
Test script for Generation Parameter Explorer

Tests basic functionality of the generation parameter explorer tool.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_imports():
    """Test that all required modules can be imported."""
    print("="*80)
    print("TEST: Import generation_explorer module")
    print("="*80)

    try:
        from tools.generation_explorer import GenerationExplorer
        print("✓ Successfully imported GenerationExplorer class")
        return True
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False


def test_pytorch_availability():
    """Test if PyTorch is available."""
    print("\n" + "="*80)
    print("TEST: PyTorch availability")
    print("="*80)

    try:
        import torch
        print(f"✓ PyTorch is available (version {torch.__version__})")

        # Check device availability
        if torch.cuda.is_available():
            print(f"  CUDA is available: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            print("  MPS (Apple Silicon) is available")
        else:
            print("  CPU only (no GPU acceleration)")

        return True
    except ImportError:
        print("✗ PyTorch is not available")
        print("  Install with: pip install torch")
        return False


def test_with_model():
    """Test with an actual model if available."""
    print("\n" + "="*80)
    print("TEST: Model loading and basic generation")
    print("="*80)

    try:
        import torch
        from nanochat.checkpoint_manager import load_model
        from nanochat.tokenizer import get_tokenizer
        from tools.generation_explorer import GenerationExplorer

        # Try to load a model
        print("Attempting to load base model...")
        device = torch.device("cpu")  # Use CPU for testing

        try:
            model, tokenizer, meta_data = load_model("base", device, phase="eval")
            model.eval()
            print("✓ Model loaded successfully")

            # Create explorer
            explorer = GenerationExplorer(model, tokenizer)
            print("✓ GenerationExplorer created successfully")

            # Test basic generation
            print("\nTesting basic generation with prompt: 'The capital of France is'")
            print("-" * 80)
            explorer.sample_with_probabilities(
                "The capital of France is",
                max_tokens=10,
                temperature=0.1,  # Low temperature for deterministic output
                show_probs=False
            )
            print("✓ Basic generation test passed")

            return True

        except FileNotFoundError:
            print("✗ No trained model found")
            print("\n  To test with a model:")
            print("  1. Train a model using nanochat training scripts")
            print("  2. Or download a pretrained checkpoint")
            print("  3. Run the test again")
            return False

    except Exception as e:
        print(f"✗ Error during model testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tokenizer_integration():
    """Test tokenizer integration."""
    print("\n" + "="*80)
    print("TEST: Tokenizer integration")
    print("="*80)

    try:
        from nanochat.tokenizer import get_tokenizer

        tokenizer = get_tokenizer()
        print("✓ Tokenizer loaded successfully")

        # Test encoding
        test_text = "Hello, world!"
        tokens = tokenizer.encode(test_text, prepend="<|bos|>")
        print(f"  Encoded '{test_text}' to {len(tokens)} tokens")

        # Test special token
        try:
            end_token = tokenizer.encode_special("<|assistant_end|>")
            print(f"  Assistant end token ID: {end_token}")
            print("✓ Special token encoding works")
        except Exception as e:
            print(f"⚠ Special token encoding failed: {e}")

        return True

    except Exception as e:
        print(f"✗ Tokenizer test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("TESTING GENERATION PARAMETER EXPLORER")
    print("="*80)
    print()

    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    # Test PyTorch
    pytorch_available = test_pytorch_availability()
    results.append(("PyTorch", pytorch_available))

    # Test tokenizer
    results.append(("Tokenizer", test_tokenizer_integration()))

    # Test with model (only if PyTorch is available)
    if pytorch_available:
        results.append(("Model & Generation", test_with_model()))
    else:
        print("\n⚠ Skipping model tests (PyTorch not available)")

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:25s} {status}")

    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)

    print(f"\nPassed: {passed_tests}/{total_tests}")

    if passed_tests == total_tests:
        print("\n✓ All tests passed!")
    else:
        print("\n⚠ Some tests failed or skipped")

    print("\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)
    print()
    print("# Interactive mode (requires trained model)")
    print("python tools/generation_explorer.py --source base --interactive")
    print()
    print("# Generate with probability display")
    print("python tools/generation_explorer.py --source base \\")
    print("    --prompt 'The capital of France is' --show-probs")
    print()
    print("# Compare different temperatures")
    print("python tools/generation_explorer.py --source base \\")
    print("    --prompt 'Once upon a time' --compare-temp")
    print()
    print("# Compare different top-k values")
    print("python tools/generation_explorer.py --source base \\")
    print("    --prompt 'In the beginning' --compare-topk")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
