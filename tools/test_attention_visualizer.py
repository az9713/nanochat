"""
Tests for attention_visualizer.py

Comprehensive test suite to verify attention visualization functionality.
Tests are designed to work with or without PyTorch installed.
"""

import os
import sys
from io import StringIO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that the module can be imported."""
    print("=" * 80)
    print("TEST 1: Module Import")
    print("=" * 80)

    try:
        # Just test if the file can be found and basic imports work
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "attention_visualizer",
            "tools/attention_visualizer.py"
        )
        if spec is None:
            print("✗ Could not find attention_visualizer.py")
            return False

        print("✓ attention_visualizer.py found")
        return True
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False


def test_pytorch_availability():
    """Test if PyTorch is available."""
    print("\n" + "=" * 80)
    print("TEST 2: PyTorch Availability")
    print("=" * 80)

    try:
        import torch
        print(f"✓ PyTorch is available (version {torch.__version__})")

        # Check device availability
        if torch.cuda.is_available():
            print(f"  CUDA is available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("  MPS (Apple Silicon) is available")
        else:
            print("  CPU only (no GPU acceleration)")

        return True
    except ImportError:
        print("✗ PyTorch is not available")
        print("  Install with: pip install torch")
        print("  Note: This tool requires PyTorch to function")
        return False


def test_with_pytorch():
    """Run comprehensive tests if PyTorch is available."""
    print("\n" + "=" * 80)
    print("TEST 3: Full Functionality Tests (requires PyTorch)")
    print("=" * 80)

    try:
        import torch
        from nanochat.gpt import GPT, GPTConfig
        from nanochat.tokenizer import get_tokenizer
        from tools.attention_visualizer import AttentionVisualizer
    except ImportError as e:
        print(f"✗ Cannot import required modules: {e}")
        return False

    try:
        # Test 3.1: Create test model and visualizer
        print("\n  3.1: Creating test model and visualizer...")
        config = GPTConfig(
            sequence_len=128,
            vocab_size=1000,
            n_layer=4,
            n_head=4,
            n_embd=128
        )
        model = GPT(config)
        model.eval()

        tokenizer = get_tokenizer()
        visualizer = AttentionVisualizer(model, tokenizer)
        print("  ✓ Visualizer created successfully")

        # Test 3.2: Compute attention weights
        print("\n  3.2: Computing attention weights...")
        text = "Hello world"
        layer_idx = 0
        tokens, attn_weights = visualizer.compute_attention_weights(text, layer_idx)

        assert len(tokens) > 0, "No tokens returned"
        assert attn_weights.dim() == 2, f"Expected 2D tensor, got {attn_weights.dim()}D"
        assert attn_weights.size(0) == len(tokens), "Attention matrix size mismatch"
        print(f"  ✓ Attention computed: {len(tokens)} tokens, shape {attn_weights.shape}")

        # Test 3.3: Verify attention properties
        print("\n  3.3: Verifying attention properties...")

        # Check row sums (should be ~1.0)
        row_sums = attn_weights.sum(dim=1)
        for i, row_sum in enumerate(row_sums):
            assert abs(row_sum.item() - 1.0) < 0.01, \
                f"Row {i} sums to {row_sum:.4f}, expected ~1.0"
        print("  ✓ Row sums are ~1.0 (valid probability distribution)")

        # Check causal masking
        causal_ok = True
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                if attn_weights[i, j].item() != 0.0:
                    causal_ok = False
                    break
        assert causal_ok, "Causal masking violated"
        print("  ✓ Causal masking enforced")

        # Test 3.4: Test all layers
        print("\n  3.4: Testing all layers...")
        for layer_idx in range(model.config.n_layer):
            tokens, attn = visualizer.compute_attention_weights(text, layer_idx)
            assert attn.size(0) == len(tokens)
        print(f"  ✓ All {model.config.n_layer} layers work")

        # Test 3.5: Test specific heads
        print("\n  3.5: Testing individual heads...")
        for head_idx in range(model.config.n_head):
            tokens, attn = visualizer.compute_attention_weights(
                text, 0, head_idx=head_idx
            )
            assert attn.size(0) == len(tokens)
        print(f"  ✓ All {model.config.n_head} heads work")

        # Test 3.6: Test error handling
        print("\n  3.6: Testing error handling...")
        try:
            visualizer.compute_attention_weights(text, layer_idx=999)
            print("  ✗ Should have raised ValueError for invalid layer")
            return False
        except ValueError:
            print("  ✓ Correctly rejects invalid layer index")

        try:
            visualizer.compute_attention_weights(text, layer_idx=0, head_idx=999)
            print("  ✗ Should have raised ValueError for invalid head")
            return False
        except ValueError:
            print("  ✓ Correctly rejects invalid head index")

        # Test 3.7: Test text visualization
        print("\n  3.7: Testing text visualization...")
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            visualizer.visualize_text(text, layer_idx=0, top_k=3)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "ATTENTION VISUALIZATION" in output
        assert "Layer: 0" in output
        print("  ✓ Text visualization generates output")

        # Test 3.8: Test heatmap (if matplotlib available)
        print("\n  3.8: Testing heatmap generation...")
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                visualizer.visualize_heatmap(text, layer_idx=0, output_file=tmp_path)
                sys.stdout = old_stdout

                assert os.path.exists(tmp_path)
                assert os.path.getsize(tmp_path) > 0
                print("  ✓ Heatmap generation works")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except ImportError:
            print("  ⚠ Matplotlib not available, skipping heatmap test")

        print("\n✓ All PyTorch-dependent tests passed")
        return True

    except Exception as e:
        print(f"\n✗ PyTorch tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tokenizer_integration():
    """Test that tokenizer works independently."""
    print("\n" + "=" * 80)
    print("TEST 4: Tokenizer Integration")
    print("=" * 80)

    try:
        from nanochat.tokenizer import get_tokenizer

        tokenizer = get_tokenizer()
        print("✓ Tokenizer loaded successfully")

        # Test encoding
        test_text = "Hello, world!"
        tokens = tokenizer.encode(test_text, prepend="<|bos|>")
        print(f"  Encoded '{test_text}' to {len(tokens)} tokens")

        return True
    except Exception as e:
        print(f"✗ Tokenizer test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 80)
    print("ATTENTION VISUALIZER TEST SUITE")
    print("=" * 80)
    print()

    results = []

    # Test 1: Imports (always run)
    results.append(("Module Import", test_imports()))

    # Test 2: PyTorch availability
    pytorch_available = test_pytorch_availability()
    results.append(("PyTorch", pytorch_available))

    # Test 3: Full tests (only if PyTorch available)
    if pytorch_available:
        results.append(("Full Functionality", test_with_pytorch()))
    else:
        print("\n⚠ Skipping comprehensive tests (PyTorch not available)")
        print("  Install PyTorch to run full test suite: pip install torch")

    # Test 4: Tokenizer (independent of PyTorch for model creation)
    results.append(("Tokenizer", test_tokenizer_integration()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:25s} {status}")

    passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")

    if not pytorch_available:
        print("\n⚠ Note: Some tests were skipped due to missing PyTorch")
        print("  The tool requires PyTorch to function")

    if passed == total:
        print("\n🎉 All available tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


def print_usage_examples():
    """Print usage examples."""
    print("\n" + "=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    print()
    print("# Text-based visualization for layer 0")
    print("python tools/attention_visualizer.py --text 'Hello world' --layer 0")
    print()
    print("# Heatmap for layer 5, head 3 (requires matplotlib)")
    print("python tools/attention_visualizer.py --text 'The capital of France is' \\")
    print("    --layer 5 --head 3 --heatmap")
    print()
    print("# Save heatmap to file")
    print("python tools/attention_visualizer.py --text 'Once upon a time' \\")
    print("    --layer 10 --heatmap --output attention.png")
    print()
    print("# Use specific model")
    print("python tools/attention_visualizer.py --source base --model-tag d20 \\")
    print("    --text 'Hello' --layer 0")
    print()
    print("=" * 80)


if __name__ == "__main__":
    exit_code = run_all_tests()
    print_usage_examples()
    sys.exit(exit_code)
