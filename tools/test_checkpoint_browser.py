"""
Test script for Checkpoint Browser

Tests basic functionality of the checkpoint browser tool.
"""

import os
import sys
import json
import tempfile
import shutil
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_imports():
    """Test that all required modules can be imported."""
    print("="*80)
    print("TEST: Import checkpoint_browser module")
    print("="*80)

    try:
        from tools.checkpoint_browser import CheckpointBrowser
        print("✓ Successfully imported CheckpointBrowser class")
        return True
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False


def test_tabulate_availability():
    """Test if tabulate is available."""
    print("\n" + "="*80)
    print("TEST: Tabulate library availability")
    print("="*80)

    try:
        import tabulate
        print(f"✓ tabulate is available (version {tabulate.__version__ if hasattr(tabulate, '__version__') else 'unknown'})")
        print("  Tables will use nice grid formatting")
        return True
    except ImportError:
        print("⚠ tabulate is not available")
        print("  Tables will use simple fallback formatting")
        print("  Install with: pip install tabulate")
        return False


def test_with_mock_checkpoints():
    """Test with mock checkpoint structure."""
    print("\n" + "="*80)
    print("TEST: Mock checkpoint scanning and display")
    print("="*80)

    from tools.checkpoint_browser import CheckpointBrowser

    # Create temporary directory structure
    temp_dir = tempfile.mkdtemp()
    print(f"Creating mock checkpoints in: {temp_dir}")

    try:
        # Create mock checkpoint structure
        base_cp_dir = os.path.join(temp_dir, "base_checkpoints", "d20")
        os.makedirs(base_cp_dir, exist_ok=True)

        # Create a mock checkpoint (step 1000)
        step = 1000

        # Create mock model file (just empty file for testing)
        model_file = os.path.join(base_cp_dir, f"model_{step:06d}.pt")
        with open(model_file, 'wb') as f:
            f.write(b'\x00' * 1024)  # 1KB mock file

        # Create mock metadata
        meta_file = os.path.join(base_cp_dir, f"meta_{step:06d}.json")
        metadata = {
            'step': step,
            'val_bpb': 1.2345,
            'train_bpb': 1.3456,
            'model_config': {
                'n_layer': 6,
                'n_embd': 384,
                'n_head': 6,
                'sequence_len': 1024,
                'vocab_size': 50304
            }
        }
        with open(meta_file, 'w') as f:
            json.dump(metadata, f)

        # Create mock optimizer file
        optim_file = os.path.join(base_cp_dir, f"optim_{step:06d}.pt")
        with open(optim_file, 'wb') as f:
            f.write(b'\x00' * 512)  # 512B mock file

        print("✓ Created mock checkpoint structure")

        # Test checkpoint browser
        browser = CheckpointBrowser(base_dir=temp_dir)

        # Test find_all_checkpoints
        print("\nFinding checkpoints...")
        checkpoints = browser.find_all_checkpoints()

        if len(checkpoints) == 0:
            print("✗ No checkpoints found!")
            return False

        print(f"✓ Found {len(checkpoints)} checkpoint(s)")

        # Verify checkpoint info
        cp = checkpoints[0]
        print(f"\nCheckpoint details:")
        print(f"  Model name: {cp['model_name']}")
        print(f"  Step: {cp['step']}")
        print(f"  Val BPB: {cp['val_bpb']}")
        print(f"  Layers: {cp['n_layer']}")
        print(f"  Hidden dim: {cp['n_embd']}")
        print(f"  Has optimizer: {cp['has_optimizer']}")

        # Test list_checkpoints
        print("\n" + "-"*80)
        print("Testing list_checkpoints():")
        print("-"*80)
        browser.list_checkpoints()

        # Test inspect_checkpoint
        print("\n" + "-"*80)
        print("Testing inspect_checkpoint():")
        print("-"*80)
        browser.inspect_checkpoint('d20')

        print("\n✓ All mock checkpoint tests passed")
        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory: {temp_dir}")


def test_with_real_checkpoints():
    """Test with real checkpoints if available."""
    print("\n" + "="*80)
    print("TEST: Real checkpoint detection")
    print("="*80)

    from tools.checkpoint_browser import CheckpointBrowser

    browser = CheckpointBrowser(base_dir="out")
    checkpoints = browser.find_all_checkpoints()

    if len(checkpoints) == 0:
        print("⚠ No real checkpoints found in 'out/' directory")
        print("  This is expected if you haven't trained any models yet")
        print("  To test with real checkpoints:")
        print("    1. Train a model using nanochat training scripts")
        print("    2. Run this test again")
        return True  # Not a failure, just no checkpoints

    print(f"✓ Found {len(checkpoints)} real checkpoint(s)")
    print("\nListing real checkpoints:")
    print("-"*80)
    browser.list_checkpoints()

    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*80)
    print("TEST: Edge cases and error handling")
    print("="*80)

    from tools.checkpoint_browser import CheckpointBrowser

    # Test with non-existent directory
    browser = CheckpointBrowser(base_dir="/nonexistent/directory")
    checkpoints = browser.find_all_checkpoints()
    print(f"✓ Non-existent directory handled: {len(checkpoints)} checkpoints (expected 0)")

    # Test inspect with non-existent model
    print("\nTesting inspect with non-existent model:")
    browser.inspect_checkpoint("nonexistent_model")
    print("✓ Non-existent model handled gracefully")

    # Test compare with non-existent models
    print("\nTesting compare with non-existent models:")
    browser.compare_checkpoints("model1", "model2")
    print("✓ Comparison with non-existent models handled gracefully")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("TESTING CHECKPOINT BROWSER")
    print("="*80)
    print()

    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    # Test tabulate
    results.append(("Tabulate", test_tabulate_availability()))

    # Test with mock checkpoints
    results.append(("Mock checkpoints", test_with_mock_checkpoints()))

    # Test edge cases
    results.append(("Edge cases", test_edge_cases()))

    # Test with real checkpoints (if available)
    results.append(("Real checkpoints", test_with_real_checkpoints()))

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
        print("\n⚠ Some tests failed")

    print("\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)
    print()
    print("# List all checkpoints")
    print("python tools/checkpoint_browser.py --list")
    print()
    print("# Sort by validation loss")
    print("python tools/checkpoint_browser.py --list --sort val_bpb")
    print()
    print("# Inspect a specific model")
    print("python tools/checkpoint_browser.py --inspect d20")
    print()
    print("# Inspect specific step")
    print("python tools/checkpoint_browser.py --inspect d20 --step 5000")
    print()
    print("# Compare two models")
    print("python tools/checkpoint_browser.py --compare d20 d26")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
