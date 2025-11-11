"""
Tests for lr_finder.py

Comprehensive test suite to verify learning rate finder functionality.
"""

import os
import sys
import json
import tempfile
import shutil
from io import StringIO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.lr_finder import LRFinder


def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 80)
    print("TEST 1: Module Imports")
    print("=" * 80)

    from tools.lr_finder import LRFinder
    print("✓ All imports successful")


def test_lr_finder_initialization():
    """Test LRFinder initialization."""
    print("\n" + "=" * 80)
    print("TEST 2: LRFinder Initialization")
    print("=" * 80)

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test successful initialization
        finder = LRFinder(tmpdir)
        print(f"✓ LRFinder created for directory: {tmpdir}")

        # Test with non-existent directory
        try:
            finder = LRFinder("/nonexistent/directory")
            raise AssertionError("Should have raised ValueError for non-existent directory")
        except ValueError as e:
            print(f"✓ Correctly raised ValueError for non-existent directory")


def create_test_checkpoints(tmpdir: str, num_checkpoints: int = 5) -> None:
    """
    Create test checkpoint metadata files.

    Args:
        tmpdir: Temporary directory
        num_checkpoints: Number of checkpoints to create
    """
    # Create checkpoints with varying LR and loss
    for i in range(num_checkpoints):
        step = i * 500
        lr = 0.001 * (0.9 ** i)  # Decaying LR
        loss = 3.0 - (i * 0.3)   # Decreasing loss

        meta = {
            'step': step,
            'learning_rate': lr,
            'train_loss': loss,
            'val_bpb': loss + 0.1,
            'config': {
                'n_layer': 6,
                'n_embd': 384
            }
        }

        filename = f"meta_{step:06d}.json"
        filepath = os.path.join(tmpdir, filename)

        with open(filepath, 'w') as f:
            json.dump(meta, f)


def test_checkpoint_discovery():
    """Test checkpoint discovery functionality."""
    print("\n" + "=" * 80)
    print("TEST 3: Checkpoint Discovery")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test checkpoints
        create_test_checkpoints(tmpdir, num_checkpoints=5)

        # Initialize finder
        finder = LRFinder(tmpdir)

        # Find checkpoints
        checkpoints = finder.find_checkpoints()

        assert len(checkpoints) == 5, f"Expected 5 checkpoints, found {len(checkpoints)}"
        print(f"✓ Found {len(checkpoints)} checkpoints")

        # Verify checkpoint structure
        for ckpt in checkpoints:
            assert 'step' in ckpt, "Missing 'step' in checkpoint"
            assert 'filepath' in ckpt, "Missing 'filepath' in checkpoint"
            assert 'meta' in ckpt, "Missing 'meta' in checkpoint"

        print("✓ All checkpoints have correct structure")

        # Verify checkpoints are sorted by step
        steps = [ckpt['step'] for ckpt in checkpoints]
        assert steps == sorted(steps), "Checkpoints not sorted by step"
        print("✓ Checkpoints sorted correctly")


def test_lr_loss_extraction():
    """Test extraction of LR and loss data."""
    print("\n" + "=" * 80)
    print("TEST 4: LR and Loss Extraction")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test checkpoints
        create_test_checkpoints(tmpdir, num_checkpoints=5)

        # Initialize finder
        finder = LRFinder(tmpdir)
        finder.find_checkpoints()

        # Extract data
        steps, lrs, losses = finder.extract_lr_loss_data()

        assert len(steps) == 5, f"Expected 5 steps, got {len(steps)}"
        assert len(lrs) == 5, f"Expected 5 LRs, got {len(lrs)}"
        assert len(losses) == 5, f"Expected 5 losses, got {len(losses)}"

        print(f"✓ Extracted {len(steps)} data points")

        # Verify LR is decaying
        assert lrs[0] > lrs[-1], "LR should be decaying"
        print("✓ LR decay detected")

        # Verify loss is decreasing
        assert losses[0] > losses[-1], "Loss should be decreasing"
        print("✓ Loss decrease detected")


def test_schedule_type_detection():
    """Test detection of different LR schedule types."""
    print("\n" + "=" * 80)
    print("TEST 5: Schedule Type Detection")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: Constant LR
        for i in range(5):
            meta = {
                'step': i * 100,
                'learning_rate': 0.001,
                'train_loss': 3.0 - (i * 0.1)
            }
            with open(os.path.join(tmpdir, f"meta_{i*100:06d}.json"), 'w') as f:
                json.dump(meta, f)

        finder = LRFinder(tmpdir)
        finder.find_checkpoints()
        steps, lrs, _ = finder.extract_lr_loss_data()
        schedule_type = finder._detect_schedule_type(lrs)

        assert schedule_type == "constant", f"Expected 'constant', got '{schedule_type}'"
        print("✓ Constant schedule detected correctly")

        # Clear directory
        for f in os.listdir(tmpdir):
            os.remove(os.path.join(tmpdir, f))

        # Test 2: Decay schedule
        for i in range(5):
            meta = {
                'step': i * 100,
                'learning_rate': 0.001 * (0.9 ** i),
                'train_loss': 3.0 - (i * 0.1)
            }
            with open(os.path.join(tmpdir, f"meta_{i*100:06d}.json"), 'w') as f:
                json.dump(meta, f)

        finder = LRFinder(tmpdir)
        finder.find_checkpoints()
        steps, lrs, _ = finder.extract_lr_loss_data()
        schedule_type = finder._detect_schedule_type(lrs)

        assert schedule_type == "decay", f"Expected 'decay', got '{schedule_type}'"
        print("✓ Decay schedule detected correctly")

        # Clear directory
        for f in os.listdir(tmpdir):
            os.remove(os.path.join(tmpdir, f))

        # Test 3: Warmup schedule
        for i in range(5):
            meta = {
                'step': i * 100,
                'learning_rate': 0.0001 * (2 ** i),
                'train_loss': 3.0 - (i * 0.1)
            }
            with open(os.path.join(tmpdir, f"meta_{i*100:06d}.json"), 'w') as f:
                json.dump(meta, f)

        finder = LRFinder(tmpdir)
        finder.find_checkpoints()
        steps, lrs, _ = finder.extract_lr_loss_data()
        schedule_type = finder._detect_schedule_type(lrs)

        assert schedule_type == "warmup", f"Expected 'warmup', got '{schedule_type}'"
        print("✓ Warmup schedule detected correctly")


def test_analysis():
    """Test LR schedule analysis."""
    print("\n" + "=" * 80)
    print("TEST 6: LR Schedule Analysis")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test checkpoints
        create_test_checkpoints(tmpdir, num_checkpoints=5)

        # Initialize finder
        finder = LRFinder(tmpdir)
        finder.find_checkpoints()

        # Analyze
        analysis = finder.analyze_lr_schedule()

        assert analysis['status'] == 'success', "Analysis should succeed"
        print("✓ Analysis completed successfully")

        assert analysis['num_checkpoints'] == 5, "Should analyze 5 checkpoints"
        print(f"✓ Analyzed {analysis['num_checkpoints']} checkpoints")

        assert 'lr_range' in analysis, "Missing lr_range"
        assert 'min' in analysis['lr_range'], "Missing min LR"
        assert 'max' in analysis['lr_range'], "Missing max LR"
        print("✓ LR range computed")

        assert 'best_checkpoint' in analysis, "Missing best_checkpoint"
        assert 'step' in analysis['best_checkpoint'], "Missing best step"
        assert 'learning_rate' in analysis['best_checkpoint'], "Missing best LR"
        assert 'train_loss' in analysis['best_checkpoint'], "Missing best loss"
        print("✓ Best checkpoint identified")

        assert 'schedule_type' in analysis, "Missing schedule_type"
        print(f"✓ Schedule type: {analysis['schedule_type']}")


def test_empty_directory():
    """Test handling of empty directory."""
    print("\n" + "=" * 80)
    print("TEST 7: Empty Directory Handling")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize finder with empty directory
        finder = LRFinder(tmpdir)
        checkpoints = finder.find_checkpoints()

        assert len(checkpoints) == 0, "Should find no checkpoints"
        print("✓ Empty directory handled correctly")

        # Analyze should return no_data status
        analysis = finder.analyze_lr_schedule()
        assert analysis['status'] == 'no_data', "Should return no_data status"
        print("✓ Analysis correctly reports no data")


def test_malformed_json():
    """Test handling of malformed JSON files."""
    print("\n" + "=" * 80)
    print("TEST 8: Malformed JSON Handling")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create good checkpoint
        meta = {
            'step': 0,
            'learning_rate': 0.001,
            'train_loss': 3.0
        }
        with open(os.path.join(tmpdir, "meta_000000.json"), 'w') as f:
            json.dump(meta, f)

        # Create malformed JSON
        with open(os.path.join(tmpdir, "meta_000100.json"), 'w') as f:
            f.write("{ invalid json }")

        # Initialize finder
        finder = LRFinder(tmpdir)

        # Should skip malformed file with warning
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()

        checkpoints = finder.find_checkpoints()

        sys.stdout = old_stdout

        # Should find only the good checkpoint
        assert len(checkpoints) == 1, f"Should find 1 checkpoint, found {len(checkpoints)}"
        print("✓ Malformed JSON skipped gracefully")

        # Check that warning was printed
        output = captured.getvalue()
        assert "Warning" in output or "Could not load" in output, "Should print warning"
        print("✓ Warning message generated")


def test_print_functions():
    """Test output formatting functions."""
    print("\n" + "=" * 80)
    print("TEST 9: Output Formatting")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test checkpoints
        create_test_checkpoints(tmpdir, num_checkpoints=5)

        # Initialize finder
        finder = LRFinder(tmpdir)
        finder.find_checkpoints()
        analysis = finder.analyze_lr_schedule()

        # Test print_analysis
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()

        finder.print_analysis(analysis)

        sys.stdout = old_stdout
        output = captured.getvalue()

        assert "LEARNING RATE ANALYSIS" in output, "Missing analysis header"
        assert "Learning rate range" in output, "Missing LR range"
        assert "Best checkpoint" in output, "Missing best checkpoint info"
        print("✓ Analysis output formatted correctly")

        # Test generate_recommendations
        sys.stdout = captured = StringIO()

        finder.generate_recommendations(analysis)

        sys.stdout = old_stdout
        output = captured.getvalue()

        assert "RECOMMENDATIONS" in output, "Missing recommendations header"
        assert "Best observed learning rate" in output, "Missing best LR"
        print("✓ Recommendations generated")


def test_plot_generation():
    """Test plot generation (if matplotlib available)."""
    print("\n" + "=" * 80)
    print("TEST 10: Plot Generation")
    print("=" * 80)

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ Matplotlib not available, skipping plot test")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test checkpoints
        create_test_checkpoints(tmpdir, num_checkpoints=10)

        # Initialize finder
        finder = LRFinder(tmpdir)
        finder.find_checkpoints()
        analysis = finder.analyze_lr_schedule()

        # Generate plot to temporary file
        plot_file = os.path.join(tmpdir, "test_plot.png")

        # Suppress output
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        finder.plot_lr_loss(analysis, output_file=plot_file)

        sys.stdout = old_stdout

        # Check that file was created
        assert os.path.exists(plot_file), "Plot file not created"
        assert os.path.getsize(plot_file) > 0, "Plot file is empty"

        print("✓ Plot generation successful")
        print(f"  - File created: {plot_file}")
        print(f"  - File size: {os.path.getsize(plot_file)} bytes")


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 80)
    print("LR FINDER TEST SUITE")
    print("=" * 80)
    print()

    tests = [
        test_imports,
        test_lr_finder_initialization,
        test_checkpoint_discovery,
        test_lr_loss_extraction,
        test_schedule_type_detection,
        test_analysis,
        test_empty_directory,
        test_malformed_json,
        test_print_functions,
        test_plot_generation,
    ]

    failed = []
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n✗ Test {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test_func.__name__)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    total = len(tests)
    passed = total - len(failed)
    print(f"\nPassed: {passed}/{total}")

    if not failed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {len(failed)} test(s) failed:")
        for test_name in failed:
            print(f"  - {test_name}")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
