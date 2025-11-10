"""
Test script for Training Dashboard

Tests basic functionality of the training dashboard tool.
"""

import os
import sys
import json
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_imports():
    """Test that all required modules can be imported."""
    print("="*80)
    print("TEST: Import training_dashboard module")
    print("="*80)

    try:
        from tools.training_dashboard import TrainingDashboard
        print("✓ Successfully imported TrainingDashboard class")
        return True
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False


def create_mock_checkpoint(checkpoint_dir: str, step: int, train_loss: float,
                          val_bpb: float = None, lr: float = 0.001):
    """
    Create a mock checkpoint metadata file.

    Args:
        checkpoint_dir: Directory to create checkpoint in
        step: Training step number
        train_loss: Training loss value
        val_bpb: Validation BPB (optional)
        lr: Learning rate
    """
    metadata = {
        "step": step,
        "train_loss": train_loss,
        "learning_rate": lr,
        "config": {
            "n_layer": 6,
            "n_embd": 384,
            "sequence_len": 1024,
        }
    }

    if val_bpb is not None:
        metadata["val_bpb"] = val_bpb

    meta_file = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def test_checkpoint_discovery():
    """Test finding and loading checkpoints."""
    print("\n" + "="*80)
    print("TEST: Checkpoint discovery")
    print("="*80)

    from tools.training_dashboard import TrainingDashboard

    # Create temporary checkpoint directory
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")

    try:
        # Create mock checkpoints
        create_mock_checkpoint(temp_dir, step=100, train_loss=3.5, lr=0.001)
        create_mock_checkpoint(temp_dir, step=200, train_loss=3.0, val_bpb=3.1, lr=0.001)
        create_mock_checkpoint(temp_dir, step=300, train_loss=2.8, val_bpb=2.9, lr=0.0009)

        # Initialize dashboard
        dashboard = TrainingDashboard(temp_dir)

        # Find checkpoints
        checkpoints = dashboard.find_all_checkpoints()

        assert len(checkpoints) == 3, f"Expected 3 checkpoints, got {len(checkpoints)}"
        print(f"✓ Found 3 checkpoints")

        # Check sorting by step
        steps = [ckpt["step"] for ckpt in checkpoints]
        assert steps == [100, 200, 300], f"Steps not sorted correctly: {steps}"
        print(f"✓ Checkpoints sorted correctly by step")

        # Check metadata loading
        assert checkpoints[0]["metadata"]["train_loss"] == 3.5
        assert checkpoints[1]["metadata"]["val_bpb"] == 3.1
        assert checkpoints[2]["metadata"]["learning_rate"] == 0.0009
        print(f"✓ Metadata loaded correctly")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")


def test_metrics_extraction():
    """Test extracting metrics from checkpoints."""
    print("\n" + "="*80)
    print("TEST: Metrics extraction")
    print("="*80)

    from tools.training_dashboard import TrainingDashboard

    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")

    try:
        # Create checkpoints with varying metrics
        create_mock_checkpoint(temp_dir, step=100, train_loss=3.0, lr=0.001)
        create_mock_checkpoint(temp_dir, step=200, train_loss=2.5, val_bpb=2.6, lr=0.0009)
        create_mock_checkpoint(temp_dir, step=300, train_loss=2.2, val_bpb=2.3, lr=0.0008)

        dashboard = TrainingDashboard(temp_dir)
        checkpoints = dashboard.find_all_checkpoints()
        metrics = dashboard.extract_metrics(checkpoints)

        # Verify metrics structure
        assert "steps" in metrics
        assert "train_loss" in metrics
        assert "val_bpb" in metrics
        assert "learning_rate" in metrics
        print("✓ Metrics dictionary has correct keys")

        # Verify values
        assert metrics["steps"] == [100, 200, 300]
        print("✓ Steps extracted correctly")

        assert metrics["train_loss"] == [3.0, 2.5, 2.2]
        print("✓ Train losses extracted correctly")

        assert metrics["val_bpb"] == [None, 2.6, 2.3]
        print("✓ Validation BPB extracted correctly (including None)")

        assert metrics["learning_rate"] == [0.001, 0.0009, 0.0008]
        print("✓ Learning rates extracted correctly")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")


def test_summary_output():
    """Test summary printing."""
    print("\n" + "="*80)
    print("TEST: Summary output")
    print("="*80)

    from tools.training_dashboard import TrainingDashboard

    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")

    try:
        # Create checkpoints
        create_mock_checkpoint(temp_dir, step=0, train_loss=4.0, lr=0.001)
        create_mock_checkpoint(temp_dir, step=500, train_loss=3.0, val_bpb=3.1, lr=0.001)
        create_mock_checkpoint(temp_dir, step=1000, train_loss=2.5, val_bpb=2.6, lr=0.0008)

        dashboard = TrainingDashboard(temp_dir)

        print("\nPrinting summary:")
        dashboard.print_summary()
        print("✓ Summary printed successfully")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")


def test_progress_table():
    """Test progress table display."""
    print("\n" + "="*80)
    print("TEST: Progress table")
    print("="*80)

    from tools.training_dashboard import TrainingDashboard

    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")

    try:
        # Create checkpoints
        create_mock_checkpoint(temp_dir, step=100, train_loss=3.0, lr=0.001)
        create_mock_checkpoint(temp_dir, step=200, train_loss=2.5, val_bpb=2.6, lr=0.0009)

        dashboard = TrainingDashboard(temp_dir)

        print("\nShowing progress:")
        dashboard.show_progress()
        print("✓ Progress table displayed successfully")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")


def test_csv_export():
    """Test CSV export functionality."""
    print("\n" + "="*80)
    print("TEST: CSV export")
    print("="*80)

    from tools.training_dashboard import TrainingDashboard

    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")

    try:
        # Create checkpoints
        create_mock_checkpoint(temp_dir, step=100, train_loss=3.0, lr=0.001)
        create_mock_checkpoint(temp_dir, step=200, train_loss=2.5, val_bpb=2.6, lr=0.0009)

        dashboard = TrainingDashboard(temp_dir)

        # Export to CSV
        csv_path = os.path.join(temp_dir, "test_metrics.csv")
        dashboard.export_csv(output_file=csv_path)

        # Verify CSV was created
        assert os.path.exists(csv_path), "CSV file was not created"
        print(f"✓ CSV file created: {csv_path}")

        # Read and verify content
        with open(csv_path, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 3, f"Expected 3 lines (header + 2 rows), got {len(lines)}"
        assert lines[0].strip() == "step,train_loss,val_bpb,learning_rate"
        print("✓ CSV header correct")

        assert "100,3.0," in lines[1]
        assert "200,2.5,2.6," in lines[2]
        print("✓ CSV data rows correct")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")


def test_plot_generation():
    """Test plot generation (if matplotlib available)."""
    print("\n" + "="*80)
    print("TEST: Plot generation")
    print("="*80)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        has_matplotlib = True
    except ImportError:
        print("ℹ Matplotlib not available, skipping plot test")
        return True

    from tools.training_dashboard import TrainingDashboard

    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")

    try:
        # Create checkpoints
        create_mock_checkpoint(temp_dir, step=100, train_loss=3.0, lr=0.001)
        create_mock_checkpoint(temp_dir, step=200, train_loss=2.5, val_bpb=2.6, lr=0.0009)
        create_mock_checkpoint(temp_dir, step=300, train_loss=2.2, val_bpb=2.3, lr=0.0008)

        dashboard = TrainingDashboard(temp_dir)

        # Generate plot
        plot_path = os.path.join(temp_dir, "test_plot.png")
        dashboard.plot_metrics(output_file=plot_path)

        # Verify plot was created
        assert os.path.exists(plot_path), "Plot file was not created"
        print(f"✓ Plot file created: {plot_path}")

        # Verify it's a PNG file (check header)
        with open(plot_path, 'rb') as f:
            header = f.read(8)
        assert header == b'\x89PNG\r\n\x1a\n', "File is not a valid PNG"
        print("✓ Plot is a valid PNG file")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*80)
    print("TEST: Edge cases and error handling")
    print("="*80)

    from tools.training_dashboard import TrainingDashboard

    # Test 1: Non-existent directory
    try:
        dashboard = TrainingDashboard("/nonexistent/directory")
        print("✗ Should have raised error for non-existent directory")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for non-existent directory: {e}")

    # Test 2: Empty directory (no checkpoints)
    temp_dir = tempfile.mkdtemp()
    try:
        dashboard = TrainingDashboard(temp_dir)
        checkpoints = dashboard.find_all_checkpoints()
        assert len(checkpoints) == 0, "Should find no checkpoints in empty directory"
        print("✓ Handles empty directory correctly")

        # Should handle gracefully when no checkpoints
        dashboard.print_summary()
        print("✓ print_summary() handles empty directory")

        dashboard.show_progress()
        print("✓ show_progress() handles empty directory")

    finally:
        shutil.rmtree(temp_dir)

    # Test 3: Malformed JSON
    temp_dir = tempfile.mkdtemp()
    try:
        # Create malformed JSON file
        bad_meta = os.path.join(temp_dir, "meta_000100.json")
        with open(bad_meta, 'w') as f:
            f.write("{invalid json")

        dashboard = TrainingDashboard(temp_dir)
        checkpoints = dashboard.find_all_checkpoints()

        # Should skip malformed files with warning
        assert len(checkpoints) == 0, "Should skip malformed JSON files"
        print("✓ Handles malformed JSON files gracefully")

    finally:
        shutil.rmtree(temp_dir)

    # Test 4: Missing optional fields
    temp_dir = tempfile.mkdtemp()
    try:
        # Create checkpoint with minimal metadata
        minimal_meta = {
            "step": 100,
            "train_loss": 3.0,
            # Missing: val_bpb, learning_rate
        }
        meta_file = os.path.join(temp_dir, "meta_000100.json")
        with open(meta_file, 'w') as f:
            json.dump(minimal_meta, f)

        dashboard = TrainingDashboard(temp_dir)
        checkpoints = dashboard.find_all_checkpoints()
        metrics = dashboard.extract_metrics(checkpoints)

        assert metrics["val_bpb"][0] is None
        assert metrics["learning_rate"][0] is None
        print("✓ Handles missing optional fields correctly")

    finally:
        shutil.rmtree(temp_dir)

    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("TESTING TRAINING DASHBOARD")
    print("="*80)
    print()

    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    # Test functionality
    results.append(("Checkpoint discovery", test_checkpoint_discovery()))
    results.append(("Metrics extraction", test_metrics_extraction()))
    results.append(("Summary output", test_summary_output()))
    results.append(("Progress table", test_progress_table()))
    results.append(("CSV export", test_csv_export()))
    results.append(("Plot generation", test_plot_generation()))
    results.append(("Edge cases", test_edge_cases()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30s} {status}")

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
    print("# Show training summary")
    print("python tools/training_dashboard.py out/base_checkpoints/d20")
    print()
    print("# Show progress table")
    print("python tools/training_dashboard.py out/base_checkpoints/d20 --progress")
    print()
    print("# Generate visualization plot")
    print("python tools/training_dashboard.py out/base_checkpoints/d20 --plot")
    print()
    print("# Export metrics to CSV")
    print("python tools/training_dashboard.py out/base_checkpoints/d20 --export-csv")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
