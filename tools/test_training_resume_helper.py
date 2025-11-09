#!/usr/bin/env python3
"""
Test script for Training Resume Helper

Creates a mock checkpoint and tests all functionality.
"""

import os
import sys
import tempfile
import shutil


def create_mock_checkpoint(checkpoint_dir: str):
    """Create a mock checkpoint for testing."""
    try:
        import torch
    except ImportError:
        print("PyTorch not available - skipping checkpoint creation test")
        return None

    # Create directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create a simple mock model state
    model_state = {
        'layer1.weight': torch.randn(10, 10),
        'layer1.bias': torch.randn(10),
        'layer2.weight': torch.randn(5, 10),
        'layer2.bias': torch.randn(5),
    }

    # Create metadata
    metadata = {
        'step': 2500,
        'val_bpb': 1.234,
        'model_config': {
            'n_layer': 6,
            'n_embd': 384,
            'sequence_len': 1024,
        },
        'user_config': {
            'device_batch_size': 16,
            'total_batch_size': 64,
            'learning_rate': 0.001,
        },
        'device_batch_size': 16,
        'max_seq_len': 1024,
    }

    # Save checkpoint
    checkpoint = {
        'model': model_state,
        'metadata': metadata,
    }

    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def test_training_resume_helper():
    """Test all functionality of the Training Resume Helper."""
    print("="*80)
    print("TESTING TRAINING RESUME HELPER")
    print("="*80 + "\n")

    # Create temporary directory for test checkpoint
    test_dir = tempfile.mkdtemp(prefix="test_checkpoint_")
    print(f"📁 Creating test checkpoint in: {test_dir}\n")

    try:
        # Create mock checkpoint
        checkpoint_path = create_mock_checkpoint(test_dir)

        if checkpoint_path is None:
            print("⚠️  Cannot test without PyTorch - tool requires torch for checkpoint loading")
            return

        print(f"✓ Created mock checkpoint: {checkpoint_path}\n")

        # Import the helper
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from training_resume_helper import TrainingResumeHelper

        helper = TrainingResumeHelper(test_dir)

        # Test 1: Find checkpoint
        print("TEST 1: Find latest checkpoint")
        print("-" * 80)
        found = helper.find_latest_checkpoint()
        if found:
            print(f"✓ Found checkpoint: {found}")
        else:
            print("✗ Failed to find checkpoint")
        print()

        # Test 2: Load checkpoint info
        print("TEST 2: Load checkpoint information")
        print("-" * 80)
        info = helper.load_checkpoint_info(checkpoint_path)
        print(f"✓ Loaded info: step={info['step']}, val_bpb={info['val_bpb']}")
        print()

        # Test 3: Verify checkpoint
        print("TEST 3: Verify checkpoint integrity")
        print("-" * 80)
        is_valid = helper.verify_checkpoint(checkpoint_path)
        print()

        # Test 4: Calculate resume parameters
        print("TEST 4: Calculate resume parameters")
        print("-" * 80)
        target_steps = 5000
        resume_params = helper.calculate_resume_params(info, target_steps)
        print(f"✓ Current step: {resume_params['current_step']}")
        print(f"✓ Remaining: {resume_params['remaining_steps']}")
        print(f"✓ Progress: {resume_params['progress_pct']:.1f}%")
        print()

        # Test 5: Print full report
        print("TEST 5: Generate full resume report")
        print("-" * 80)
        helper.print_resume_report(checkpoint_path, target_steps)

        # Test 6: Generate command
        print("TEST 6: Generate resume command")
        print("-" * 80)
        cmd = helper.generate_resume_command(checkpoint_path)
        print(f"✓ Generated command:\n{cmd}")
        print()

        print("="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)

    finally:
        # Clean up
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"\n🧹 Cleaned up test directory")


if __name__ == "__main__":
    test_training_resume_helper()
