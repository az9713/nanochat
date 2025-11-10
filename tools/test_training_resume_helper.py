#!/usr/bin/env python3
"""
Test script for Training Resume Helper

Creates mock checkpoints in nanochat format and tests all functionality.
"""

import os
import sys
import json
import tempfile
import shutil


def create_mock_checkpoint(checkpoint_dir: str, step: int = 2500):
    """
    Create a mock checkpoint for testing in nanochat format.

    Nanochat saves checkpoints as:
    - model_<step>.pt (model weights)
    - meta_<step>.json (metadata)
    - optim_<step>.pt (optimizer state, optional)
    """
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

    # Create metadata as a dict (will be saved to JSON)
    metadata = {
        'step': step,
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

    # Save model file (model_<step>.pt)
    model_path = os.path.join(checkpoint_dir, f'model_{step:06d}.pt')
    torch.save(model_state, model_path)

    # Save metadata file (meta_<step>.json)
    meta_path = os.path.join(checkpoint_dir, f'meta_{step:06d}.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    return step, model_path, meta_path


def test_training_resume_helper():
    """Test all functionality of the Training Resume Helper."""
    print("="*80)
    print("TESTING TRAINING RESUME HELPER")
    print("="*80 + "\n")

    # Create temporary directory for test checkpoint
    test_dir = tempfile.mkdtemp(prefix="test_checkpoint_")
    print(f"📁 Creating test checkpoint in: {test_dir}\n")

    try:
        # Create mock checkpoint in nanochat format
        checkpoint_result = create_mock_checkpoint(test_dir)

        if checkpoint_result is None:
            print("⚠️  Cannot test without PyTorch - tool requires torch for checkpoint loading")
            return

        step, model_path, meta_path = checkpoint_result
        print(f"✓ Created mock checkpoint at step {step}:")
        print(f"  Model: {os.path.basename(model_path)}")
        print(f"  Metadata: {os.path.basename(meta_path)}\n")

        # Import the helper
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from training_resume_helper import TrainingResumeHelper

        helper = TrainingResumeHelper(test_dir)

        # Test 1: Find checkpoint
        print("TEST 1: Find latest checkpoint")
        print("-" * 80)
        found = helper.find_latest_checkpoint()
        if found:
            found_step, found_model, found_meta, has_optim = found
            print(f"✓ Found checkpoint at step {found_step}")
            print(f"  Model: {os.path.basename(found_model)}")
            print(f"  Metadata: {os.path.basename(found_meta)}")
            print(f"  Has optimizer: {has_optim}")
        else:
            print("✗ Failed to find checkpoint")
        print()

        # Test 2: Load checkpoint info
        print("TEST 2: Load checkpoint information")
        print("-" * 80)
        info = helper.load_checkpoint_info(meta_path)
        print(f"✓ Loaded info: step={info['step']}, val_bpb={info['val_bpb']}")
        print()

        # Test 3: Verify checkpoint
        print("TEST 3: Verify checkpoint integrity")
        print("-" * 80)
        is_valid = helper.verify_checkpoint(step, model_path, meta_path)
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

        # Test 5: Print full report (without optimizer - test both cases)
        print("TEST 5: Generate full resume report (no optimizer)")
        print("-" * 80)
        helper.print_resume_report(step, model_path, meta_path, has_optimizer=False, target_steps=target_steps)

        # Test 6: Generate resume instructions (both with and without optimizer)
        print("TEST 6a: Generate resume instructions (without optimizer)")
        print("-" * 80)
        instructions_no_opt = helper.generate_resume_instructions(step, meta_path, has_optimizer=False)
        print(f"✓ Generated instructions without optimizer (truncated):")
        print(instructions_no_opt[:250] + "...")
        print()

        print("TEST 6b: Generate resume instructions (with optimizer)")
        print("-" * 80)
        instructions_with_opt = helper.generate_resume_instructions(step, meta_path, has_optimizer=True)
        print(f"✓ Generated instructions with optimizer (truncated):")
        print(instructions_with_opt[:250] + "...")
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
