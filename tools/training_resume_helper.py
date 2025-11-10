#!/usr/bin/env python3
"""
Training Resume Helper

Automatically detect and resume interrupted training.
Analyzes checkpoints and provides resume information.

Usage:
    python tools/training_resume_helper.py out/checkpoint_dir
    python tools/training_resume_helper.py out/checkpoint_dir --target-steps 5400
    python tools/training_resume_helper.py out/checkpoint_dir --verify
    python tools/training_resume_helper.py out/checkpoint_dir --command
"""

import os
import argparse
import sys
import json
import glob
from pathlib import Path
from typing import Optional, Dict, Tuple


class TrainingResumeHelper:
    """Helper for resuming interrupted training."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir

    def find_latest_checkpoint(self) -> Optional[Tuple[int, str, str, bool]]:
        """
        Find the most recent checkpoint in the directory.

        Nanochat saves checkpoints as:
        - model_<step>.pt (model weights)
        - meta_<step>.json (metadata)
        - optim_<step>.pt (optimizer state, optional)

        Returns:
            Tuple of (step, model_path, meta_path, has_optimizer), or None if not found
        """
        # Look for model_*.pt files
        model_files = glob.glob(os.path.join(self.checkpoint_dir, "model_*.pt"))

        if not model_files:
            return None

        # Extract step numbers and find the latest
        checkpoints = []
        for model_path in model_files:
            basename = os.path.basename(model_path)
            try:
                # Extract step from model_<step>.pt
                step_str = basename.split('_')[1].split('.')[0]
                step = int(step_str)

                # Check if corresponding meta file exists
                meta_path = os.path.join(self.checkpoint_dir, f"meta_{step:06d}.json")
                if os.path.exists(meta_path):
                    # Check if optimizer file exists (optional)
                    optim_path = os.path.join(self.checkpoint_dir, f"optim_{step:06d}.pt")
                    has_optimizer = os.path.exists(optim_path)
                    checkpoints.append((step, model_path, meta_path, has_optimizer))
            except (IndexError, ValueError):
                continue

        if not checkpoints:
            return None

        # Return the checkpoint with the highest step number
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return checkpoints[0]

    def load_checkpoint_info(self, meta_path: str) -> Dict:
        """
        Load checkpoint metadata from JSON file.

        Args:
            meta_path: Path to meta_<step>.json file

        Returns:
            Dictionary with checkpoint information
        """
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        return {
            'step': metadata.get('step', 0),
            'val_bpb': metadata.get('val_bpb', None),
            'model_config': metadata.get('model_config', {}),
            'user_config': metadata.get('user_config', {}),
            'device_batch_size': metadata.get('device_batch_size', None),
            'max_seq_len': metadata.get('max_seq_len', None),
        }

    def verify_checkpoint(self, step: int, model_path: str, meta_path: str) -> bool:
        """
        Verify checkpoint integrity.

        Args:
            step: Checkpoint step number
            model_path: Path to model_<step>.pt
            meta_path: Path to meta_<step>.json

        Returns:
            True if checkpoint is valid
        """
        try:
            import torch
        except ImportError:
            print("❌ Error: PyTorch is required for checkpoint verification.", file=sys.stderr)
            return False

        try:
            # Check model file exists and can be loaded
            if not os.path.exists(model_path):
                print(f"✗ Model file not found: {model_path}")
                return False

            model_state = torch.load(model_path, map_location='cpu')
            if not isinstance(model_state, dict) or len(model_state) == 0:
                print("✗ Invalid or empty model state")
                return False

            # Check metadata file exists and can be loaded
            if not os.path.exists(meta_path):
                print(f"✗ Metadata file not found: {meta_path}")
                return False

            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            if not isinstance(metadata, dict):
                print("✗ Invalid metadata format")
                return False

            # Check that step matches
            if metadata.get('step') != step:
                print(f"✗ Step mismatch: file says {step}, metadata says {metadata.get('step')}")
                return False

            print(f"✓ Checkpoint at step {step} is valid")
            print(f"  Model file: {os.path.basename(model_path)}")
            print(f"  Metadata file: {os.path.basename(meta_path)}")
            return True

        except Exception as e:
            print(f"✗ Error verifying checkpoint: {e}")
            return False

    def calculate_resume_params(self, checkpoint_info: Dict, target_steps: int) -> Dict:
        """
        Calculate parameters for resuming training.

        Args:
            checkpoint_info: Info from checkpoint
            target_steps: Total target training steps

        Returns:
            Dictionary with resume parameters
        """
        current_step = checkpoint_info['step']
        remaining_steps = target_steps - current_step

        # Calculate progress percentage
        progress_pct = 100.0 * current_step / target_steps if target_steps > 0 else 0

        # Suggest learning rate (typically continue with warmdown if near end)
        warmdown_threshold = 0.8  # Last 20% of training
        in_warmdown = progress_pct >= (warmdown_threshold * 100)

        return {
            'current_step': current_step,
            'remaining_steps': remaining_steps,
            'progress_pct': progress_pct,
            'in_warmdown': in_warmdown,
            'target_steps': target_steps
        }

    def print_resume_report(self, step: int, model_path: str, meta_path: str, has_optimizer: bool, target_steps: int = None):
        """
        Print a report about resuming from this checkpoint.

        Args:
            step: Checkpoint step number
            model_path: Path to model_<step>.pt
            meta_path: Path to meta_<step>.json
            has_optimizer: Whether optimizer file exists
            target_steps: Target total steps (optional)
        """
        print(f"\n{'='*80}")
        print(f"TRAINING RESUME REPORT")
        print(f"{'='*80}\n")

        # Load checkpoint info
        info = self.load_checkpoint_info(meta_path)

        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"Model file: {os.path.basename(model_path)}")
        print(f"Metadata file: {os.path.basename(meta_path)}")
        print(f"Optimizer file: {'optim_' + f'{step:06d}.pt' if has_optimizer else 'N/A (not saved)'}")
        print(f"\nLast saved step: {info['step']:,}")
        if info['val_bpb']:
            print(f"Validation BPB: {info['val_bpb']:.4f}")
        else:
            print(f"Validation BPB: N/A")

        # Model config
        model_config = info['model_config']
        if model_config:
            print(f"\nModel Configuration:")
            print(f"  Layers: {model_config.get('n_layer', 'N/A')}")
            print(f"  Hidden dim: {model_config.get('n_embd', 'N/A')}")
            print(f"  Sequence length: {model_config.get('sequence_len', 'N/A')}")

        # Training config
        if info['user_config']:
            print(f"\nTraining Configuration:")
            for key in ['device_batch_size', 'total_batch_size', 'learning_rate']:
                if key in info['user_config']:
                    print(f"  {key}: {info['user_config'][key]}")

        # Resume parameters
        if target_steps:
            resume_params = self.calculate_resume_params(info, target_steps)

            print(f"\nResume Parameters:")
            print(f"  Current step: {resume_params['current_step']:,}")
            print(f"  Target steps: {resume_params['target_steps']:,}")
            print(f"  Remaining: {resume_params['remaining_steps']:,}")
            print(f"  Progress: {resume_params['progress_pct']:.1f}%")
            print(f"  In warmdown: {'Yes' if resume_params['in_warmdown'] else 'No'}")

            # Educational insights
            print(f"\n💡 Learning Insights:")
            if resume_params['in_warmdown']:
                print("  • You're in the warmdown phase - learning rate should be decreasing")
                print("  • Model performance gains will be smaller but important for final quality")
            else:
                print("  • Still in main training phase")
                print(f"  • Approximately {100 - resume_params['progress_pct']:.0f}% of training remaining")

        print(f"\n{'='*80}\n")

    def generate_resume_instructions(self, step: int, meta_path: str, has_optimizer: bool, script: str = "base_train.py") -> str:
        """
        Generate instructions for manually resuming training.

        Note: Nanochat training scripts don't have built-in resume functionality.
        You need to manually load the checkpoint in your training script.

        Args:
            step: Checkpoint step number
            meta_path: Path to meta_<step>.json
            has_optimizer: Whether optimizer file exists
            script: Training script name

        Returns:
            Instructions string for resuming training
        """
        info = self.load_checkpoint_info(meta_path)

        checkpoint_dir = self.checkpoint_dir

        # Generate conditional optimizer loading based on whether file exists
        if has_optimizer:
            load_optimizer_code = f"""   model_data, optim_data, meta_data = load_checkpoint(
       checkpoint_dir="{checkpoint_dir}",
       step={step},
       device=device,
       load_optimizer=True  # Optimizer checkpoint exists
   )

   # Load into your model
   model.load_state_dict(model_data)

   # Load optimizer state to continue training
   optimizer.load_state_dict(optim_data)"""
        else:
            load_optimizer_code = f"""   model_data, optim_data, meta_data = load_checkpoint(
       checkpoint_dir="{checkpoint_dir}",
       step={step},
       device=device,
       load_optimizer=False  # No optimizer checkpoint saved
   )

   # Load into your model
   model.load_state_dict(model_data)

   # No optimizer state available - will start with fresh optimizer
   # (fine for fine-tuning or transfer learning)"""

        instructions = f"""
To resume training from step {step:,}:

1. Load the checkpoint in your training script using checkpoint_manager:

   from nanochat.checkpoint_manager import load_checkpoint

{load_optimizer_code}

2. Set the starting step to continue from:

   start_step = {step} + 1

3. Adjust your training loop:

   for step in range(start_step, num_iterations + 1):
       # ... training code ...

4. Or use the checkpoint as initialization for fine-tuning with different settings

NOTE: Nanochat training scripts ({script}) currently don't have automatic
resume via CLI arguments. Manual code modifications are required.
"""
        return instructions


def main():
    parser = argparse.ArgumentParser(
        description='Training Resume Helper - Analyze and resume training from checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show checkpoint information
  python tools/training_resume_helper.py out/checkpoint_dir

  # Calculate remaining steps
  python tools/training_resume_helper.py out/checkpoint_dir --target-steps 5400

  # Verify checkpoint integrity
  python tools/training_resume_helper.py out/checkpoint_dir --verify

  # Generate resume command
  python tools/training_resume_helper.py out/checkpoint_dir --command
        """
    )

    parser.add_argument("checkpoint_dir", help="Checkpoint directory")
    parser.add_argument("--target-steps", type=int, help="Target total training steps")
    parser.add_argument("--verify", action="store_true", help="Verify checkpoint integrity")
    parser.add_argument("--command", action="store_true", help="Generate resume command")
    parser.add_argument("--script", default="base_train.py", help="Training script name (default: base_train.py)")

    args = parser.parse_args()

    helper = TrainingResumeHelper(args.checkpoint_dir)

    # Find latest checkpoint
    checkpoint_result = helper.find_latest_checkpoint()

    if not checkpoint_result:
        print(f"❌ No checkpoint found in {args.checkpoint_dir}")
        print(f"\nLooking for files matching: model_*.pt and meta_*.json")
        print(f"\nNanochat saves checkpoints as:")
        print(f"  - model_<step>.pt  (model weights)")
        print(f"  - meta_<step>.json (metadata)")
        print(f"  - optim_<step>.pt  (optimizer state, optional)")
        sys.exit(1)

    step, model_path, meta_path, has_optimizer = checkpoint_result

    if args.verify:
        if helper.verify_checkpoint(step, model_path, meta_path):
            print("\n✅ Checkpoint is ready to use!")
            if has_optimizer:
                print("   (includes optimizer state)")
            else:
                print("   (no optimizer state - fine for fine-tuning)")
        else:
            print("\n❌ Checkpoint has issues - may not resume correctly")
            sys.exit(1)
    elif args.command:
        instructions = helper.generate_resume_instructions(step, meta_path, has_optimizer, args.script)
        print("\n" + "="*80)
        print("RESUME INSTRUCTIONS")
        print("="*80)
        print(instructions)
        print("="*80 + "\n")
    else:
        helper.print_resume_report(step, model_path, meta_path, has_optimizer, args.target_steps)


if __name__ == "__main__":
    main()
