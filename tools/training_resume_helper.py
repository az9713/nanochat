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
from pathlib import Path
from typing import Optional, Dict


class TrainingResumeHelper:
    """Helper for resuming interrupted training."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir

    def find_latest_checkpoint(self) -> Optional[str]:
        """
        Find the most recent checkpoint in the directory.

        Returns:
            Path to latest checkpoint, or None if not found
        """
        checkpoint_file = os.path.join(self.checkpoint_dir, "checkpoint.pt")

        if os.path.exists(checkpoint_file):
            return checkpoint_file

        return None

    def load_checkpoint_info(self, checkpoint_path: str) -> Dict:
        """
        Load checkpoint metadata without loading full model.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Dictionary with checkpoint information
        """
        # Import torch only when needed
        try:
            import torch
        except ImportError:
            print("❌ Error: PyTorch is required for checkpoint loading.", file=sys.stderr)
            print("Install with: pip install torch", file=sys.stderr)
            sys.exit(1)

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        metadata = checkpoint.get('metadata', {})

        return {
            'step': metadata.get('step', 0),
            'val_bpb': metadata.get('val_bpb', None),
            'model_config': metadata.get('model_config', {}),
            'user_config': metadata.get('user_config', {}),
            'device_batch_size': metadata.get('device_batch_size', None),
            'max_seq_len': metadata.get('max_seq_len', None),
        }

    def verify_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Verify checkpoint integrity.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            True if checkpoint is valid
        """
        try:
            import torch
        except ImportError:
            print("❌ Error: PyTorch is required for checkpoint verification.", file=sys.stderr)
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Check required fields
            required_fields = ['model', 'metadata']
            for field in required_fields:
                if field not in checkpoint:
                    print(f"✗ Missing required field: {field}")
                    return False

            # Check model state dict
            model_state = checkpoint['model']
            if not isinstance(model_state, dict) or len(model_state) == 0:
                print("✗ Invalid or empty model state")
                return False

            print("✓ Checkpoint is valid")
            return True

        except Exception as e:
            print(f"✗ Error loading checkpoint: {e}")
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

    def print_resume_report(self, checkpoint_path: str, target_steps: int = None):
        """
        Print a report about resuming from this checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            target_steps: Target total steps (optional)
        """
        print(f"\n{'='*80}")
        print(f"TRAINING RESUME REPORT")
        print(f"{'='*80}\n")

        # Load checkpoint info
        info = self.load_checkpoint_info(checkpoint_path)

        print(f"Checkpoint: {checkpoint_path}")
        print(f"Last saved step: {info['step']:,}")
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

    def generate_resume_command(self, checkpoint_path: str, script: str = "base_train.py") -> str:
        """
        Generate command to resume training.

        Args:
            checkpoint_path: Path to checkpoint
            script: Training script name

        Returns:
            Command string to resume training
        """
        info = self.load_checkpoint_info(checkpoint_path)

        # Build command
        cmd_parts = [
            f"python -m scripts.{script.replace('.py', '')}",
            f"--resume_from={checkpoint_path}",
        ]

        # Add key parameters from checkpoint
        if info['device_batch_size']:
            cmd_parts.append(f"--device_batch_size={info['device_batch_size']}")

        return " \\\n    ".join(cmd_parts)


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
    checkpoint_path = helper.find_latest_checkpoint()

    if not checkpoint_path:
        print(f"❌ No checkpoint found in {args.checkpoint_dir}")
        print(f"\nLooking for: {os.path.join(args.checkpoint_dir, 'checkpoint.pt')}")
        sys.exit(1)

    if args.verify:
        if helper.verify_checkpoint(checkpoint_path):
            print("\n✅ Checkpoint is ready to use!")
        else:
            print("\n❌ Checkpoint has issues - may not resume correctly")
            sys.exit(1)
    elif args.command:
        cmd = helper.generate_resume_command(checkpoint_path, args.script)
        print("\n" + "="*80)
        print("RESUME COMMAND")
        print("="*80 + "\n")
        print(cmd)
        print("\n" + "="*80 + "\n")
    else:
        helper.print_resume_report(checkpoint_path, args.target_steps)


if __name__ == "__main__":
    main()
