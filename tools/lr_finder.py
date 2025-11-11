"""
Learning Rate Finder

Analyze learning rate schedules and suggest optimal learning rates for training.
This tool helps determine the best learning rate before running full training experiments.

Provides two modes:
1. Analyze existing training runs to understand LR impact on loss
2. Generate recommendations for LR range tests
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LRFinder:
    """
    Learning Rate Finder tool.

    Analyzes training dynamics to help identify optimal learning rates.
    """

    def __init__(self, checkpoint_dir: str):
        """
        Initialize LR Finder.

        Args:
            checkpoint_dir: Directory containing checkpoint metadata files
        """
        if not os.path.exists(checkpoint_dir):
            raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")

        self.checkpoint_dir = checkpoint_dir
        self.checkpoints = []

    def find_checkpoints(self) -> List[Dict]:
        """
        Find and load all checkpoint metadata files.

        Returns:
            List of checkpoint dictionaries with metadata
        """
        checkpoints = []

        # Find all meta_*.json files
        for filename in sorted(os.listdir(self.checkpoint_dir)):
            if filename.startswith("meta_") and filename.endswith(".json"):
                filepath = os.path.join(self.checkpoint_dir, filename)

                try:
                    with open(filepath, 'r') as f:
                        meta = json.load(f)

                    # Extract step from filename (meta_NNNNNN.json)
                    step_str = filename.replace("meta_", "").replace(".json", "")
                    step = int(step_str)

                    checkpoints.append({
                        'step': step,
                        'filepath': filepath,
                        'meta': meta
                    })
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Warning: Could not load {filename}: {e}")
                    continue

        self.checkpoints = checkpoints
        return checkpoints

    def extract_lr_loss_data(self) -> Tuple[List[float], List[float], List[float]]:
        """
        Extract learning rate and loss data from checkpoints.

        Returns:
            Tuple of (steps, learning_rates, train_losses)
        """
        steps = []
        learning_rates = []
        train_losses = []

        for ckpt in self.checkpoints:
            step = ckpt['step']
            meta = ckpt['meta']

            # Extract learning rate
            lr = meta.get('learning_rate')
            if lr is not None:
                # Extract training loss
                train_loss = meta.get('train_loss')
                if train_loss is not None:
                    steps.append(step)
                    learning_rates.append(lr)
                    train_losses.append(train_loss)

        return steps, learning_rates, train_losses

    def analyze_lr_schedule(self) -> Dict:
        """
        Analyze the learning rate schedule from training run.

        Returns:
            Dictionary with analysis results
        """
        if len(self.checkpoints) == 0:
            return {
                'status': 'no_data',
                'message': 'No checkpoint data available'
            }

        steps, lrs, losses = self.extract_lr_loss_data()

        if len(steps) == 0:
            return {
                'status': 'no_data',
                'message': 'No learning rate or loss data in checkpoints'
            }

        # Find LR range
        min_lr = min(lrs) if lrs else None
        max_lr = max(lrs) if lrs else None

        # Find best checkpoint (lowest loss)
        best_idx = losses.index(min(losses)) if losses else None
        best_lr = lrs[best_idx] if best_idx is not None else None
        best_step = steps[best_idx] if best_idx is not None else None
        best_loss = losses[best_idx] if best_idx is not None else None

        # Detect LR schedule type
        schedule_type = self._detect_schedule_type(lrs)

        return {
            'status': 'success',
            'num_checkpoints': len(steps),
            'lr_range': {
                'min': min_lr,
                'max': max_lr
            },
            'best_checkpoint': {
                'step': best_step,
                'learning_rate': best_lr,
                'train_loss': best_loss
            },
            'schedule_type': schedule_type,
            'steps': steps,
            'learning_rates': lrs,
            'train_losses': losses
        }

    def _detect_schedule_type(self, lrs: List[float]) -> str:
        """
        Detect the type of learning rate schedule.

        Args:
            lrs: List of learning rates

        Returns:
            String describing the schedule type
        """
        if len(lrs) < 2:
            return "unknown"

        # Check if constant
        if len(set(lrs)) == 1:
            return "constant"

        # Check if monotonically decreasing
        if all(lrs[i] >= lrs[i+1] for i in range(len(lrs)-1)):
            return "decay"

        # Check if monotonically increasing
        if all(lrs[i] <= lrs[i+1] for i in range(len(lrs)-1)):
            return "warmup"

        # Check for warmup + decay pattern
        # Find the maximum LR
        max_idx = lrs.index(max(lrs))
        if max_idx > 0 and max_idx < len(lrs) - 1:
            # Check if increasing before max and decreasing after
            warmup_phase = all(lrs[i] <= lrs[i+1] for i in range(max_idx))
            decay_phase = all(lrs[i] >= lrs[i+1] for i in range(max_idx, len(lrs)-1))

            if warmup_phase and decay_phase:
                return "warmup_decay"

        return "custom"

    def print_analysis(self, analysis: Dict) -> None:
        """
        Print analysis results.

        Args:
            analysis: Analysis dictionary from analyze_lr_schedule()
        """
        print("\n" + "=" * 100)
        print("LEARNING RATE ANALYSIS")
        print("=" * 100)

        if analysis['status'] != 'success':
            print(f"\n{analysis['message']}")
            return

        print(f"\nCheckpoint directory: {self.checkpoint_dir}")
        print(f"Number of checkpoints analyzed: {analysis['num_checkpoints']}")

        print(f"\nLearning rate range:")
        print(f"  Min LR: {analysis['lr_range']['min']:.6f}")
        print(f"  Max LR: {analysis['lr_range']['max']:.6f}")

        print(f"\nLR schedule type: {analysis['schedule_type']}")

        best = analysis['best_checkpoint']
        print(f"\nBest checkpoint (lowest training loss):")
        print(f"  Step:          {best['step']}")
        print(f"  Learning rate: {best['learning_rate']:.6f}")
        print(f"  Train loss:    {best['train_loss']:.6f}")

        print("\n" + "=" * 100 + "\n")

    def plot_lr_loss(self, analysis: Dict, output_file: Optional[str] = None) -> None:
        """
        Plot learning rate vs loss curves.

        Args:
            analysis: Analysis dictionary
            output_file: Path to save plot (None = display)
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("Error: matplotlib is required for plotting.")
            print("Install with: pip install matplotlib")
            return

        if analysis['status'] != 'success':
            print(f"Cannot plot: {analysis['message']}")
            return

        steps = analysis['steps']
        lrs = analysis['learning_rates']
        losses = analysis['train_losses']

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Learning rate over time
        ax1.plot(steps, lrs, 'b-', linewidth=2)
        ax1.set_xlabel('Training Step', fontsize=12)
        ax1.set_ylabel('Learning Rate', fontsize=12)
        ax1.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Mark best checkpoint
        best_idx = losses.index(min(losses))
        ax1.axvline(x=steps[best_idx], color='red', linestyle='--', alpha=0.7,
                    label=f'Best checkpoint (step {steps[best_idx]})')
        ax1.legend()

        # Plot 2: Loss over time
        ax2.plot(steps, losses, 'g-', linewidth=2)
        ax2.set_xlabel('Training Step', fontsize=12)
        ax2.set_ylabel('Training Loss', fontsize=12)
        ax2.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Mark best checkpoint
        ax2.plot(steps[best_idx], losses[best_idx], 'ro', markersize=10,
                 label=f'Best loss: {losses[best_idx]:.6f}')
        ax2.legend()

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {output_file}")
        else:
            plt.show()

        plt.close()

    def generate_recommendations(self, analysis: Dict) -> None:
        """
        Generate learning rate recommendations based on analysis.

        Args:
            analysis: Analysis dictionary
        """
        print("\n" + "=" * 100)
        print("LEARNING RATE RECOMMENDATIONS")
        print("=" * 100)

        if analysis['status'] != 'success':
            print(f"\n{analysis['message']}")
            print("\nCannot generate recommendations without training data.")
            self._print_general_recommendations()
            return

        best_lr = analysis['best_checkpoint']['learning_rate']
        schedule_type = analysis['schedule_type']

        print(f"\nBased on your training run analysis:")
        print(f"\n1. Best observed learning rate: {best_lr:.6f}")
        print(f"   This LR achieved the lowest training loss in your run.")

        print(f"\n2. Detected schedule: {schedule_type}")

        if schedule_type == "constant":
            print("   Consider adding LR decay for better convergence.")
            print(f"   Try: Start at {best_lr:.6f}, decay to {best_lr * 0.1:.6f}")

        elif schedule_type == "decay":
            print("   Good! Decaying LR often helps convergence.")
            print("   Consider adding warmup for stability.")

        elif schedule_type == "warmup_decay":
            print("   Excellent! This is a recommended schedule pattern.")
            print("   Your current schedule looks good.")

        print(f"\n3. For new experiments, try:")
        print(f"   - Conservative: {best_lr * 0.5:.6f} (50% of best)")
        print(f"   - Recommended: {best_lr:.6f} (observed best)")
        print(f"   - Aggressive:  {best_lr * 2:.6f} (2x best)")

        self._print_general_recommendations()

    def _print_general_recommendations(self) -> None:
        """Print general LR finding recommendations."""
        print("\n" + "-" * 100)
        print("GENERAL RECOMMENDATIONS")
        print("-" * 100)

        print("\n📖 How to find optimal learning rate:")
        print("\n1. LR Range Test (Leslie Smith method):")
        print("   - Start with very small LR (e.g., 1e-8)")
        print("   - Increase exponentially each step to large LR (e.g., 10)")
        print("   - Run for 100-1000 steps")
        print("   - Plot loss vs LR")
        print("   - Choose LR where loss decreases most steeply")

        print("\n2. Rules of thumb:")
        print("   - Too low: Loss decreases very slowly")
        print("   - Too high: Loss diverges or oscillates")
        print("   - Just right: Steady, fast decrease")

        print("\n3. Common starting points:")
        print("   - Adam/AdamW: 3e-4 to 1e-3")
        print("   - SGD: 1e-2 to 1e-1")
        print("   - For fine-tuning: 1e-5 to 1e-4")

        print("\n4. Scheduler recommendations:")
        print("   - Warmup: 2-10% of total steps")
        print("   - Cosine decay: Good default choice")
        print("   - Linear decay: Simple and effective")

        print("\n" + "=" * 100 + "\n")


def main():
    """Main entry point for LR Finder tool."""
    parser = argparse.ArgumentParser(
        description="Learning Rate Finder - Analyze LR schedules and find optimal learning rates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze existing training run
  python tools/lr_finder.py out/base_checkpoints/d20

  # Analyze and generate plot
  python tools/lr_finder.py out/base_checkpoints/d20 --plot

  # Save plot to file
  python tools/lr_finder.py out/base_checkpoints/d20 --plot --output lr_analysis.png

  # Get recommendations only
  python tools/lr_finder.py out/base_checkpoints/d20 --recommend
        """
    )

    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Directory containing checkpoint metadata files"
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for plot (default: display interactively)"
    )

    parser.add_argument(
        "--recommend",
        action="store_true",
        help="Show LR recommendations"
    )

    args = parser.parse_args()

    # Validate checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {args.checkpoint_dir}")
        print("\nAvailable checkpoint directories:")

        # Try to find checkpoint directories
        base_dir = os.path.dirname(args.checkpoint_dir)
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path):
                    print(f"  - {item_path}")
        sys.exit(1)

    try:
        # Initialize LR Finder
        finder = LRFinder(args.checkpoint_dir)

        # Find checkpoints
        checkpoints = finder.find_checkpoints()
        if len(checkpoints) == 0:
            print(f"Error: No checkpoint metadata files found in {args.checkpoint_dir}")
            print("Expected files like: meta_000000.json, meta_000500.json, etc.")
            sys.exit(1)

        # Analyze LR schedule
        analysis = finder.analyze_lr_schedule()

        # Always print analysis
        finder.print_analysis(analysis)

        # Generate plot if requested
        if args.plot:
            finder.plot_lr_loss(analysis, args.output)

        # Show recommendations if requested or by default
        if args.recommend or not args.plot:
            finder.generate_recommendations(analysis)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
