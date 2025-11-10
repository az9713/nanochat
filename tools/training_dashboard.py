"""
Training Progress Dashboard

Monitor and visualize training progress by reading checkpoint metadata and logs.
This standalone tool doesn't require modifying training scripts.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TrainingDashboard:
    """Tool for monitoring and visualizing training progress."""

    def __init__(self, checkpoint_dir: str):
        """
        Initialize dashboard for a checkpoint directory.

        Args:
            checkpoint_dir: Path to checkpoint directory (e.g., out/base_checkpoints/d20)
        """
        if not os.path.exists(checkpoint_dir):
            raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")

        self.checkpoint_dir = checkpoint_dir

    def find_all_checkpoints(self) -> List[Dict]:
        """
        Find all checkpoints and load their metadata.

        Returns:
            List of checkpoint info dictionaries sorted by step
        """
        checkpoints = []

        # Find all meta_*.json files
        meta_files = glob.glob(os.path.join(self.checkpoint_dir, "meta_*.json"))

        for meta_file in meta_files:
            basename = os.path.basename(meta_file)
            try:
                # Extract step number from meta_NNNNNN.json
                step_str = basename.replace("meta_", "").replace(".json", "")
                step = int(step_str)
            except (ValueError, IndexError):
                continue

            # Load metadata
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)

                checkpoint_info = {
                    "step": step,
                    "meta_file": meta_file,
                    "metadata": metadata,
                }

                checkpoints.append(checkpoint_info)

            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {meta_file}: {e}")
                continue

        # Sort by step
        checkpoints.sort(key=lambda x: x["step"])

        return checkpoints

    def extract_metrics(self, checkpoints: List[Dict]) -> Dict[str, List]:
        """
        Extract training metrics from checkpoints.

        Args:
            checkpoints: List of checkpoint dictionaries

        Returns:
            Dictionary with lists of steps, losses, etc.
        """
        metrics = {
            "steps": [],
            "train_loss": [],
            "val_bpb": [],
            "learning_rate": [],
        }

        for ckpt in checkpoints:
            meta = ckpt["metadata"]
            step = ckpt["step"]

            metrics["steps"].append(step)

            # Extract metrics (with defaults for missing values)
            metrics["train_loss"].append(meta.get("train_loss", None))
            metrics["val_bpb"].append(meta.get("val_bpb", None))
            metrics["learning_rate"].append(meta.get("learning_rate", None))

        return metrics

    def print_summary(self):
        """Print a summary of the training run."""
        checkpoints = self.find_all_checkpoints()

        if not checkpoints:
            print(f"\nNo checkpoints found in: {self.checkpoint_dir}\n")
            return

        print("\n" + "=" * 100)
        print(f"TRAINING DASHBOARD")
        print("=" * 100 + "\n")

        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"Total checkpoints: {len(checkpoints)}\n")

        # First and last checkpoint
        first_ckpt = checkpoints[0]
        last_ckpt = checkpoints[-1]

        print(f"Training range:")
        print(f"  First checkpoint: Step {first_ckpt['step']}")
        print(f"  Last checkpoint:  Step {last_ckpt['step']}")
        print()

        # Latest metrics
        latest_meta = last_ckpt["metadata"]
        print(f"Latest metrics (step {last_ckpt['step']}):")

        if "train_loss" in latest_meta:
            print(f"  Train loss: {latest_meta['train_loss']:.4f}")
        if "val_bpb" in latest_meta:
            print(f"  Val BPB:    {latest_meta['val_bpb']:.4f}")
        if "learning_rate" in latest_meta:
            print(f"  LR:         {latest_meta['learning_rate']:.6f}")
        print()

        # Best validation checkpoint
        val_checkpoints = [c for c in checkpoints if "val_bpb" in c["metadata"]]
        if val_checkpoints:
            best_ckpt = min(val_checkpoints, key=lambda c: c["metadata"]["val_bpb"])
            print(f"Best validation checkpoint:")
            print(f"  Step:    {best_ckpt['step']}")
            print(f"  Val BPB: {best_ckpt['metadata']['val_bpb']:.4f}")
            print()

        # Model configuration
        if "config" in latest_meta:
            config = latest_meta["config"]
            print(f"Model configuration:")
            if "n_layer" in config:
                print(f"  Layers:       {config['n_layer']}")
            if "n_embd" in config:
                print(f"  Embedding:    {config['n_embd']}")
            if "sequence_len" in config:
                print(f"  Seq length:   {config['sequence_len']}")
            print()

        print("=" * 100 + "\n")

    def show_progress(self):
        """Show training progress over time."""
        checkpoints = self.find_all_checkpoints()

        if not checkpoints:
            print(f"\nNo checkpoints found in: {self.checkpoint_dir}\n")
            return

        metrics = self.extract_metrics(checkpoints)

        print("\n" + "=" * 100)
        print("TRAINING PROGRESS")
        print("=" * 100 + "\n")

        print(f"{'Step':<10} {'Train Loss':<15} {'Val BPB':<15} {'Learning Rate':<15}")
        print("-" * 100)

        for i, step in enumerate(metrics["steps"]):
            train_loss = metrics["train_loss"][i]
            val_bpb = metrics["val_bpb"][i]
            lr = metrics["learning_rate"][i]

            train_loss_str = f"{train_loss:.4f}" if train_loss is not None else "N/A"
            val_bpb_str = f"{val_bpb:.4f}" if val_bpb is not None else "N/A"
            lr_str = f"{lr:.6f}" if lr is not None else "N/A"

            print(f"{step:<10} {train_loss_str:<15} {val_bpb_str:<15} {lr_str:<15}")

        print("\n" + "=" * 100 + "\n")

    def plot_metrics(self, output_file: Optional[str] = None):
        """
        Generate a visualization of training metrics.

        Args:
            output_file: Output path for plot (default: checkpoint_dir/training_dashboard.png)
        """
        checkpoints = self.find_all_checkpoints()

        if not checkpoints:
            print(f"\nNo checkpoints found in: {self.checkpoint_dir}\n")
            return

        # Try to import matplotlib
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("Error: matplotlib is required for plotting.")
            print("Install with: pip install matplotlib")
            return

        metrics = self.extract_metrics(checkpoints)

        # Filter out None values
        steps = metrics["steps"]
        train_loss = [(s, l) for s, l in zip(steps, metrics["train_loss"]) if l is not None]
        val_bpb = [(s, v) for s, v in zip(steps, metrics["val_bpb"]) if v is not None]
        lr = [(s, r) for s, r in zip(steps, metrics["learning_rate"]) if r is not None]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'Training Dashboard\n{os.path.basename(self.checkpoint_dir)}',
                    fontsize=14, fontweight='bold')

        # Plot 1: Loss curves
        ax = axes[0]
        if train_loss:
            train_steps, train_vals = zip(*train_loss)
            ax.plot(train_steps, train_vals, label='Train Loss', color='blue', linewidth=2)
        if val_bpb:
            val_steps, val_vals = zip(*val_bpb)
            ax.plot(val_steps, val_vals, label='Val BPB', color='red',
                   marker='o', linewidth=2, markersize=4)
        ax.set_xlabel('Training Step', fontsize=11)
        ax.set_ylabel('Loss / BPB', fontsize=11)
        ax.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Plot 2: Learning rate schedule
        ax = axes[1]
        if lr:
            lr_steps, lr_vals = zip(*lr)
            ax.plot(lr_steps, lr_vals, color='green', linewidth=2)
            ax.set_xlabel('Training Step', fontsize=11)
            ax.set_ylabel('Learning Rate', fontsize=11)
            ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
        else:
            ax.text(0.5, 0.5, 'No learning rate data available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()

        # Save plot
        if output_file is None:
            output_file = os.path.join(self.checkpoint_dir, "training_dashboard.png")

        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Training dashboard plot saved to: {output_file}\n")

    def export_csv(self, output_file: Optional[str] = None):
        """
        Export metrics to CSV format.

        Args:
            output_file: Output CSV path (default: checkpoint_dir/training_metrics.csv)
        """
        checkpoints = self.find_all_checkpoints()

        if not checkpoints:
            print(f"\nNo checkpoints found in: {self.checkpoint_dir}\n")
            return

        if output_file is None:
            output_file = os.path.join(self.checkpoint_dir, "training_metrics.csv")

        metrics = self.extract_metrics(checkpoints)

        # Write CSV
        with open(output_file, 'w') as f:
            # Header
            f.write("step,train_loss,val_bpb,learning_rate\n")

            # Data rows
            for i, step in enumerate(metrics["steps"]):
                train_loss = metrics["train_loss"][i] if metrics["train_loss"][i] is not None else ""
                val_bpb = metrics["val_bpb"][i] if metrics["val_bpb"][i] is not None else ""
                lr = metrics["learning_rate"][i] if metrics["learning_rate"][i] is not None else ""

                f.write(f"{step},{train_loss},{val_bpb},{lr}\n")

        print(f"\n✓ Metrics exported to: {output_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Training Progress Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show summary of training run
  python tools/training_dashboard.py out/base_checkpoints/d20

  # Show progress table
  python tools/training_dashboard.py out/base_checkpoints/d20 --progress

  # Generate visualization plot
  python tools/training_dashboard.py out/base_checkpoints/d20 --plot

  # Export metrics to CSV
  python tools/training_dashboard.py out/base_checkpoints/d20 --export-csv

  # Generate plot with custom output path
  python tools/training_dashboard.py out/base_checkpoints/d20 --plot --output my_plot.png
        """
    )

    parser.add_argument("checkpoint_dir", help="Path to checkpoint directory")
    parser.add_argument("--progress", action="store_true",
                       help="Show progress table")
    parser.add_argument("--plot", action="store_true",
                       help="Generate training dashboard plot")
    parser.add_argument("--export-csv", action="store_true",
                       help="Export metrics to CSV")
    parser.add_argument("--output", "-o", help="Output file path for plot or CSV")

    args = parser.parse_args()

    try:
        dashboard = TrainingDashboard(args.checkpoint_dir)

        if args.progress:
            dashboard.show_progress()
        elif args.plot:
            dashboard.plot_metrics(output_file=args.output)
        elif args.export_csv:
            dashboard.export_csv(output_file=args.output)
        else:
            # Default: show summary
            dashboard.print_summary()

    except ValueError as e:
        print(f"\nError: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
