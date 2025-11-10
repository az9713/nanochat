"""
Checkpoint Browser and Comparator

Browse, inspect, and compare model checkpoints in nanochat's format.
"""

import os
import json
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Try to import tabulate, fall back to simple formatting if not available
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


class CheckpointBrowser:
    """Tool for browsing and comparing model checkpoints."""

    def __init__(self, base_dir: str = "out"):
        self.base_dir = base_dir
        self.checkpoint_dirs = [
            os.path.join(base_dir, "base_checkpoints"),
            os.path.join(base_dir, "chat_checkpoints"),
        ]

    def find_all_checkpoints(self) -> List[Dict]:
        """
        Scan for all checkpoints and gather metadata.

        Returns:
            List of checkpoint info dictionaries
        """
        checkpoints = []

        for checkpoint_dir in self.checkpoint_dirs:
            if not os.path.exists(checkpoint_dir):
                continue

            # Iterate through model directories (e.g., d20, d26)
            try:
                subdirs = os.listdir(checkpoint_dir)
            except PermissionError:
                continue

            for model_name in subdirs:
                model_path = os.path.join(checkpoint_dir, model_name)
                if not os.path.isdir(model_path):
                    continue

                # Find all model checkpoint files in this directory
                model_files = glob.glob(os.path.join(model_path, "model_*.pt"))

                for model_file in model_files:
                    basename = os.path.basename(model_file)
                    try:
                        # Extract step number from filename
                        step_str = basename.split('_')[1].split('.')[0]
                        step = int(step_str)
                    except (IndexError, ValueError):
                        continue

                    # Find corresponding meta file
                    meta_file = os.path.join(model_path, f"meta_{step:06d}.json")
                    if not os.path.exists(meta_file):
                        continue

                    # Check for optimizer file
                    optim_file = os.path.join(model_path, f"optim_{step:06d}.pt")
                    has_optimizer = os.path.exists(optim_file)

                    try:
                        # Load metadata from JSON
                        with open(meta_file, 'r') as f:
                            metadata = json.load(f)

                        # Get model config
                        model_config = metadata.get('model_config', {})

                        # Calculate file sizes
                        model_size_mb = os.path.getsize(model_file) / (1024 * 1024)
                        meta_size_mb = os.path.getsize(meta_file) / (1024 * 1024)
                        optim_size_mb = os.path.getsize(optim_file) / (1024 * 1024) if has_optimizer else 0
                        total_size_mb = model_size_mb + meta_size_mb + optim_size_mb

                        # Get modification time
                        modified_time = datetime.fromtimestamp(os.path.getmtime(model_file))

                        # Calculate number of parameters from metadata
                        num_params = model_config.get('n_embd', 0) ** 2 * model_config.get('n_layer', 0)
                        # This is a rough estimate; actual calculation would require loading the model

                        checkpoint_info = {
                            'model_name': model_name,
                            'step': step,
                            'model_file': model_file,
                            'meta_file': meta_file,
                            'optim_file': optim_file if has_optimizer else None,
                            'has_optimizer': has_optimizer,
                            'type': 'base' if 'base_checkpoints' in checkpoint_dir else 'chat',
                            'val_bpb': metadata.get('val_bpb', None),
                            'train_bpb': metadata.get('train_bpb', None),
                            'n_layer': model_config.get('n_layer', None),
                            'n_embd': model_config.get('n_embd', None),
                            'n_head': model_config.get('n_head', None),
                            'sequence_len': model_config.get('sequence_len', None),
                            'vocab_size': model_config.get('vocab_size', None),
                            'model_size_mb': model_size_mb,
                            'total_size_mb': total_size_mb,
                            'modified': modified_time,
                            'metadata': metadata,
                        }

                        checkpoints.append(checkpoint_info)

                    except (json.JSONDecodeError, KeyError, OSError) as e:
                        print(f"Warning: Could not load checkpoint {model_file}: {e}")
                        continue

        return checkpoints

    def _format_table(self, headers: List[str], rows: List[List], title: str = None):
        """
        Format table using tabulate if available, otherwise use simple formatting.
        """
        if HAS_TABULATE:
            if title:
                print("\n" + "=" * 120)
                print(title)
                print("=" * 120 + "\n")
            print(tabulate(rows, headers=headers, tablefmt='grid'))
        else:
            # Simple fallback formatting
            if title:
                print("\n" + "=" * 120)
                print(title)
                print("=" * 120 + "\n")

            # Calculate column widths
            col_widths = [len(h) for h in headers]
            for row in rows:
                for i, cell in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

            # Print header
            header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
            print(header_line)
            print("-" * len(header_line))

            # Print rows
            for row in rows:
                row_line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
                print(row_line)

    def list_checkpoints(self, sort_by: str = 'modified'):
        """
        List all checkpoints in a formatted table.

        Args:
            sort_by: Field to sort by ('modified', 'val_bpb', 'size_mb', 'step')
        """
        checkpoints = self.find_all_checkpoints()

        if not checkpoints:
            print("No checkpoints found!")
            print(f"Searched in: {', '.join(self.checkpoint_dirs)}")
            return

        # Sort checkpoints - handle None values safely
        if sort_by == 'modified':
            checkpoints.sort(key=lambda x: x['modified'], reverse=True)
        elif sort_by == 'val_bpb':
            checkpoints.sort(key=lambda x: (x['val_bpb'] is None, x['val_bpb'] if x['val_bpb'] else float('inf')))
        elif sort_by == 'size_mb':
            checkpoints.sort(key=lambda x: x['total_size_mb'], reverse=True)
        elif sort_by == 'step':
            checkpoints.sort(key=lambda x: x['step'], reverse=True)

        # Prepare table data
        headers = ['Model', 'Type', 'Step', 'Val BPB', 'Layers', 'Dim', 'Size (MB)', 'Optim', 'Modified']
        rows = []

        for cp in checkpoints:
            rows.append([
                cp['model_name'],
                cp['type'],
                f"{cp['step']:,}",
                f"{cp['val_bpb']:.4f}" if cp['val_bpb'] is not None else 'N/A',
                cp['n_layer'] if cp['n_layer'] is not None else 'N/A',
                cp['n_embd'] if cp['n_embd'] is not None else 'N/A',
                f"{cp['total_size_mb']:.1f}",
                '✓' if cp['has_optimizer'] else '✗',
                cp['modified'].strftime('%Y-%m-%d %H:%M'),
            ])

        self._format_table(headers, rows, title=f"CHECKPOINTS (sorted by {sort_by})")

        print(f"\nTotal checkpoints: {len(checkpoints)}")
        total_size_gb = sum(cp['total_size_mb'] for cp in checkpoints) / 1024
        print(f"Total disk usage: {total_size_gb:.2f} GB")

        if not HAS_TABULATE:
            print("\n💡 Tip: Install 'tabulate' for better formatting: pip install tabulate")
        print()

    def inspect_checkpoint(self, model_name: str, step: Optional[int] = None):
        """
        Show detailed information about a specific checkpoint.

        Args:
            model_name: Model name (e.g., 'd20')
            step: Optional specific step to inspect (if None, shows latest)
        """
        checkpoints = self.find_all_checkpoints()

        # Filter by model name
        matching = [cp for cp in checkpoints if cp['model_name'] == model_name]

        if not matching:
            print(f"No checkpoints found for model '{model_name}'")
            return

        # If step specified, find exact match; otherwise use latest
        if step is not None:
            checkpoint = next((cp for cp in matching if cp['step'] == step), None)
            if not checkpoint:
                print(f"No checkpoint found for model '{model_name}' at step {step}")
                print(f"Available steps: {', '.join(str(cp['step']) for cp in matching)}")
                return
        else:
            # Get latest checkpoint
            checkpoint = max(matching, key=lambda x: x['step'])

        print("\n" + "=" * 80)
        print(f"CHECKPOINT DETAILS: {checkpoint['model_name']} (step {checkpoint['step']:,})")
        print("=" * 80 + "\n")

        # File information
        print("Files:")
        print(f"  Model:      {os.path.basename(checkpoint['model_file'])} ({checkpoint['model_size_mb']:.1f} MB)")
        print(f"  Metadata:   {os.path.basename(checkpoint['meta_file'])}")
        if checkpoint['has_optimizer']:
            optim_size = os.path.getsize(checkpoint['optim_file']) / (1024 * 1024)
            print(f"  Optimizer:  {os.path.basename(checkpoint['optim_file'])} ({optim_size:.1f} MB)")
        else:
            print(f"  Optimizer:  Not saved")
        print(f"  Total size: {checkpoint['total_size_mb']:.1f} MB")
        print(f"  Modified:   {checkpoint['modified'].strftime('%Y-%m-%d %H:%M:%S')}")

        # Training information
        print(f"\nTraining Information:")
        print(f"  Type:       {checkpoint['type']}")
        print(f"  Step:       {checkpoint['step']:,}")
        if checkpoint['val_bpb'] is not None:
            print(f"  Val BPB:    {checkpoint['val_bpb']:.4f}")
        if checkpoint['train_bpb'] is not None:
            print(f"  Train BPB:  {checkpoint['train_bpb']:.4f}")

        # Model architecture
        print(f"\nModel Architecture:")
        if checkpoint['n_layer'] is not None:
            print(f"  Layers:     {checkpoint['n_layer']}")
        if checkpoint['n_embd'] is not None:
            print(f"  Hidden dim: {checkpoint['n_embd']}")
        if checkpoint['n_head'] is not None:
            print(f"  Heads:      {checkpoint['n_head']}")
        if checkpoint['sequence_len'] is not None:
            print(f"  Seq length: {checkpoint['sequence_len']}")
        if checkpoint['vocab_size'] is not None:
            print(f"  Vocab size: {checkpoint['vocab_size']}")

        # Additional metadata
        metadata = checkpoint['metadata']
        if 'user_config' in metadata and metadata['user_config']:
            print(f"\nUser Configuration:")
            for key, value in metadata['user_config'].items():
                print(f"  {key:20s} = {value}")

        print()

    def compare_checkpoints(self, name1: str, name2: str, step1: Optional[int] = None, step2: Optional[int] = None):
        """
        Compare two checkpoints side-by-side.

        Args:
            name1: First model name
            name2: Second model name
            step1: Optional step for first model
            step2: Optional step for second model
        """
        checkpoints = self.find_all_checkpoints()

        # Find first checkpoint
        matching1 = [cp for cp in checkpoints if cp['model_name'] == name1]
        if not matching1:
            print(f"Model '{name1}' not found!")
            return
        if step1 is not None:
            cp1 = next((cp for cp in matching1 if cp['step'] == step1), None)
            if not cp1:
                print(f"Step {step1} not found for model '{name1}'")
                return
        else:
            cp1 = max(matching1, key=lambda x: x['step'])

        # Find second checkpoint
        matching2 = [cp for cp in checkpoints if cp['model_name'] == name2]
        if not matching2:
            print(f"Model '{name2}' not found!")
            return
        if step2 is not None:
            cp2 = next((cp for cp in matching2 if cp['step'] == step2), None)
            if not cp2:
                print(f"Step {step2} not found for model '{name2}'")
                return
        else:
            cp2 = max(matching2, key=lambda x: x['step'])

        print("\n" + "=" * 100)
        print(f"CHECKPOINT COMPARISON")
        print("=" * 100 + "\n")

        # Comparison table
        headers = ['Metric', f'{cp1["model_name"]} (step {cp1["step"]})', f'{cp2["model_name"]} (step {cp2["step"]})', 'Difference']
        rows = []

        # Architecture comparisons
        if cp1['n_layer'] is not None and cp2['n_layer'] is not None:
            rows.append([
                'Layers',
                str(cp1['n_layer']),
                str(cp2['n_layer']),
                f"{cp2['n_layer'] - cp1['n_layer']:+d}"
            ])

        if cp1['n_embd'] is not None and cp2['n_embd'] is not None:
            rows.append([
                'Hidden dimension',
                str(cp1['n_embd']),
                str(cp2['n_embd']),
                f"{cp2['n_embd'] - cp1['n_embd']:+d}"
            ])

        if cp1['n_head'] is not None and cp2['n_head'] is not None:
            rows.append([
                'Attention heads',
                str(cp1['n_head']),
                str(cp2['n_head']),
                f"{cp2['n_head'] - cp1['n_head']:+d}"
            ])

        # Training metrics
        if cp1['val_bpb'] is not None and cp2['val_bpb'] is not None:
            diff_bpb = cp2['val_bpb'] - cp1['val_bpb']
            better = "←" if diff_bpb < 0 else "→" if diff_bpb > 0 else "="
            rows.append([
                'Val BPB',
                f"{cp1['val_bpb']:.4f}",
                f"{cp2['val_bpb']:.4f}",
                f"{diff_bpb:+.4f} {better}"
            ])

        # Size comparison
        diff_size = cp2['total_size_mb'] - cp1['total_size_mb']
        rows.append([
            'Total size (MB)',
            f"{cp1['total_size_mb']:.1f}",
            f"{cp2['total_size_mb']:.1f}",
            f"{diff_size:+.1f}"
        ])

        # Optimizer
        rows.append([
            'Has optimizer',
            '✓' if cp1['has_optimizer'] else '✗',
            '✓' if cp2['has_optimizer'] else '✗',
            '='
        ])

        self._format_table(headers, rows)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Checkpoint Browser and Comparator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all checkpoints
  python tools/checkpoint_browser.py --list

  # Sort by validation loss
  python tools/checkpoint_browser.py --list --sort val_bpb

  # Inspect a specific model (latest checkpoint)
  python tools/checkpoint_browser.py --inspect d20

  # Inspect specific step
  python tools/checkpoint_browser.py --inspect d20 --step 5000

  # Compare two models
  python tools/checkpoint_browser.py --compare d20 d26
        """
    )
    parser.add_argument("--list", action="store_true", help="List all checkpoints")
    parser.add_argument("--sort", default="modified",
                       choices=['modified', 'val_bpb', 'size_mb', 'step'],
                       help="Sort checkpoints by field (default: modified)")
    parser.add_argument("--inspect", metavar="MODEL", help="Inspect a specific model")
    parser.add_argument("--step", type=int, help="Specific step to inspect/compare")
    parser.add_argument("--compare", nargs=2, metavar=("MODEL1", "MODEL2"),
                       help="Compare two models")
    parser.add_argument("--base-dir", default="out", help="Base directory for checkpoints (default: out)")

    args = parser.parse_args()

    # Validate step usage
    if args.step is not None and not (args.inspect or args.compare):
        print("Error: --step can only be used with --inspect or --compare")
        return

    browser = CheckpointBrowser(base_dir=args.base_dir)

    if args.list:
        browser.list_checkpoints(sort_by=args.sort)
    elif args.inspect:
        browser.inspect_checkpoint(args.inspect, step=args.step)
    elif args.compare:
        browser.compare_checkpoints(args.compare[0], args.compare[1], step1=args.step)
    else:
        # Default: list checkpoints
        browser.list_checkpoints()


if __name__ == "__main__":
    main()
