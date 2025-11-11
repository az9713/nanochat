"""
Simple Attention Visualizer

Visualize attention patterns in trained nanochat models to understand
what tokens the model attends to during inference.

This is an educational tool to help beginners understand the Transformer
attention mechanism by showing which tokens attend to which other tokens.
"""

import argparse
import os
import sys
from typing import Optional, Tuple, List
import json

import torch
import torch.nn.functional as F

# Add parent directory to path to import nanochat
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import load_checkpoint, find_checkpoint
from nanochat.gpt import GPT, apply_rotary_emb, norm


class AttentionVisualizer:
    """
    Tool for visualizing attention patterns in GPT models.

    This class loads a trained model and extracts attention weights
    to help understand what the model is "looking at" during inference.
    """

    def __init__(self, model: GPT, tokenizer):
        """
        Initialize the attention visualizer.

        Args:
            model: Trained GPT model
            tokenizer: Tokenizer instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.model.eval()  # Set to eval mode

    def compute_attention_weights(
        self,
        text: str,
        layer_idx: int,
        head_idx: Optional[int] = None
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Compute attention weights for a given text and layer.

        Args:
            text: Input text to analyze
            layer_idx: Which transformer layer to extract attention from (0-indexed)
            head_idx: Which attention head to visualize (None = average all heads)

        Returns:
            Tuple of (token_strings, attention_matrix)
            - token_strings: List of decoded tokens
            - attention_matrix: [seq_len, seq_len] tensor of attention weights
        """
        # Validate layer index
        if layer_idx < 0 or layer_idx >= self.model.config.n_layer:
            raise ValueError(
                f"layer_idx must be between 0 and {self.model.config.n_layer - 1}, "
                f"got {layer_idx}"
            )

        # Validate head index if provided
        if head_idx is not None:
            if head_idx < 0 or head_idx >= self.model.config.n_head:
                raise ValueError(
                    f"head_idx must be between 0 and {self.model.config.n_head - 1}, "
                    f"got {head_idx}"
                )

        # Tokenize input
        tokens = self.tokenizer.encode(text, prepend="<|bos|>")
        token_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        # Get token strings for display
        token_strings = []
        for token_id in tokens:
            token_str = self.tokenizer.decode([token_id])
            token_strings.append(token_str)

        # Forward pass to get hidden states at the target layer
        with torch.no_grad():
            # Manually compute forward pass up to target layer
            x = self.model.transformer.wte(token_ids)  # token embeddings
            x = norm(x)  # norm after embedding

            # Get rotary embeddings for current sequence length
            T = x.size(1)
            # Access rotary embedding buffers directly (model.cos, model.sin)
            # Shape: [1, seq_len, 1, head_dim//2]
            cos = self.model.cos[:, :T, :, :]
            sin = self.model.sin[:, :T, :, :]
            cos_sin = (cos, sin)

            # Forward through layers up to and including target layer
            for i in range(layer_idx + 1):
                block = self.model.transformer.h[i]

                # If this is our target layer, compute attention manually
                if i == layer_idx:
                    attn_weights = self._compute_layer_attention(
                        x, block.attn, cos_sin, head_idx
                    )

                # Continue forward pass
                x = x + block.attn(norm(x), cos_sin, kv_cache=None)
                x = x + block.mlp(norm(x))

        return token_strings, attn_weights

    def _compute_layer_attention(
        self,
        x: torch.Tensor,
        attn_module,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        head_idx: Optional[int]
    ) -> torch.Tensor:
        """
        Manually compute attention weights for a specific layer.

        Args:
            x: Input tensor [B, T, C]
            attn_module: CausalSelfAttention module
            cos_sin: Rotary embedding (cos, sin) tuple
            head_idx: Which head to extract (None = average all)

        Returns:
            Attention weights [T, T]
        """
        B, T, C = x.size()
        x = norm(x)  # Apply norm (same as in Block forward)

        # Project to Q, K, V
        q = attn_module.c_q(x).view(B, T, attn_module.n_head, attn_module.head_dim)
        k = attn_module.c_k(x).view(B, T, attn_module.n_kv_head, attn_module.head_dim)
        v = attn_module.c_v(x).view(B, T, attn_module.n_kv_head, attn_module.head_dim)

        # Apply rotary embeddings and norms (same as in CausalSelfAttention)
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = norm(q)
        k = norm(k)

        # Transpose to [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # Handle MQA/GQA: expand k if needed
        if attn_module.n_head != attn_module.n_kv_head:
            # Repeat k and v to match number of query heads
            n_rep = attn_module.n_head // attn_module.n_kv_head
            k = k.repeat_interleave(n_rep, dim=1)

        # Compute attention scores: [B, H, T, T]
        # scores[i,j] = how much token i attends to token j
        scores = torch.matmul(q, k.transpose(-2, -1)) / (attn_module.head_dim ** 0.5)

        # Apply causal mask (prevent attending to future tokens)
        causal_mask = torch.triu(
            torch.ones(T, T, device=scores.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))

        # Apply softmax to get attention probabilities
        attn_probs = F.softmax(scores, dim=-1)

        # Extract specific head or average across all heads
        if head_idx is not None:
            # Single head: [B, T, T] -> [T, T]
            attn_weights = attn_probs[0, head_idx, :, :]
        else:
            # Average across all heads: [B, H, T, T] -> [T, T]
            attn_weights = attn_probs[0].mean(dim=0)

        return attn_weights

    def visualize_text(
        self,
        text: str,
        layer_idx: int,
        head_idx: Optional[int] = None,
        top_k: int = 5
    ) -> None:
        """
        Display text-based attention visualization.

        Shows for each token what it attends to most strongly.

        Args:
            text: Input text to analyze
            layer_idx: Which layer to visualize
            head_idx: Which head (None = average all heads)
            top_k: How many top attended tokens to show per token
        """
        tokens, attn_weights = self.compute_attention_weights(text, layer_idx, head_idx)

        # Print header
        print("\n" + "=" * 100)
        print("ATTENTION VISUALIZATION (Text-based)")
        print("=" * 100)
        print(f"\nInput text: \"{text}\"")
        print(f"Layer: {layer_idx}")
        if head_idx is not None:
            print(f"Head: {head_idx}")
        else:
            print(f"Head: Average across all {self.model.config.n_head} heads")
        print(f"\nShowing top-{top_k} attended tokens for each position:")
        print("-" * 100)

        # For each token, show what it attends to
        for i, token in enumerate(tokens):
            print(f"\nToken {i}: {repr(token)}")

            # Get attention scores for this token (row i)
            attn_scores = attn_weights[i, :i+1]  # Only up to current position (causal)

            # Get top-k attended tokens
            if len(attn_scores) > 0:
                top_k_actual = min(top_k, len(attn_scores))
                top_values, top_indices = torch.topk(attn_scores, top_k_actual)

                print(f"  Attends to:")
                for rank, (idx, score) in enumerate(zip(top_indices, top_values), 1):
                    idx = idx.item()
                    score = score.item()
                    attended_token = tokens[idx]
                    # Show bar visualization
                    bar_length = int(score * 40)  # Scale to 40 chars max
                    bar = "█" * bar_length
                    print(f"    {rank}. Token {idx}: {repr(attended_token):20s} "
                          f"[{score:5.1%}] {bar}")
            else:
                print(f"  (No previous tokens to attend to)")

        print("\n" + "=" * 100 + "\n")

    def visualize_heatmap(
        self,
        text: str,
        layer_idx: int,
        head_idx: Optional[int] = None,
        output_file: Optional[str] = None
    ) -> None:
        """
        Generate a heatmap visualization of attention weights.

        Args:
            text: Input text to analyze
            layer_idx: Which layer to visualize
            head_idx: Which head (None = average all heads)
            output_file: Path to save the plot (if None, display interactively)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
        except ImportError:
            print("Error: matplotlib is required for heatmap visualization.")
            print("Install with: pip install matplotlib")
            print("Or use --text mode for text-based visualization.")
            return

        tokens, attn_weights = self.compute_attention_weights(text, layer_idx, head_idx)

        # Convert to numpy for matplotlib
        attn_matrix = attn_weights.cpu().numpy()

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot heatmap
        im = ax.imshow(attn_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)

        # Set ticks and labels
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)

        # Labels
        ax.set_xlabel('Attended Token (Key)', fontsize=12)
        ax.set_ylabel('Attending Token (Query)', fontsize=12)

        # Title
        head_str = f"Head {head_idx}" if head_idx is not None else "Averaged Heads"
        ax.set_title(
            f'Attention Weights - Layer {layer_idx}, {head_str}\n"{text}"',
            fontsize=14,
            pad=20
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)

        # Add grid
        ax.set_xticks([x - 0.5 for x in range(1, len(tokens))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(tokens))], minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=1)

        plt.tight_layout()

        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\nHeatmap saved to: {output_file}")
        else:
            print("\nDisplaying heatmap... (close window to continue)")
            plt.show()

        plt.close()


def main():
    """Main entry point for the attention visualizer tool."""
    parser = argparse.ArgumentParser(
        description="Visualize attention patterns in nanochat models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text-based visualization for layer 0
  python tools/attention_visualizer.py --text "Hello world" --layer 0

  # Heatmap for layer 5, head 3
  python tools/attention_visualizer.py --text "The capital of France is" --layer 5 --head 3 --heatmap

  # Save heatmap to file
  python tools/attention_visualizer.py --text "Once upon a time" --layer 10 --heatmap --output attention.png

  # Use specific model
  python tools/attention_visualizer.py --source base --model-tag d20 --text "Hello" --layer 0
        """
    )

    # Model selection
    parser.add_argument(
        "--source",
        type=str,
        default="base",
        choices=["base", "sft"],
        help="Model source (base or sft)"
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default=None,
        help="Model tag (e.g., d20, d26). If not specified, uses most recent."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on (default: cpu for faster loading)"
    )

    # Input text
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input text to analyze"
    )

    # Attention parameters
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Which transformer layer to visualize (0-indexed)"
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Which attention head to visualize (default: average all heads)"
    )

    # Visualization mode
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Generate heatmap visualization (requires matplotlib)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for heatmap (default: display interactively)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top attended tokens to show in text mode (default: 5)"
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading {args.source} model...", end=" ", flush=True)
    checkpoint_dir = f"out/{args.source}_checkpoints"

    if args.model_tag:
        checkpoint_dir = os.path.join(checkpoint_dir, args.model_tag)

    if not os.path.exists(checkpoint_dir):
        print(f"\nError: Checkpoint directory not found: {checkpoint_dir}")
        print("Available checkpoints:")
        base_dir = f"out/{args.source}_checkpoints"
        if os.path.exists(base_dir):
            for d in os.listdir(base_dir):
                if os.path.isdir(os.path.join(base_dir, d)):
                    print(f"  - {d}")
        sys.exit(1)

    try:
        checkpoint_info = find_checkpoint(checkpoint_dir)
        model, meta = load_checkpoint(checkpoint_info, device=args.device)
        print(f"✓ Loaded checkpoint from step {meta.get('step', 'unknown')}")
    except Exception as e:
        print(f"\nError loading model: {e}")
        sys.exit(1)

    # Load tokenizer
    tokenizer = get_tokenizer()

    # Create visualizer
    visualizer = AttentionVisualizer(model, tokenizer)

    # Generate visualization
    try:
        if args.heatmap:
            visualizer.visualize_heatmap(
                args.text,
                args.layer,
                args.head,
                args.output
            )
        else:
            # Text-based visualization (default)
            visualizer.visualize_text(
                args.text,
                args.layer,
                args.head,
                args.top_k
            )
    except ValueError as e:
        print(f"\nError: {e}")
        print(f"\nModel has {model.config.n_layer} layers (0-{model.config.n_layer - 1})")
        print(f"Model has {model.config.n_head} attention heads (0-{model.config.n_head - 1})")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
