"""
Generation Parameter Explorer

Interactively explore how sampling parameters affect model outputs.
"""

import argparse
import torch
import torch.nn.functional as F
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine


class GenerationExplorer:
    """Tool for exploring generation parameters."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.engine = Engine(model, tokenizer)

    def sample_with_probabilities(self, prompt: str, max_tokens: int = 50,
                                  temperature: float = 1.0, top_k: int = None,
                                  show_probs: bool = True, num_alternatives: int = 5):
        """
        Generate text and show probability distribution at each step.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            show_probs: Whether to show probabilities
            num_alternatives: Number of alternative tokens to show
        """
        # Encode prompt
        prompt_tokens = self.tokenizer.encode(prompt, prepend="<|bos|>")

        print(f"\n{'='*100}")
        print(f"GENERATION WITH PROBABILITIES")
        print(f"{'='*100}\n")
        print(f"Prompt: \"{prompt}\"")
        print(f"Temperature: {temperature}, Top-k: {top_k}\n")
        print(f"{'-'*100}\n")

        # Generate token by token
        generated_tokens = []
        current_tokens = prompt_tokens.copy()

        for step in range(max_tokens):
            # Forward pass
            with torch.no_grad():
                ids = torch.tensor([current_tokens], dtype=torch.long, device=self.model.get_device())
                logits = self.model(ids)
                logits = logits[0, -1, :]  # Last position

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Apply top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = -float('Inf')

            # Get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample
            if temperature > 0:
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(logits).item()

            # Show probabilities
            if show_probs:
                # Get top alternatives
                top_probs, top_indices = torch.topk(probs, num_alternatives)

                print(f"Step {step + 1}:")
                print(f"  Sampled: [{next_token}] \"{self.tokenizer.decode([next_token])}\" (p={probs[next_token]:.4f})")

                print(f"  Top {num_alternatives} alternatives:")
                for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    token_str = self.tokenizer.decode([idx.item()])
                    marker = "←" if idx.item() == next_token else " "
                    print(f"    {i+1}. [{idx.item():5d}] \"{token_str:20s}\" p={prob:.4f} {marker}")
                print()

            # Add to sequence
            generated_tokens.append(next_token)
            current_tokens.append(next_token)

            # Stop at end token
            if next_token == self.tokenizer.encode_special("<|assistant_end|>"):
                break

        # Print full generation
        print(f"{'-'*100}")
        print(f"\nFull generation:")
        print(f"\"{self.tokenizer.decode(generated_tokens)}\"")
        print()

    def compare_temperatures(self, prompt: str, temperatures: list = [0.1, 0.5, 0.9, 1.2, 1.5],
                           max_tokens: int = 50, num_samples: int = 3):
        """
        Compare outputs at different temperatures.

        Args:
            prompt: Input prompt
            temperatures: List of temperatures to try
            max_tokens: Maximum tokens per generation
            num_samples: Number of samples per temperature
        """
        print(f"\n{'='*100}")
        print(f"TEMPERATURE COMPARISON")
        print(f"{'='*100}\n")
        print(f"Prompt: \"{prompt}\"\n")

        prompt_tokens = self.tokenizer.encode(prompt, prepend="<|bos|>")

        for temp in temperatures:
            print(f"\n{'-'*100}")
            print(f"Temperature: {temp}")
            print(f"{'-'*100}\n")

            for i in range(num_samples):
                samples, _ = self.engine.generate_batch(
                    prompt_tokens,
                    num_samples=1,
                    max_tokens=max_tokens,
                    temperature=temp,
                    seed=42 + i
                )

                output = self.tokenizer.decode(samples[0][len(prompt_tokens):])
                print(f"  Sample {i+1}: {output}")

            print()

    def compare_top_k(self, prompt: str, top_k_values: list = [10, 50, 100, None],
                     max_tokens: int = 50):
        """
        Compare outputs with different top-k values.

        Args:
            prompt: Input prompt
            top_k_values: List of top-k values to try (None = no filtering)
            max_tokens: Maximum tokens per generation
        """
        print(f"\n{'='*100}")
        print(f"TOP-K COMPARISON")
        print(f"{'='*100}\n")
        print(f"Prompt: \"{prompt}\"\n")

        prompt_tokens = self.tokenizer.encode(prompt, prepend="<|bos|>")

        for k in top_k_values:
            print(f"\n{'-'*100}")
            print(f"Top-k: {k if k is not None else 'None (no filtering)'}")
            print(f"{'-'*100}\n")

            samples, _ = self.engine.generate_batch(
                prompt_tokens,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=0.9,
                top_k=k,
                seed=42
            )

            output = self.tokenizer.decode(samples[0][len(prompt_tokens):])
            print(f"  Output: {output}\n")

    def interactive_mode(self):
        """Interactive mode for exploring generation parameters."""
        print(f"\n{'='*100}")
        print(f"GENERATION PARAMETER EXPLORER - Interactive Mode")
        print(f"{'='*100}\n")
        print("Commands:")
        print("  - Type a prompt to generate with current settings")
        print("  - 'temp <value>' to set temperature (e.g., 'temp 0.9')")
        print("  - 'topk <value>' to set top-k (e.g., 'topk 50')")
        print("  - 'probs' to toggle probability display")
        print("  - 'compare-temp <prompt>' to compare temperatures")
        print("  - 'quit' to exit\n")

        # Default settings
        temperature = 0.9
        top_k = None
        show_probs = False

        while True:
            try:
                print(f"Current settings: temperature={temperature}, top_k={top_k}, show_probs={show_probs}")
                user_input = input("\n> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if user_input.lower().startswith('temp '):
                    temperature = float(user_input.split()[1])
                    print(f"Set temperature to {temperature}")
                    continue

                if user_input.lower().startswith('topk '):
                    value = user_input.split()[1]
                    top_k = None if value.lower() == 'none' else int(value)
                    print(f"Set top-k to {top_k}")
                    continue

                if user_input.lower() == 'probs':
                    show_probs = not show_probs
                    print(f"Probability display: {'ON' if show_probs else 'OFF'}")
                    continue

                if user_input.lower().startswith('compare-temp '):
                    prompt = user_input[13:].strip()
                    self.compare_temperatures(prompt)
                    continue

                # Default: generate with current settings
                self.sample_with_probabilities(
                    user_input,
                    temperature=temperature,
                    top_k=top_k,
                    show_probs=show_probs
                )

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generation Parameter Explorer")
    parser.add_argument("--source", default="base", help="Model source (base, mid, sft, rl)")
    parser.add_argument("--model-tag", help="Model tag (e.g., d20)")
    parser.add_argument("--step", type=int, help="Checkpoint step")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--prompt", help="Prompt to use")
    parser.add_argument("--temperature", type=float, default=0.9, help="Temperature")
    parser.add_argument("--top-k", type=int, help="Top-k filtering")
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--compare-temp", action="store_true", help="Compare temperatures")
    parser.add_argument("--compare-topk", action="store_true", help="Compare top-k values")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--show-probs", action="store_true", help="Show probabilities")

    args = parser.parse_args()

    # Determine device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Load model
    print(f"Loading model from source '{args.source}' on device '{device}'...")
    try:
        model, tokenizer, meta_data = load_model(
            args.source,
            device,
            phase="eval",
            model_tag=args.model_tag,
            step=args.step
        )
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nUsage tips:")
        print("  - Make sure you have a trained model checkpoint")
        print("  - Use --source to specify model type (base, mid, sft, rl)")
        print("  - Use --model-tag to specify model (e.g., d20)")
        print("  - Use --step to specify checkpoint step (optional)")
        return

    explorer = GenerationExplorer(model, tokenizer)

    if args.interactive:
        explorer.interactive_mode()
    elif args.compare_temp and args.prompt:
        explorer.compare_temperatures(args.prompt, max_tokens=args.max_tokens)
    elif args.compare_topk and args.prompt:
        explorer.compare_top_k(args.prompt, max_tokens=args.max_tokens)
    elif args.prompt:
        explorer.sample_with_probabilities(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            show_probs=args.show_probs
        )
    else:
        print("Please provide --prompt or use --interactive mode")
        print("\nExamples:")
        print("  # Interactive mode")
        print("  python tools/generation_explorer.py --source base --interactive")
        print("")
        print("  # Generate with probability display")
        print("  python tools/generation_explorer.py --source base --prompt 'The capital of France is' --show-probs")
        print("")
        print("  # Compare temperatures")
        print("  python tools/generation_explorer.py --source base --prompt 'Once upon a time' --compare-temp")


if __name__ == "__main__":
    main()
