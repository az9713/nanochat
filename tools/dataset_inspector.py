#!/usr/bin/env python3
"""
Dataset Inspector

Inspect, analyze, and validate training datasets for nanochat.
Perfect for understanding your data before spending hours training.

Usage:
    python tools/dataset_inspector.py dataset.jsonl
    python tools/dataset_inspector.py dataset.jsonl --samples 10
    python tools/dataset_inspector.py dataset.jsonl --analyze-lengths
    python tools/dataset_inspector.py dataset.jsonl --validate
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from typing import List, Dict, Optional


class DatasetInspector:
    """Tool for inspecting and analyzing datasets."""

    def __init__(self, tokenizer=None):
        """
        Initialize dataset inspector.

        Args:
            tokenizer: Optional tokenizer for length analysis
        """
        self.tokenizer = tokenizer

    def load_jsonl(self, filepath: str, max_samples: Optional[int] = None) -> List[Dict]:
        """
        Load conversations from JSONL file.

        Args:
            filepath: Path to JSONL file
            max_samples: Maximum number of samples to load (None = all)

        Returns:
            List of conversation dictionaries
        """
        conversations = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    try:
                        conversations.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed JSON on line {i+1}: {e}", file=sys.stderr)
        except FileNotFoundError:
            print(f"Error: File not found: {filepath}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error loading file: {e}", file=sys.stderr)
            sys.exit(1)

        return conversations

    def show_samples(self, filepath: str, num_samples: int = 5):
        """
        Show random samples from dataset.

        Args:
            filepath: Path to dataset file
            num_samples: Number of samples to show
        """
        conversations = self.load_jsonl(filepath)

        if not conversations:
            print("No conversations found in dataset!")
            return

        samples = random.sample(conversations, min(num_samples, len(conversations)))

        print(f"\n{'='*80}")
        print(f"DATASET SAMPLES ({len(samples)} of {len(conversations)} total)")
        print(f"{'='*80}\n")

        for i, conv in enumerate(samples, 1):
            print(f"Sample {i}:")
            print("-" * 80)

            messages = conv.get('messages', [])
            for msg in messages:
                role = msg.get('role', 'unknown').upper()
                content = msg.get('content', '')

                # Handle both string and structured content
                if isinstance(content, str):
                    # Truncate very long messages for display
                    display_content = content[:200] + "..." if len(content) > 200 else content
                    print(f"{role}: {display_content}")
                elif isinstance(content, list):
                    # Structured content (with tool use)
                    print(f"{role}:")
                    for part in content:
                        part_type = part.get('type', 'unknown').upper()
                        part_text = part.get('text', '')[:100]
                        print(f"  [{part_type}] {part_text}")
                else:
                    print(f"{role}: [non-string content]")

            print()

    def analyze_lengths(self, filepath: str):
        """
        Analyze token and character lengths in dataset.

        Args:
            filepath: Path to dataset file
        """
        conversations = self.load_jsonl(filepath)

        if not conversations:
            print("No conversations found in dataset!")
            return

        conv_token_lengths = []
        conv_char_lengths = []
        user_lengths = []
        assistant_lengths = []

        for conv in conversations:
            messages = conv.get('messages', [])

            # If we have a tokenizer, count tokens; otherwise count characters
            if self.tokenizer:
                try:
                    ids, _ = self.tokenizer.render_conversation(conv)
                    conv_token_lengths.append(len(ids))
                except Exception as e:
                    # If tokenization fails, skip this conversation
                    print(f"Warning: Skipping conversation due to tokenization error: {e}", file=sys.stderr)
                    continue

            # Always count characters as a backup
            total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
            conv_char_lengths.append(total_chars)

            # Analyze individual messages
            for msg in messages:
                role = msg.get('role')
                content = msg.get('content', '')

                if isinstance(content, str):
                    char_len = len(content)
                    if role == 'user':
                        user_lengths.append(char_len)
                    elif role == 'assistant':
                        assistant_lengths.append(char_len)

        print(f"\n{'='*80}")
        print(f"LENGTH ANALYSIS")
        print(f"{'='*80}\n")

        print(f"Total conversations: {len(conversations)}")

        # Show token lengths if tokenizer available
        if conv_token_lengths:
            print(f"\nFull Conversation Lengths (in tokens):")
            print(f"  Min:     {min(conv_token_lengths):,}")
            print(f"  Max:     {max(conv_token_lengths):,}")
            print(f"  Mean:    {sum(conv_token_lengths)/len(conv_token_lengths):,.1f}")
            print(f"  Median:  {sorted(conv_token_lengths)[len(conv_token_lengths)//2]:,}")

        # Show character lengths
        if conv_char_lengths:
            print(f"\nFull Conversation Lengths (in characters):")
            print(f"  Min:     {min(conv_char_lengths):,}")
            print(f"  Max:     {max(conv_char_lengths):,}")
            print(f"  Mean:    {sum(conv_char_lengths)/len(conv_char_lengths):,.1f}")

        # Message-level statistics
        if user_lengths:
            print(f"\nUser Message Lengths (characters):")
            print(f"  Min:     {min(user_lengths):,}")
            print(f"  Max:     {max(user_lengths):,}")
            print(f"  Mean:    {sum(user_lengths)/len(user_lengths):,.1f}")

        if assistant_lengths:
            print(f"\nAssistant Message Lengths (characters):")
            print(f"  Min:     {min(assistant_lengths):,}")
            print(f"  Max:     {max(assistant_lengths):,}")
            print(f"  Mean:    {sum(assistant_lengths)/len(assistant_lengths):,.1f}")

        # Show distribution histogram (using tokens if available, else chars)
        lengths_to_plot = conv_token_lengths if conv_token_lengths else conv_char_lengths
        unit = "tokens" if conv_token_lengths else "chars"

        if lengths_to_plot:
            print(f"\nLength Distribution ({unit}):")
            bins = [0, 100, 200, 500, 1000, 2000, 5000, 10000]
            distribution = defaultdict(int)

            for length in lengths_to_plot:
                for i, bin_max in enumerate(bins[1:], 1):
                    if length <= bin_max:
                        bin_label = f"{bins[i-1]}-{bin_max}"
                        distribution[bin_label] += 1
                        break
                else:
                    distribution[f">{bins[-1]}"] += 1

            max_count = max(distribution.values()) if distribution else 1
            for bin_label in [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)] + [f">{bins[-1]}"]:
                count = distribution.get(bin_label, 0)
                if count > 0:
                    bar = '#' * max(1, int(count * 50 / max_count))
                    print(f"  {bin_label:12s}: {count:5d} {bar}")

        print(f"\n💡 Learning Insight:")
        if conv_token_lengths:
            avg_len = sum(conv_token_lengths) / len(conv_token_lengths)
            if avg_len < 512:
                print("  Your conversations are relatively short. Good for quick training!")
            elif avg_len < 2048:
                print("  Your conversations are medium length. Balanced for most use cases.")
            else:
                print("  Your conversations are long. Consider using larger context windows.")

    def validate_format(self, filepath: str):
        """
        Validate dataset format and find issues.

        Args:
            filepath: Path to dataset file
        """
        conversations = self.load_jsonl(filepath)

        if not conversations:
            print("No conversations found in dataset!")
            return

        print(f"\n{'='*80}")
        print(f"FORMAT VALIDATION")
        print(f"{'='*80}\n")

        issues = []
        issue_types = Counter()

        for i, conv in enumerate(conversations):
            # Check required fields
            if 'messages' not in conv:
                issues.append((i, "Missing 'messages' field"))
                issue_types['missing_messages'] += 1
                continue

            messages = conv['messages']

            # Check message count
            if len(messages) < 2:
                issues.append((i, f"Only {len(messages)} message(s) - need at least 2"))
                issue_types['too_short'] += 1

            # Check alternating roles
            # Handle optional system message at start (as per tokenizer.render_conversation)
            start_idx = 0
            if len(messages) > 0 and messages[0].get('role') == 'system':
                # System message must be followed by user message
                if len(messages) < 2:
                    issues.append((i, "System message must be followed by at least a user message"))
                    issue_types['wrong_role'] += 1
                elif messages[1].get('role') != 'user':
                    issues.append((i, f"Message 1: system message must be followed by user, got '{messages[1].get('role')}'"))
                    issue_types['wrong_role'] += 1
                start_idx = 1  # Start checking alternation from message 1 (after system)

            # Check alternation starting from start_idx (should be user, assistant, user, assistant...)
            for j in range(start_idx, len(messages)):
                actual_role = messages[j].get('role')
                # Relative position after skipping system message
                relative_pos = j - start_idx
                expected_role = 'user' if relative_pos % 2 == 0 else 'assistant'

                if actual_role != expected_role:
                    issues.append((i, f"Message {j}: expected '{expected_role}', got '{actual_role}'"))
                    issue_types['wrong_role'] += 1

            # Check content exists for all messages
            for j, msg in enumerate(messages):
                if 'content' not in msg or not msg['content']:
                    issues.append((i, f"Message {j}: empty or missing content"))
                    issue_types['empty_content'] += 1

        # Print summary
        print(f"Checked {len(conversations)} conversations\n")

        if issues:
            print(f"❌ Found {len(issues)} issues:\n")
            for issue_type, count in issue_types.most_common():
                print(f"  {issue_type:20s}: {count:5d}")

            print(f"\nFirst 10 issues (showing conversation index):")
            for idx, issue_msg in issues[:10]:
                print(f"  Conv {idx}: {issue_msg}")

            print(f"\n💡 Fix these issues before training to avoid errors!")
        else:
            print("✅ No format issues found! Dataset looks good.")

        # Additional statistics
        print(f"\n{'='*80}")
        print(f"DATASET STATISTICS")
        print(f"{'='*80}\n")

        total_messages = sum(len(conv.get('messages', [])) for conv in conversations)
        print(f"Total conversations: {len(conversations):,}")
        print(f"Total messages:      {total_messages:,}")

        if conversations:
            print(f"Avg messages/conv:   {total_messages/len(conversations):.1f}")

        # Count tool usage
        tool_convs = 0
        for conv in conversations:
            for msg in conv.get('messages', []):
                content = msg.get('content')
                if isinstance(content, list):
                    # Has structured content - likely uses tools
                    for part in content:
                        if part.get('type') in ['python', 'python_output', 'code']:
                            tool_convs += 1
                            break
                    if tool_convs > 0:
                        break

        if tool_convs > 0:
            print(f"\nTool Usage:")
            print(f"  Conversations with tools: {tool_convs:,} ({100*tool_convs/len(conversations):.1f}%)")

        # Role distribution
        role_counts = Counter()
        for conv in conversations:
            for msg in conv.get('messages', []):
                role_counts[msg.get('role', 'unknown')] += 1

        if role_counts:
            print(f"\nMessage Role Distribution:")
            for role, count in role_counts.most_common():
                print(f"  {role:15s}: {count:,}")

    def export_samples(self, filepath: str, output_file: str, num_samples: int = 100):
        """
        Export random samples to a file for manual review.

        Args:
            filepath: Input dataset path
            output_file: Output file path
            num_samples: Number of samples to export
        """
        conversations = self.load_jsonl(filepath)

        if not conversations:
            print("No conversations found in dataset!")
            return

        samples = random.sample(conversations, min(num_samples, len(conversations)))

        with open(output_file, 'w', encoding='utf-8') as f:
            for i, conv in enumerate(samples, 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"SAMPLE {i}\n")
                f.write(f"{'='*80}\n\n")

                messages = conv.get('messages', [])
                for msg in messages:
                    role = msg.get('role', 'unknown').upper()
                    content = msg.get('content', '')

                    if isinstance(content, str):
                        f.write(f"{role}: {content}\n\n")
                    elif isinstance(content, list):
                        f.write(f"{role}:\n")
                        for part in content:
                            part_type = part.get('type', 'unknown').upper()
                            part_text = part.get('text', '')
                            f.write(f"  [{part_type}] {part_text}\n")
                        f.write("\n")

        print(f"✅ Exported {len(samples)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Dataset Inspector - Analyze and validate training datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show 5 random samples
  python tools/dataset_inspector.py dataset.jsonl

  # Show 10 samples
  python tools/dataset_inspector.py dataset.jsonl --samples 10

  # Analyze token/character lengths
  python tools/dataset_inspector.py dataset.jsonl --analyze-lengths

  # Validate format for errors
  python tools/dataset_inspector.py dataset.jsonl --validate

  # Export 100 samples for manual review
  python tools/dataset_inspector.py dataset.jsonl --export samples.txt

  # Do everything
  python tools/dataset_inspector.py dataset.jsonl --validate --analyze-lengths --samples 10
        """
    )

    parser.add_argument("dataset", help="Path to dataset file (JSONL format)")
    parser.add_argument("--samples", type=int, default=5,
                       help="Number of samples to show (default: 5)")
    parser.add_argument("--analyze-lengths", action="store_true",
                       help="Analyze token and character lengths")
    parser.add_argument("--validate", action="store_true",
                       help="Validate dataset format")
    parser.add_argument("--export", metavar="FILE",
                       help="Export samples to file for manual review")
    parser.add_argument("--export-count", type=int, default=100,
                       help="Number of samples to export (default: 100)")
    parser.add_argument("--no-tokenizer", action="store_true",
                       help="Skip loading tokenizer (faster, but no token counts)")

    args = parser.parse_args()

    # Load tokenizer if needed and not disabled
    tokenizer = None
    if args.analyze_lengths and not args.no_tokenizer:
        try:
            print("Loading tokenizer...", file=sys.stderr)
            from nanochat.tokenizer import get_tokenizer
            tokenizer = get_tokenizer()
            print("Tokenizer loaded!\n", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}", file=sys.stderr)
            print("Continuing with character-based analysis only.\n", file=sys.stderr)

    inspector = DatasetInspector(tokenizer=tokenizer)

    # Default behavior: show samples
    if not (args.validate or args.analyze_lengths or args.export):
        inspector.show_samples(args.dataset, args.samples)
    else:
        # Run requested analyses
        if args.validate:
            inspector.validate_format(args.dataset)

        if args.analyze_lengths:
            inspector.analyze_lengths(args.dataset)

        if args.export:
            inspector.export_samples(args.dataset, args.export, args.export_count)

        # Also show samples if explicitly requested
        if args.samples and args.samples != 5:  # User changed default
            inspector.show_samples(args.dataset, args.samples)


if __name__ == '__main__':
    main()
