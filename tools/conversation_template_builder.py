"""
Conversation Template Builder

Create, validate, and test conversation templates for fine-tuning.
"""

import argparse
import json
import sys
import os
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nanochat.tokenizer import get_tokenizer


class ConversationTemplateBuilder:
    """Tool for creating and validating conversation templates."""

    def __init__(self, tokenizer=None):
        """Initialize the builder with a tokenizer."""
        if tokenizer is None:
            self.tokenizer = get_tokenizer()
        else:
            self.tokenizer = tokenizer

    def create_simple_conversation(self, user_message: str, assistant_message: str) -> Dict[str, Any]:
        """
        Create a simple 2-turn conversation.

        Args:
            user_message: User's message
            assistant_message: Assistant's response

        Returns:
            Conversation dictionary
        """
        return {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message}
            ]
        }

    def create_multi_turn_conversation(self, turns: List[tuple]) -> Dict[str, Any]:
        """
        Create a multi-turn conversation from a list of (role, content) tuples.

        Args:
            turns: List of (role, content) tuples where role is 'user' or 'assistant'

        Returns:
            Conversation dictionary

        Raises:
            ValueError: If turns are invalid
        """
        if not turns:
            raise ValueError("Cannot create conversation with zero turns")

        # Validate that turns alternate between user and assistant
        messages = []
        for i, (role, content) in enumerate(turns):
            if role not in ["user", "assistant", "system"]:
                raise ValueError(f"Invalid role '{role}' at turn {i}. Must be 'user', 'assistant', or 'system'")

            # System messages can only be at the start
            if role == "system" and i != 0:
                raise ValueError(f"System message at turn {i} must be at the start (turn 0)")

            # After system message (if present), must alternate user/assistant starting with user
            if i == 1 and len(messages) > 0 and messages[0]["role"] == "system":
                if role != "user":
                    raise ValueError(f"After system message, first message must be from user, got '{role}'")
            elif i > 0:
                # Check alternation (accounting for optional system message)
                start_idx = 1 if messages[0]["role"] == "system" else 0
                expected_role = "user" if (i - start_idx) % 2 == 0 else "assistant"
                if role != expected_role:
                    raise ValueError(
                        f"Turn {i}: Expected '{expected_role}' but got '{role}'. "
                        f"Conversations must alternate between user and assistant"
                    )

            messages.append({"role": role, "content": content})

        # First message (after optional system) must be from user
        if messages[0]["role"] == "system":
            if len(messages) < 2:
                raise ValueError("Conversation with system message must have at least one user/assistant message")
            first_real_msg = messages[1]
        else:
            first_real_msg = messages[0]

        if first_real_msg["role"] != "user":
            raise ValueError("First message (after optional system) must be from user")

        return {"messages": messages}

    def create_tool_conversation(self, user_message: str, python_code: str,
                                 python_output: str, assistant_response: str) -> Dict[str, Any]:
        """
        Create a conversation with Python tool usage.

        Args:
            user_message: User's request
            python_code: Python code the assistant runs
            python_output: Output from Python execution
            assistant_response: Assistant's final response

        Returns:
            Conversation dictionary
        """
        return {
            "messages": [
                {"role": "user", "content": user_message},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "python", "text": python_code},
                        {"type": "python_output", "text": python_output},
                        {"type": "text", "text": assistant_response}
                    ]
                }
            ]
        }

    def validate_conversation(self, conversation: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate a conversation structure.

        Args:
            conversation: Conversation dictionary to validate

        Returns:
            (is_valid, error_message) tuple
        """
        if not isinstance(conversation, dict):
            return False, "Conversation must be a dictionary"

        if "messages" not in conversation:
            return False, "Conversation must have 'messages' key"

        messages = conversation["messages"]
        if not isinstance(messages, list):
            return False, "'messages' must be a list"

        if len(messages) == 0:
            return False, "Conversation must have at least one message"

        # Check message structure
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                return False, f"Message {i} is not a dictionary"

            if "role" not in msg:
                return False, f"Message {i} missing 'role' field"

            if "content" not in msg:
                return False, f"Message {i} missing 'content' field"

            role = msg["role"]
            if role not in ["user", "assistant", "system"]:
                return False, f"Message {i} has invalid role '{role}'"

            content = msg["content"]

            # Validate role-specific content
            if role == "user":
                if not isinstance(content, str):
                    return False, f"Message {i} (user) content must be a string"
            elif role == "assistant":
                if isinstance(content, str):
                    pass  # Simple string is valid
                elif isinstance(content, list):
                    # List of parts (for tool usage)
                    for j, part in enumerate(content):
                        if not isinstance(part, dict):
                            return False, f"Message {i}, part {j} is not a dictionary"
                        if "type" not in part:
                            return False, f"Message {i}, part {j} missing 'type' field"
                        if "text" not in part:
                            return False, f"Message {i}, part {j} missing 'text' field"
                        if part["type"] not in ["text", "python", "python_output"]:
                            return False, f"Message {i}, part {j} has invalid type '{part['type']}'"
                else:
                    return False, f"Message {i} (assistant) content must be string or list"
            elif role == "system":
                if not isinstance(content, str):
                    return False, f"Message {i} (system) content must be a string"

        # Validate alternation (system can be at start)
        start_idx = 1 if messages[0]["role"] == "system" else 0

        # First real message must be from user
        if start_idx < len(messages):
            if messages[start_idx]["role"] != "user":
                return False, f"First message (after optional system) must be from user, got '{messages[start_idx]['role']}'"

        # Check alternation
        for i in range(start_idx, len(messages)):
            expected_role = "user" if (i - start_idx) % 2 == 0 else "assistant"
            actual_role = messages[i]["role"]
            if actual_role != expected_role:
                return False, (
                    f"Message {i}: Expected '{expected_role}' but got '{actual_role}'. "
                    f"Messages must alternate between user and assistant"
                )

        return True, None

    def show_tokenization(self, conversation: Dict[str, Any], max_tokens: int = 4096):
        """
        Show how a conversation gets tokenized.

        Args:
            conversation: Conversation dictionary
            max_tokens: Maximum tokens for encoding
        """
        print("\n" + "=" * 100)
        print("CONVERSATION TOKENIZATION")
        print("=" * 100 + "\n")

        # Validate first
        is_valid, error = self.validate_conversation(conversation)
        if not is_valid:
            print(f"❌ Invalid conversation: {error}\n")
            return

        print("✓ Conversation is valid\n")

        # Show the conversation structure
        print("Conversation Structure:")
        print("-" * 100)
        for i, msg in enumerate(conversation["messages"]):
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                content_preview = content[:100] + "..." if len(content) > 100 else content
                print(f"  {i+1}. [{role.upper()}]: {content_preview}")
            else:
                print(f"  {i+1}. [{role.upper()}]:")
                for j, part in enumerate(content):
                    text_preview = part["text"][:80] + "..." if len(part["text"]) > 80 else part["text"]
                    print(f"       - {part['type']}: {text_preview}")
        print()

        # Tokenize
        try:
            ids, mask = self.tokenizer.render_conversation(conversation, max_tokens=max_tokens)
        except Exception as e:
            print(f"❌ Error tokenizing conversation: {e}\n")
            return

        # Show statistics
        num_supervised = sum(mask)
        num_total = len(ids)
        print("Tokenization Statistics:")
        print("-" * 100)
        print(f"  Total tokens:       {num_total}")
        print(f"  Supervised tokens:  {num_supervised} (tokens the model learns from)")
        print(f"  Unsupervised:       {num_total - num_supervised} (prompt/context tokens)")
        print(f"  Supervision ratio:  {num_supervised / num_total * 100:.1f}%")
        print()

        # Show tokenized format with colors
        print("Tokenized Format:")
        print("-" * 100)
        print("(Green = supervised, Red = unsupervised)")
        print()
        visualization = self.tokenizer.visualize_tokenization(ids, mask, with_token_id=False)
        print(visualization)
        print("\n")

    def interactive_mode(self):
        """Interactive mode for building conversations."""
        print("\n" + "=" * 100)
        print("CONVERSATION TEMPLATE BUILDER - Interactive Mode")
        print("=" * 100 + "\n")
        print("Commands:")
        print("  'simple' - Create a simple 2-turn conversation")
        print("  'multi' - Create a multi-turn conversation")
        print("  'tool' - Create a conversation with tool usage")
        print("  'load <file>' - Load conversation from JSON file")
        print("  'save <file>' - Save last conversation to JSON file")
        print("  'validate <file>' - Validate a conversation JSON file")
        print("  'show' - Show last conversation")
        print("  'tokenize' - Tokenize and visualize last conversation")
        print("  'help' - Show this help message")
        print("  'quit' - Exit")
        print()

        last_conversation = None

        while True:
            try:
                user_input = input("> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  'simple' - Create a simple 2-turn conversation")
                    print("  'multi' - Create a multi-turn conversation")
                    print("  'tool' - Create a conversation with tool usage")
                    print("  'load <file>' - Load conversation from JSON file")
                    print("  'save <file>' - Save last conversation to JSON file")
                    print("  'validate <file>' - Validate a conversation JSON file")
                    print("  'show' - Show last conversation")
                    print("  'tokenize' - Tokenize and visualize last conversation")
                    print("  'quit' - Exit\n")
                    continue

                if user_input.lower() == 'simple':
                    print("\nCreating simple 2-turn conversation...")
                    user_msg = input("User message: ").strip()
                    if not user_msg:
                        print("Error: User message cannot be empty")
                        continue
                    assistant_msg = input("Assistant message: ").strip()
                    if not assistant_msg:
                        print("Error: Assistant message cannot be empty")
                        continue

                    last_conversation = self.create_simple_conversation(user_msg, assistant_msg)
                    print("\n✓ Conversation created!")
                    self.show_tokenization(last_conversation)
                    continue

                if user_input.lower() == 'multi':
                    print("\nCreating multi-turn conversation...")
                    print("Enter messages one at a time. Type 'done' when finished.")
                    print("Format: <role> <content>")
                    print("Example: user Hello there!")
                    print("         assistant Hi! How can I help?\n")

                    turns = []
                    while True:
                        turn_input = input(f"Turn {len(turns) + 1} (or 'done'): ").strip()
                        if turn_input.lower() == 'done':
                            break

                        # Parse role and content
                        parts = turn_input.split(maxsplit=1)
                        if len(parts) < 2:
                            print("Error: Must provide both role and content. Try again.")
                            continue

                        role, content = parts[0].lower(), parts[1]
                        if role not in ['user', 'assistant', 'system']:
                            print("Error: Role must be 'user', 'assistant', or 'system'. Try again.")
                            continue

                        turns.append((role, content))

                    if not turns:
                        print("Error: No turns provided")
                        continue

                    try:
                        last_conversation = self.create_multi_turn_conversation(turns)
                        print("\n✓ Conversation created!")
                        self.show_tokenization(last_conversation)
                    except ValueError as e:
                        print(f"\n❌ Error: {e}")
                    continue

                if user_input.lower() == 'tool':
                    print("\nCreating conversation with tool usage...")
                    user_msg = input("User message: ").strip()
                    if not user_msg:
                        print("Error: User message cannot be empty")
                        continue
                    python_code = input("Python code: ").strip()
                    if not python_code:
                        print("Error: Python code cannot be empty")
                        continue
                    python_output = input("Python output: ").strip()
                    if not python_output:
                        print("Error: Python output cannot be empty")
                        continue
                    assistant_msg = input("Assistant response: ").strip()
                    if not assistant_msg:
                        print("Error: Assistant response cannot be empty")
                        continue

                    last_conversation = self.create_tool_conversation(
                        user_msg, python_code, python_output, assistant_msg
                    )
                    print("\n✓ Conversation created!")
                    self.show_tokenization(last_conversation)
                    continue

                if user_input.lower().startswith('load '):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) < 2:
                        print("Error: Usage: load <filename>")
                        continue

                    filename = parts[1]
                    try:
                        with open(filename, 'r') as f:
                            last_conversation = json.load(f)
                        print(f"\n✓ Loaded conversation from {filename}")
                        self.show_tokenization(last_conversation)
                    except FileNotFoundError:
                        print(f"Error: File '{filename}' not found")
                    except json.JSONDecodeError as e:
                        print(f"Error: Invalid JSON in file: {e}")
                    except Exception as e:
                        print(f"Error loading file: {e}")
                    continue

                if user_input.lower().startswith('save '):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) < 2:
                        print("Error: Usage: save <filename>")
                        continue

                    if last_conversation is None:
                        print("Error: No conversation to save. Create one first.")
                        continue

                    filename = parts[1]
                    try:
                        with open(filename, 'w') as f:
                            json.dump(last_conversation, f, indent=2)
                        print(f"\n✓ Saved conversation to {filename}")
                    except Exception as e:
                        print(f"Error saving file: {e}")
                    continue

                if user_input.lower().startswith('validate '):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) < 2:
                        print("Error: Usage: validate <filename>")
                        continue

                    filename = parts[1]
                    try:
                        with open(filename, 'r') as f:
                            conversation = json.load(f)

                        is_valid, error = self.validate_conversation(conversation)
                        if is_valid:
                            print(f"\n✓ Conversation in '{filename}' is valid!")
                            self.show_tokenization(conversation)
                        else:
                            print(f"\n❌ Invalid conversation: {error}")
                    except FileNotFoundError:
                        print(f"Error: File '{filename}' not found")
                    except json.JSONDecodeError as e:
                        print(f"Error: Invalid JSON in file: {e}")
                    except Exception as e:
                        print(f"Error: {e}")
                    continue

                if user_input.lower() == 'show':
                    if last_conversation is None:
                        print("Error: No conversation to show. Create one first.")
                        continue

                    print("\n" + "=" * 100)
                    print("CONVERSATION")
                    print("=" * 100 + "\n")
                    print(json.dumps(last_conversation, indent=2))
                    print()
                    continue

                if user_input.lower() == 'tokenize':
                    if last_conversation is None:
                        print("Error: No conversation to tokenize. Create one first.")
                        continue

                    self.show_tokenization(last_conversation)
                    continue

                print(f"Unknown command: '{user_input}'. Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def batch_validate(self, input_file: str, output_file: Optional[str] = None):
        """
        Validate multiple conversations from a JSONL file.

        Args:
            input_file: Input JSONL file with one conversation per line
            output_file: Optional output file for validation report
        """
        print(f"\n{'='*100}")
        print(f"BATCH VALIDATION: {input_file}")
        print(f"{'='*100}\n")

        try:
            with open(input_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"❌ Error: File '{input_file}' not found\n")
            return
        except Exception as e:
            print(f"❌ Error reading file: {e}\n")
            return

        results = []
        valid_count = 0
        invalid_count = 0

        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            try:
                conversation = json.loads(line)
                is_valid, error = self.validate_conversation(conversation)

                result = {
                    "line": i,
                    "valid": is_valid,
                    "error": error
                }
                results.append(result)

                if is_valid:
                    valid_count += 1
                    print(f"  Line {i}: ✓ Valid")
                else:
                    invalid_count += 1
                    print(f"  Line {i}: ❌ Invalid - {error}")

            except json.JSONDecodeError as e:
                invalid_count += 1
                print(f"  Line {i}: ❌ JSON Error - {e}")
                results.append({
                    "line": i,
                    "valid": False,
                    "error": f"JSON decode error: {e}"
                })

        # Summary
        print(f"\n{'-'*100}")
        print(f"Summary:")
        print(f"  Total: {len(results)}")
        print(f"  Valid: {valid_count}")
        print(f"  Invalid: {invalid_count}")
        print()

        # Save report if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump({
                        "summary": {
                            "total": len(results),
                            "valid": valid_count,
                            "invalid": invalid_count
                        },
                        "results": results
                    }, f, indent=2)
                print(f"✓ Validation report saved to {output_file}\n")
            except Exception as e:
                print(f"❌ Error saving report: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Conversation Template Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python tools/conversation_template_builder.py --interactive

  # Create simple conversation
  python tools/conversation_template_builder.py --simple "What is 2+2?" "2+2 equals 4."

  # Validate a conversation file
  python tools/conversation_template_builder.py --validate conversation.json

  # Batch validate JSONL file
  python tools/conversation_template_builder.py --batch-validate dataset.jsonl
        """
    )

    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Interactive mode for building conversations")
    parser.add_argument("--simple", nargs=2, metavar=("USER_MSG", "ASSISTANT_MSG"),
                       help="Create a simple 2-turn conversation")
    parser.add_argument("--validate", metavar="FILE",
                       help="Validate a conversation JSON file")
    parser.add_argument("--batch-validate", metavar="FILE",
                       help="Validate conversations from JSONL file")
    parser.add_argument("--output", "-o", metavar="FILE",
                       help="Output file for batch validation report")

    args = parser.parse_args()

    builder = ConversationTemplateBuilder()

    if args.interactive:
        builder.interactive_mode()
    elif args.simple:
        user_msg, assistant_msg = args.simple
        conversation = builder.create_simple_conversation(user_msg, assistant_msg)
        builder.show_tokenization(conversation)
    elif args.validate:
        try:
            with open(args.validate, 'r') as f:
                conversation = json.load(f)

            is_valid, error = builder.validate_conversation(conversation)
            if is_valid:
                print(f"\n✓ Conversation is valid!")
                builder.show_tokenization(conversation)
            else:
                print(f"\n❌ Invalid conversation: {error}\n")
        except FileNotFoundError:
            print(f"Error: File '{args.validate}' not found")
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON: {e}")
        except Exception as e:
            print(f"Error: {e}")
    elif args.batch_validate:
        builder.batch_validate(args.batch_validate, args.output)
    else:
        # Default: show help
        parser.print_help()


if __name__ == "__main__":
    main()
