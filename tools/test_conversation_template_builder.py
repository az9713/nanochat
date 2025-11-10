"""
Test script for Conversation Template Builder

Tests basic functionality of the conversation template builder tool.
"""

import os
import sys
import json
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_imports():
    """Test that all required modules can be imported."""
    print("="*80)
    print("TEST: Import conversation_template_builder module")
    print("="*80)

    try:
        from tools.conversation_template_builder import ConversationTemplateBuilder
        print("✓ Successfully imported ConversationTemplateBuilder class")
        return True
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False


def test_tokenizer_availability():
    """Test if tokenizer can be loaded."""
    print("\n" + "="*80)
    print("TEST: Tokenizer availability")
    print("="*80)

    try:
        from nanochat.tokenizer import get_tokenizer
        tokenizer = get_tokenizer()
        print(f"✓ Tokenizer loaded successfully")
        print(f"  Vocabulary size: {tokenizer.vocab_size}")
        return True
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return False


def test_simple_conversation():
    """Test creating a simple 2-turn conversation."""
    print("\n" + "="*80)
    print("TEST: Simple conversation creation")
    print("="*80)

    from tools.conversation_template_builder import ConversationTemplateBuilder

    builder = ConversationTemplateBuilder()

    # Create simple conversation
    conversation = builder.create_simple_conversation(
        "What is 2+2?",
        "2+2 equals 4."
    )

    # Validate structure
    assert "messages" in conversation, "Missing 'messages' key"
    assert len(conversation["messages"]) == 2, f"Expected 2 messages, got {len(conversation['messages'])}"
    assert conversation["messages"][0]["role"] == "user", "First message should be from user"
    assert conversation["messages"][1]["role"] == "assistant", "Second message should be from assistant"

    print("✓ Simple conversation created successfully")
    print(f"  User message: {conversation['messages'][0]['content']}")
    print(f"  Assistant message: {conversation['messages'][1]['content']}")

    return True


def test_multi_turn_conversation():
    """Test creating multi-turn conversations."""
    print("\n" + "="*80)
    print("TEST: Multi-turn conversation creation")
    print("="*80)

    from tools.conversation_template_builder import ConversationTemplateBuilder

    builder = ConversationTemplateBuilder()

    # Test valid 4-turn conversation
    turns = [
        ("user", "Hello!"),
        ("assistant", "Hi there! How can I help you?"),
        ("user", "What's the weather like?"),
        ("assistant", "I don't have access to weather information.")
    ]

    conversation = builder.create_multi_turn_conversation(turns)

    assert len(conversation["messages"]) == 4, f"Expected 4 messages, got {len(conversation['messages'])}"
    print("✓ Valid 4-turn conversation created")

    # Test with system message
    turns_with_system = [
        ("system", "You are a helpful assistant."),
        ("user", "Hello!"),
        ("assistant", "Hi there!")
    ]

    conversation = builder.create_multi_turn_conversation(turns_with_system)
    assert len(conversation["messages"]) == 3, f"Expected 3 messages, got {len(conversation['messages'])}"
    assert conversation["messages"][0]["role"] == "system", "First message should be system"
    print("✓ Conversation with system message created")

    # Test invalid: wrong alternation
    try:
        invalid_turns = [
            ("user", "Hello!"),
            ("user", "Another user message")  # Wrong! Should be assistant
        ]
        builder.create_multi_turn_conversation(invalid_turns)
        print("✗ Failed to catch invalid alternation")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected invalid alternation: {e}")

    # Test invalid: assistant first
    try:
        invalid_turns = [
            ("assistant", "Hello!")  # Wrong! Should start with user
        ]
        builder.create_multi_turn_conversation(invalid_turns)
        print("✗ Failed to catch assistant-first message")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected assistant-first message: {e}")

    # Test invalid: empty turns
    try:
        builder.create_multi_turn_conversation([])
        print("✗ Failed to catch empty turns")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected empty turns: {e}")

    # Test invalid: only system message (no user/assistant)
    try:
        invalid_turns = [
            ("system", "You are helpful")  # Only system, no actual conversation
        ]
        builder.create_multi_turn_conversation(invalid_turns)
        print("✗ Failed to catch system-only conversation")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected system-only conversation: {e}")

    return True


def test_tool_conversation():
    """Test creating conversations with tool usage."""
    print("\n" + "="*80)
    print("TEST: Tool conversation creation")
    print("="*80)

    from tools.conversation_template_builder import ConversationTemplateBuilder

    builder = ConversationTemplateBuilder()

    conversation = builder.create_tool_conversation(
        "Calculate 15 * 23",
        "15 * 23",
        "345",
        "The result is 345."
    )

    # Check structure
    assert len(conversation["messages"]) == 2, f"Expected 2 messages, got {len(conversation['messages'])}"
    assert conversation["messages"][0]["role"] == "user"
    assert conversation["messages"][1]["role"] == "assistant"

    # Check assistant content is a list
    assistant_content = conversation["messages"][1]["content"]
    assert isinstance(assistant_content, list), "Assistant content should be a list"
    assert len(assistant_content) == 3, f"Expected 3 parts, got {len(assistant_content)}"

    # Check part types
    assert assistant_content[0]["type"] == "python"
    assert assistant_content[1]["type"] == "python_output"
    assert assistant_content[2]["type"] == "text"

    print("✓ Tool conversation created successfully")
    print(f"  Python code: {assistant_content[0]['text']}")
    print(f"  Python output: {assistant_content[1]['text']}")
    print(f"  Final response: {assistant_content[2]['text']}")

    return True


def test_validation():
    """Test conversation validation."""
    print("\n" + "="*80)
    print("TEST: Conversation validation")
    print("="*80)

    from tools.conversation_template_builder import ConversationTemplateBuilder

    builder = ConversationTemplateBuilder()

    # Valid conversation
    valid_conversation = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
    }
    is_valid, error = builder.validate_conversation(valid_conversation)
    assert is_valid, f"Valid conversation rejected: {error}"
    print("✓ Valid conversation accepted")

    # Invalid: not a dict
    is_valid, error = builder.validate_conversation([])
    assert not is_valid, "List should be rejected"
    print(f"✓ Non-dict rejected: {error}")

    # Invalid: missing 'messages'
    is_valid, error = builder.validate_conversation({"data": []})
    assert not is_valid, "Missing 'messages' should be rejected"
    print(f"✓ Missing 'messages' rejected: {error}")

    # Invalid: empty messages
    is_valid, error = builder.validate_conversation({"messages": []})
    assert not is_valid, "Empty messages should be rejected"
    print(f"✓ Empty messages rejected: {error}")

    # Invalid: wrong role
    invalid_conversation = {
        "messages": [
            {"role": "invalid_role", "content": "Hello"}
        ]
    }
    is_valid, error = builder.validate_conversation(invalid_conversation)
    assert not is_valid, "Invalid role should be rejected"
    print(f"✓ Invalid role rejected: {error}")

    # Invalid: user content not string
    invalid_conversation = {
        "messages": [
            {"role": "user", "content": ["not", "a", "string"]}
        ]
    }
    is_valid, error = builder.validate_conversation(invalid_conversation)
    assert not is_valid, "User content as list should be rejected"
    print(f"✓ Invalid user content type rejected: {error}")

    # Invalid: wrong alternation
    invalid_conversation = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Hello again"}  # Should be assistant
        ]
    }
    is_valid, error = builder.validate_conversation(invalid_conversation)
    assert not is_valid, "Wrong alternation should be rejected"
    print(f"✓ Wrong alternation rejected: {error}")

    # Invalid: system-only conversation (no user/assistant messages)
    invalid_conversation = {
        "messages": [
            {"role": "system", "content": "You are helpful"}
        ]
    }
    is_valid, error = builder.validate_conversation(invalid_conversation)
    assert not is_valid, "System-only conversation should be rejected"
    print(f"✓ System-only conversation rejected: {error}")

    return True


def test_tokenization():
    """Test conversation tokenization."""
    print("\n" + "="*80)
    print("TEST: Conversation tokenization")
    print("="*80)

    from tools.conversation_template_builder import ConversationTemplateBuilder

    builder = ConversationTemplateBuilder()

    conversation = builder.create_simple_conversation(
        "What is 2+2?",
        "The answer is 4."
    )

    print("\nTokenizing conversation...")
    try:
        builder.show_tokenization(conversation)
        print("✓ Tokenization successful")
        return True
    except Exception as e:
        print(f"✗ Tokenization failed: {e}")
        return False


def test_file_operations():
    """Test saving and loading conversations."""
    print("\n" + "="*80)
    print("TEST: File operations")
    print("="*80)

    from tools.conversation_template_builder import ConversationTemplateBuilder

    builder = ConversationTemplateBuilder()

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")

    try:
        # Create a conversation
        conversation = builder.create_simple_conversation(
            "Test question",
            "Test answer"
        )

        # Save to file
        test_file = os.path.join(temp_dir, "test_conversation.json")
        with open(test_file, 'w') as f:
            json.dump(conversation, f, indent=2)
        print(f"✓ Saved conversation to {test_file}")

        # Load from file
        with open(test_file, 'r') as f:
            loaded_conversation = json.load(f)

        # Validate loaded conversation
        is_valid, error = builder.validate_conversation(loaded_conversation)
        assert is_valid, f"Loaded conversation is invalid: {error}"
        print("✓ Loaded and validated conversation from file")

        # Test batch validation with JSONL
        jsonl_file = os.path.join(temp_dir, "test_batch.jsonl")
        with open(jsonl_file, 'w') as f:
            # Write 3 conversations
            for i in range(3):
                conv = builder.create_simple_conversation(f"Question {i}", f"Answer {i}")
                f.write(json.dumps(conv) + "\n")

        print(f"✓ Created JSONL file with 3 conversations")

        # Test batch validation
        report_file = os.path.join(temp_dir, "validation_report.json")
        builder.batch_validate(jsonl_file, report_file)

        # Check report was created
        assert os.path.exists(report_file), "Validation report not created"
        with open(report_file, 'r') as f:
            report = json.load(f)

        assert report["summary"]["total"] == 3, f"Expected 3 conversations, got {report['summary']['total']}"
        assert report["summary"]["valid"] == 3, f"Expected all valid, got {report['summary']['valid']}"
        print("✓ Batch validation successful")

        return True

    except Exception as e:
        print(f"✗ File operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*80)
    print("TEST: Edge cases and error handling")
    print("="*80)

    from tools.conversation_template_builder import ConversationTemplateBuilder

    builder = ConversationTemplateBuilder()

    # Test with very long content
    long_content = "A" * 10000
    conversation = builder.create_simple_conversation("Question", long_content)
    is_valid, error = builder.validate_conversation(conversation)
    assert is_valid, f"Long content should be valid: {error}"
    print("✓ Long content handled correctly")

    # Test with special characters
    special_content = "Hello! 你好 مرحبا שלום 🎉 \n\t\r"
    conversation = builder.create_simple_conversation("Question", special_content)
    is_valid, error = builder.validate_conversation(conversation)
    assert is_valid, f"Special characters should be valid: {error}"
    print("✓ Special characters handled correctly")

    # Test with empty strings (should be valid but might tokenize to nothing)
    empty_conversation = builder.create_simple_conversation("", "")
    is_valid, error = builder.validate_conversation(empty_conversation)
    assert is_valid, f"Empty strings should be structurally valid: {error}"
    print("✓ Empty strings handled correctly")

    # Test system message not at start
    try:
        turns = [
            ("user", "Hello"),
            ("system", "You are helpful")  # System not at start
        ]
        builder.create_multi_turn_conversation(turns)
        print("✗ Failed to catch system message not at start")
        return False
    except ValueError as e:
        print(f"✓ System message not at start rejected: {e}")

    # Test invalid role
    try:
        turns = [
            ("invalid_role", "Hello")
        ]
        builder.create_multi_turn_conversation(turns)
        print("✗ Failed to catch invalid role")
        return False
    except ValueError as e:
        print(f"✓ Invalid role rejected: {e}")

    # Test batch validation with non-existent file
    print("\nTesting non-existent file...")
    builder.batch_validate("/nonexistent/file.jsonl")
    print("✓ Non-existent file handled gracefully")

    # Test validation with malformed JSON
    temp_dir = tempfile.mkdtemp()
    try:
        malformed_file = os.path.join(temp_dir, "malformed.jsonl")
        with open(malformed_file, 'w') as f:
            f.write('{"messages": [}\n')  # Malformed JSON

        builder.batch_validate(malformed_file)
        print("✓ Malformed JSON handled gracefully")

    finally:
        shutil.rmtree(temp_dir)

    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("TESTING CONVERSATION TEMPLATE BUILDER")
    print("="*80)
    print()

    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    # Test tokenizer
    results.append(("Tokenizer", test_tokenizer_availability()))

    # Test functionality
    results.append(("Simple conversation", test_simple_conversation()))
    results.append(("Multi-turn conversation", test_multi_turn_conversation()))
    results.append(("Tool conversation", test_tool_conversation()))
    results.append(("Validation", test_validation()))
    results.append(("Tokenization", test_tokenization()))
    results.append(("File operations", test_file_operations()))
    results.append(("Edge cases", test_edge_cases()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30s} {status}")

    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)

    print(f"\nPassed: {passed_tests}/{total_tests}")

    if passed_tests == total_tests:
        print("\n✓ All tests passed!")
    else:
        print("\n⚠ Some tests failed")

    print("\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)
    print()
    print("# Interactive mode")
    print("python tools/conversation_template_builder.py --interactive")
    print()
    print("# Create simple conversation")
    print("python tools/conversation_template_builder.py --simple \"What is 2+2?\" \"It's 4.\"")
    print()
    print("# Validate a conversation file")
    print("python tools/conversation_template_builder.py --validate conversation.json")
    print()
    print("# Batch validate JSONL file")
    print("python tools/conversation_template_builder.py --batch-validate dataset.jsonl")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
