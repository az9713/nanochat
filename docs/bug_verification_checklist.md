# Bug Verification Checklist

A comprehensive checklist for verifying code against 8 bug types. Use this checklist for every feature implementation to ensure thorough verification.

## Quick Reference

1. ✅ **Functional Bugs** - Core features work correctly
2. ✅ **Logical Bugs** - Logic and control flow are sound
3. ✅ **Workflow Bugs** - User experience is intuitive
4. ✅ **Unit-Level Bugs** - Edge cases and pattern consistency
5. ✅ **System-Level Integration Bugs** - API usage is correct
6. ✅ **Out of Bound Bugs** - Array access and guard logic are safe
7. ✅ **Security Bugs** - No injection or unsafe operations
8. ✅ **Guard Logic Bugs** - Validation logic is correct (NEW)

---

## Bug Type #1: Functional Bugs

**Definition:** Core functionality does not work as intended.

### Verification Checklist

- [ ] **List all intended features** from specification
- [ ] **Test each feature independently**
  - [ ] Feature 1: ___________________ → Works? ☐
  - [ ] Feature 2: ___________________ → Works? ☐
  - [ ] Feature 3: ___________________ → Works? ☐
- [ ] **Verify return values** are correct type and content
- [ ] **Check state changes** are persisted correctly
- [ ] **Test feature interactions** (does feature A break feature B?)

### Example
```python
# Function: create_simple_conversation()
✓ Creates dictionary with "messages" key
✓ Creates exactly 2 messages
✓ First message has role="user"
✓ Second message has role="assistant"
✓ Content fields populated correctly
```

---

## Bug Type #2: Logical Bugs

**Definition:** Logic errors in conditionals, loops, and control flow.

### Verification Checklist

- [ ] **Trace all conditional branches**
  ```python
  if condition:
      # What happens here? Document it
  else:
      # What happens here? Document it
  # What happens after? Document it
  ```

- [ ] **Check for implicit else cases**
  ```python
  if condition:
      do_something()
  # ← DANGER: What if condition is false?
  # Is there missing logic here?
  ```

- [ ] **Verify loop conditions**
  - [ ] Does loop start at correct index?
  - [ ] Does loop end at correct index?
  - [ ] Can loop become infinite?

- [ ] **Check boolean logic**
  - [ ] Are AND/OR operators correct?
  - [ ] Is negation (not) correct?
  - [ ] De Morgan's laws applied correctly?

- [ ] **Verify alternation/state machine logic**
  - [ ] States transition correctly?
  - [ ] Invalid transitions rejected?

### Example - The Bug We Caught
```python
# BUGGY LOGIC:
if i == 1 and len(messages) > 0 and messages[0]["role"] == "system":
    if role != "user":
        raise ValueError(...)
elif i > 0:
    # Check alternation...

# Problem: Complex conditions, easy to miss edge cases
# Solution: Simplify, use start_idx pattern
```

---

## Bug Type #3: Workflow Bugs

**Definition:** User interaction is confusing, unclear, or broken.

### Verification Checklist

- [ ] **Error messages are clear and actionable**
  - ❌ Bad: "Error"
  - ✓ Good: "Error: User message cannot be empty"

- [ ] **Help text is complete**
  - [ ] All commands documented?
  - [ ] Usage examples provided?

- [ ] **User feedback is immediate**
  - [ ] Success messages shown?
  - [ ] Progress indicators for long operations?

- [ ] **Command parsing is robust**
  - [ ] Handles missing arguments?
  - [ ] Handles extra whitespace?
  - [ ] Case-insensitive where appropriate?

- [ ] **Flow makes sense**
  - [ ] Can user accomplish task in logical steps?
  - [ ] Are there "dead ends" where user gets stuck?

### Example
```python
# Good workflow:
> load file.json
Error: File 'file.json' not found
Hint: Use 'load <filename>' with an existing file

# Bad workflow:
> load file.json
FileNotFoundError: [Errno 2] No such file or directory: 'file.json'
```

---

## Bug Type #4: Unit-Level Bugs

**Definition:** Edge cases, boundary values, and pattern inconsistencies.

### Verification Checklist

#### Part A: Edge Cases
- [ ] **Empty inputs**
  - [ ] Empty string: ""
  - [ ] Empty list: []
  - [ ] Empty dict: {}

- [ ] **None values**
  - [ ] What if parameter is None?
  - [ ] What if dict value is None?

- [ ] **Boundary values**
  - [ ] Zero, negative numbers
  - [ ] Very large numbers
  - [ ] Max length strings/arrays

- [ ] **Invalid types**
  - [ ] String where int expected?
  - [ ] List where string expected?

#### Part B: Pattern Consistency

- [ ] **Pattern search completed**
  ```bash
  # When you fix a bug, search for similar patterns:
  grep -n "array\[i\]" file.py
  grep -n "messages\[1\]" file.py
  grep -n "start_idx" file.py
  ```

- [ ] **All instances checked**
  - [ ] Instance 1: Line ___ → Fixed? ☐
  - [ ] Instance 2: Line ___ → Fixed? ☐
  - [ ] Instance 3: Line ___ → Fixed? ☐

- [ ] **Consistent error handling**
  - [ ] Same error type used everywhere?
  - [ ] Same error message format?

#### Part C: Function-Level Testing

- [ ] **Test each function independently**
  - [ ] High-level function: `create_X()` → Tested? ☐
  - [ ] Low-level function: `validate_X()` → Tested? ☐
  - [ ] Helper function: `_internal_X()` → Tested? ☐

### Example - Pattern Search
```python
# Bug found in create_multi_turn_conversation() line 93
# MUST search for similar pattern:

$ grep -n "messages\[1\]" conversation_template_builder.py
93:    first_real_msg = messages[1]  # ← Bug #1 (fixed)
199:   if messages[start_idx]...      # ← Bug #2 (found via search!)

# Fix BOTH instances!
```

---

## Bug Type #5: System-Level Integration Bugs

**Definition:** Incorrect API usage, missing error handling, wrong assumptions about external systems.

### Verification Checklist

- [ ] **API contracts verified**
  - [ ] Function signature correct?
  - [ ] All required parameters passed?
  - [ ] Optional parameters understood?
  - [ ] Return type handled correctly?

- [ ] **Return values checked**
  ```python
  result = external_function()
  # Don't assume success!
  if result is None:
      # Handle error
  ```

- [ ] **Exceptions caught appropriately**
  - [ ] FileNotFoundError for file operations?
  - [ ] JSONDecodeError for JSON parsing?
  - [ ] ValueError for invalid inputs?

- [ ] **External dependencies available**
  - [ ] Import statements work?
  - [ ] Optional dependencies handled gracefully?

- [ ] **File operations are safe**
  - [ ] Use context managers (`with open(...)`)
  - [ ] Handle permission errors
  - [ ] Handle disk full errors

### Example
```python
# GOOD: Proper API usage
try:
    ids, mask = self.tokenizer.render_conversation(
        conversation,
        max_tokens=max_tokens  # ← Correct parameter name
    )
except Exception as e:
    print(f"Error tokenizing: {e}")
    return

# BAD: Wrong parameter name
ids, mask = self.tokenizer.render_conversation(
    conversation,
    max_len=max_tokens  # ← Wrong! Should be max_tokens
)
```

---

## Bug Type #6: Out of Bound Bugs & Guard Logic Errors

**Definition:** Array accesses without proper bounds checking OR bounds checks with incorrect logic.

### Verification Checklist

#### Part A: Find All Array Accesses
```bash
# Search for array/list/dict accesses:
grep -n "\[0\]" file.py
grep -n "\[1\]" file.py
grep -n "\[i\]" file.py
grep -n "\[.*\]" file.py  # All bracket accesses
```

#### Part B: For Each Access, Verify

- [ ] **Access: `array[index]` at line ___**

  - [ ] **Step 1: Bounds check exists?**
    - YES → Proceed to Step 2
    - NO → ADD bounds check

  - [ ] **Step 2: Trace both branches**
    ```python
    # Example:
    if start_idx < len(messages):
        # TRUE branch: What happens?
        validate(messages[start_idx])
    # FALSE branch: What happens?
    # ← Nothing! Validation skipped! BUG!
    ```

  - [ ] **Step 3: Logic direction correct?**
    - [ ] Rejects invalid: `if idx >= len(arr): return error` ✓
    - [ ] Skips validation: `if idx < len(arr): validate` ❌

  - [ ] **Step 4: Comparison operator correct?**
    - [ ] `<` vs `<=`
    - [ ] `>` vs `>=`
    - [ ] `==` vs `!=`

  - [ ] **Step 5: Both cases handled explicitly?**
    ```python
    # GOOD: Both cases explicit
    if idx >= len(arr):
        return False, "Error"  # Out of bounds → reject
    # Now safe to access arr[idx]

    # BAD: Implicit else
    if idx < len(arr):
        validate(arr[idx])
    # ← What happens when idx >= len(arr)? Not clear!
    ```

  - [ ] **Step 6: Test coverage?**
    - [ ] Test when index is valid
    - [ ] Test when index is out of bounds

#### Part C: Special Cases

- [ ] **Dictionary access: `dict[key]`**
  - [ ] Use `dict.get(key, default)` instead?
  - [ ] Or check `if key in dict`?

- [ ] **Negative indices: `array[-1]`**
  - [ ] Intentional (last element)?
  - [ ] Or potential bug?

- [ ] **Slice operations: `array[start:end]`**
  - [ ] Can start/end be out of bounds?
  - [ ] Empty slice handled?

### Example - The Bug We Missed

```python
# Location: validate_conversation(), line 198

# BUGGY CODE:
if start_idx < len(messages):  # ← Logic inverted!
    if messages[start_idx]["role"] != "user":
        return False, "Error"
# When start_idx >= len(messages):
#   - No error returned
#   - Validation skipped
#   - Later code crashes with IndexError

# VERIFICATION TRACE:
# start_idx < len(messages) = TRUE:  Validates messages[start_idx] ✓
# start_idx < len(messages) = FALSE: Does nothing, falls through ❌

# FIXED CODE:
if start_idx >= len(messages):  # ← Correct logic!
    return False, "Error"  # Explicitly reject
# Now when start_idx >= len(messages), error is returned
```

---

## Bug Type #7: Security Bugs

**Definition:** Code vulnerable to injection, path traversal, or malicious input.

### Verification Checklist

- [ ] **No command injection**
  - [ ] No `eval()` on user input
  - [ ] No `exec()` on user input
  - [ ] No `subprocess` with shell=True and user input
  - [ ] No `os.system()` with user input

- [ ] **No SQL injection**
  - [ ] Use parameterized queries
  - [ ] Never string concatenation for SQL

- [ ] **No path traversal**
  - [ ] Validate file paths
  - [ ] No `../` in user-provided paths
  - [ ] Use `os.path.abspath()` and check prefix

- [ ] **Safe deserialization**
  - [ ] Use `json.load()` not `pickle.load()` for untrusted data
  - [ ] Validate JSON structure after loading

- [ ] **Input sanitization**
  - [ ] HTML escaped if displayed in browser?
  - [ ] Special characters handled?

- [ ] **File operations safe**
  - [ ] Don't follow symlinks if security-sensitive
  - [ ] Check file permissions
  - [ ] Don't overwrite system files

### Example
```python
# GOOD: Safe JSON parsing
try:
    data = json.load(f)
    is_valid, error = validate_structure(data)
    if not is_valid:
        return error
except json.JSONDecodeError as e:
    return f"Invalid JSON: {e}"

# BAD: Unsafe deserialization
import pickle
data = pickle.load(f)  # ← Can execute arbitrary code!
```

---

## Bug Type #8: Guard Logic & Control Flow Bugs (NEW)

**Definition:** Validation logic that is inverted, incomplete, or has implicit else cases that lead to undefined behavior.

### Verification Checklist

#### Part A: Find All Guard Statements
```bash
# Search for guards:
grep -n "if.*:" file.py
grep -n "elif.*:" file.py
grep -n "try:" file.py
grep -n "assert" file.py
```

#### Part B: For Each Guard, Verify

- [ ] **Guard statement at line ___**

  - [ ] **Step 1: What is being guarded?**
    ```python
    if condition:
        # This code is guarded
    ```

  - [ ] **Step 2: Write out TRUE branch**
    ```
    When condition is TRUE:
    - Does: _______________
    - Returns: _______________
    - Side effects: _______________
    ```

  - [ ] **Step 3: Write out FALSE branch**
    ```
    When condition is FALSE:
    - Does: _______________
    - Returns: _______________
    - Side effects: _______________
    ```

  - [ ] **Step 4: Is FALSE branch handled?**
    - Explicit else: ✓
    - Implicit (nothing): ❌ DANGER!
    - Return/raise in TRUE: ✓

  - [ ] **Step 5: Check for validation skipping**
    ```python
    # Pattern A: GOOD (rejects bad cases)
    if is_invalid:
        return error
    # Safe to proceed

    # Pattern B: BAD (skips validation)
    if is_valid:
        validate()
    # ← What if invalid? No error!
    ```

  - [ ] **Step 6: Verify condition logic**
    - [ ] Should be `<` or `<=`?
    - [ ] Should be `>` or `>=`?
    - [ ] Should be `==` or `!=`?
    - [ ] Should be `and` or `or`?
    - [ ] Should be negated with `not`?

#### Part C: Common Anti-Patterns

- [ ] **Anti-pattern 1: Validation only in happy path**
  ```python
  # BAD:
  if valid_case:
      validate_and_process()
  # ← Invalid case not handled!

  # GOOD:
  if invalid_case:
      return error
  validate_and_process()
  ```

- [ ] **Anti-pattern 2: Early return without validation**
  ```python
  # BAD:
  if special_case:
      return quick_result  # ← Skipped validation!

  validate()
  return normal_result

  # GOOD:
  validate()  # Always validate first
  if special_case:
      return quick_result
  return normal_result
  ```

- [ ] **Anti-pattern 3: Inverted condition**
  ```python
  # BAD:
  if idx < len(arr):  # ← Handles valid case
      validate(arr[idx])
  # Invalid case falls through!

  # GOOD:
  if idx >= len(arr):  # ← Rejects invalid case
      return error
  validate(arr[idx])  # Safe
  ```

### Example - Guard Logic Bug

```python
# BUG: Validation skipped for out-of-bounds

# BUGGY:
if start_idx < len(messages):
    # Valid case: validate
    if messages[start_idx]["role"] != "user":
        return False, "Error"
# Invalid case: no error returned, falls through

# ANALYSIS:
# TRUE (start_idx < len):  Validates ✓
# FALSE (start_idx >= len): Does nothing ❌
# Result: Out-of-bounds case not rejected

# FIXED:
if start_idx >= len(messages):
    # Invalid case: reject immediately
    return False, "Error"
# Valid case: safe to validate
if messages[start_idx]["role"] != "user":
    return False, "Error"

# ANALYSIS:
# TRUE (start_idx >= len):  Returns error ✓
# FALSE (start_idx < len):  Validates ✓
# Result: Both cases handled correctly
```

---

## Complete Verification Workflow

Use this workflow for every feature implementation:

### Phase 1: Implementation
1. ✅ Write code
2. ✅ Write tests
3. ✅ Run tests

### Phase 2: Initial Verification (Quick Pass)
1. ✅ Bug Type #1: Does it work?
2. ✅ Bug Type #3: Is UX good?
3. ✅ Bug Type #7: Any obvious security issues?

### Phase 3: Deep Verification (Comprehensive)

For each function in the implementation:

1. **Read function completely**
2. **List all array accesses** → Check Bug Type #6
3. **List all guard statements** → Check Bug Type #8
4. **Trace all branches** → Check Bug Type #2
5. **List all external calls** → Check Bug Type #5
6. **Check edge cases** → Check Bug Type #4
7. **Pattern search for similar code** → Check Bug Type #4

### Phase 4: Cross-File Verification

1. **Search for patterns across ALL files**
   ```bash
   grep -r "pattern" tools/
   ```

2. **Check for consistency**
   - Same error messages?
   - Same validation logic?
   - Same error types?

### Phase 5: Test Coverage

1. **For each guard statement, ensure test exists**
   - Guard at line 91: Test exists? ☐
   - Guard at line 198: Test exists? ☐

2. **For each edge case, ensure test exists**
   - Empty input: Test exists? ☐
   - Boundary value: Test exists? ☐

### Phase 6: Documentation

1. ✅ Document all findings
2. ✅ Update test cases
3. ✅ Add comments for tricky logic

---

## Bug Priority Matrix

When bugs are found, prioritize fixes:

| Bug Type | Severity | Priority |
|----------|----------|----------|
| #7 Security | Critical | P0 - Fix immediately |
| #6 Out of Bounds (crashes) | Critical | P0 - Fix immediately |
| #8 Guard Logic (data corruption) | High | P1 - Fix before merge |
| #1 Functional (broken feature) | High | P1 - Fix before merge |
| #2 Logical (wrong results) | High | P1 - Fix before merge |
| #5 Integration (API misuse) | Medium | P2 - Fix in PR review |
| #4 Unit-Level (edge cases) | Medium | P2 - Fix in PR review |
| #3 Workflow (confusing UX) | Low | P3 - Fix when convenient |

---

## Real-World Example: Full Verification

### Code Under Review
```python
def validate_conversation(self, conversation):
    messages = conversation["messages"]
    start_idx = 1 if messages[0]["role"] == "system" else 0

    if start_idx < len(messages):
        if messages[start_idx]["role"] != "user":
            return False, "Error"

    return True, None
```

### Verification Checklist Applied

#### Bug Type #6: Out of Bounds
- [ ] Access: `messages[0]` line 2
  - Empty list handled? → Need to check len first ❌

- [ ] Access: `messages[start_idx]` line 5
  - Bounds check exists? → Yes, line 4 ✓
  - Logic correct? → **TRACE BRANCHES**
    - TRUE (start_idx < len): Validates ✓
    - FALSE (start_idx >= len): Does nothing ❌
  - Both branches explicit? → No! ❌

#### Bug Type #8: Guard Logic
- [ ] Guard: `if start_idx < len(messages)` line 4
  - TRUE branch: Validates messages[start_idx]
  - FALSE branch: Falls through, returns True
  - **PROBLEM**: Invalid case returns True! ❌

#### Bug Type #4: Pattern Search
```bash
$ grep -n "messages\[0\]" file.py
2: messages[0]["role"]  # No len check!

$ grep -n "start_idx" file.py
3: start_idx = ...
4: if start_idx < len...  # Logic inverted!
```

### Bugs Found: 2

1. **Critical**: Line 4 - Guard logic inverted, skips validation
2. **High**: Line 2 - No empty list check before messages[0]

### Fixes Applied
```python
def validate_conversation(self, conversation):
    messages = conversation["messages"]

    # Fix #2: Check for empty list
    if len(messages) == 0:
        return False, "Conversation must have at least one message"

    start_idx = 1 if messages[0]["role"] == "system" else 0

    # Fix #1: Invert logic to reject invalid case
    if start_idx >= len(messages):
        return False, "Conversation with system message must have at least one user/assistant message"

    if messages[start_idx]["role"] != "user":
        return False, "First message must be from user"

    return True, None
```

---

## Conclusion

**Key Principles:**

1. **Guards must reject, not skip** - Check for invalid cases explicitly
2. **Trace both branches** - Write out what happens in true/false cases
3. **Search for patterns** - If you fix one, search for all instances
4. **Test both paths** - Test valid AND invalid cases
5. **Be explicit** - Avoid implicit else cases

**Remember:** The goal is not just to have guards, but to have **correct** guards that actually protect against bugs.

---

## Revision History

- v1.0 (2025-11-10): Initial version with 8 bug types
- Added Bug Type #8: Guard Logic & Control Flow Bugs
- Enhanced Bug Type #6: Added guard logic verification
- Enhanced Bug Type #4: Added pattern consistency checks
