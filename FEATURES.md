# Nanochat Learning Fork - Additional Features

This is a learning-focused fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) designed for beginners who want to deeply understand LLMs and PyTorch. All additions focus on educational value and hands-on learning.

## About This Fork

**Original Repository**: [karpathy/nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy

**This Fork's Purpose**:
- Provide comprehensive beginner-friendly documentation
- Add practical tools for learning and experimentation
- Enable hands-on feature building to understand LLM internals
- Maintain simplicity and minimal dependencies

**Target Audience**: Complete beginners to LLMs and PyTorch who want to learn by doing.

## What's Added

### 📚 Comprehensive Documentation (`docs/`)

Eight detailed guides that teach LLMs from the ground up, assuming NO prior knowledge:

1. **Introduction to LLMs and PyTorch** - What are LLMs? What is PyTorch? How does ML work?
2. **Tokenization** - How text becomes numbers, BPE algorithm explained
3. **Architecture** - Complete Transformer architecture breakdown with code walkthroughs
4. **Training Pipeline** - How models learn, distributed training, optimizers
5. **Inference** - Text generation, sampling strategies, KV cache optimization
6. **Tools and Capabilities** - Calculator and code execution features
7. **Evaluation** - CORE score, benchmarks, metrics explained
8. **Quick Start Guide** - Installation, first training, troubleshooting
9. **Feature Implementation Guide** - 10 features you can build to learn

**Key Principle**: All documentation is self-contained. You should be able to understand the entire system using ONLY these docs and the code, with no external resources needed.

### 🛠️ Learning Tools (`tools/`)

Practical utilities for understanding model behavior and planning experiments:

#### ✅ Implemented Features

##### 1. Interactive Tokenizer Playground (`tokenizer_playground.py`)
Visualize and understand how text is tokenized into tokens.

**What it does:**
- Colorized visualization of tokens in your terminal
- Show detailed token information (IDs, byte counts, types)
- Display all special tokens used in conversations
- Compare tokenization efficiency of different texts
- Interactive mode for experimentation
- Vocabulary statistics and breakdown

**Why it's useful:**
- Understand how "Hello world" becomes token IDs
- See token boundaries visually with color-coding
- Learn about special tokens for chat and tool use
- Debug tokenization issues
- Optimize prompts for token efficiency
- Concrete understanding of BPE algorithm results

**Usage:**
```bash
# Tokenize a single text
python tools/tokenizer_playground.py "Hello world!"

# Interactive mode - experiment with different texts
python tools/tokenizer_playground.py --interactive
python tools/tokenizer_playground.py -i

# Show all special tokens
python tools/tokenizer_playground.py --special

# Show vocabulary information
python tools/tokenizer_playground.py --vocab

# Compare multiple texts
python tools/tokenizer_playground.py --compare "Hello" "Hi" "Hey there"
```

**Example output:**
```
======================================================================
TOKENIZATION VISUALIZATION
======================================================================

Original Text:
  "Hello world!"

Quick Stats:
  Total tokens:      3
  Total characters:  12
  Total bytes:       12
  Compression ratio: 0.250 tokens/byte
  Efficiency:        4.00 chars/token

Colored Token Breakdown:
(Each color represents a different token)

Hello world!  <-- Each word shown in different colors


Detailed Token Information:

Index  Token ID   Text                                Bytes    Type
--------------------------------------------------------------------------------
0      1000       Hello                               5        Alphabetic
1      1001        world                              6        Mixed/Other
2      33         !                                   1        Single byte

======================================================================
```

**Dependencies:** None (Python standard library only, uses nanochat's tokenizer)

**Learning outcomes:**
- Understand tokenization visually
- See how BPE creates subword units
- Learn about special tokens in conversations
- Understand token efficiency and compression
- Debug tokenization-related issues

##### 2. Training Progress Dashboard (`training_dashboard.py`)
Monitor and visualize training progress by reading checkpoint metadata and logs.

**What it does:**
- Read checkpoint metadata files (meta_*.json) from training runs
- Extract training metrics (steps, train_loss, val_bpb, learning_rate)
- Display training summary with best checkpoint identification
- Show progress table with all checkpoints and metrics
- Generate visualization plots of loss curves and learning rate schedule
- Export metrics to CSV for external analysis
- Standalone tool that doesn't require modifying training scripts

**Why it's useful:**
- Track training progress without modifying training code
- Quickly identify best checkpoints by validation loss
- Visualize training dynamics (loss curves, LR schedule)
- Monitor experiments remotely by checking checkpoint directory
- Export data for custom analysis and plotting
- Debug training issues (loss spikes, learning rate problems)
- Compare multiple training runs side-by-side
- Learn about training dynamics and optimization

**Usage:**
```bash
# Show training summary (default)
python tools/training_dashboard.py out/base_checkpoints/d20

# Show progress table with all checkpoints
python tools/training_dashboard.py out/base_checkpoints/d20 --progress

# Generate visualization plot
python tools/training_dashboard.py out/base_checkpoints/d20 --plot

# Export metrics to CSV
python tools/training_dashboard.py out/base_checkpoints/d20 --export-csv

# Custom output path for plot or CSV
python tools/training_dashboard.py out/base_checkpoints/d20 --plot --output my_plot.png
python tools/training_dashboard.py out/base_checkpoints/d20 --export-csv --output metrics.csv
```

**Example output (summary mode):**
```
====================================================================================================
TRAINING DASHBOARD
====================================================================================================

Checkpoint directory: out/base_checkpoints/d20
Total checkpoints: 15

Training range:
  First checkpoint: Step 0
  Last checkpoint:  Step 7200

Latest metrics (step 7200):
  Train loss: 1.2345
  Val BPB:    1.1823
  LR:         0.000100

Best validation checkpoint:
  Step:    5400
  Val BPB: 1.1642

Model configuration:
  Layers:       20
  Embedding:    512
  Seq length:   1024

====================================================================================================
```

**Example output (progress mode):**
```
====================================================================================================
TRAINING PROGRESS
====================================================================================================

Step       Train Loss      Val BPB         Learning Rate
----------------------------------------------------------------------------------------------------
0          4.2156          N/A             0.001000
500        3.1234          3.2145          0.001000
1000       2.8765          2.9234          0.001000
1500       2.6543          2.7123          0.000950
...
====================================================================================================
```

**Plot output:**
Generates a PNG file with two subplots:
1. Training and validation loss curves over time
2. Learning rate schedule over time

Both plots include:
- Clear axis labels and titles
- Grid for easy reading
- Legend for multiple metrics
- Professional formatting

**Dependencies:**
- Python standard library only (core functionality)
- matplotlib (optional, for plotting - gracefully falls back if not available)

**Learning outcomes:**
- Understand checkpoint metadata structure (meta_*.json format)
- Learn about training metrics and their significance
- Practice working with JSON data and file I/O
- See how to track and visualize training progress
- Understand learning rate schedules and warmdown phases
- Learn about validation metrics (BPB) vs training loss
- Practice creating data analysis and visualization tools
- Understand matplotlib basics for scientific plotting
- Learn graceful degradation (optional dependencies)

##### 4. Dataset Inspector (`dataset_inspector.py`)
Analyze and validate training datasets before running expensive training jobs.

**What it does:**
- Show random samples from your dataset
- Validate JSONL format and conversation structure (supports optional system messages)
- Analyze token and character length distributions
- Check for common formatting errors (role alternation, empty content)
- Generate statistics about your data
- Export samples for manual review

**Why it's useful:**
- Catch data quality issues before training
- Understand what patterns your model will learn
- Ensure conversations are properly formatted
- Estimate optimal sequence length settings
- Save hours of debugging failed training runs
- Balance dataset across different task types

**Usage:**
```bash
# Show 5 random samples (default)
python tools/dataset_inspector.py dataset.jsonl

# Show 10 samples
python tools/dataset_inspector.py dataset.jsonl --samples 10

# Validate dataset format
python tools/dataset_inspector.py dataset.jsonl --validate

# Analyze length distributions
python tools/dataset_inspector.py dataset.jsonl --analyze-lengths

# Export 100 samples for manual review
python tools/dataset_inspector.py dataset.jsonl --export review.txt --export-count 100

# Do everything at once
python tools/dataset_inspector.py dataset.jsonl --validate --analyze-lengths --samples 10
```

**Example output:**
```
================================================================================
FORMAT VALIDATION
================================================================================

Checked 1,000 conversations

✅ No format issues found! Dataset looks good.

================================================================================
DATASET STATISTICS
================================================================================

Total conversations: 1,000
Total messages:      2,450
Avg messages/conv:   2.5

Message Role Distribution:
  user           : 1,000
  assistant      : 1,450

================================================================================
LENGTH ANALYSIS
================================================================================

Total conversations: 1,000

Full Conversation Lengths (in tokens):
  Min:     45
  Max:     1,856
  Mean:    312.4
  Median:  276

User Message Lengths (characters):
  Min:     12
  Max:     524
  Mean:    98.3

Assistant Message Lengths (characters):
  Min:     15
  Max:     1,203
  Mean:    245.7

Length Distribution (tokens):
  0-100       :    125 ############
  100-200     :    345 #######################
  200-500     :    428 ############################
  500-1000    :     89 #####
  1000-2000   :     13 #

💡 Learning Insight:
  Your conversations are medium length. Balanced for most use cases.
```

**Dependencies:** None (Python standard library only, optional tokenizer integration)

**Learning outcomes:**
- Understand training data quality requirements
- Learn JSONL and conversation format structure
- Practice data validation and statistics
- Recognize common data formatting issues
- Appreciate the importance of data quality

##### 5. Model Size & Cost Calculator (`model_calculator.py`)
Calculate parameters, memory, and training costs for any model configuration.

**What it does:**
- Counts parameters for all model components (embeddings, attention, MLP)
- Estimates memory requirements (fp32, fp16, training, inference)
- Predicts training time and computational cost (FLOPs)
- Provides educational insights about parameter distribution

**Why it's useful:**
- Understand how model size scales with dimensions
- Avoid GPU OOM errors by predicting memory needs
- Plan training experiments and timelines
- Learn where parameters come from in Transformers

**Usage:**
```bash
# Use preset configurations
python tools/model_calculator.py --preset nanochat-tiny
python tools/model_calculator.py --preset gpt2-small

# Custom configuration
python tools/model_calculator.py --depth 12 --hidden-dim 768 --vocab-size 32000

# Customize training parameters
python tools/model_calculator.py --preset gpt2-small --batch-size 32 --total-tokens 20000000000
```

**Example output:**
```
======================================================================
MODEL SIZE & COST CALCULATOR
======================================================================

📊 MODEL CONFIGURATION
----------------------------------------------------------------------
  Layers (depth):        6
  Hidden dimension:      384
  Vocabulary size:       32,000
  Attention heads:       6

🔢 PARAMETER BREAKDOWN
----------------------------------------------------------------------
  Token embeddings:           12,288,000 params
  Per-layer breakdown:
    - Attention:                 589,824 params
    - MLP:                     1,179,648 params
    - LayerNorm:                   1,536 params
    - Total per layer:         1,771,008 params
  All 6 layers:             10,626,048 params
  Final LayerNorm:                   768 params
  LM head:                    12,288,000 params
  ────────────────────────────────────────────────────────────
  TOTAL PARAMETERS:           35,202,816 params
                                   35.20 M

💾 MEMORY REQUIREMENTS
----------------------------------------------------------------------
  Model weights (fp32):        0.13 GB
  Model weights (fp16):        0.07 GB
  Training (fp32+opt):         0.52 GB
  Inference (fp16):            0.07 GB

⏱️  TRAINING ESTIMATES
----------------------------------------------------------------------
  Training tokens:       10.0B tokens
  Batch size:            64 sequences
  Sequence length:       1024 tokens
  Tokens per batch:      65,536
  Total steps:           152,587
  Throughput:            100,000 tokens/sec
  Training time:         27.8 hours (1.2 days)
  Total FLOPs:           2112.2 PetaFLOPs

💡 LEARNING INSIGHTS
----------------------------------------------------------------------
  • Embeddings use ~34.9% of parameters
  • Attention layers use ~10.1% of parameters
  • MLP layers use ~20.1% of parameters
  • Training needs ~4x more memory than inference
  • Each parameter sees 284 tokens during training
======================================================================
```

**Dependencies:** None (Python standard library only)

**Learning outcomes:**
- Understand parameter counting in Transformers
- Learn about memory requirements for different precisions
- See how batch size and sequence length affect training
- Calculate FLOPs for computational cost estimation

##### 3. Checkpoint Browser & Comparator (`checkpoint_browser.py`)
Browse, inspect, and compare model checkpoints across training runs.

**What it does:**
- Scan and list all checkpoints in your training directories
- Display checkpoint metadata (step, validation loss, model architecture)
- Compare two checkpoints side-by-side
- Inspect detailed information about specific checkpoints
- Track disk usage across all checkpoints
- Support for nanochat's multi-file checkpoint format

**Why it's useful:**
- Find your best checkpoint by validation loss
- Compare different model sizes and configurations
- Track experiment history and model evolution
- Understand what each checkpoint contains
- Manage disk space by identifying old checkpoints
- Verify checkpoint integrity before resuming training

**Usage:**
```bash
# List all checkpoints
python tools/checkpoint_browser.py --list

# Sort by validation loss (find best checkpoint)
python tools/checkpoint_browser.py --list --sort val_bpb

# Inspect a specific model (shows latest checkpoint)
python tools/checkpoint_browser.py --inspect d20

# Inspect specific step
python tools/checkpoint_browser.py --inspect d20 --step 5000

# Compare two checkpoints
python tools/checkpoint_browser.py --compare d20 d26
```

**Example output:**
```
========================================================================================
CHECKPOINTS (sorted by val_bpb)
========================================================================================

Model    | Type | Step  | Val BPB | Layers | Dim | Size (MB) | Optim | Modified
--------------------------------------------------------------------------------------
d26      | base | 7,200 | 1.1823  | 26     | 832 | 3982.1    | ✓     | 2025-11-07 09:15
d20      | base | 5,400 | 1.2145  | 20     | 512 | 2145.3    | ✓     | 2025-11-08 14:32
d20_sft  | chat | 2,000 | 1.3421  | 20     | 512 | 2145.3    | ✓     | 2025-11-06 18:45

Total checkpoints: 3
Total disk usage: 8.27 GB
```

**Dependencies:** None (Python standard library only; tabulate optional for better formatting)

**Learning outcomes:**
- Understand nanochat's checkpoint file structure (model_*.pt + meta_*.json + optim_*.pt)
- Learn about checkpoint metadata and what gets saved during training
- Practice working with file systems and directory scanning
- See how to compare model configurations
- Understand the relationship between model size and checkpoint size
- Learn about glob patterns and file matching

##### 6. Generation Parameter Explorer (`generation_explorer.py`)
Interactively explore how sampling parameters affect model text generation.

**What it does:**
- Load trained models and generate text with different parameters
- Show probability distributions at each generation step
- Compare outputs at different temperatures side-by-side
- Compare different top-k filtering values
- Visualize what tokens the model considers likely
- Interactive mode for experimentation

**Why it's useful:**
- Understand how temperature/top-k/top-p affect output quality
- Find optimal generation settings for your use case
- See model confidence and probability distributions
- Debug why outputs are too random or too repetitive
- Learn sampling strategies (greedy, temperature, top-k)
- Understand the randomness vs quality trade-off

**Usage:**
```bash
# Interactive mode - experiment with different settings
python tools/generation_explorer.py --source base --interactive

# Generate with probability display
python tools/generation_explorer.py --source base \
    --prompt "The capital of France is" --show-probs

# Compare different temperatures
python tools/generation_explorer.py --source base \
    --prompt "Once upon a time" --compare-temp

# Compare different top-k values
python tools/generation_explorer.py --source base \
    --prompt "In the beginning" --compare-topk

# Use specific model and device
python tools/generation_explorer.py --source sft --model-tag d20 \
    --device cuda --prompt "What is 2+2?" --show-probs
```

**Example output:**
```
====================================================================================================
GENERATION WITH PROBABILITIES
====================================================================================================

Prompt: "The capital of France is"
Temperature: 0.9, Top-k: None

----------------------------------------------------------------------------------------------------

Step 1:
  Sampled: [4342] " Paris" (p=0.9234)
  Top 5 alternatives:
    1. [ 4342] " Paris              " p=0.9234 ←
    2. [ 1234] " the                " p=0.0421
    3. [ 5678] " located            " p=0.0156
    4. [ 9012] " a                  " p=0.0098
    5. [ 3456] " one                " p=0.0045

Step 2:
  Sampled: [287] "," (p=0.3421)
  Top 5 alternatives:
    1. [  287] ",                  " p=0.3421 ←
    2. [  288] ".                  " p=0.2987
    3. [ 1098] " which              " p=0.1234
    4. [  345] ";                  " p=0.0876
    5. [  543] " and                " p=0.0654

----------------------------------------------------------------------------------------------------

Full generation:
" Paris, the city of lights"
```

With `--compare-temp`, compares outputs at different temperatures (0.1, 0.5, 0.9, 1.2, 1.5) to show how temperature affects randomness and creativity.

**Dependencies:** PyTorch (for model loading and inference)

**Learning outcomes:**
- Understand sampling strategies (temperature, top-k, top-p)
- Learn about probability distributions in language models
- See how model confidence varies by token
- Practice model loading and inference in PyTorch
- Understand the trade-off between determinism and creativity
- Learn when to use greedy vs sampling-based generation

##### 7. Training Resume Helper (`training_resume_helper.py`)
Analyze checkpoints and provide guidance for resuming interrupted training runs.

**What it does:**
- Find latest checkpoint in nanochat's format (model_*.pt + meta_*.json + optional optim_*.pt)
- Verify checkpoint integrity and file structure
- Display checkpoint metadata (step, loss, model config)
- Calculate training progress and remaining steps
- Generate manual resume instructions with code examples (conditional based on optimizer presence)
- Detect warmdown phase for learning rate adjustment

**Why it's useful:**
- Never lose training progress from crashes
- Identify the correct checkpoint to resume from
- Calculate remaining training time accurately
- Understand nanochat's checkpoint file structure
- Learn about training state management (model, optimizer, metadata)
- Get code examples for manual checkpoint loading
- Debug checkpoint loading issues

**Usage:**
```bash
# Show checkpoint information
python tools/training_resume_helper.py out/checkpoint_dir

# Calculate progress toward target steps
python tools/training_resume_helper.py out/checkpoint_dir --target-steps 5400

# Verify checkpoint integrity
python tools/training_resume_helper.py out/checkpoint_dir --verify

# Generate resume instructions (manual code examples)
python tools/training_resume_helper.py out/checkpoint_dir --command
```

**Example output:**
```
================================================================================
TRAINING RESUME REPORT
================================================================================

Checkpoint directory: out/base_checkpoints/d20
Model file: model_002500.pt
Metadata file: meta_002500.json
Optimizer file: optim_002500.pt

Last saved step: 2,500
Validation BPB: 1.2340

Model Configuration:
  Layers: 6
  Hidden dim: 384
  Sequence length: 1024

Training Configuration:
  device_batch_size: 16
  total_batch_size: 64
  learning_rate: 0.001

Resume Parameters:
  Current step: 2,500
  Target steps: 5,000
  Remaining: 2,500
  Progress: 50.0%
  In warmdown: No

💡 Learning Insights:
  • Still in main training phase
  • Approximately 50% of training remaining

================================================================================
```

With `--command`, generates manual resume instructions showing how to use `load_checkpoint()` from `checkpoint_manager.py` to manually load and resume training. Instructions are conditional based on whether optimizer checkpoint exists.

**Dependencies:** PyTorch (for checkpoint loading)

**Learning outcomes:**
- Understand nanochat's checkpoint file format (model_*.pt + meta_*.json + optional optim_*.pt)
- Learn about training state management (model weights, optimizer state, metadata)
- Practice working with PyTorch state_dicts and JSON metadata
- See how to calculate training progress and detect training phases
- Understand manual checkpoint loading using checkpoint_manager.py
- Learn about warmdown phase in training schedules
- Understand why optimizer checkpoints are optional (fine-tuning vs continuation)

##### 10. Conversation Template Builder (`conversation_template_builder.py`)
Create, validate, and visualize conversation templates for fine-tuning nanochat models.

**What it does:**
- Create simple 2-turn conversations (user → assistant)
- Create multi-turn conversations with proper role alternation
- Create conversations with Python tool usage
- Validate conversation structure and format
- Show how conversations get tokenized with color-coded visualization
- Display supervision statistics (which tokens the model learns from)
- Batch validate JSONL datasets
- Interactive mode for building conversations step-by-step
- Save/load conversations as JSON files

**Why it's useful:**
- Prepare training data for fine-tuning (SFT phase)
- Understand nanochat's conversation format
- See exactly how conversations become token sequences
- Validate datasets before expensive training runs
- Learn which tokens are supervised vs unsupervised
- Debug tokenization issues in chat data
- Create examples for testing model behavior

**Usage:**
```bash
# Interactive mode - build conversations step by step
python tools/conversation_template_builder.py --interactive

# Create a simple conversation
python tools/conversation_template_builder.py --simple "What is 2+2?" "It's 4."

# Validate a conversation file
python tools/conversation_template_builder.py --validate conversation.json

# Batch validate a JSONL dataset
python tools/conversation_template_builder.py --batch-validate dataset.jsonl --output report.json
```

**Example output:**
```
====================================================================================================
CONVERSATION TOKENIZATION
====================================================================================================

✓ Conversation is valid

Conversation Structure:
----------------------------------------------------------------------------------------------------
  1. [USER]: What is 2+2?
  2. [ASSISTANT]: The answer is 4.

Tokenization Statistics:
----------------------------------------------------------------------------------------------------
  Total tokens:       18
  Supervised tokens:  8 (tokens the model learns from)
  Unsupervised:       10 (prompt/context tokens)
  Supervision ratio:  44.4%

Tokenized Format:
----------------------------------------------------------------------------------------------------
(Green = supervised, Red = unsupervised)

<|bos|>|<|user_start|>|What| is| 2|+|2|?|<|user_end|>|<|assistant_start|>|The| answer| is| 4|.|<|assistant_end|>
[Red tokens: bos, user_start, content, user_end, assistant_start | Green tokens: assistant content | Red: assistant_end]
```

**Interactive mode commands:**
- `simple` - Create 2-turn conversation
- `multi` - Create multi-turn conversation
- `tool` - Create conversation with Python tool usage
- `load <file>` - Load conversation from JSON
- `save <file>` - Save last conversation
- `validate <file>` - Validate a conversation file
- `show` - Display last conversation
- `tokenize` - Show tokenization of last conversation

**Dependencies:** Standard library only (for conversation creation), nanochat tokenizer (for tokenization visualization)

**Learning outcomes:**
- Understand nanochat's conversation format (messages with roles)
- Learn about supervised vs unsupervised tokens in fine-tuning
- See how special tokens (<|user_start|>, <|assistant_end|>, etc.) work
- Practice JSON data structure design for ML
- Understand conversation alternation rules (user → assistant → user...)
- Learn about tool usage format (<|python_start|>, <|output_start|>)
- Prepare high-quality training data for SFT phase
- Debug tokenization and formatting issues before training

##### 8. Simple Attention Visualizer (`attention_visualizer.py`)
Visualize attention patterns in trained models to understand the Transformer attention mechanism.

**What it does:**
- Load trained nanochat models from checkpoints
- Compute attention weights for any transformer layer
- Visualize attention for specific heads or averaged across all heads
- Show text-based visualization with top-k attended tokens
- Generate heatmap visualizations of attention matrices (requires matplotlib)
- Display which tokens the model "looks at" during inference
- Support for both base and SFT models

**Why it's useful:**
- Understand the Transformer attention mechanism concretely
- See what tokens the model focuses on for predictions
- Debug model behavior and understand reasoning
- Learn how different layers capture different patterns
- Compare attention across different heads
- Visualize information flow through the model
- Educational tool for understanding self-attention
- Verify model is learning meaningful relationships

**Usage:**
```bash
# Text-based visualization for layer 0
python tools/attention_visualizer.py --text "Hello world" --layer 0

# Visualize specific head in layer 5
python tools/attention_visualizer.py --text "The capital of France is" --layer 5 --head 3

# Generate heatmap and save to file
python tools/attention_visualizer.py --text "Once upon a time" --layer 10 --heatmap --output attention.png

# Use specific model checkpoint
python tools/attention_visualizer.py --source base --model-tag d20 --text "Hello" --layer 0

# Show more attended tokens in text mode
python tools/attention_visualizer.py --text "The quick brown fox" --layer 0 --top-k 10
```

**Example output (text mode):**
```
====================================================================================================
ATTENTION VISUALIZATION (Text-based)
====================================================================================================

Input text: "Hello world"
Layer: 0
Head: Average across all 6 heads

Showing top-5 attended tokens for each position:
----------------------------------------------------------------------------------------------------

Token 0: '<|bos|>'
  Attends to:
    1. Token 0: '<|bos|>'          [100.0%] ████████████████████████████████████████

Token 1: 'Hello'
  Attends to:
    1. Token 1: 'Hello'            [ 65.3%] ██████████████████████████
    2. Token 0: '<|bos|>'          [ 34.7%] █████████████

Token 2: ' world'
  Attends to:
    1. Token 2: ' world'           [ 48.2%] ███████████████████
    2. Token 1: 'Hello'            [ 31.5%] ████████████
    3. Token 0: '<|bos|>'          [ 20.3%] ████████

====================================================================================================
```

**Heatmap output:**
Generates a color-coded heatmap showing attention weights as a matrix:
- Rows: Attending tokens (queries)
- Columns: Attended tokens (keys)
- Color intensity: Attention weight (0-1)
- Includes token labels on both axes
- Professional formatting with colorbar and grid

**Dependencies:**
- Python standard library (core functionality)
- PyTorch (required - for model loading and inference)
- matplotlib (optional, for heatmap visualization - gracefully falls back if unavailable)

**Learning outcomes:**
- Understand Transformer self-attention mechanism visually
- See how queries attend to keys in the attention computation
- Learn about attention heads and their different roles
- Understand causal masking (no attention to future tokens)
- Practice working with attention weight matrices
- Learn how attention patterns change across layers
- Understand multi-head attention averaging
- See concrete examples of what the model "pays attention to"
- Debug and interpret model predictions
- Understand rotary embeddings and QK normalization effects

##### 9. Learning Rate Finder (`lr_finder.py`)
Analyze learning rate schedules and find optimal learning rates for training experiments.

**What it does:**
- Analyze existing training runs to understand LR impact on loss
- Extract learning rate and loss data from checkpoint metadata
- Detect LR schedule type (constant, decay, warmup, warmup_decay, custom)
- Identify best checkpoint by lowest training loss
- Generate recommendations for optimal learning rates
- Plot LR schedule and loss curves over training (requires matplotlib)
- Provide general guidance on LR range testing

**Why it's useful:**
- Find optimal learning rate before full training runs
- Save time and compute by avoiding bad LR choices
- Understand how LR affects training dynamics
- Learn to identify good LR schedules
- Debug training issues related to learning rate
- Get data-driven recommendations for new experiments
- Understand LR range test methodology (Leslie Smith)
- Compare different LR schedules empirically

**Usage:**
```bash
# Analyze existing training run
python tools/lr_finder.py out/base_checkpoints/d20

# Analyze and generate visualization plot
python tools/lr_finder.py out/base_checkpoints/d20 --plot

# Save plot to file
python tools/lr_finder.py out/base_checkpoints/d20 --plot --output lr_analysis.png

# Get recommendations only
python tools/lr_finder.py out/base_checkpoints/d20 --recommend
```

**Example output:**
```
====================================================================================================
LEARNING RATE ANALYSIS
====================================================================================================

Checkpoint directory: out/base_checkpoints/d20
Number of checkpoints analyzed: 15

Learning rate range:
  Min LR: 0.000100
  Max LR: 0.001000

LR schedule type: warmup_decay

Best checkpoint (lowest training loss):
  Step:          5400
  Learning rate: 0.000500
  Train loss:    1.1642

====================================================================================================

====================================================================================================
LEARNING RATE RECOMMENDATIONS
====================================================================================================

Based on your training run analysis:

1. Best observed learning rate: 0.000500
   This LR achieved the lowest training loss in your run.

2. Detected schedule: warmup_decay
   Excellent! This is a recommended schedule pattern.
   Your current schedule looks good.

3. For new experiments, try:
   - Conservative: 0.000250 (50% of best)
   - Recommended: 0.000500 (observed best)
   - Aggressive:  0.001000 (2x best)

----------------------------------------------------------------------------------------------------
GENERAL RECOMMENDATIONS
----------------------------------------------------------------------------------------------------

📖 How to find optimal learning rate:

1. LR Range Test (Leslie Smith method):
   - Start with very small LR (e.g., 1e-8)
   - Increase exponentially each step to large LR (e.g., 10)
   - Run for 100-1000 steps
   - Plot loss vs LR
   - Choose LR where loss decreases most steeply

2. Rules of thumb:
   - Too low: Loss decreases very slowly
   - Too high: Loss diverges or oscillates
   - Just right: Steady, fast decrease

3. Common starting points:
   - Adam/AdamW: 3e-4 to 1e-3
   - SGD: 1e-2 to 1e-1
   - For fine-tuning: 1e-5 to 1e-4

====================================================================================================
```

**Plot output:**
Generates a figure with two subplots:
1. Learning rate schedule over training steps
2. Training loss over training steps

Both plots mark the best checkpoint with a vertical line/marker for easy identification.

**Dependencies:**
- Python standard library only (core functionality)
- matplotlib (optional, for plotting - gracefully falls back if unavailable)

**Learning outcomes:**
- Understand the importance of learning rate in training
- Learn about different LR schedules (constant, decay, warmup, warmup_decay)
- Practice analyzing training dynamics
- Understand LR range test methodology
- Learn to identify optimal learning rates from data
- Understand the relationship between LR and loss
- Practice reading and analyzing checkpoint metadata
- Learn rules of thumb for different optimizers
- Understand trade-offs between conservative and aggressive LRs

#### 🔜 Planned Features

All 10 features implemented! 🎉

## Design Principles

All additions follow these principles:

1. **Educational First** - Every feature teaches you something about LLMs
2. **Minimal Dependencies** - Use standard library when possible, avoid bloat
3. **Simple Implementation** - Code should be readable by beginners
4. **No GPU Required** (for tools) - Learning tools work on any machine
5. **Self-Contained** - Documentation explains everything needed

## How to Use This Fork

### For Learning:
1. Read the documentation in `docs/` sequentially
2. Use the tools in `tools/` to experiment
3. Build the features from `docs/09_feature_implementation_guide.md`
4. Modify and extend features to deepen understanding

### For Experimentation:
1. Use `tools/model_calculator.py` to plan your experiment
2. Follow `docs/08_quickstart.md` to run training
3. Use the web interface or CLI to test your model
4. Build additional tools as needed

### For Contributing:
This is a personal learning fork. Feel free to fork it further for your own learning journey!

## Differences from Original

### Added:
- Complete beginner documentation (8 guides + 1 feature guide)
- Learning tools directory with utilities
- Feature implementation guide with 10 hands-on projects

### Unchanged:
- All core nanochat functionality
- Training pipeline and scripts
- Model architecture
- Evaluation framework

### Philosophy:
The original nanochat is minimalist and production-focused. This fork adds a comprehensive learning layer on top without modifying the core system.

## Status

- ✅ Documentation: Complete (9 guides covering all aspects)
- ✅ Tools: 10/10 features implemented 🎉
  - Feature 1: Interactive Tokenizer Playground ✅
  - Feature 2: Training Progress Dashboard ✅
  - Feature 3: Checkpoint Browser & Comparator ✅
  - Feature 4: Dataset Inspector ✅
  - Feature 5: Model Size & Cost Calculator ✅
  - Feature 6: Generation Parameter Explorer ✅
  - Feature 7: Training Resume Helper ✅
  - Feature 8: Simple Attention Visualizer ✅
  - Feature 9: Learning Rate Finder ✅
  - Feature 10: Conversation Template Builder ✅
- 🎓 Complete feature set for learning LLMs and PyTorch!

## Acknowledgments

Huge thanks to [Andrej Karpathy](https://github.com/karpathy) for creating nanochat - a beautifully simple and educational LLM implementation that makes learning accessible.

This fork wouldn't exist without his excellent work on making AI education approachable.

## License

Same as original nanochat repository (MIT License).

---

**Note**: This is a learning fork. For production use, refer to the [original nanochat repository](https://github.com/karpathy/nanochat).
