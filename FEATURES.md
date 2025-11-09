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

##### 7. Training Resume Helper (`training_resume_helper.py`)
Analyze checkpoints and automatically resume interrupted training runs.

**What it does:**
- Find and verify checkpoint integrity
- Display checkpoint metadata (step, loss, model config)
- Calculate training progress and remaining steps
- Generate resume commands with correct parameters
- Detect warmdown phase for learning rate adjustment

**Why it's useful:**
- Never lose training progress from crashes
- Automatically resume from the correct checkpoint
- Calculate remaining training time accurately
- Understand checkpoint structure and metadata
- Learn about training state management
- Debug checkpoint loading issues

**Usage:**
```bash
# Show checkpoint information
python tools/training_resume_helper.py out/checkpoint_dir

# Calculate progress toward target steps
python tools/training_resume_helper.py out/checkpoint_dir --target-steps 5400

# Verify checkpoint integrity
python tools/training_resume_helper.py out/checkpoint_dir --verify

# Generate resume command
python tools/training_resume_helper.py out/checkpoint_dir --command
```

**Example output:**
```
================================================================================
TRAINING RESUME REPORT
================================================================================

Checkpoint: out/checkpoint_dir/checkpoint.pt
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

**Dependencies:** PyTorch (for checkpoint loading)

**Learning outcomes:**
- Understand checkpoint save/load mechanics
- Learn about training state management (model, optimizer, metadata)
- Practice error handling and validation
- See how to calculate training progress
- Understand warmdown phase in training schedules

#### 🔜 Planned Features (See `docs/09_feature_implementation_guide.md`)

2. **Training Progress Dashboard** - Real-time visualization of training metrics
3. **Checkpoint Browser & Comparator** - Explore saved models and compare performance
6. **Generation Parameter Explorer** - Experiment with temperature, top-k, top-p
8. **Simple Attention Visualizer** - See what the model attends to
9. **Learning Rate Finder** - Find optimal learning rate automatically
10. **Conversation Template Builder** - Create and test custom chat templates

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
- ✅ Tools: 4/10 features implemented
  - Feature 1: Interactive Tokenizer Playground ✅
  - Feature 4: Dataset Inspector ✅
  - Feature 5: Model Size & Cost Calculator ✅
  - Feature 7: Training Resume Helper ✅
- 🔄 Actively adding more learning features

## Acknowledgments

Huge thanks to [Andrej Karpathy](https://github.com/karpathy) for creating nanochat - a beautifully simple and educational LLM implementation that makes learning accessible.

This fork wouldn't exist without his excellent work on making AI education approachable.

## License

Same as original nanochat repository (MIT License).

---

**Note**: This is a learning fork. For production use, refer to the [original nanochat repository](https://github.com/karpathy/nanochat).
