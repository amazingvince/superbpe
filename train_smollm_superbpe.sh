#!/bin/bash
# Train SuperBPE tokenizer on SmolLM2 dataset using the optimal configuration from the paper
# Configuration: vocab_size=200k, transition point t=180k

set -e  # Exit on error

# Configuration
DATASET_NAME="EleutherAI/SmolLM2-1.7B-stage-4-100B"
TEXT_COLUMN="text"
NUM_BYTES=$((10**10))  # 10GB
VOCAB_SIZE=200000
NUM_INHERIT_MERGES=180000  # Optimal t=180k from the paper

# Output directories
STAGE1_DIR="tokenizers/smollm2_bpe_stage1"
STAGE2_DIR="tokenizers/smollm2_superbpe_t180k"

echo "=== SuperBPE Training Script for SmolLM2 ==="
echo "Dataset: $DATASET_NAME"
echo "Vocabulary size: $VOCAB_SIZE"
echo "Transition point: t=$NUM_INHERIT_MERGES"
echo ""

# Stage 1: Train with whitespace pretokenization
echo "=== Stage 1: Training BPE with whitespace pretokenization ==="
echo "Output directory: $STAGE1_DIR"
echo ""

python -m train_tokenizer \
    --output_dir "$STAGE1_DIR" \
    --hf_dataset "$DATASET_NAME" \
    --text_column "$TEXT_COLUMN" \
    --num_bytes "$NUM_BYTES" \
    --vocab_size "$VOCAB_SIZE" \
    --do_whitespace_pretokenization true

echo ""
echo "Stage 1 complete!"
echo ""

# Stage 2: Continue training without whitespace pretokenization
echo "=== Stage 2: Extending to SuperBPE (no whitespace pretokenization) ==="
echo "Output directory: $STAGE2_DIR"
echo "Inheriting first $NUM_INHERIT_MERGES merges from stage 1"
echo ""

# Create output directory
mkdir -p "$STAGE2_DIR"

# Inherit the first 180k merges from the BPE tokenizer
echo "Copying first $NUM_INHERIT_MERGES merges..."
head -n "$NUM_INHERIT_MERGES" "$STAGE1_DIR/merges.txt" > "$STAGE2_DIR/merges.txt"

# Copy metadata (the script will automatically detect it's a HF dataset)
echo "Copying metadata..."
cp "$STAGE1_DIR/meta.json" "$STAGE2_DIR/meta.json"

# Continue training to 200k total vocabulary
python -m train_tokenizer \
    --output_dir "$STAGE2_DIR" \
    --vocab_size "$VOCAB_SIZE" \
    --do_whitespace_pretokenization false

echo ""
echo "Stage 2 complete!"
echo ""

# Update decoder configuration
echo "=== Updating decoder configuration ==="
echo "Updating $STAGE2_DIR/tokenizer.json"

# Create a Python script to update the decoder field
python -c "
import json

tokenizer_path = '$STAGE2_DIR/tokenizer.json'
with open(tokenizer_path, 'r') as f:
    tokenizer_config = json.load(f)

# Update decoder configuration
tokenizer_config['decoder'] = {
    'type': 'ByteLevel',
    'add_prefix_space': True,
    'trim_offsets': True,
    'use_regex': True
}

with open(tokenizer_path, 'w') as f:
    json.dump(tokenizer_config, f, indent=2)

print('Decoder configuration updated successfully!')
"

echo ""
echo "=== Training Complete! ==="
echo ""
echo "SuperBPE tokenizer saved to: $STAGE2_DIR"
echo "Configuration:"
echo "  - Vocabulary size: $VOCAB_SIZE tokens"
echo "  - Transition point: t=$NUM_INHERIT_MERGES"
echo "  - First $NUM_INHERIT_MERGES tokens: learned with whitespace pretokenization (subwords)"
echo "  - Last $((VOCAB_SIZE - NUM_INHERIT_MERGES)) tokens: learned without whitespace pretokenization (superwords)"
echo ""
echo "Expected performance (based on paper):"
echo "  - +4.0% improvement over BPE baseline"
echo "  - 27% reduction in inference compute"
echo ""
echo "To use this tokenizer, load it from: $STAGE2_DIR/tokenizer.json"