"""
Example script for training a tokenizer on Hugging Face datasets.
"""

# Example 1: Train tokenizer on EleutherAI/SmolLM2-1.7B-stage-4-100B dataset
# This uses streaming mode to handle large datasets efficiently
print("Example command for training on SmolLM2 dataset:")
print("""
python -m train_tokenizer \\
    --output_dir tokenizers/smollm2_bpe \\
    --hf_dataset "EleutherAI/SmolLM2-1.7B-stage-4-100B" \\
    --text_column "text" \\
    --num_bytes 10000000000 \\
    --vocab_size 200000 \\
    --do_whitespace_pretokenization true
""")

# Example 2: Train SuperBPE tokenizer in two stages using HF dataset
print("\nExample commands for two-stage SuperBPE training:")
print("""
# Stage 1: Train with whitespace pretokenization
python -m train_tokenizer \\
    --output_dir tokenizers/smollm2_bpe_stage1 \\
    --hf_dataset "EleutherAI/SmolLM2-1.7B-stage-4-100B" \\
    --num_bytes 10000000000 \\
    --vocab_size 200000 \\
    --do_whitespace_pretokenization true

# Stage 2: Continue training without whitespace pretokenization
orig_tokenizer_dir=tokenizers/smollm2_bpe_stage1
num_inherit_merges=180000
output_dir=tokenizers/smollm2_superbpe

mkdir -p $output_dir

# Inherit the first num_inherit_merges from the BPE tokenizer
head -n $num_inherit_merges $orig_tokenizer_dir/merges.txt > $output_dir/merges.txt

# Copy metadata (it will be updated to use HF dataset)
cp $orig_tokenizer_dir/meta.json $output_dir/meta.json

python -m train_tokenizer \\
    --output_dir $output_dir \\
    --vocab_size 200000 \\
    --do_whitespace_pretokenization false
""")

# Example 3: Use different HF datasets
print("\nOther Hugging Face dataset examples:")
print("""
# Example with a different dataset
python -m train_tokenizer \\
    --output_dir tokenizers/my_custom_tokenizer \\
    --hf_dataset "allenai/c4" \\
    --text_column "text" \\
    --num_bytes 5000000000 \\
    --vocab_size 100000 \\
    --do_whitespace_pretokenization true

# Example with Wikipedia dataset
python -m train_tokenizer \\
    --output_dir tokenizers/wikipedia_tokenizer \\
    --hf_dataset "wikipedia" \\
    --text_column "text" \\
    --num_bytes 1000000000 \\
    --vocab_size 50000 \\
    --do_whitespace_pretokenization true
""")

print("\nNote: The --hf_dataset parameter accepts any dataset available on Hugging Face Hub.")
print("Make sure the --text_column parameter matches the column name in your dataset.")
print("Use --num_bytes to limit the amount of data processed (useful for large datasets).")