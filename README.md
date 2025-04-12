# SuperBPE: Space Travel for Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2503.13423-b31b1b.svg)](https://arxiv.org/pdf/2503.13423) [![website](https://img.shields.io/badge/Website-superbpe.github.io-C16C8A)](https://superbpe.github.io/) [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-FFD21E)](https://huggingface.co/collections/UW/superbpe-67db2338062faa07c7473ffa)

This repository contains the tokenizer training code. Code for other aspects of the project (e.g. evals, model scaling, data processing, wandb, train configs) will be added soon!

## Setup
First, clone the project with:
```bash
git clone --recurse-submodules https://github.com/PythonNut/superbpe.git
```
We use a custom [fork](https://github.com/alisawuffles/tokenizers-superbpe) of [huggingface/tokenizers](https://github.com/huggingface/tokenizers) which conflicts with the original.
Because of this, we recommend *always installing this project in its own virtual environment.*

### Setup virtual environment

#### Using `conda`
```bash
conda create -n superbpe python=3.12 rust
conda activate superbpe
pip install -r requirements.txt
```

#### Using `venv`
You will need to [install rust](https://www.rust-lang.org/tools/install) and Python 3.12.
Then, you can do:
```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data download
Our tokenizer training data is available [here](https://huggingface.co/datasets/UW/olmo-mix-1124-subset-p99).
You can download it using [`huggingface-cli`](https://huggingface.co/docs/huggingface_hub/en/guides/cli) (after logging into your HuggingFace account) using:
```
mkdir olmo-mix-1124-subset-p99
cd olmo-mix-1124-subset-p99
huggingface-cli download UW/olmo-mix-1124-subset-p99 --repo-type dataset --local-dir .
```

## Tokenizer training
Training a SuperBPE tokenizer involves two stages:

1. **Stage 1:** Learn subwords by enforcing whitespace pretokenization (equivalent to regular BPE training).

```bash
python -m train_tokenizer \
    --output_dir tokenizers/olmo2_bpe \
    --corpus_dir olmo-mix-1124-subset-p99/train \
    --num_bytes $((10**10)) \
    --vocab_size 200000 \
    --do_whitespace_pretokenization true
```

2. **Stage 2:** Learn superwords by resuming tokenizer training, but this time skip the whitespace pretokenization step.

```bash
orig_tokenizer_dir=tokenizers/olmo2_bpe
num_inherit_merges=180000
output_dir=tokenizers/olmo2_superbpe

mkdir -p $output_dir

# inherit the first num_inherit_merges from the BPE tokenizer
head -n $num_inherit_merges $orig_tokenizer_dir/merges.txt > $output_dir/merges.txt

# specifies the same training files used in stage 1
cp $orig_tokenizer_dir/meta.json $output_dir/meta.json

python -m train_tokenizer \
    --output_dir $output_dir \
    --vocab_size 200000 \
    --do_whitespace_pretokenization false
```
