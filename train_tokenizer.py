"""
Train tokenizer on a single data category.
"""

import os
from pathlib import Path
import time
import json
import re
import random
import click
from utils import (
    ensure_dir,
    get_files_with_num_bytes,
    get_truncated_file,
    train_or_extend_tokenizer,
    get_hf_dataset_iterator,
)

random.seed(0)


@click.command()
@click.option(
    "--output_dir",
    type=str,
    help="Where to save the trained tokenizer.",
)
@click.option(
    "--num_bytes",
    type=int,
    default=None,
    help="The maximum number of bytes to use for tokenizer training.",
)
@click.option(
    "--corpus_dir",
    type=str,
    default=None,
    help="Directory containing text files to use for training the tokenizer.",
)
@click.option(
    "--hf_dataset",
    type=str,
    default=None,
    help="Hugging Face dataset name (e.g., 'EleutherAI/SmolLM2-1.7B-stage-4-100B').",
)
@click.option(
    "--text_column",
    type=str,
    default="text",
    help="Column name containing text in the HF dataset.",
)
@click.option(
    "--vocab_size",
    type=int,
    default=100000,
    help="The number of tokens in the vocabulary.",
)
@click.option(
    "--do_whitespace_pretokenization",
    type=bool,
    default=True,
    help="Whether to do whitespace pretokenization.",
)
def main(
    output_dir: str,
    num_bytes: int,
    corpus_dir: str,
    vocab_size: int,
    do_whitespace_pretokenization: bool,
    hf_dataset: str,
    text_column: str,
):
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    print(f"We are training a tokenizer for {output_dir}", flush=True)

    # We look for merges.txt in the current dir to determine whether we are extending
    # the tokenizer or training from scratch, so we need to cd into the output directory.
    os.chdir(output_dir)

    # Check if we're using HF dataset or local files
    if hf_dataset:
        print(f"Using Hugging Face dataset: {hf_dataset}")
        # For HF datasets, we'll use an iterator
        train_data = get_hf_dataset_iterator(
            dataset_name=hf_dataset,
            num_bytes=num_bytes,
            text_column=text_column,
            split="train",
            streaming=True,
        )
        actual_num_bytes = num_bytes  # Will be updated by the iterator
        
        # Write metadata for HF dataset
        with open("meta.json", "w") as fo:
            meta = {
                "dataset_type": "huggingface",
                "dataset_name": hf_dataset,
                "text_column": text_column,
                "total_bytes": actual_num_bytes,
            }
            if os.path.exists("merges.txt"):
                os.system("cp merges.txt initial_merges.txt")
                meta["num_initial_merges"] = (
                    sum(1 for line in open("initial_merges.txt")) - 1
                )
            json.dump(meta, fo, indent=5)
    else:
        if os.path.exists("meta.json"):
            print(
                "Output directory contains meta.json, so we will use the files from there."
            )
            meta = json.load(open("meta.json"))
            
            # Check if it's a HF dataset meta file
            if meta.get("dataset_type") == "huggingface":
                train_data = get_hf_dataset_iterator(
                    dataset_name=meta["dataset_name"],
                    num_bytes=meta.get("total_bytes"),
                    text_column=meta.get("text_column", "text"),
                    split="train",
                    streaming=True,
                )
                actual_num_bytes = meta["total_bytes"]
            else:
                train_files, actual_num_bytes = meta["train_files"], meta["total_bytes"]
                for file in train_files:
                    if not os.path.exists(file):
                        assert "truncated" in file, f"{file} not found"
                        wanted_filesize = int(re.search(r"_truncated_(\d+)", file).group(1))
                        file = re.sub(r"_truncated_\d+", "", file)
                        get_truncated_file(file, wanted_filesize)
                train_data = train_files
        else:
            if not corpus_dir:
                raise ValueError("Either --corpus_dir or --hf_dataset must be provided")
            train_files, actual_num_bytes = get_files_with_num_bytes(corpus_dir, num_bytes)
            train_data = train_files

            # Write metadata for file-based training
            with open("meta.json", "w") as fo:
                meta = {
                    "dataset_type": "files",
                    "total_bytes": actual_num_bytes,
                    "train_files": train_files,
                }
                if os.path.exists("merges.txt"):
                    os.system("cp merges.txt initial_merges.txt")
                    meta["num_initial_merges"] = (
                        sum(1 for line in open("initial_merges.txt")) - 1
                    )
                json.dump(meta, fo, indent=5)

    # Train tokenizer
    start_time = time.time()

    print("Training with HF tokenizers...")
    tokenizer = train_or_extend_tokenizer(
        train_data,
        vocab_size=vocab_size,
        do_whitespace_pretokenization=do_whitespace_pretokenization,
    )
    tokenizer.model.save(".")  # saves merges.txt and vocab.json
    tokenizer.save("tokenizer.json")

    print(f"Train time: {time.time() - start_time}", flush=True)
    print("Tokenizer info saved to " + str(output_dir), flush=True)

    # Delete files that were constructed just for this
    # for f in train_files:
    #     if "truncated" in f:
    #         os.remove(f)


if __name__ == "__main__":
    main()
