import re
import sqlite3
import pandas as pd
from datasets import load_dataset, Dataset
from typing import Dict, Union
from transformers import AutoTokenizer
from random import randint
from itertools import chain
from functools import partial
from langdetect import detect
import csv
# Model and Tokenizer Configuration
model_id = "Qwen/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

def process_sql_dataset(dataset_name="NumbersStation/NSText2SQL", source_filter="atis"):
    """
    Process a SQL dataset by filtering and preparing context and instructions.
    
    Args:
        dataset_name (str): Name of the dataset to load from Hugging Face Hub
        source_filter (str): Prefix to filter dataset sources
    
    Returns:
        processed_dataset: Dataset with added context and refined instructions
    """
    # Load dataset from the hub
    dataset = load_dataset(dataset_name, split="train")
    
    # Filter dataset based on source
    dataset = dataset.filter(lambda example: example["source"].startswith(source_filter))
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size}")
    
    # Create database connection (assuming SQLite database path)
    dbfile = "sqllite/atis-db.sqlite"
    con = sqlite3.connect(dbfile)
    
    # Retrieve all table names
    df_all_tables = pd.read_sql_query("""SELECT name FROM sqlite_master WHERE type='table';""", con)
    
    # Collect and clean create statements
    all_create_statements = []
    for index, row in df_all_tables.iterrows():
        table_name = row["name"]
        create_statement = pd.read_sql_query(f"SELECT sql FROM sqlite_schema WHERE name = '{table_name}'", con).iat[0, 0]
        
        # Clean up create statements
        create_statement = re.sub(r"\(\d+\)", "", create_statement)
        create_statement = re.sub(r"'\d+'", "", create_statement)
        create_statement = create_statement.replace("DEFAULT NULL", "")
        create_statement = create_statement.replace("NOT NULL DEFAULT", "")
        create_statement = create_statement.replace("\n", "")
        create_statement = create_statement.replace("\'\'", "")
        
        all_create_statements.append(create_statement)
    
    def split_dataset_column(sample):
        """
        Modify each sample by adding context and refining instruction
        """
        sample["context"] = "; ".join(all_create_statements)
        sample["instruction"] = sample["instruction"].split("-- Using valid SQLite, answer the following questions for the tables provided above.\n\n-- ")[-1]
        return sample
    
    # Apply processing to dataset
    processed_dataset = dataset.map(split_dataset_column)
    processed_dataset = processed_dataset.filter(lambda sample: detect(sample["instruction"]) == 'en')
    
    return processed_dataset

def split_dataset_by_size(
    dataset: Dataset,
    train_size: Union[int, float] = 0.7,
    eval_size: Union[int, float] = 0.2,
    test_size: Union[int, float] = 0.1,
    train_seed: int = 42,
    eval_seed: int = 42
) -> Dict[str, Dataset]:
    """
    Split dataset deterministically into train, eval, and test sets with flexible sizing.

    Args:
        dataset (Dataset): Processed dataset to split
        train_size (int or float): Number or proportion of samples for training
        eval_size (int or float): Number or proportion of samples for evaluation
        test_size (int or float): Number or proportion of samples for testing
        train_seed (int): Seed for selecting the fixed train dataset
        eval_seed (int): Seed for selecting eval and test datasets

    Returns:
        Dict containing train, eval, and test datasets
    """
    # Total dataset size
    total_size = len(dataset)

    # Convert proportional sizes to actual numbers if needed
    def convert_to_size(size_param, total):
        if isinstance(size_param, float):
            return int(total * size_param)
        return size_param

    train_size = convert_to_size(train_size, total_size)
    eval_size = convert_to_size(eval_size, total_size)
    test_size = convert_to_size(test_size, total_size)

    # Validate sizes
    if train_size + eval_size + test_size > total_size:
        raise ValueError(f"Combined sizes ({train_size + eval_size + test_size}) exceed total dataset size ({total_size})")

    # Shuffle the entire dataset for consistent splitting
    shuffled_dataset = dataset.shuffle(seed=train_seed)

    # Select train dataset
    train_dataset = shuffled_dataset.select(range(train_size))

    # Select remaining dataset (excluding train)
    remaining_dataset = shuffled_dataset.select(range(train_size, total_size))

    # Shuffle remaining dataset with eval seed
    remaining_dataset = remaining_dataset.shuffle(seed=eval_seed)

    # Select eval and test datasets from remaining
    eval_dataset = remaining_dataset.select(range(eval_size))
    test_dataset = remaining_dataset.select(range(eval_size, eval_size + test_size))

    # Print split information
    print(f"Dataset Split Information:")
    print(f"Total samples: {total_size}")
    print(f"Train samples: {len(train_dataset)} ({train_size})")
    print(f"Eval samples: {len(eval_dataset)} ({eval_size})")
    print(f"Test samples: {len(test_dataset)} ({test_size})")

    return {
        "dataset": shuffled_dataset,
        "train": train_dataset,
        "eval": eval_dataset,
        "test": test_dataset
    }

def format_dolly(sample):
    """
    Format sample in Dolly-style prompt template

    Args:
        sample (dict): Dataset sample with instruction, input, and output

    Returns:
        str: Formatted prompt
    """
    instruction = f"### Instruction\n{sample['instruction']}"
    input = f"### Input\n{sample['input']}" if len(sample["input"]) > 0 else None
    response = f"### Answer\n{sample['output']}"
    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, input, response] if i is not None])
    return prompt

def template_dataset(sample):
    """
    Add formatted text to dataset sample

    Args:
        sample (dict): Dataset sample

    Returns:
        dict: Sample with added text field
    """
    sample["text"] = f"{format_dolly(sample)}{tokenizer.eos_token}"
    return sample

def chunk(sample, chunk_length=2048):
    """
    Chunk tokenized dataset into fixed-length sequences

    Args:
        sample (dict): Tokenized dataset sample
        chunk_length (int): Length of chunks to create

    Returns:
        dict: Chunked dataset sample
    """
    # Define global remainder variable to save remainder from batches to use in next batch
    global remainder

    # Concatenate all texts and add remainder from previous batch
    concatenated_examples = {k: list(chain(*sample[k])) for k in sample.keys()}
    concatenated_examples = {k: remainder[k] + concatenated_examples[k] for k in concatenated_examples.keys()}

    # Get total number of tokens for batch
    batch_total_length = len(concatenated_examples[list(sample.keys())[0]])

    # Get max number of chunks for batch
    if batch_total_length >= chunk_length:
        batch_chunk_length = (batch_total_length // chunk_length) * chunk_length

    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_length] for i in range(0, batch_chunk_length, chunk_length)]
        for k, t in concatenated_examples.items()
    }

    # Add remainder to global variable for next batch
    remainder = {k: concatenated_examples[k][batch_chunk_length:] for k in concatenated_examples.keys()}

    # Prepare labels
    result["labels"] = result["input_ids"].copy()
    return result

def process_dataset(dataset: Dataset, chunk_length: int = 2048) -> Dataset:
  # Initialize remainder dictionary for chunking
  global remainder
  remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}

  # Prepare train dataset
  # Apply prompt template per sample
  dataset = dataset.map(template_dataset, remove_columns=list(dataset.features))

  # Tokenize and chunk train dataset
  lm_dataset = dataset.map(
      lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(dataset.features)
  ).map(
      partial(chunk, chunk_length=chunk_length),
      batched=True,
  )

  # Print total number of samples
  print(f"Total train samples: {len(lm_dataset)}")

  return lm_dataset

def save_to_csv(dataset, filename):
    """
    Save dataset to a CSV file.
    
    Args:
        dataset (Dataset): The dataset to save.
        filename (str): The name of the CSV file.
    """
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(dataset.column_names)
        # Write the data
        for row in dataset:
            writer.writerow([row[column] for column in dataset.column_names])