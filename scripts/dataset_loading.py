from datasets import load_dataset, Dataset
from typing import Dict, Union
from functools import partial
from itertools import chain
def get_dataset(
    dataset_path="lavita/medical-qa-datasets",
    dataset_name="chatdoctor_healthcaremagic",
    dataset_size=None
):
    """
    Process a SQL dataset by filtering and preparing context and instructions.

    Args:
        dataset_name (str): Name of the dataset to load from Hugging Face Hub
        source_filter (str): Prefix to filter dataset sources

    Returns:
        processed_dataset: Dataset with added context and refined instructions
    """
    # Load dataset from the hub and filter dataset based on source
    dataset = load_dataset(path=dataset_path, name=dataset_name, split="train")

    if dataset_size:
        dataset = dataset.select(range(dataset_size))
    else:
        dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size}")
    return dataset