import os
import json
from typing import Dict, List, Tuple, Optional
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from .tokenizer import TokenizerHandler
from ..utils.logging import get_logger

logger = get_logger(__name__)

class TextSimplificationDataset:
    """Handler for text simplification dataset operations."""
    
    def __init__(
        self,
        config: Dict,
        tokenizer: PreTrainedTokenizer
    ):
        """Initialize dataset handler.
        
        Args:
            config: Configuration dictionary
            tokenizer: HuggingFace tokenizer instance
        """
        self.config = config
        self.data_config = config["data"]
        self.tokenizer_handler = TokenizerHandler(tokenizer, config)
        
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """Load a JSONL file into a list of dictionaries.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of dictionaries containing the data
        """
        full_path = os.path.join(self.data_config["root_dir"], file_path)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        except FileNotFoundError:
            logger.error(f"File not found: {full_path}")
            raise
            
    def process_raw_data(self, raw_data: List[Dict]) -> Dataset:
        """Process raw data into a HuggingFace Dataset.
        
        Args:
            raw_data: List of dictionaries containing the data
            
        Returns:
            HuggingFace Dataset
        """
        processed_data = []
        for item in raw_data:
            processed_data.append({
                "src": item["src"],
                "trg": item["trg"]
            })
        return Dataset.from_list(processed_data)
        
    def prepare_dataset(
        self,
        dataset: Dataset,
        split: str
    ) -> Dataset:
        """Prepare dataset for training/evaluation.
        
        Args:
            dataset: HuggingFace Dataset
            split: Dataset split name
            
        Returns:
            Processed Dataset
        """
        logger.info(f"Preparing {split} dataset...")
        
        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer_handler.tokenize_batch(
                examples["src"],
                examples["trg"]
            )
            
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Set format for PyTorch
        tokenized_dataset.set_format(type="torch")
        
        return tokenized_dataset
        
    def load_or_prepare_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load datasets from cache or prepare from raw files.
        
        Returns:
            Tuple of (train, validation, test) datasets
        """
        
        cache_dir = self.data_config["cache_dir"]
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_paths = {
            "train": os.path.join(cache_dir, "train"),
            "val": os.path.join(cache_dir, "val"),
            "test": os.path.join(cache_dir, "test")
        }
        
        # Try loading from cache first
        if all(os.path.exists(path) for path in cache_paths.values()):
            logger.info("Loading datasets from cache...")
            return (
                load_from_disk(cache_paths["train"]),
                load_from_disk(cache_paths["val"]),
                load_from_disk(cache_paths["test"])
            )
            
        # Process raw datasets
        logger.info("Processing raw datasets...")
        datasets = {}
        for split, file_key in [
            ("train", "train_file"),
            ("val", "val_file"),
            ("test", "test_file")
        ]:
            raw_data = self.load_jsonl(self.data_config[file_key])
            dataset = self.process_raw_data(raw_data)
            datasets[split] = self.prepare_dataset(dataset, split)
            datasets[split].save_to_disk(cache_paths[split])
            logger.info(f"Saved {split} dataset to {cache_paths[split]}")
            
        return datasets["train"], datasets["val"], datasets["test"]
        
    def get_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        num_workers: Optional[int] = None
    ) -> DataLoader:
        """Create a DataLoader for the dataset.
        
        Args:
            dataset: HuggingFace Dataset
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            
        Returns:
            PyTorch DataLoader
        """
        if num_workers is None:
            num_workers = self.config["training"]["num_workers"]
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )