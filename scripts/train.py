import argparse
import os
from typing import Dict

from src.utils.config import Config
from src.utils.logging import setup_logging, get_logger
from src.data.dataset import TextSimplificationDataset
from src.models.t5_model import T5Model
from transformers import T5Tokenizer

logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train text simplification model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/t5_config.yaml",
        help="Path to config file"
    )
    return parser.parse_args()

def setup_environment(config: Dict) -> None:
    """Setup training environment.
    
    Args:
        config: Configuration dictionary
    """
    # Set CUDA devices
    if "gpu_devices" in config["training"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["training"]["gpu_devices"]
        
    # Create output directories
    os.makedirs(config["model"]["save_dir"], exist_ok=True)
    os.makedirs(config["data"]["cache_dir"], exist_ok=True)
    
    # Setup random seed for reproducibility
    if "seed" in config["training"]:
        Config.setup_environment(config)

def main() -> None:
    """Main training function."""
    # Parse arguments and load config
    args = parse_args()
    config = Config.load_config(args.config)
    
    # Setup logging and environment
    setup_logging(config=config)
    logger.info(f"Loaded config from {args.config}")
    setup_environment(config)
    
    try:
        # Initialize tokenizer and model
        logger.info("Initializing tokenizer and model...")
        tokenizer = T5Tokenizer.from_pretrained(config["model"]["name"])
        model = T5Model(config, tokenizer=tokenizer)
        
        # Initialize dataset
        logger.info("Initializing dataset...")
        dataset = TextSimplificationDataset(config, tokenizer)
        
        # Load or prepare datasets
        train_dataset, val_dataset, test_dataset = dataset.load_or_prepare_datasets()
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")
        
        # Start training
        logger.info("Starting training...")
        model.train(
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        logger.info("Training completed successfully!")
        
        # Save final model and config
        model.save_model()
        Config.save_config(
            config,
            os.path.join(config["model"]["save_dir"], "config.yaml")
        )
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise
    
if __name__ == "__main__":
    main()