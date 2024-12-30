import argparse
import os
from typing import Dict, List, Tuple
from tqdm import tqdm

from src.utils.config import Config
from src.utils.logging import setup_logging, get_logger
from src.utils.metrics import MetricsCalculator
from src.data.dataset import TextSimplificationDataset
from src.models.t5_model import T5Model
from transformers import T5Tokenizer

logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate text simplification model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for generation (overrides config)"
    )
    return parser.parse_args()

def setup_environment(config: Dict) -> None:
    """Setup evaluation environment.
    
    Args:
        config: Configuration dictionary
    """
    if "gpu_device" in config["model"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["model"]["gpu_device"]
        
    # Create results directory
    if "results_dir" in config.get("logging", {}):
        os.makedirs(config["logging"]["results_dir"], exist_ok=True)

def load_test_data(config: Dict, tokenizer: T5Tokenizer) -> Tuple[List[str], List[str]]:
    """Load test data from cache.
    
    Args:
        config: Configuration dictionary
        tokenizer: T5Tokenizer instance
        
    Returns:
        Tuple of (source texts, reference texts)
    """
    logger.info("Loading test dataset...")
    dataset = TextSimplificationDataset(config, tokenizer)
    _, _, test_dataset = dataset.load_or_prepare_datasets()
    
    # Extract source and reference texts
    source_texts = []
    reference_texts = []
    
    for batch in tqdm(test_dataset, desc="Extracting texts"):
        # Decode input_ids to get source texts
        source = tokenizer.decode(batch["input_ids"], skip_special_tokens=True)
        source = source.replace("simplify: ", "")  # Remove prefix
        source_texts.append(source)
        
        # Decode labels to get reference texts
        reference = tokenizer.decode(batch["labels"], skip_special_tokens=True)
        reference_texts.append(reference)
    
    return source_texts, reference_texts

def generate_predictions(
    model: T5Model,
    source_texts: List[str],
    batch_size: int
) -> List[str]:
    """Generate predictions for test data.
    
    Args:
        model: T5Model instance
        source_texts: List of source texts
        batch_size: Batch size for generation
        
    Returns:
        List of predicted texts
    """
    logger.info("Generating predictions...")
    return model.generate(
        input_texts=source_texts,
        batch_size=batch_size
    )

def main() -> None:
    """Main evaluation function."""
    # Parse arguments and load config
    args = parse_args()
    config = Config.load_config(args.config)
    
    # Override batch size if provided
    if args.batch_size is not None:
        config["generation"]["batch_size"] = args.batch_size
    
    # Setup logging and environment
    setup_logging(config=config)
    logger.info(f"Loaded config from {args.config}")
    setup_environment(config)
    
    try:
        # Initialize model and metrics calculator
        logger.info("Initializing model and metrics calculator...")
        tokenizer = T5Tokenizer.from_pretrained(config["model"]["checkpoint_path"])
        model = T5Model(config, tokenizer=tokenizer)
        metrics_calculator = MetricsCalculator(config)
        
        # Load test data
        source_texts, reference_texts = load_test_data(config, tokenizer)
        logger.info(f"Loaded {len(source_texts)} test examples")
        
        # Generate predictions
        predictions = generate_predictions(
            model,
            source_texts,
            config["generation"]["batch_size"]
        )
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        average_metrics, detailed_scores = metrics_calculator.calculate_all_metrics(
            predictions=predictions,
            references=reference_texts
        )
        
        # Log and save results
        metrics_calculator.log_metrics(average_metrics, detailed_scores)
        metrics_calculator.save_results(
            predictions=predictions,
            references=reference_texts,
            average_metrics=average_metrics,
            detailed_scores=detailed_scores
        )
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()