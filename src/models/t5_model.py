import os
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments
)
from ..utils.logging import get_logger
from ..data.tokenizer import TokenizerHandler

logger = get_logger(__name__)

class T5Model:
    """Handler for T5 model operations."""
    
    def __init__(
        self,
        config: Dict,
        tokenizer: Optional[T5Tokenizer] = None,
        model: Optional[T5ForConditionalGeneration] = None
    ):
        """Initialize T5 model handler.
        
        Args:
            config: Configuration dictionary
            tokenizer: Optional pre-initialized tokenizer
            model: Optional pre-initialized model
        """
        self.config = config
        self.model_config = config["model"]
        self.training_config = config.get("training", {})
        
        # Initialize or load tokenizer
        self.tokenizer = tokenizer or self._initialize_tokenizer()
        self.tokenizer_handler = TokenizerHandler(self.tokenizer, config)
        
        # Initialize or load model
        self.model = model or self._initialize_model()
        self.device = self._get_device()
        self.model.to(self.device)
        
        logger.info(f"Model initialized on device: {self.device}")
        
    def _initialize_tokenizer(self) -> T5Tokenizer:
        """Initialize or load the tokenizer.
        
        Returns:
            T5Tokenizer instance
        """
        model_path = self.model_config.get("checkpoint_path", self.model_config["name"])
        return T5Tokenizer.from_pretrained(model_path)
        
    def _initialize_model(self) -> T5ForConditionalGeneration:
        """Initialize or load the model.
        
        Returns:
            T5ForConditionalGeneration instance 
        """
        model_path = self.model_config.get("checkpoint_path", self.model_config["name"])
        logger.info(f"Loading model: {model_path}")
        
        return T5ForConditionalGeneration.from_pretrained(model_path)
        
    def _get_device(self) -> torch.device:
        """Get the appropriate device.
        
        Returns:
            torch.device instance
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
        
    def get_training_args(self) -> TrainingArguments:
        """Get training arguments from config.
        
        Returns:
            TrainingArguments instance
        """
        return TrainingArguments(
            output_dir=self.model_config["save_dir"],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=float(self.training_config["learning_rate"]),
            per_device_train_batch_size=self.training_config["train_batch_size"],
            per_device_eval_batch_size=self.training_config["eval_batch_size"],
            gradient_accumulation_steps=self.training_config["gradient_accumulation_steps"],
            num_train_epochs=self.training_config["num_epochs"],
            weight_decay=self.training_config["weight_decay"],
            save_total_limit=self.training_config["save_total_limit"],
            logging_dir=os.path.join(self.model_config["save_dir"], "logs"),
            logging_steps=self.training_config["logging_steps"],
            load_best_model_at_end=True,
            fp16=self.training_config.get("fp16", False),
            dataloader_num_workers=self.training_config["num_workers"],
            warmup_steps=self.training_config.get("warmup_steps", 0)
        )
        
    def train(
        self,
        train_dataset: DataLoader,
        eval_dataset: Optional[DataLoader] = None
    ) -> None:
        """Train the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        training_args = self.get_training_args()
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model
        logger.info("Saving model...")
        self.save_model()
        
    def save_model(self, path: Optional[str] = None) -> None:
        """Save the model and tokenizer.
        
        Args:
            path: Optional custom save path
        """
        save_path = path or self.model_config["save_dir"]
        os.makedirs(save_path, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
        
    def generate(
        self,
        input_texts: List[str],
        batch_size: Optional[int] = None,
        **generation_kwargs
    ) -> List[str]:
        """Generate simplified texts.
        
        Args:
            input_texts: List of input texts
            batch_size: Optional batch size for generation
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        if batch_size is None:
            batch_size = self.config.get("generation", {}).get("batch_size", 8)
            
        all_outputs = []
        
        # Process in batches
        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i:i + batch_size]
            
            # Tokenize inputs
            inputs = self.tokenizer_handler.tokenize_text(batch_texts)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.config["generation"]["max_length"],
                    num_beams=self.config["generation"]["num_beams"],
                    length_penalty=self.config["generation"]["length_penalty"],
                    early_stopping=self.config["generation"]["early_stopping"],
                    **generation_kwargs
                )
            
            # Decode outputs
            decoded_outputs = self.tokenizer_handler.decode_batch(outputs)
            all_outputs.extend(decoded_outputs)
            
        return all_outputs
        
    @torch.no_grad()
    def evaluate_batch(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a single batch.
        
        Args:
            batch: Dictionary containing input_ids, attention_mask, and labels
            
        Returns:
            Tuple of (loss, logits)
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        return outputs.loss, outputs.logits