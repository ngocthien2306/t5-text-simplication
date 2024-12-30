from typing import Dict, List, Union
import torch
from transformers import PreTrainedTokenizer
from ..utils.logging import get_logger

logger = get_logger(__name__)

class TokenizerHandler:
    """Handler for tokenization operations."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, config: Dict):
        """Initialize tokenizer handler.
        
        Args:
            tokenizer: HuggingFace tokenizer instance
            config: Configuration dictionary
        """
        self.tokenizer = tokenizer
        self.config = config
        self.model_config = config["model"]
        self.prefix = config.get("tokenizer", {}).get("prefix", "simplify: ")
        
    def tokenize_text(
        self,
        text: Union[str, List[str]],
        max_length: int = None,
        padding: str = "max_length",
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Tokenize input text.
        
        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate sequences
            
        Returns:
            Dictionary containing tokenized inputs
        """
        if max_length is None:
            max_length = self.model_config["max_input_length"]
            
        if isinstance(text, str):
            text = [text]
            
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )
        
        return encoded
        
    def tokenize_batch(
        self,
        source_texts: List[str],
        target_texts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of source and target texts.
        
        Args:
            source_texts: List of source texts
            target_texts: List of target texts
            
        Returns:
            Dictionary containing tokenized inputs and labels
        """
        # Add prefix to source texts
        source_texts = [self.prefix + text for text in source_texts]
        
        # Tokenize inputs
        inputs = self.tokenize_text(
            source_texts,
            max_length=self.model_config["max_input_length"]
        )
        
        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenize_text(
                target_texts,
                max_length=self.model_config["max_target_length"]
            )
            
        # Combine inputs and labels
        inputs["labels"] = targets["input_ids"]
        
        return inputs
    
    def decode_batch(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode batch of token IDs to texts.
        
        Args:
            token_ids: Tensor of token IDs
            skip_special_tokens: Whether to remove special tokens
            
        Returns:
            List of decoded texts
        """
        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )