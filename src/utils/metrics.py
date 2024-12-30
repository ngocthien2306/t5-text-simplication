import numpy as np
from typing import Dict, List, Tuple, Any
import torch
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import json
import os
from datetime import datetime
from .logging import get_logger

logger = get_logger(__name__)

class MetricsCalculator:
    """Handler for calculating and logging evaluation metrics."""
    
    def __init__(self, config: Dict):
        """Initialize metrics calculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.eval_config = config.get("evaluation", {})
        self.metrics_config = self.eval_config.get("metrics", {})
        
        # Initialize metric calculators
        self._initialize_metrics()
        
        # Setup result saving
        self.results_dir = self.config.get("logging", {}).get("results_dir", "./results")
        os.makedirs(self.results_dir, exist_ok=True)
        
    def _initialize_metrics(self) -> None:
        """Initialize metric calculators based on config."""
        # ROUGE scorer
        if self.metrics_config.get("rouge", True):
            self.rouge_scorer = ROUGEScore()
            
        # BLEU smoothing function
        if self.metrics_config.get("bleu", True):
            self.smooth = SmoothingFunction().method4
            
        # Store BERTScore config
        if self.metrics_config.get("bert_score", True):
            self.bert_score_config = self.eval_config.get("bert_score", {})
            
    def calculate_bleu(
        self,
        predictions: List[str],
        references: List[str]
    ) -> List[float]:
        """Calculate BLEU scores for each prediction.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            List of BLEU scores
        """
        if not self.metrics_config.get("bleu", True):
            return []
            
        logger.info("Calculating BLEU scores...")
        scores = []
        
        for pred, ref in zip(predictions, references):
            score = sentence_bleu(
                [ref.split()],
                pred.split(),
                smoothing_function=self.smooth
            )
            scores.append(score)
            
        return scores
        
    def calculate_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, List[float]]:
        """Calculate ROUGE scores for each prediction.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary of ROUGE scores
        """
        if not self.metrics_config.get("rouge", True):
            return {}
            
        logger.info("Calculating ROUGE scores...")
        scores = {
            "rouge1_fmeasure": [],
            "rouge2_fmeasure": [],
            "rougeL_fmeasure": []
        }
        
        for pred, ref in zip(predictions, references):
            rouge_scores = self.rouge_scorer(pred, ref)
            for key in scores:
                scores[key].append(rouge_scores[key].item())
                
        return scores
        
    def calculate_bert_score(
        self,
        predictions: List[str],
        references: List[str]
    ) -> List[float]:
        """Calculate BERTScore for each prediction.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            List of BERTScores
        """
        if not self.metrics_config.get("bert_score", True):
            return []
            
        logger.info("Calculating BERTScore...")
        _, _, F1 = bert_score(
            predictions,
            references,
            model_type=self.bert_score_config.get("model", "microsoft/deberta-xlarge-mnli"),
            batch_size=self.bert_score_config.get("batch_size", 32),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        return F1.tolist()
        
    def calculate_all_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
        """Calculate all configured metrics.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Tuple of (average metrics, detailed scores)
        """
        # Calculate individual metric scores
        bleu_scores = self.calculate_bleu(predictions, references)
        rouge_scores = self.calculate_rouge(predictions, references)
        bert_scores = self.calculate_bert_score(predictions, references)
        
        # Compile detailed scores
        detailed_scores = {
            "bleu": bleu_scores,
            "rouge1": rouge_scores.get("rouge1_fmeasure", []),
            "rouge2": rouge_scores.get("rouge2_fmeasure", []),
            "rougeL": rouge_scores.get("rougeL_fmeasure", []),
            "bert_score": bert_scores
        }
        
        # Calculate averages
        average_metrics = {}
        for metric, scores in detailed_scores.items():
            if scores:  # Only calculate if scores exist
                average_metrics[metric] = float(np.mean(scores))
                
        return average_metrics, detailed_scores
        
    def log_metrics(
        self,
        average_metrics: Dict[str, float],
        detailed_scores: Dict[str, List[float]]
    ) -> None:
        """Log metrics results.
        
        Args:
            average_metrics: Dictionary of average metric scores
            detailed_scores: Dictionary of detailed metric scores
        """
        logger.info("\nEvaluation Results:")
        logger.info("=" * 50)
        
        for metric, score in average_metrics.items():
            logger.info(f"Average {metric}: {score:.4f}")
            
        # Log detailed statistics for each metric
        logger.info("\nDetailed Statistics:")
        logger.info("=" * 50)
        
        for metric, scores in detailed_scores.items():
            if scores:
                logger.info(f"\n{metric} Statistics:")
                logger.info(f"Mean: {np.mean(scores):.4f}")
                logger.info(f"Median: {np.median(scores):.4f}")
                logger.info(f"Std: {np.std(scores):.4f}")
                logger.info(f"Min: {np.min(scores):.4f}")
                logger.info(f"Max: {np.max(scores):.4f}")
                
    def save_results(
        self,
        predictions: List[str],
        references: List[str],
        average_metrics: Dict[str, float],
        detailed_scores: Dict[str, List[float]]
    ) -> None:
        """Save evaluation results to files.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            average_metrics: Dictionary of average metric scores
            detailed_scores: Dictionary of detailed metric scores
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.results_dir, f"evaluation_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics
        metrics_file = os.path.join(results_dir, "metrics.json")
        results = {
            "average_metrics": average_metrics,
            "detailed_scores": detailed_scores
        }
        
        with open(metrics_file, "w") as f:
            json.dump(results, f, indent=2)
            
        # Save predictions and references
        if self.config.get("logging", {}).get("save_predictions", True):
            examples_file = os.path.join(results_dir, "predictions.txt")
            with open(examples_file, "w") as f:
                for i, (pred, ref) in enumerate(zip(predictions, references)):
                    f.write(f"Example {i+1}:\n")
                    f.write(f"Reference: {ref}\n")
                    f.write(f"Prediction: {pred}\n")
                    f.write("-" * 50 + "\n")
                    
        logger.info(f"Results saved to {results_dir}")