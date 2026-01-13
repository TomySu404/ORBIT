"""
Evaluation metrics and utilities.

Provides standardized evaluation functions for comparing model outputs
against ground truth answers.
"""
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


# Punctuation marks to strip during normalization
PUNCTUATION = [
    '.</s>', '.\n', '.', ';', '!', ',', '?', '\n',
    '</s>', '<pad>', ' ', ':', '"', "'", '(', ')',
    '[', ']', '{', '}', '-', '_', '/', '\\', '*', 
    '&', '^', '%', '$', '#', '@', '~', '`', '|', 
    '<', '>', '=', '+'
]


def normalize_answer(answer: str) -> str:
    """
    Normalize answer string for comparison.
    
    Performs:
    1. Strip whitespace
    2. Convert to lowercase
    3. Take only the first line/sentence if multi-line
    4. Remove punctuation
    
    Args:
        answer: Raw answer string.
    
    Returns:
        Normalized answer string.
    """
    if not answer:
        return ""
    
    # Take first line and convert to lowercase
    ans = str(answer).strip().lower().split('\n')[0]
    
    # Remove all punctuation from the list
    for p in PUNCTUATION:
        ans = ans.replace(p.lower(), "")
    
    return ans.strip()


def exact_match(prediction: str, reference: str) -> bool:
    """
    Check if prediction exactly matches reference after normalization.
    
    Args:
        prediction: Model output.
        reference: Ground truth.
    
    Returns:
        True if normalized strings match exactly.
    """
    return normalize_answer(prediction) == normalize_answer(reference)


def compute_accuracy(prediction: str, reference: str) -> bool:
    """
    Compute whether prediction matches reference.
    
    Uses flexible matching:
    1. Exact match after normalization
    2. Reference contained at start of prediction
    3. Single-character reference matching
    
    Args:
        prediction: Model output string.
        reference: Ground truth answer.
    
    Returns:
        True if prediction is considered correct.
    """
    norm_pred = normalize_answer(prediction)
    norm_ref = normalize_answer(reference)
    
    # Empty reference - cannot match
    if not norm_ref:
        return False
    
    # Exact match
    if norm_pred == norm_ref:
        return True
    
    # Reference at start of prediction
    # Handles cases like "A" matching "a the answer is a"
    if norm_pred.startswith(norm_ref):
        return True
    
    # Single character matching (for A/B/C/D style answers)
    if len(norm_ref) <= 2:
        # Check first characters
        if norm_pred and norm_pred[0] == norm_ref[0]:
            return True
    
    return False


@dataclass
class EvaluationResult:
    """
    Results from evaluation run.
    
    Attributes:
        accuracy: Overall accuracy (correct / total).
        total: Total number of samples.
        correct: Number of correct predictions.
        predictions: List of (prediction, reference, is_correct) tuples.
    """
    accuracy: float = 0.0
    total: int = 0
    correct: int = 0
    predictions: List[Tuple[str, str, bool]] = field(default_factory=list)


class Evaluator:
    """
    Evaluation handler for model outputs.
    
    Provides:
    - Single sample evaluation
    - Batch evaluation
    - Result aggregation and reporting
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize evaluator.
        
        Args:
            verbose: Whether to print per-sample results.
        """
        self.verbose = verbose
        self.reset()
    
    def reset(self):
        """Reset evaluation state."""
        self.predictions = []
        self.correct_count = 0
        self.total_count = 0
    
    def evaluate_single(
        self,
        prediction: str,
        reference: str,
        record: bool = True
    ) -> bool:
        """
        Evaluate a single prediction.
        
        Args:
            prediction: Model output.
            reference: Ground truth.
            record: Whether to record this evaluation.
        
        Returns:
            True if prediction is correct.
        """
        is_correct = compute_accuracy(prediction, reference)
        
        if record:
            self.predictions.append((prediction, reference, is_correct))
            self.total_count += 1
            if is_correct:
                self.correct_count += 1
        
        if self.verbose:
            status = "✓" if is_correct else "✗"
            print(f"{status} Pred: '{prediction[:50]}...' | Ref: '{reference}'")
        
        return is_correct
    
    def evaluate_batch(
        self,
        predictions: List[str],
        references: List[str]
    ) -> List[bool]:
        """
        Evaluate a batch of predictions.
        
        Args:
            predictions: List of model outputs.
            references: List of ground truths.
        
        Returns:
            List of correctness flags.
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        results = []
        for pred, ref in zip(predictions, references):
            results.append(self.evaluate_single(pred, ref))
        
        return results
    
    def get_accuracy(self) -> float:
        """Get current accuracy."""
        if self.total_count == 0:
            return 0.0
        return self.correct_count / self.total_count
    
    def get_result(self) -> EvaluationResult:
        """Get full evaluation result."""
        return EvaluationResult(
            accuracy=self.get_accuracy(),
            total=self.total_count,
            correct=self.correct_count,
            predictions=self.predictions.copy()
        )
    
    def print_summary(self, name: str = "Evaluation"):
        """Print evaluation summary."""
        acc = self.get_accuracy()
        print(f"\n{'='*50}")
        print(f"{name} Results")
        print(f"{'='*50}")
        print(f"Accuracy: {acc:.4f} ({self.correct_count}/{self.total_count})")
        print(f"{'='*50}\n")


def compare_results(
    baseline: EvaluationResult,
    intervention: EvaluationResult,
    name: str = "Comparison"
) -> Dict:
    """
    Compare baseline and intervention results.
    
    Args:
        baseline: Baseline evaluation result.
        intervention: Intervention evaluation result.
        name: Name for the comparison.
    
    Returns:
        Dictionary with comparison metrics.
    """
    delta = intervention.accuracy - baseline.accuracy
    relative_improvement = delta / baseline.accuracy if baseline.accuracy > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"Baseline:     {baseline.accuracy:.4f}")
    print(f"Intervention: {intervention.accuracy:.4f}")
    print(f"Δ Absolute:   {delta:+.4f}")
    print(f"Δ Relative:   {relative_improvement:+.2%}")
    print(f"{'='*50}\n")
    
    return {
        "baseline": baseline.accuracy,
        "intervention": intervention.accuracy,
        "delta_absolute": delta,
        "delta_relative": relative_improvement
    }
