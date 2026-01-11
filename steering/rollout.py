"""
Rollout-based contrastive pair generation.

This module implements the first key improvement:
Instead of using manually constructed (question + correct_answer) vs (question + wrong_answer),
we let the model generate multiple responses and naturally form contrastive pairs.

Key benefits:
1. No distribution shift between training and inference
2. More pairs per sample (better statistics, lower variance)
3. Automatic classification of correct/incorrect responses
"""
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from ..config import RolloutConfig


@dataclass
class ContrastivePair:
    """
    A single contrastive pair for difference computation.
    
    Attributes:
        positive: Response classified as correct.
        negative: Response classified as incorrect.
        used_reread: Whether re-read mechanism was used to generate positive.
    """
    positive: str
    negative: str
    used_reread: bool = False


@dataclass
class RolloutResult:
    """
    Result of rollout generation for a single sample.
    
    Attributes:
        question: Original input question/prompt.
        correct_answer: Ground truth answer.
        rollout_responses: All generated responses.
        correct_responses: Responses matching the correct answer.
        incorrect_responses: Responses not matching the correct answer.
        contrastive_pairs: All (positive, negative) pairs formed.
        used_reread_mechanism: Whether fallback re-read was needed.
    """
    question: str
    correct_answer: str
    rollout_responses: List[str] = field(default_factory=list)
    correct_responses: List[str] = field(default_factory=list)
    incorrect_responses: List[str] = field(default_factory=list)
    contrastive_pairs: List[ContrastivePair] = field(default_factory=list)
    used_reread_mechanism: bool = False


class RolloutGenerator:
    """
    Generates contrastive pairs using rollout (sampling) from the model.
    
    Mathematical motivation:
    Let p_ar(h|q) be the activation distribution under auto-regressive generation.
    Let p_tf(h|q,a) be the activation distribution under teacher forcing.
    
    Previous methods compute: E[h|q,a+] - E[h|q,a-] under p_tf
    Our method computes: E[h|q,a+] - E[h|q,a-] under p_ar
    
    Since inference uses auto-regressive generation, our estimate has no distribution shift,
    leading to better alignment between the steering vector and actual model behavior.
    """
    
    # Punctuation marks to strip when comparing answers
    PUNCTUATION = [
        '.</s>', '.\n', '.', ';', '!', ',', '?', '\n', 
        '</s>', '<pad>', ' ', ':', '"', "'", '(', ')'
    ]
    
    def __init__(self, model, config: RolloutConfig):
        """
        Initialize rollout generator.
        
        Args:
            model: ModelWrapper instance for generation.
            config: Rollout configuration.
        """
        self.model = model
        self.config = config
    
    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize answer string for comparison.
        
        Strips whitespace, punctuation, and converts to lowercase.
        
        Args:
            answer: Raw answer string.
        
        Returns:
            Normalized answer string.
        """
        ans = answer.strip().lower()
        for p in self.PUNCTUATION:
            ans = ans.replace(p.lower(), "")
        return ans.strip()
    
    def _is_correct(self, response: str, reference: str) -> bool:
        """
        Check if a response matches the reference answer.
        
        Uses normalized comparison with support for partial matching
        (reference contained in response).
        
        Args:
            response: Model-generated response.
            reference: Ground truth answer.
        
        Returns:
            True if response is considered correct.
        """
        norm_response = self._normalize_answer(response)
        norm_reference = self._normalize_answer(reference)
        
        # Empty check
        if not norm_reference:
            return False
        
        # Exact match
        if norm_response == norm_reference:
            return True
        
        # Check if reference is contained at the start of response
        # This handles cases like "A" matching "A. The answer is..."
        if norm_response.startswith(norm_reference):
            return True
        
        # Check if reference is a single token and response starts with it
        if len(norm_reference) <= 2 and norm_response[:len(norm_reference)] == norm_reference:
            return True
        
        return False
    
    def _reread_generate(self, question: str, correct_answer: str) -> str:
        """
        Re-read mechanism: Generate a response biased toward the correct answer.
        
        This is the fallback when rollout fails to produce any correct responses.
        It uses the correct answer directly as the "generated" response.
        
        Mathematical note:
        For samples where the model cannot naturally produce correct responses,
        we mark them with a flag and optionally down-weight them during aggregation.
        This prevents the steering vector from being dominated by "forced" examples.
        
        Args:
            question: Original question.
            correct_answer: The correct answer to use.
        
        Returns:
            The correct answer string.
        """
        # Simply return the correct answer
        # This is marked as "re-read" and will be down-weighted
        return correct_answer
    
    def generate_rollouts(self, question: str) -> List[str]:
        """
        Generate multiple diverse responses via temperature sampling.
        
        Args:
            question: Input question/prompt.
        
        Returns:
            List of generated response strings.
        """
        responses = []
        
        # Generate responses one at a time for diversity
        # (batch generation tends to produce more similar outputs)
        for _ in range(self.config.num_rollouts):
            response = self.model.generate(
                question,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                num_return_sequences=1
            )[0]
            responses.append(response)
        
        return responses
    
    def build_contrastive_pairs(
        self,
        question: str,
        correct_answer: str,
        wrong_answer: Optional[str] = None
    ) -> RolloutResult:
        """
        Build contrastive pairs from rollout responses.
        
        Algorithm:
        1. Generate n rollout responses using temperature sampling
        2. Classify each response as correct or incorrect
        3. Form all (correct, incorrect) pairs
        4. If no correct responses, fall back to re-read mechanism
        
        Mathematical benefit:
        If rollout produces k correct and (n-k) incorrect responses,
        we get k * (n-k) pairs per sample instead of just 1.
        This reduces the variance of our difference estimate by a factor of k*(n-k).
        
        Args:
            question: Input question.
            correct_answer: Ground truth answer.
            wrong_answer: Optional known wrong answer (used if all rollouts are correct).
        
        Returns:
            RolloutResult containing all pairs and metadata.
        """
        # Step 1: Generate rollout responses
        rollout_responses = self.generate_rollouts(question)
        
        # Step 2: Classify responses into correct/incorrect
        correct_responses = []
        incorrect_responses = []
        
        for resp in rollout_responses:
            if self._is_correct(resp, correct_answer):
                correct_responses.append(resp)
            else:
                incorrect_responses.append(resp)
        
        # Step 3: Handle edge cases
        used_reread = False
        
        # Case A: No correct responses found
        if len(correct_responses) == 0:
            if self.config.use_reread_fallback:
                # Use re-read mechanism (direct insertion of correct answer)
                reread_response = self._reread_generate(question, correct_answer)
                correct_responses.append(reread_response)
                used_reread = True
            else:
                # Return empty result - sample will be skipped
                return RolloutResult(
                    question=question,
                    correct_answer=correct_answer,
                    rollout_responses=rollout_responses,
                    correct_responses=[],
                    incorrect_responses=incorrect_responses,
                    contrastive_pairs=[],
                    used_reread_mechanism=False
                )
        
        # Case B: No incorrect responses found (model is too good on this sample)
        if len(incorrect_responses) == 0:
            if wrong_answer:
                # Use the provided wrong answer
                incorrect_responses.append(wrong_answer)
            else:
                # Cannot form contrastive pairs
                return RolloutResult(
                    question=question,
                    correct_answer=correct_answer,
                    rollout_responses=rollout_responses,
                    correct_responses=correct_responses,
                    incorrect_responses=[],
                    contrastive_pairs=[],
                    used_reread_mechanism=used_reread
                )
        
        # Step 4: Build all pairwise combinations
        pairs = []
        for pos in correct_responses:
            for neg in incorrect_responses:
                pairs.append(ContrastivePair(
                    positive=pos,
                    negative=neg,
                    # Mark pairs involving re-read response
                    used_reread=used_reread and (pos == correct_responses[0])
                ))
        
        return RolloutResult(
            question=question,
            correct_answer=correct_answer,
            rollout_responses=rollout_responses,
            correct_responses=correct_responses,
            incorrect_responses=incorrect_responses,
            contrastive_pairs=pairs,
            used_reread_mechanism=used_reread
        )
