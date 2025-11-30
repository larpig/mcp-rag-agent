"""
RAGAS metrics computation module.

This module provides wrapper functions for computing RAGAS metrics
that don't require retrieved contexts.
"""

from typing import Dict, List, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, answer_similarity, answer_correctness
from langchain_openai import ChatOpenAI


class RAGASEvaluator:
    """Wrapper for RAGAS evaluation metrics."""
    
    def __init__(self, model_name: str, api_key: str):
        """
        Initialize the RAGAS evaluator.
        
        Args:
            model_name: Name of the OpenAI model to use for evaluation
            api_key: OpenAI API key
        """
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.0
        )
        
        # Configure metrics to use our LLM
        self.metrics = [
            answer_relevancy,
            answer_similarity,
            answer_correctness
        ]
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        ground_truth: str
    ) -> Dict[str, float]:
        """
        Evaluate a single question-answer pair.
        
        Args:
            question: The input question
            answer: The generated answer from the agent
            ground_truth: The reference/expected answer
            
        Returns:
            Dictionary with metric names and scores
        """
        # Create a dataset with a single sample
        data = {
            "question": [question],
            "answer": [answer],
            "ground_truth": [ground_truth]
        }
        
        dataset = Dataset.from_dict(data)
        
        # Evaluate using RAGAS
        try:
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.llm
            )
            
            # Extract scores - RAGAS returns lists even for single items
            # Also convert numpy types to Python floats
            def extract_scalar(value):
                """Extract scalar value from list or numpy type."""
                if isinstance(value, list) and len(value) > 0:
                    value = value[0]
                # Convert numpy types to Python float
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
            
            scores = {
                "answer_relevancy": extract_scalar(result["answer_relevancy"]),
                "answer_similarity": extract_scalar(result["answer_similarity"]),
                "answer_correctness": extract_scalar(result["answer_correctness"])
            }
            
            return scores
            
        except Exception as e:
            # Return None values if evaluation fails
            return {
                "answer_relevancy": None,
                "answer_similarity": None,
                "answer_correctness": None,
                "error": str(e)
            }
    
    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        ground_truths: List[str]
    ) -> Dict[str, List[float]]:
        """
        Evaluate a batch of question-answer pairs.
        
        Args:
            questions: List of input questions
            answers: List of generated answers
            ground_truths: List of reference answers
            
        Returns:
            Dictionary with metric names and lists of scores
        """
        data = {
            "question": questions,
            "answer": answers,
            "ground_truth": ground_truths
        }
        
        dataset = Dataset.from_dict(data)
        
        # Evaluate using RAGAS
        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm
        )
        
        return result
