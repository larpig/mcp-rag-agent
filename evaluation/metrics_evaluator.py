"""
Metrics evaluation module.

This module computes RAGAS metrics for already-generated answers.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from mcp_rag_agent.core.config import config

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import RAGASEvaluator

class MetricsEvaluator:
    """Computes RAGAS metrics for generated answers."""
    
    def __init__(self, answers_file: Path, output_file: Path):
        """
        Initialize the metrics evaluator.
        
        Args:
            answers_file: Path to CSV file with generated answers
            output_file: Path to save evaluation results CSV
        """
        self.answers_file = answers_file
        self.output_file = output_file
        
        # Initialize RAGAS evaluator
        self.ragas_evaluator = RAGASEvaluator(
            model_name=config.evaluation_model,
            api_key=config.model_api_key
        )
    
    def load_answers(self) -> pd.DataFrame:
        """
        Load generated answers from CSV file.
        
        Returns:
            DataFrame with generated answers
        """
        print(f"Loading answers from {self.answers_file}...")
        df = pd.read_csv(self.answers_file)
        print(f"Loaded {len(df)} answers.")
        return df
    
    def save_incremental(self, df: pd.DataFrame):
        """
        Save results with metrics incrementally to CSV.
        
        Args:
            df: DataFrame to save
        """
        df.to_csv(self.output_file, index=False)
    
    def compute_metrics_for_row(self, row: pd.Series, idx: int, total: int) -> Dict[str, Any]:
        """
        Compute metrics for a row that already has a generated answer.
        
        Args:
            row: Row from the DataFrame containing question, answer, and reference
            idx: Current row index (1-based)
            total: Total number of rows
            
        Returns:
            Dictionary with all data including computed metrics
        """
        question = row["question"]
        reference = row["reference"]
        generated_answer = row["generated_answer"]
        agent_error = row.get("agent_error")
        
        print(f"\n[{idx}/{total}] Computing metrics for: {question[:80]}...")
        
        # Skip metrics if there was an agent error or no answer
        # Check if agent_error is not None and not empty string
        has_agent_error = pd.notna(agent_error) and str(agent_error).strip() != ""
        has_no_answer = pd.isna(generated_answer) or str(generated_answer).strip() == ""
        
        if has_agent_error or has_no_answer:
            error_msg = "Skipped due to agent error" if has_agent_error else "No answer generated"
            print(f"  Skipping metrics: {error_msg}")
            return {
                "#": row["#"],
                "question": question,
                "reference": row["reference"],
                "source": row["source"],
                "generated_answer": generated_answer,
                "answer_relevancy": None,
                "answer_similarity": None,
                "answer_correctness": None,
                "agent_error": agent_error if has_agent_error else None,
                "metrics_error": error_msg,
                "timestamp": row["timestamp"]
            }
        
        # Compute RAGAS metrics
        print("  Computing RAGAS metrics...")
        metrics = self.ragas_evaluator.evaluate_single(
            question=question,
            answer=generated_answer,
            ground_truth=reference
        )
        
        # Extract metrics and potential error
        metrics_error = metrics.pop("error", None)
        
        print(f"  Metrics: {metrics}")
        
        # Return complete row data
        return {
            "#": row["#"],
            "question": question,
            "reference": row["reference"],
            "source": row["source"],
            "generated_answer": generated_answer,
            "answer_relevancy": metrics.get("answer_relevancy"),
            "answer_similarity": metrics.get("answer_similarity"),
            "answer_correctness": metrics.get("answer_correctness"),
            "agent_error": agent_error,
            "metrics_error": metrics_error,
            "timestamp": row["timestamp"]
        }
    
    def save_summary_statistics(self, final_df: pd.DataFrame):
        """
        Save summary statistics to a separate file.
        
        Args:
            final_df: DataFrame with all evaluation results
        """
        # Generate summary filename
        summary_file = self.output_file.parent / self.output_file.name.replace(
            "evaluation_results_", "summary_statistics_"
        )
        
        # Count errors
        agent_errors = final_df["agent_error"].notna().sum()
        metrics_errors = final_df["metrics_error"].notna().sum()
        
        # Get successful evaluations
        successful = final_df[
            (final_df["agent_error"].isna()) & 
            (final_df["metrics_error"].isna())
        ]
        
        # Calculate statistics
        total_cases = len(final_df)
        successful_count = len(successful)
        
        summary = {
            "Total Test Cases": [total_cases],
            "Successful Evaluations": [successful_count],
            "Agent Errors": [int(agent_errors)],
            "Metrics Errors": [int(metrics_errors)],
            "Success Rate (%)": [round((successful_count / total_cases) * 100, 2) if total_cases > 0 else 0]
        }
        
        # Add mean metrics for successful evaluations
        if len(successful) > 0:
            for metric in ["answer_relevancy", "answer_similarity", "answer_correctness"]:
                if metric in successful.columns:
                    mean_val = successful[metric].mean()
                    summary[f"Mean {metric.replace('_', ' ').title()}"] = [round(mean_val, 4)]
                else:
                    summary[f"Mean {metric.replace('_', ' ').title()}"] = [None]
        else:
            summary["Mean Answer Relevancy"] = [None]
            summary["Mean Answer Similarity"] = [None]
            summary["Mean Answer Correctness"] = [None]
        
        # Save to CSV
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(summary_file, index=False)
        
        return summary_file
    
    def compute_all_metrics(self):
        """Compute metrics for all generated answers."""
        print("=" * 80)
        print("COMPUTING METRICS")
        print("=" * 80)
        
        # Load answers
        answers_df = self.load_answers()
        
        # Compute metrics for each row
        final_results = []
        total = len(answers_df)
        for idx, row in answers_df.iterrows():
            metrics_result = self.compute_metrics_for_row(row, idx + 1, total)
            final_results.append(metrics_result)
            
            # Save final results incrementally
            final_df = pd.DataFrame(final_results)
            self.save_incremental(final_df)
            print(f"  Saved results to {self.output_file}")
        
        print(f"\nMetrics computation complete. Saved to: {self.output_file}")
        
        # Calculate and display summary statistics
        final_df = pd.DataFrame(final_results)
        
        # Count errors
        agent_errors = final_df["agent_error"].notna().sum()
        metrics_errors = final_df["metrics_error"].notna().sum()
        
        print(f"\nAgent invocation errors: {agent_errors}")
        print(f"Metrics computation errors: {metrics_errors}")
        
        # Display metric statistics for successful evaluations
        successful = final_df[
            (final_df["agent_error"].isna()) & 
            (final_df["metrics_error"].isna())
        ]
        
        if len(successful) > 0:
            print(f"\nSuccessful evaluations: {len(successful)}")
            print("\nMetric Statistics:")
            print("-" * 40)
            for metric in ["answer_relevancy", "answer_similarity", "answer_correctness"]:
                if metric in successful.columns:
                    mean_val = successful[metric].mean()
                    print(f"{metric}: {mean_val:.4f}")
        
        # Save summary statistics to file
        summary_file = self.save_summary_statistics(final_df)
        print(f"\nSummary statistics saved to: {summary_file}")
        
        print("=" * 80)
        
        return self.output_file
