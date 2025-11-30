"""
Main entry point for the RAG agent evaluation.

This script orchestrates the two-phase evaluation process:
1. Generate answers (can be skipped if answers file already exists)
2. Compute metrics from the answers
"""

import asyncio
from datetime import datetime
from pathlib import Path

from answer_generator import AnswerGenerator
from metrics_evaluator import MetricsEvaluator

async def main():
    """Main entry point for evaluation."""
    # Define file paths
    input_file = Path("data/evaluation_documents/expected_behaviour.xlsx")
    results_dir = Path("evaluation/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    answers_file = results_dir / f"answers_{timestamp}.csv"
    final_file = results_dir / f"evaluation_results_{timestamp}.csv"
    
    # Check if we should use an existing answers file
    existing_answers = sorted(results_dir.glob("answers_*.csv"))
    
    skip_generation = False
    if existing_answers:
        latest_answers = existing_answers[-1]
        print("=" * 80)
        print("EXISTING ANSWERS FILE FOUND")
        print("=" * 80)
        print(f"Latest answers file: {latest_answers.name}")
        print(f"Full path: {latest_answers}")
        
        # Ask user if they want to use existing file or generate new answers
        response = input("\nUse this file and skip answer generation? (y/n): ").strip().lower()
        
        if response == 'y':
            print("\n✓ Using existing answers file. Skipping Phase 1.")
            answers_file = latest_answers
            # Update final file to match the timestamp of answers
            timestamp_from_answers = latest_answers.stem.replace("answers_", "")
            final_file = results_dir / f"evaluation_results_{timestamp_from_answers}.csv"
            skip_generation = True
        else:
            print("\n✓ Generating new answers...")
    else:
        print("\nNo existing answers found. Will generate new answers.\n")
    
    # PHASE 1: Generate answers (if needed)
    if not skip_generation:
        print("\n" + "=" * 80)
        print("STARTING EVALUATION PIPELINE")
        print("=" * 80)
        
        generator = AnswerGenerator(input_file, answers_file)
        await generator.generate_all_answers()
    
    # PHASE 2: Compute metrics
    print("\n" + "=" * 80)
    print("PHASE 2: COMPUTING METRICS")
    print("=" * 80)
    
    evaluator = MetricsEvaluator(answers_file, final_file)
    evaluator.compute_all_metrics()
    
    # Final summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Answers file: {answers_file}")
    print(f"Final results: {final_file}")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
