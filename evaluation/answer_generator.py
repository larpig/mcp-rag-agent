"""
Answer generation module for RAG agent evaluation.

This module handles generating answers from the RAG agent for test questions.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from mcp_rag_agent.agent import agent


class AnswerGenerator:
    """Generates answers from the RAG agent for evaluation questions."""
    
    def __init__(self, input_file: Path, output_file: Path):
        """
        Initialize the answer generator.
        
        Args:
            input_file: Path to Excel file with test cases
            output_file: Path to save generated answers CSV
        """
        self.input_file = input_file
        self.output_file = output_file
        self.agent = agent  # Use the pre-initialized MCP-free agent
        
    def load_test_cases(self) -> pd.DataFrame:
        """
        Load test cases from Excel file.
        
        Returns:
            DataFrame with test cases
        """
        print(f"Loading test cases from {self.input_file}...")
        df = pd.read_excel(self.input_file, sheet_name="Sheet1")
        print(f"Loaded {len(df)} test cases.")
        return df
    
    def save_incremental(self, df: pd.DataFrame):
        """
        Save answers incrementally to CSV.
        
        Args:
            df: DataFrame to save
        """
        df.to_csv(self.output_file, index=False)
    
    async def generate_answer_for_question(self, question: str, timeout: int = 120) -> Dict[str, Any]:
        """
        Generate an answer for a single question.
        
        Args:
            question: The question to ask the agent
            timeout: Maximum time in seconds to wait for response (default: 120)
            
        Returns:
            Dictionary with 'answer' and optional 'error' keys
        """
        try:
            # Invoke agent with timeout
            result = await asyncio.wait_for(
                self.agent.ainvoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": question,
                            }
                        ]
                    }
                ),
                timeout=timeout
            )
            
            # Extract the answer
            final_message = result["messages"][-1]
            generated_answer = final_message.content
            
            return {"answer": generated_answer, "error": None}
            
        except asyncio.TimeoutError:
            error_msg = f"Agent invocation timed out after {timeout} seconds"
            return {"answer": None, "error": error_msg}
        except Exception as e:
            error_msg = f"Agent invocation failed: {str(e)[:200]}"
            return {"answer": None, "error": error_msg}
    
    async def generate_all_answers(self):
        """Generate answers for all test cases."""
        print("=" * 80)
        print("PHASE 1: GENERATING ANSWERS")
        print("=" * 80)
        
        # Load test cases
        df = self.load_test_cases()
        
        # Generate answers for all questions
        answers = []
        total = len(df)
        
        for idx, row in df.iterrows():
            question = row["question"]
            reference = row["reference"]
            source = row["source"]
            
            print(f"\n[{idx + 1}/{total}] Generating answer for: {question[:80]}...")
            
            # Generate answer
            result = await self.generate_answer_for_question(question)
            generated_answer = result["answer"]
            agent_error = result["error"]
            
            if generated_answer:
                print(f"  Generated answer: {generated_answer[:80]}...")
            else:
                print(f"  ERROR: {agent_error}")
            
            # Store result
            answer_result = {
                "#": row["#"],
                "question": question,
                "reference": reference,
                "source": source,
                "generated_answer": generated_answer,
                "agent_error": agent_error,
                "timestamp": datetime.now().isoformat()
            }
            answers.append(answer_result)
            
            # Save answers incrementally
            answers_df = pd.DataFrame(answers)
            self.save_incremental(answers_df)
            print(f"  Saved to {self.output_file}")
        
        print(f"\n{'=' * 80}")
        print(f"Answer generation complete!")
        print(f"Total questions: {total}")
        print(f"Answers saved to: {self.output_file}")
        
        # Count errors
        answers_df = pd.DataFrame(answers)
        agent_errors = answers_df["agent_error"].notna().sum()
        successful = total - agent_errors
        print(f"Successful: {successful}/{total}")
        if agent_errors > 0:
            print(f"Errors: {agent_errors}")
        print("=" * 80)
        
        return self.output_file
