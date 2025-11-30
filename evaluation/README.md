# RAG Agent Evaluation Module

This module provides automated evaluation of the RAG-based agent using RAGAS (RAG Assessment) metrics.

## Overview

The evaluation module implements a **two-phase approach**:

1. **Phase 1: Answer Generation** - Invokes the RAG agent for each test question and saves answers to CSV
2. **Phase 2: Metrics Computation** - Computes RAGAS metrics by comparing generated answers to ground truth

Results are saved incrementally after each step to prevent data loss.

## Architecture

```
evaluation/
├── __init__.py              # Package initialization
├── answer_generator.py      # Phase 1: Answer generation from RAG agent
├── metrics_evaluator.py     # Phase 2: RAGAS metrics computation
├── metrics.py              # RAGAS metrics wrapper
├── main.py                 # Entry point - orchestrates both phases
├── README.md              # This file
└── results/               # Output directory for CSV files
    ├── answers_*.csv              # Generated answers (Phase 1 output)
    └── evaluation_results_*.csv   # Final results with metrics (Phase 2 output)
```

## Metrics

The module computes the following context-free RAGAS metrics:

- **Answer Relevancy**: Measures how relevant the generated answer is to the input question
- **Answer Similarity**: Computes semantic similarity between the generated answer and reference answer
- **Answer Correctness**: Evaluates factual correctness combining similarity and accuracy

These metrics don't require retrieved document contexts, making them suitable for evaluating the agent's final responses.

## Installation

Install the required dependencies (already in main requirements.txt):

```bash
pip install -r requirements.txt
```

Key packages for evaluation:
- `ragas>=0.1.0` - RAG evaluation framework
- `openpyxl>=3.1.0` - Excel file reading
- `pandas>=2.0.0` - Data manipulation
- `datasets>=2.14.0` - Required by RAGAS

## Configuration

The evaluation uses the `evaluation_model` setting from `src/mcp_rag_agent/core/config.py`. By default, this is set to `gpt-4o-mini` via the `EVALUATION_MODEL_NAME` environment variable.

To change the evaluation model, update your `.env` file:

```bash
EVALUATION_MODEL_NAME=gpt-4o-mini
```

## Usage

Run the evaluation from the project root:

```bash
python evaluation/main.py
```

### Two-Phase Execution

**First Run** (no existing answers):
1. Generates answers for all test questions
2. Saves to `answers_TIMESTAMP.csv`
3. Computes metrics
4. Saves to `evaluation_results_TIMESTAMP.csv`

**Subsequent Runs** (existing answers found):
```
EXISTING ANSWERS FILE FOUND
Latest answers file: answers_20241130_081500.csv

Use this file and skip answer generation? (y/n):
```

- Type `y` to skip Phase 1 and only compute metrics (useful for trying different evaluation models)
- Type `n` to generate fresh answers

## Input Format

Test cases are read from `data/evaluation_documents/expected_behaviour.xlsx`:

| # | question | reference | source |
|---|----------|-----------|--------|
| 1 | In the UK, how many days of annual leave do employees receive? | 25 days per year. | 3 - Annual Leave.txt |

**Columns:**
- `#`: Test case number
- `question`: Question to ask the agent
- `reference`: Ground truth/expected answer
- `source`: Document containing the answer

## Output Format

### Phase 1 Output: `answers_TIMESTAMP.csv`

Contains original data plus generated answers:

```csv
#,question,reference,source,generated_answer,agent_error,timestamp
1,"In the UK, how many days...",25 days per year.,3 - Annual Leave.txt,"Employees in the UK receive 25 days...",,2024-11-30T08:15:30.123456
```

### Phase 2 Output: `evaluation_results_TIMESTAMP.csv`

Contains all Phase 1 columns plus computed metrics:

```csv
#,question,reference,source,generated_answer,answer_relevancy,answer_similarity,answer_correctness,agent_error,metrics_error,timestamp
1,"In the UK, how many days...",25 days per year.,3 - Annual Leave.txt,"Employees receive 25 days...",0.9876,0.9543,0.9234,,,2024-11-30T08:15:45.789012
```

**New columns:**
- `answer_relevancy`: Relevancy score (0-1)
- `answer_similarity`: Similarity score (0-1)
- `answer_correctness`: Correctness score (0-1)
- `agent_error`: Error message if answer generation failed
- `metrics_error`: Error message if metrics computation failed
- `timestamp`: ISO format timestamp

## Key Features

✅ **MCP-Free Implementation** - Uses the refactored agent without MCP dependencies  
✅ **Two-Phase Architecture** - Separates answer generation from metrics computation  
✅ **Incremental Saving** - Saves progress after each question/metric  
✅ **Skip Answer Generation** - Reuse existing answers for different evaluations  
✅ **Error Handling** - Gracefully handles and logs errors  
✅ **Summary Statistics** - Displays average metrics and error counts  

## Error Handling

The evaluation module handles errors gracefully:

- **Agent errors**: Captured in `agent_error` column, metrics skipped for that row
- **Metrics errors**: Captured in `metrics_error` column
- **Incremental saving**: Progress is preserved even if the process is interrupted

Evaluation continues for remaining test cases even if some fail.

## Example Output

```
================================================================================
PHASE 1: GENERATING ANSWERS
================================================================================
Loading test cases from data\evaluation_documents\expected_behaviour.xlsx...
Loaded 10 test cases.

[1/10] Generating answer for: In the UK, how many days of annual leave do employees receive?...
  Generated answer: Employees in the UK receive 25 days of annual leave per year...
  Saved to evaluation\results\answers_20241130_081500.csv

...

================================================================================
Answer generation complete!
Total questions: 10
Answers saved to: evaluation\results\answers_20241130_081500.csv
Successful: 10/10
================================================================================

================================================================================
PHASE 2: COMPUTING METRICS
================================================================================
Loading answers from evaluation\results\answers_20241130_081500.csv...
Loaded 10 answers.

[1/10] Computing metrics for: In the UK, how many days of annual leave...
  Computing RAGAS metrics...
  Metrics: {'answer_relevancy': 0.9876, 'answer_similarity': 0.9543, 'answer_correctness': 0.9234}
  Saved results to evaluation\results\evaluation_results_20241130_081500.csv

...

Metrics computation complete. Saved to: evaluation\results\evaluation_results_20241130_081500.csv

Agent invocation errors: 0
Metrics computation errors: 0

Successful evaluations: 10

Metric Statistics:
----------------------------------------
answer_relevancy: 0.9234
answer_similarity: 0.8876
answer_correctness: 0.8654
================================================================================
```

## Module Components

### answer_generator.py

**Class:** `AnswerGenerator`

- Loads test cases from Excel
- Uses the pre-initialized MCP-free agent
- Generates answers for each question with timeout protection
- Saves answers incrementally to CSV

**Key method:** `generate_all_answers()` - Orchestrates Phase 1

### metrics_evaluator.py

**Class:** `MetricsEvaluator`

- Loads generated answers from CSV
- Computes RAGAS metrics for each answer
- Skips metrics for answers with errors
- Saves results incrementally to CSV

**Key method:** `compute_all_metrics()` - Orchestrates Phase 2

### metrics.py

**Class:** `RAGASEvaluator`

- Wraps RAGAS library functionality
- Provides `evaluate_single()` method for computing metrics
- Handles RAGAS-specific error scenarios

### main.py

Entry point that:
1. Checks for existing answer files
2. Prompts user to reuse or regenerate
3. Runs Phase 1 (if needed)
4. Runs Phase 2 (always)
5. Displays final summary

## Performance Considerations

- **Answer Generation**: 5-30 seconds per question depending on complexity
- **Metrics Computation**: 2-5 seconds per question (requires LLM calls)
- **Expected Runtime**: 2-5 minutes for 10 test cases (full run)
- **Reusing Answers**: ~30 seconds for 10 test cases (metrics only)

## Troubleshooting

### Common Issues

**Import errors**: 
```bash
# Ensure you're in the project root
cd d:/Projects/mcp-rag-agent
python evaluation/main.py
```

**Agent initialization fails**: 
- Check MongoDB is running
- Verify `.env` configuration
- Ensure documents are indexed

**RAGAS API errors**: 
- Verify OpenAI API key is valid
- Check API quota/rate limits

**Excel file not found**: 
- Ensure `data/evaluation_documents/expected_behaviour.xlsx` exists

## Future Enhancements

Potential improvements:
- Add context-aware metrics when agent exposes retrieved documents
- Support batch evaluation for improved throughput
- Add visualization with charts and graphs
- Compare multiple agent configurations
- CI/CD integration for automated regression testing
- Support for additional metrics (faithfulness, context recall, etc.)

## Dependencies

The evaluation module depends on:
- The MCP-free RAG agent from `src/mcp_rag_agent/agent/`
- Configuration from `src/mcp_rag_agent/core/config.py`
- Test data from `data/evaluation_documents/expected_behaviour.xlsx`

## Version History

- **v2.0**: Refactored to use MCP-free agent, clean two-phase architecture
- **v1.0**: Initial implementation with MCP-based agent
