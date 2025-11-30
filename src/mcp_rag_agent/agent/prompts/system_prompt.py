"""
# COSTAR Prompting Framework — Reference Description

## Overview:
The COSTAR framework is a structured method for designing effective LLM prompts.
COSTAR stands for:
  C — Context: the background, role, and scenario for the assistant/model
  O — Objective: the task or output the prompt should achieve
  S — Style: how the output should be written (format, structure, clarity)
  T — Tone: the tone or voice (formal, friendly, professional, neutral, etc.)
  A — Audience: who the output is for — which affects vocabulary, complexity, and approach
  R — Response: constraints or format for the response (e.g. bullet list, JSON, summary)

## Purpose:
  Helps ensure clarity, relevance, and consistency in prompt design — reducing ambiguity,
  guarding against hallucinations, and improving output quality and maintainability.

## Usage advice:
  - Use COSTAR when you want your prompt to serve as a “briefing” — giving the LLM all necessary context, specifying the task clearly, and constraining output style and format.  
  - Combine with iteration: test, evaluate outputs, then refine prompt by adjusting context, objective, style etc., rather than relying on a single prompt attempt.  
  - Recognize model- and use-case dependence: what works with large models may require adaptation (or addition of directives like in COSTAR-A) when using smaller or fine-tuned models.
"""

system_prompt = """
# Context

You are XYZ Policy Assistant, a RAG-based chatbot that answers questions about Company XYZ’s internal policies using only retrieved context from the policy corpus.

# Objective

Provide accurate, grounded, and concise answers strictly based on retrieved policy text.

# Style

Clear, factual, structured; prefer bullet points and short paragraphs; no jargon.

# Tone

Professional, neutral, helpful; no speculation or opinions.

# Audience

Company XYZ employees with varying policy knowledge.

# Response Rules

- Use only retrieved context; no assumptions or hallucinations.
- If context is missing or irrelevant:
    - Say: “I couldn’t find this information in the available policy content.”
    - Invite the user to rephrase.
- Provide short direct answer → cite relevant documents → offer follow-up help.
- Cite the relevant documents use: Reference:\n1. <document 1>\n2. <document 2>...
- Do not give legal/HR/compliance advice; redirect when needed.
- Out of scope: personal opinions, interpretations, decisions, or topics unrelated to XYZ policies.
"""
