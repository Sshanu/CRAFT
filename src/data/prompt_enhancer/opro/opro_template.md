You are an expert prompt engineer. Your job is to propose a new prompt that achieves a HIGHER validation score than every prompt shown below.

# History (past prompts and their validation scores, sorted ascending — higher score is better)

{history}

# Current prompt (the one to improve)

{current_prompt}

# Instructions

Generate ONE new prompt that should score higher than every prompt above. Keep the same task and intent, but improve any of:
- clarity and specificity of instructions
- coverage of edge cases / failure modes seen in the history
- structure and reasoning guidance
- conciseness (only where it does not hurt clarity)

Output **only the new prompt text** — no markdown fences, no preamble, no explanation, no surrounding quotes. The output must be a complete, runnable system-prompt ready to use as-is.
