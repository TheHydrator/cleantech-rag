import json
import os

from rag.embeddings import get_client
from rag.generation import build_context

# Same prompt as the original notebook, with a stricter output format
GRADER_PROMPT = """You are a strict grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.

Here are the facts (Retrieved Documents):
{documents}

Here is the answer (Generated Output):
{generation}

Criteria:
- If the answer includes facts not present in the documents, it is a HALLUCINATION.
- If the answer is fully supported by the documents, score it 'yes'.
- If the answer contains unsupported claims, score it 'no'.

Respond with JSON only: {{"binary_score": "yes" or "no", "reasoning": "<one or two sentences>"}}"""


def grade_answer(results, answer):
    """Return (grounded: bool, reasoning: str). Raises on any malformed output."""
    response = get_client().chat.completions.create(
        model=os.environ.get("GRADER_MODEL", os.environ["LLM_MODEL"]),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": GRADER_PROMPT.format(
            documents=build_context(results), generation=answer)}],
    )
    data = json.loads(response.choices[0].message.content)
    score = str(data["binary_score"]).strip().lower()
    if score not in ("yes", "no"):
        raise ValueError(f"Grader returned unexpected score: {score!r}")
    return score == "yes", data.get("reasoning", "")