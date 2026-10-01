import json
import os

from rag.embeddings import get_client

JUDGE_PROMPT = """You are grading an answer from a question-answering system about clean technology.

Question:
{question}

Reference answer (written by an expert):
{expected}

System answer:
{answer}

Score the system answer from 1 to 5:
5 = covers all key points of the reference, no contradictions
4 = covers most key points, no contradictions
3 = partially correct, misses important points
2 = mostly wrong or very incomplete
1 = wrong, contradicts the reference, or says it does not know

Judge only factual content, not style or length.
Respond with JSON only: {{"score": <1-5>, "reason": "<one sentence>"}}"""


def judge_answer(question, expected, answer):
    response = get_client().chat.completions.create(
        model=os.environ.get("JUDGE_MODEL", os.environ["LLM_MODEL"]),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, expected=expected, answer=answer)}],
    )
    data = json.loads(response.choices[0].message.content)
    score = int(data["score"])
    if not 1 <= score <= 5:
        raise ValueError(f"Judge returned out-of-range score: {score}")
    return score, data.get("reason", ""), response.usage