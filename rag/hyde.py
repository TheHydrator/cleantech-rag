import os
from rag.embeddings import get_client

# Same prompt as the original notebook
HYDE_PROMPT = """You are an expert researcher.
Given the following question, write a brief, hypothetical paragraph that answers the question.
Do not worry about being factually correct - focus on the keywords, technical terms,
and phrasing that would appear in a perfect answer document.

Question: {question}

Hypothetical Answer Passage:"""


def generate_hypothesis(question):
    response = get_client().chat.completions.create(
        model=os.environ["LLM_MODEL"],
        temperature=0,
        messages=[{"role": "user", "content": HYDE_PROMPT.format(question=question)}],
    )
    return response.choices[0].message.content