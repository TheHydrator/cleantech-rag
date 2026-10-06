import os
from rag.embeddings import get_client

SYSTEM_PROMPT = """You answer questions about clean technology using ONLY the provided context.
If the context does not contain the answer, say "I don't know based on the available articles."
Cite the article IDs you used in square brackets, like [43333]."""


def build_context(results):
    parts = []
    for r in results:
        article_id, title, content = r[0], r[2], r[3]
        parts.append(f"[{article_id}] {title}\n{content}")
    return "\n\n---\n\n".join(parts)

def generate_answer(question, results):
    response = get_client().chat.completions.create(
        model=os.environ["LLM_MODEL"],
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{build_context(results)}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content, response.usage