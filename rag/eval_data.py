import re


def load_50_questions(path="data/CleanTech-50answers.txt"):
    """Parse the 50-question file into a list of dicts."""
    with open(path, encoding="utf-8-sig") as f:
        text = f.read()

    pattern = re.compile(
        r"\*\*(\d+)\. Query:\*\*\s*(.+?)\s*\n"
        r"\*\*Desired Output:\*\*\s*(.+?)\s*\n"
        r"\*\*Referenced Articles:\*\*\s*([\d,\s]+)",
        re.S,
    )
    return [
        {
            "num": int(num),
            "question": question.strip(),
            "expected_answer": answer.strip(),
            "article_ids": [int(x) for x in re.findall(r"\d+", ids)],
        }
        for num, question, answer, ids in pattern.findall(text)
    ]