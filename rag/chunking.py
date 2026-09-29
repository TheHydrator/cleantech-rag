import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Same settings as the original notebook
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_ARTICLE_LENGTH = 100

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    is_separator_regex=False,
)


def clean_content(raw):
    """Same cleaning as the notebook: strip outer brackets, collapse whitespace."""
    content = str(raw).strip()
    if content.startswith("[") and content.endswith("]"):
        content = content[1:-1]
    return re.sub(r"\s+", " ", content)


def chunk_article(raw_content):
    """Return a list of chunk strings, or [] if the article is too short."""
    if len(str(raw_content)) < MIN_ARTICLE_LENGTH:
        return []
    return _splitter.split_text(clean_content(raw_content))