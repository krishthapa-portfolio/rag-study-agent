"""
Token-aware chunking with overlap.

This is deliberately parameterized (chunk_size, overlap) rather than
hardcoded, because picking the right chunk size per document type is
the tuning experiment you run yourself in scripts/test_chunk_sizes.py.
"""
import tiktoken

try:
    _encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    # tiktoken downloads its BPE file on first use; if that fails
    # (offline, blocked network, first run without internet), fall
    # back to a simple whitespace tokenizer so ingestion never breaks.
    # Chunk sizes will be approximate (word count, not true token count)
    # until the real encoding can be fetched.
    _encoding = None


class _FallbackEncoding:
    @staticmethod
    def encode(text: str) -> list[str]:
        return text.split()

    @staticmethod
    def decode(tokens: list[str]) -> str:
        return " ".join(tokens)


if _encoding is None:
    _encoding = _FallbackEncoding()


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """
    Splits text into overlapping chunks measured in tokens (not characters),
    so chunk size behaves consistently across dense prose and sparse slides.
    """
    tokens = _encoding.encode(text)
    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(_encoding.decode(chunk_tokens))
        if end == len(tokens):
            break
        start = end - overlap  # step forward, keeping the overlap window

    return chunks


def token_count(text: str) -> int:
    return len(_encoding.encode(text))
