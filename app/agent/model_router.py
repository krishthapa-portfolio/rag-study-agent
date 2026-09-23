"""
Model routing: cheap/fast internal steps go to Groq (free), the one
user-facing step — final answer synthesis — goes to Anthropic (paid,
higher quality).
"""
from groq import Groq
from anthropic import Anthropic

from app.config import GROQ_API_KEY, ANTHROPIC_API_KEY

_groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
_anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

GROQ_MODEL = "openai/gpt-oss-20b"
ANTHROPIC_MODEL = "claude-sonnet-4-6"


def groq_call(system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
    if _groq_client is None:
        raise RuntimeError("GROQ_API_KEY is not set in .env")
    response = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0,
    )
    return response.choices[0].message.content or ""


def anthropic_call(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    if _anthropic_client is None:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in .env")
    response = _anthropic_client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text