Student RAG Study Agent

An AI study assistant that answers questions from your own class materials, the same idea as Notion AI or a subscription study tool, but free and self hosted, and it only ever answers from documents you actually uploaded instead of guessing.

You upload your slides, notes, and readings for a class. Then you ask it questions the way you would ask a classmate the night before an exam, and it answers using only what you gave it, with citations back to the specific slide or page. If the material doesn't cover something, it says so instead of making something up.

How it works
question
   |
   v
retrieve chunks (hybrid search: BM25 + semantic, MCP tool)
   |
   v
check if retrieved material is actually enough
   |
   +--> not enough --> reformulate the query --> retrieve again (up to 3 tries)
   |
   +--> enough
   |
   v
synthesize final answer with citations (Anthropic)

Model routing splits the work by cost and stakes. Groq handles the small internal steps, retrieval judgment and query reformulation, since they're cheap and need to be fast. Anthropic handles only the final synthesis, the one call actually worth paying for quality on. Retrieval goes through an MCP server, so the agent discovers and calls the search tool rather than having it hardcoded in.

What actually went wrong, and what I learned fixing it

This project didn't come together cleanly, and that's worth being honest about.

My first version of the retry loop opened a new connection to the retrieval tool on every call, three subprocess spin-ups per request before the agent even started thinking. Baseline latency was 106 to 122 seconds. Switching to one shared connection per request cut that to 40 to 46 seconds, about 2.7x faster.

Once I built a 21 question golden dataset and started grading with an LLM as judge, my first real score came back at 38% correct. I found the reason using LangFuse, tracing every step of a request instead of guessing. Two silent bugs turned up. My reformulation step called a reasoning model with a token limit so low it never finished thinking, so every rewritten query came back empty. My sufficiency checker had the same problem at an even tighter limit, so it almost always defaulted to insufficient, which is also why every question was burning all three retries regardless of whether the first search was already good.

Neither threw an error. The system just quietly did the wrong thing every time, invisible until I actually traced each step instead of trusting that no crash meant it was working.

After fixing both token limits and loosening an overly cautious refusal instruction, the same eval came back at 95%, 20 out of 21, with the one remaining answer graded partial.

Results

Latency: 106 to 122 seconds down to 39 to 46 seconds per request.

Eval accuracy, same 21 question golden dataset, LLM as judge: 38% correct, then 52% after fixing a class mismatch in the eval data, then 95% after fixing the two token truncation bugs.

Latency breakdown from LangFuse: synthesis dominates total request time, p50 around 10s and p95 around 14.5s, while retrieval, sufficiency checking, and reformulation combined add under 2 seconds at p95. In plain terms, almost all the wait time is the one paid call actually writing the answer, and the free reasoning steps around it cost almost no time at all.

Correctly refuses to answer outside the uploaded material, tested against a nonexistent class and a real class with an unrelated nonsense question.

Tech stack
LangGraph, agent control flow and retry loop
MCP (FastMCP), retrieval exposed as a discoverable tool
Groq, fast cheap internal reasoning steps
Anthropic, final answer synthesis
ChromaDB, vector store
Hybrid retrieval, BM25 + semantic search, fused with RRF
FastAPI, backend
LangFuse, tracing and observability
Setup

Clone the repo and create a virtual environment.

Install dependencies from requirements.txt.

Create a .env file with GROQ_API_KEY, ANTHROPIC_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL.

Run uvicorn app.main:app --reload and hit the API through the Swagger docs at /docs, or POST directly to /ask with a class_name and question.

Known limitations

Retrieval sometimes misses the right chunk on certain phrasings even when the source material covers the topic, the clearest next thing to tune, likely through chunk size or retrieval count.

The eval judge is itself an LLM and not perfectly consistent between runs on borderline answers, so the accuracy number is a strong signal rather than an exact score.

No frontend yet, everything runs through the API directly.

Where this generalizes

The real thing I built here is the pattern, not the study assistant. Retrieval grounded strictly in provided documents, a retry loop that knows when to reformulate instead of guessing, cost-aware model routing, and an eval and observability layer that catches silent failures instead of assuming no crash means correct.

That pattern transfers directly to internal knowledge bases, support tools answering only from product docs, or compliance tools that need to cite an exact source clause instead of paraphrasing from memory.