"""
Golden dataset evaluator with LLM-as-judge grading.

Runs every question in eval/golden_dataset.json through the live /ask
endpoint, then has an LLM (Groq, kept cheap and separate from the
Anthropic call the agent itself uses) grade the generated answer
against the golden answer on a simple rubric: CORRECT, PARTIAL, or
INCORRECT, with a one-line reason.

This is an OFFLINE report card, not part of the live request path.
It doesn't touch graph.py or main.py at all — it just calls your
running server the same way curl does, then grades the results
afterward.

Usage:
    Make sure `uvicorn app.main:app --reload` is running in another
    terminal, then:

    python -m eval.run_eval

Outputs:
    - Prints a per-question verdict to the console as it goes
    - Prints a final summary (accuracy breakdown)
    - Saves full results to eval/eval_results.json for your writeup
"""
import json
import os
import time
from pathlib import Path

import requests
from groq import Groq
from app.config import GROQ_API_KEY

# --- Config ---
API_URL = "http://127.0.0.1:8000/ask"
GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"

GROQ_MODEL = "openai/gpt-oss-20b"

_groq_client = Groq(api_key=GROQ_API_KEY)


def call_ask_endpoint(class_name: str, question: str) -> dict:
    """Hit the live /ask endpoint, same as a real user request."""
    response = requests.post(
        API_URL,
        json={"class_name": class_name, "question": question},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def judge_answer(question: str, golden_answer: str, generated_answer: str) -> dict:
    """
    LLM-as-judge: grades the generated answer against the golden answer.

    This is NOT semantic similarity / embedding distance — it's a
    separate model call reading both answers and applying judgment,
    which tolerates paraphrasing while still catching real errors
    (missing key facts, wrong facts, or unsupported claims).
    """
    system = (
        "You are grading a student study assistant's answer against a "
        "known-correct reference answer. Judge whether the generated "
        "answer is factually correct and covers the key points of the "
        "reference answer, even if worded differently. It does not need "
        "to match word-for-word.\n\n"
        "Reply in EXACTLY this format, nothing else:\n"
        "VERDICT: <CORRECT, PARTIAL, or INCORRECT>\n"
        "REASON: <one sentence explaining why>\n\n"
        "CORRECT = captures the key facts, no significant errors.\n"
        "PARTIAL = captures some key facts but misses others, or is "
        "vague/incomplete.\n"
        "INCORRECT = misses the key facts, contains wrong information, "
        "or (for questions with no real answer) fails to correctly "
        "decline when it should have."
    )
    user = (
        f"Question: {question}\n\n"
        f"Reference answer: {golden_answer}\n\n"
        f"Generated answer: {generated_answer}"
    )
    response = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=600,
        temperature=0,
    )
    raw = response.choices[0].message.content or ""

    verdict = "INCORRECT"  # safe default if parsing fails
    reason = raw.strip()
    for line in raw.splitlines():
        if line.upper().startswith("VERDICT:"):
            v = line.split(":", 1)[1].strip().upper()
            if v in ("CORRECT", "PARTIAL", "INCORRECT"):
                verdict = v
        if line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return {"verdict": verdict, "reason": reason}


def main():
    with open(GOLDEN_DATASET_PATH) as f:
        golden_set = json.load(f)

    results = []
    counts = {"CORRECT": 0, "PARTIAL": 0, "INCORRECT": 0}

    print(f"Running {len(golden_set)} questions through /ask...\n")

    for i, item in enumerate(golden_set, start=1):
        print(f"[{i}/{len(golden_set)}] {item['id']} — {item['question'][:60]}...")

        start = time.time()
        try:
            ask_result = call_ask_endpoint(item["class_name"], item["question"])
        except Exception as e:
            print(f"  ERROR calling /ask: {e}")
            results.append({**item, "generated_answer": None, "verdict": "ERROR", "reason": str(e)})
            counts["INCORRECT"] += 1
            continue
        elapsed = time.time() - start

        generated_answer = ask_result.get("answer", "")
        retrieval_attempts = ask_result.get("retrieval_attempts")

        judgment = judge_answer(item["question"], item["golden_answer"], generated_answer)
        counts[judgment["verdict"]] += 1

        print(f"  -> {judgment['verdict']} ({elapsed:.1f}s, {retrieval_attempts} retrieval attempt(s))")
        print(f"     {judgment['reason']}\n")

        results.append({
            **item,
            "generated_answer": generated_answer,
            "retrieval_attempts": retrieval_attempts,
            "elapsed_seconds": round(elapsed, 2),
            "verdict": judgment["verdict"],
            "reason": judgment["reason"],
        })

    total = len(golden_set)
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for verdict in ("CORRECT", "PARTIAL", "INCORRECT"):
        pct = (counts[verdict] / total * 100) if total else 0
        print(f"  {verdict}: {counts[verdict]}/{total} ({pct:.0f}%)")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()