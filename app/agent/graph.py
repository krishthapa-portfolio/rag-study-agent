"""
LangGraph agent: retrieve -> check sufficiency -> (retry with a
reformulated query, up to MAX_RETRIES) -> synthesize an answer with
citations.
"""
import sys
from typing import TypedDict

from langgraph.graph import StateGraph, END
from mcp import ClientSession, StdioServerParameters 
from contextlib import asynccontextmanager
from mcp.client.stdio import stdio_client
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()

from app.agent.model_router import groq_call, anthropic_call

MAX_RETRIES = 2


class AgentState(TypedDict):
    class_name: str
    question: str
    query: str
    attempt: int
    chunks: list
    sufficient: bool
    answer: str
    sources: list
    _session: object


_SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "app.mcp_servers.retrieval_server"],
)


@asynccontextmanager
async def _mcp_session():
    async with stdio_client(_SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def _call_retrieval_tool(session, class_name: str, query: str, n_results: int = 8) -> list:
    result = await session.call_tool(
        "search_class_documents",
        arguments={"class_name": class_name, "query": query, "n_results": n_results},
    )
    if result.structuredContent:
        return result.structuredContent.get("result", [])
    return []


async def retrieve_node(state: AgentState) -> dict:
    chunks = await _call_retrieval_tool(state["_session"], state["class_name"], state["query"])
    return {"chunks": chunks}


def sufficiency_node(state: AgentState) -> dict:
    if not state["chunks"]:
        return {"sufficient": False}

    context = "\n\n".join(c["document"][:300] for c in state["chunks"][:3])
    system = (
        "You judge whether retrieved study notes are enough to answer a "
        "student's question. Reply with exactly one word: YES or NO."
    )
    user = f"Question: {state['question']}\n\nRetrieved context:\n{context}"
    verdict = groq_call(system, user, max_tokens=150).strip().upper()
    return {"sufficient": verdict.startswith("Y")}


def reformulate_node(state: AgentState) -> dict:
    system = (
        "Rewrite the student's question as a short, keyword-rich search "
        "query likely to retrieve more relevant course material. Reply "
        "with ONLY the rewritten query, nothing else."
    )
    new_query = groq_call(system, state["question"], max_tokens=150).strip()
    if not new_query:
        new_query = state["question"]  
    return {"query": new_query, "attempt": state["attempt"] + 1}


def synthesize_node(state: AgentState) -> dict:
    if not state["chunks"]:
        answer = groq_call(
            "You are a study assistant. Tell the student plainly that no "
            "source material was found for their question in this class's "
            "documents, so you can't answer it. Keep it to one or two sentences.",
            state["question"],
            max_tokens=100,
        )
        return {"answer": answer, "sources": []}

    context = "\n\n".join(
        f"[Source: {c['metadata'].get('doc_name')}, slide/page {c['metadata'].get('page_or_slide')}]\n{c['document']}"
        for c in state["chunks"]
    )
    
    system = (
        "You are a study assistant helping a student review course material. "
        "Answer the student's question using ONLY the provided source material. "
        "Write like you're explaining it to them directly \u2014 plain prose, "
        "no markdown headers, no bullet-point-only answers, no horizontal rules "
        "or dashes as section dividers. Short paragraphs are fine, and a plain "
        "bullet list is OK if it genuinely helps, but don't structure the whole "
        "answer as a formatted document. "
        "Cite the source (doc name, slide/page) inline for each claim, in "
        "parentheses. Keep the same depth of explanation \u2014 don't cut real "
        "content \u2014 just write it as clear, natural prose instead of a report. "
        "Use the best available information in the sources, even if it's "
        "partial \u2014 synthesize what's there and answer as fully as you can. "
        "Only decline to answer if the sources are truly unrelated to the "
        "question, not merely incomplete. If part of the answer is missing, "
        "answer with what's supported and note what wasn't covered, rather "
        "than declining outright."
    )
    
    user = f"Question: {state['question']}\n\nSources:\n{context}"
    answer = anthropic_call(system, user, max_tokens=800)
    sources = [
        {
            "doc_name": c["metadata"].get("doc_name"),
            "page_or_slide": c["metadata"].get("page_or_slide"),
        }
        for c in state["chunks"]
    ]
    return {"answer": answer, "sources": sources}


def _route_after_sufficiency(state: AgentState) -> str:
    if not state["chunks"]:
        return "synthesize"  # nothing retrieved at all — retrying won't fix a nonexistent class
    if state["sufficient"] or state["attempt"] >= MAX_RETRIES:
        return "synthesize"
    return "reformulate"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("check_sufficiency", sufficiency_node)
    graph.add_node("reformulate", reformulate_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "check_sufficiency")
    graph.add_conditional_edges(
        "check_sufficiency",
        _route_after_sufficiency,
        {"synthesize": "synthesize", "reformulate": "reformulate"},
    )
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("synthesize", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


async def ask(class_name: str, question: str) -> dict:
    graph = get_graph()
    async with _mcp_session() as session:
        initial_state: AgentState = {
            "class_name": class_name,
            "question": question,
            "query": question,
            "attempt": 0,
            "chunks": [],
            "sufficient": False,
            "answer": "",
            "sources": [],
            "_session": session,
        }
        final_state = await graph.ainvoke(
            initial_state,
            config={"callbacks": [langfuse_handler]},
        )
    return {
        "answer": final_state["answer"],
        "sources": final_state["sources"],
        "retrieval_attempts": final_state["attempt"] + 1,
    }