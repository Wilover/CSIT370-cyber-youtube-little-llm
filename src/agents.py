import pandas as pd
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

from src.retrieval import hybrid_retrieval
from src.generator import answer_question, summarize_results


llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# =========================================================
# STATE
# =========================================================

class CyberAgentState(TypedDict, total=False):
    question: str
    df_path: str
    route: str
    answer: str
    evidence: list


# =========================================================
# TOOLS
# =========================================================

@tool
def sentiment_insight_tool(df_path: str) -> str:
    """Return sentiment distribution from the analyzed YouTube comments."""
    df = pd.read_csv(df_path)

    if "sentiment" not in df.columns:
        return "Sentiment analysis is not available yet."

    counts = df["sentiment"].value_counts().to_dict()

    return f"Sentiment distribution: {counts}"


@tool
def topic_insight_tool(df_path: str) -> str:
    """Return top topic clusters from BERTopic results."""
    df = pd.read_csv(df_path)

    if "topic_id" not in df.columns:
        return "Topic modeling is not available yet."

    counts = df["topic_id"].value_counts().head(10).to_dict()

    return f"Top topic clusters: {counts}"


@tool
def metadata_filter_tool(df_path: str, filter_text: str) -> str:
    """Filter comments by simple metadata such as sentiment."""
    df = pd.read_csv(df_path)

    filter_text = filter_text.lower()
    filtered = df.copy()

    if "negative" in filter_text and "sentiment" in df.columns:
        filtered = filtered[filtered["sentiment"] == "negative"]

    elif "positive" in filter_text and "sentiment" in df.columns:
        filtered = filtered[filtered["sentiment"] == "positive"]

    elif "neutral" in filter_text and "sentiment" in df.columns:
        filtered = filtered[filtered["sentiment"] == "neutral"]

    return f"Filtered rows: {len(filtered)}"


@tool
def evaluation_tool() -> str:
    """Return available evaluation methods."""
    return """
Available evaluation methods:
- retrieval relevance check
- faithfulness check
- no-answer / hallucination test
- sentiment error analysis
- topic coherence review
- MLflow experiment tracking
"""


# =========================================================
# SUPERVISOR
# =========================================================

def supervisor_node(state: CyberAgentState):
    question = state["question"]

    prompt = f"""
    You are a supervisor for a cybersecurity YouTube intelligence system.

    Choose exactly ONE route:

    Routes:
    - sentiment_agent: questions about positive, negative, neutral reactions
    - topic_agent: questions about topics, themes, clusters
    - rag_agent: factual cybersecurity questions requiring retrieved evidence
    - summary_agent: questions asking for summary or overview
    - metadata_agent: questions asking to filter comments by sentiment, topic, video, date, likes
    - evaluation_agent: questions about system quality, errors, hallucination, evaluation

    User question:
    {question}

    Return only the route name.
    """

    result = llm.invoke(prompt)
    route = result.content.strip().lower()

    valid_routes = {
        "sentiment_agent",
        "topic_agent",
        "rag_agent",
        "summary_agent",
        "metadata_agent",
        "evaluation_agent"
    }

    if route not in valid_routes:
        route = "rag_agent"

    return {"route": route}


def route_decision(state: CyberAgentState):
    return state["route"]


# =========================================================
# AGENTS
# =========================================================

def sentiment_agent(state: CyberAgentState):
    result = sentiment_insight_tool.invoke(
        {"df_path": state["df_path"]}
    )

    return {
        "answer": result,
        "evidence": []
    }


def topic_agent(state: CyberAgentState):
    result = topic_insight_tool.invoke(
        {"df_path": state["df_path"]}
    )

    return {
        "answer": result,
        "evidence": []
    }


def metadata_agent(state: CyberAgentState):
    result = metadata_filter_tool.invoke(
        {
            "df_path": state["df_path"],
            "filter_text": state["question"]
        }
    )

    return {
        "answer": result,
        "evidence": []
    }


def rag_agent(state: CyberAgentState):
    df = pd.read_csv(state["df_path"])

    retrieved = hybrid_retrieval(
        state["question"],
        df,
        k=5,
        alpha=0.7
    )

    result = answer_question(
        state["question"],
        retrieved
    )

    return {
        "answer": result["answer"],
        "evidence": retrieved
    }


def summary_agent(state: CyberAgentState):
    df = pd.read_csv(state["df_path"])

    retrieved = hybrid_retrieval(
        state["question"],
        df,
        k=10,
        alpha=0.7
    )

    summary = summarize_results(retrieved)

    return {
        "answer": summary["summary"],
        "evidence": retrieved
    }


def evaluation_agent(state: CyberAgentState):
    result = evaluation_tool.invoke({})

    return {
        "answer": result,
        "evidence": []
    }


# =========================================================
# GRAPH
# =========================================================

builder = StateGraph(CyberAgentState)

builder.add_node("supervisor", supervisor_node)

builder.add_node("sentiment_agent", sentiment_agent)
builder.add_node("topic_agent", topic_agent)
builder.add_node("rag_agent", rag_agent)
builder.add_node("summary_agent", summary_agent)
builder.add_node("metadata_agent", metadata_agent)
builder.add_node("evaluation_agent", evaluation_agent)

builder.set_entry_point("supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_decision,
    {
        "sentiment_agent": "sentiment_agent",
        "topic_agent": "topic_agent",
        "rag_agent": "rag_agent",
        "summary_agent": "summary_agent",
        "metadata_agent": "metadata_agent",
        "evaluation_agent": "evaluation_agent"
    }
)

for node in [
    "sentiment_agent",
    "topic_agent",
    "rag_agent",
    "summary_agent",
    "metadata_agent",
    "evaluation_agent"
]:
    builder.add_edge(node, END)

cyber_agent_graph = builder.compile()


# =========================================================
# PUBLIC FUNCTION
# =========================================================

def ask_cyber_system(
    question: str,
    df_path: str = "outputs/final_analysis_results.csv"
):
    result = cyber_agent_graph.invoke(
        {
            "question": question,
            "df_path": df_path
        }
    )

    return result