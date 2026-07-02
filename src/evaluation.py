import time
import mlflow
import pandas as pd


def setup_mlflow(experiment_name="Cyber_YouTube_Intelligence"):
    mlflow.set_experiment(experiment_name)
    mlflow.langchain.autolog()


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def get_retrieval_scores(retrieved_results):
    scores = []

    for item in retrieved_results:
        score = item.get(
            "hybrid_score",
            item.get("semantic_score", item.get("lexical_score", 0.0))
        )
        scores.append(safe_float(score))

    return scores


def evaluate_retrieval(retrieved_results, min_score=0.3):
    if not retrieved_results:
        return {
            "retrieval_count": 0,
            "avg_retrieval_score": 0.0,
            "max_retrieval_score": 0.0,
            "min_retrieval_score": 0.0,
            "retrieval_quality": "poor"
        }

    scores = get_retrieval_scores(retrieved_results)

    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    min_score_value = min(scores)

    quality = "good" if avg_score >= min_score else "weak"

    return {
        "retrieval_count": len(retrieved_results),
        "avg_retrieval_score": round(avg_score, 3),
        "max_retrieval_score": round(max_score, 3),
        "min_retrieval_score": round(min_score_value, 3),
        "retrieval_quality": quality
    }


def evaluate_faithfulness(answer, retrieved_results):
    """
    Simple lexical support check.
    It does not prove factual correctness, but it gives a rough signal:
    how much of the generated answer is grounded in retrieved comments.
    """

    answer = str(answer).lower()

    source_text = " ".join(
        str(
            item.get("metadata", {}).get(
                "original_text",
                item.get("text", "")
            )
        ).lower()
        for item in retrieved_results
    )

    answer_terms = {
        word.strip(".,:;!?()[]{}\"'")
        for word in answer.split()
        if len(word.strip(".,:;!?()[]{}\"'")) > 3
    }

    source_terms = {
        word.strip(".,:;!?()[]{}\"'")
        for word in source_text.split()
        if len(word.strip(".,:;!?()[]{}\"'")) > 3
    }

    if not answer_terms:
        overlap = 0.0
    else:
        overlap = len(answer_terms.intersection(source_terms)) / len(answer_terms)

    return {
        "faithfulness_overlap": round(overlap, 3),
        "faithfulness_label": "supported" if overlap >= 0.25 else "weakly_supported"
    }


def evaluate_answer_metadata(answer_result, retrieved_results):
    answer = answer_result.get("answer", answer_result.get("summary", ""))
    context = answer_result.get("context", "")

    return {
        "answer_length_chars": len(str(answer)),
        "answer_length_words": len(str(answer).split()),
        "context_length_chars": len(str(context)),
        "context_length_words": len(str(context).split()),
        "confidence": safe_float(answer_result.get("confidence", 0.0)),
        "evidence_count": len(retrieved_results)
    }


def evaluate_sentiment_distribution(df):
    if "sentiment" not in df.columns:
        return {}

    return df["sentiment"].value_counts().to_dict()


def evaluate_topic_distribution(df):
    if "topic_id" not in df.columns:
        return {}

    return df["topic_id"].value_counts().head(15).to_dict()


def evaluate_route_distribution(df):
    """
    Placeholder for future dashboard/agent logs.
    Currently route is logged per query in log_agent_run().
    """
    if "route" not in df.columns:
        return {}

    return df["route"].value_counts().to_dict()


def log_rag_run(
    question,
    answer_result,
    retrieved_results,
    route="rag_agent",
    df_path="outputs/final_analysis_results.csv",
    llm_model="llama3.2:3b",
    retrieval_strategy="hybrid",
    generation_latency=None,
    total_latency=None
):
    answer = answer_result.get("answer", answer_result.get("summary", ""))

    retrieval_eval = evaluate_retrieval(retrieved_results)
    faithfulness_eval = evaluate_faithfulness(answer, retrieved_results)
    answer_eval = evaluate_answer_metadata(answer_result, retrieved_results)

    with mlflow.start_run(run_name="rag_question_answering"):
        # Parameters
        mlflow.log_param("question", question)
        mlflow.log_param("route", route)
        mlflow.log_param("dataset_path", df_path)
        mlflow.log_param("llm_model", llm_model)
        mlflow.log_param("retrieval_strategy", retrieval_strategy)

        # Retrieval metrics
        mlflow.log_metric("retrieval_count", retrieval_eval["retrieval_count"])
        mlflow.log_metric("avg_retrieval_score", retrieval_eval["avg_retrieval_score"])
        mlflow.log_metric("max_retrieval_score", retrieval_eval["max_retrieval_score"])
        mlflow.log_metric("min_retrieval_score", retrieval_eval["min_retrieval_score"])

        # Generation / answer metrics
        mlflow.log_metric("confidence", answer_eval["confidence"])
        mlflow.log_metric("answer_length_chars", answer_eval["answer_length_chars"])
        mlflow.log_metric("answer_length_words", answer_eval["answer_length_words"])
        mlflow.log_metric("context_length_chars", answer_eval["context_length_chars"])
        mlflow.log_metric("context_length_words", answer_eval["context_length_words"])
        mlflow.log_metric("evidence_count", answer_eval["evidence_count"])

        # Faithfulness metric
        mlflow.log_metric("faithfulness_overlap", faithfulness_eval["faithfulness_overlap"])

        # Latency metrics
        if generation_latency is not None:
            mlflow.log_metric("generation_latency_sec", generation_latency)

        if total_latency is not None:
            mlflow.log_metric("total_latency_sec", total_latency)

        # Artifacts
        mlflow.log_text(str(answer), "answer.txt")
        mlflow.log_text(str(answer_result.get("context", "")), "retrieved_context.txt")
        mlflow.log_dict(retrieved_results, "retrieved_evidence.json")
        mlflow.log_dict(retrieval_eval, "retrieval_evaluation.json")
        mlflow.log_dict(faithfulness_eval, "faithfulness_evaluation.json")
        mlflow.log_dict(answer_eval, "answer_metadata.json")

    return {
        **retrieval_eval,
        **faithfulness_eval,
        **answer_eval
    }


def log_agent_run(
    question,
    agent_result,
    df_path="outputs/final_analysis_results.csv",
    llm_model="llama3.2:3b",
    retrieval_strategy="hybrid",
    total_latency=None
):
    route = agent_result.get("route", "unknown")
    answer = agent_result.get("answer", "")
    evidence = agent_result.get("evidence", [])

    answer_result = {
        "answer": answer,
        "confidence": agent_result.get("confidence", 0.0),
        "context": "",
        "sources": evidence
    }

    retrieval_eval = evaluate_retrieval(evidence)
    faithfulness_eval = evaluate_faithfulness(answer, evidence)

    with mlflow.start_run(run_name="agent_question_answering"):
        mlflow.log_param("question", question)
        mlflow.log_param("route", route)
        mlflow.log_param("dataset_path", df_path)
        mlflow.log_param("llm_model", llm_model)
        mlflow.log_param("retrieval_strategy", retrieval_strategy)

        mlflow.log_metric("retrieval_count", retrieval_eval["retrieval_count"])
        mlflow.log_metric("avg_retrieval_score", retrieval_eval["avg_retrieval_score"])
        mlflow.log_metric("max_retrieval_score", retrieval_eval["max_retrieval_score"])
        mlflow.log_metric("min_retrieval_score", retrieval_eval["min_retrieval_score"])
        mlflow.log_metric("faithfulness_overlap", faithfulness_eval["faithfulness_overlap"])
        mlflow.log_metric("answer_length_words", len(str(answer).split()))
        mlflow.log_metric("answer_length_chars", len(str(answer)))

        if total_latency is not None:
            mlflow.log_metric("total_latency_sec", total_latency)

        mlflow.log_text(str(answer), "agent_answer.txt")
        mlflow.log_dict(evidence, "agent_evidence.json")
        mlflow.log_dict(retrieval_eval, "retrieval_evaluation.json")
        mlflow.log_dict(faithfulness_eval, "faithfulness_evaluation.json")

    return {
        **retrieval_eval,
        **faithfulness_eval,
        "answer_length_words": len(str(answer).split()),
        "answer_length_chars": len(str(answer))
    }


def log_pipeline_summary(df_path="outputs/final_analysis_results.csv"):
    df = pd.read_csv(df_path)

    sentiment_distribution = evaluate_sentiment_distribution(df)
    topic_distribution = evaluate_topic_distribution(df)

    with mlflow.start_run(run_name="pipeline_summary"):
        mlflow.log_param("dataset_path", df_path)
        mlflow.log_param("rows", len(df))
        mlflow.log_param("columns_count", len(df.columns))

        if "language" in df.columns:
            mlflow.log_dict(
                df["language"].value_counts().to_dict(),
                "language_distribution.json"
            )

        if "sentiment" in df.columns:
            mlflow.log_dict(
                sentiment_distribution,
                "sentiment_distribution.json"
            )

        if "topic_id" in df.columns:
            mlflow.log_dict(
                topic_distribution,
                "topic_distribution_top15.json"
            )

        if "emotion_label" in df.columns:
            mlflow.log_dict(
                df["emotion_label"].value_counts().to_dict(),
                "emotion_distribution.json"
            )

        mlflow.log_text(
            "\n".join(df.columns),
            "dataset_columns.txt"
        )

    return {
        "rows": len(df),
        "columns": list(df.columns),
        "sentiment_distribution": sentiment_distribution,
        "topic_distribution": topic_distribution
    }