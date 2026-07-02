import os
import time
import pandas as pd

from src.preprocessing import preprocess_data
from src.ner_keywords import run_ner_keywords
from src.sentiment import run_sentiment_analysis
from src.sarcasm_emotion import run_sarcasm_emotion_analysis
from src.topic_modeling import run_topic_modeling, run_topic_modeling_per_sentiment
from src.vector_store import build_faiss_index
from src.evaluation import setup_mlflow, log_pipeline_summary, log_agent_run
from src.agents import ask_cyber_system


# Choose one:
# "preprocess", "ner", "sentiment", "emotion", "topic", "index", "agent_demo", 
# "full_build", "full_demo"
MODE = "full_demo"

USE_MLFLOW = True
DEMO_QUESTION = "What cybersecurity certifications do students recommend?"

FINAL_PATH = "outputs/final_analysis_results.csv"


def save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def ensure_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("vector_store", exist_ok=True)


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    print(f"Loading: {path}")
    return pd.read_csv(path)


def run_preprocessing():
    df = load_csv("data/youtube_comments_cs_dataset.csv")

    print("Running preprocessing...")
    df, zipf_df, zipf_stopwords = preprocess_data(
        df,
        text_column="text",
        top_n_zipf=50
    )

    save_csv(df, "outputs/processed_comments.csv")
    save_csv(zipf_df, "outputs/zipf_analysis.csv")

    with open("outputs/zipf_stopwords.txt", "w", encoding="utf-8") as f:
        for word in sorted(zipf_stopwords):
            f.write(word + "\n")

    print("Preprocessing completed.")
    return df


def run_ner():
    df = load_csv("outputs/processed_comments.csv")
    print("Running NER and keyword extraction...")
    df = run_ner_keywords(df, text_column="clean_text")
    save_csv(df, "outputs/ner_keyword_results.csv")
    print("NER completed.")
    return df


def run_sentiment():
    df = load_csv("outputs/ner_keyword_results.csv")
    print("Running sentiment analysis...")
    df = run_sentiment_analysis(df, text_column="clean_text")
    save_csv(df, "outputs/sentiment_results.csv")
    print("Sentiment completed.")
    return df


def run_emotion():
    df = load_csv("outputs/sentiment_results.csv")
    print("Running sarcasm and emotion analysis...")
    df = run_sarcasm_emotion_analysis(df, text_column="clean_text")
    save_csv(df, "outputs/sarcasm_emotion_results.csv")
    print("Emotion completed.")
    return df


def run_topic():
    df = load_csv("outputs/sarcasm_emotion_results.csv")

    print("Running topic modeling...")
    df, topic_model, topic_info = run_topic_modeling(
        df,
        text_column="final_text",
        min_topic_size=10
    )

    save_csv(df, "outputs/topic_results.csv")
    save_csv(topic_info, "outputs/topic_info.csv")
    topic_model.save("outputs/bertopic_model")

    print("Running topic modeling per sentiment...")
    topic_by_sentiment = run_topic_modeling_per_sentiment(
        df,
        text_column="final_text",
        sentiment_column="sentiment",
        min_topic_size=5
    )

    save_csv(topic_by_sentiment, "outputs/topic_by_sentiment_results.csv")
    save_csv(df, FINAL_PATH)

    print("Topic modeling completed.")
    return df


def run_index():
    df = load_csv(FINAL_PATH)
    print("Building FAISS index...")
    build_faiss_index(df, text_column="final_text")
    print("FAISS index completed.")
    return df


def run_full_build():
    print("FULL BUILD STARTED")

    run_preprocessing()
    run_ner()
    run_sentiment()
    run_emotion()
    run_topic()
    run_index()

    if USE_MLFLOW:
        setup_mlflow()
        summary = log_pipeline_summary(FINAL_PATH)
        print("\nPIPELINE SUMMARY:")
        print(summary)

    print("FULL BUILD COMPLETED")


def run_agent_demo(question=DEMO_QUESTION):
    if USE_MLFLOW:
        setup_mlflow()

    print("\nAGENTIC RAG DEMO")
    print("Question:", question)

    start_time = time.time()

    agent_result = ask_cyber_system(
        question,
        df_path=FINAL_PATH
    )

    total_latency = time.time() - start_time

    print("\nROUTE:")
    print(agent_result.get("route"))

    print("\nANSWER:")
    print(agent_result.get("answer"))

    print("\nLATENCY:")
    print(round(total_latency, 3), "sec")

    print("\nEVIDENCE:")
    evidence = agent_result.get("evidence", [])
    for i, item in enumerate(evidence, start=1):
        metadata = item.get("metadata", {})
        print(f"\nSource {i}")
        print(metadata.get("original_text", item.get("text", "")))

    if USE_MLFLOW:
        eval_result = log_agent_run(
            question=question,
            agent_result=agent_result,
            df_path=FINAL_PATH,
            llm_model="llama3.2:3b",
            retrieval_strategy="hybrid",
            total_latency=total_latency
        )

        print("\nMLFLOW EVALUATION:")
        print(eval_result)

    return agent_result


def run_full_demo():
    print("FULL DEMO STARTED")

    if not os.path.exists(FINAL_PATH):
        raise FileNotFoundError(
            "final_analysis_results.csv not found. Run MODE='full_build' first."
        )

    if not os.path.exists("vector_store/faiss.index"):
        print("FAISS index not found. Building index first...")
        run_index()

    run_agent_demo(DEMO_QUESTION)

    print("FULL DEMO COMPLETED")


def main():
    ensure_dirs()

    if MODE == "preprocess":
        run_preprocessing()

    elif MODE == "ner":
        run_ner()

    elif MODE == "sentiment":
        run_sentiment()

    elif MODE == "emotion":
        run_emotion()

    elif MODE == "topic":
        run_topic()

    elif MODE == "index":
        run_index()

    elif MODE == "agent_demo":
        run_agent_demo(DEMO_QUESTION)

    elif MODE == "full_build":
        run_full_build()

    elif MODE == "full_demo":
        run_full_demo()

    else:
        raise ValueError(f"Unknown MODE: {MODE}")

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()