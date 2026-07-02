import os
import time
import pandas as pd
import streamlit as st

from src.agents import ask_cyber_system
from src.evaluation import setup_mlflow, log_agent_run


st.set_page_config(
    page_title="Cyber YouTube Intelligence Engine",
    layout="wide"
)


DF_PATH = "outputs/final_analysis_results.csv"


@st.cache_data
def load_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def apply_filters(df):
    filtered = df.copy()

    st.sidebar.header("Filters")

    if "sentiment" in df.columns:
        sentiments = st.sidebar.multiselect(
            "Sentiment",
            options=sorted(df["sentiment"].dropna().unique())
        )
        if sentiments:
            filtered = filtered[filtered["sentiment"].isin(sentiments)]

    if "emotion_label" in df.columns:
        emotions = st.sidebar.multiselect(
            "Emotion",
            options=sorted(df["emotion_label"].dropna().unique())
        )
        if emotions:
            filtered = filtered[filtered["emotion_label"].isin(emotions)]

    if "topic_id" in df.columns:
        topics = st.sidebar.multiselect(
            "Topic ID",
            options=sorted(df["topic_id"].dropna().unique())
        )
        if topics:
            filtered = filtered[filtered["topic_id"].isin(topics)]

    if "video_id" in df.columns:
        videos = st.sidebar.multiselect(
            "Video ID",
            options=sorted(df["video_id"].dropna().unique())
        )
        if videos:
            filtered = filtered[filtered["video_id"].isin(videos)]

    if "like_count" in df.columns:
        min_likes = st.sidebar.number_input(
            "Minimum likes",
            min_value=0,
            value=0
        )
        filtered = filtered[filtered["like_count"].fillna(0) >= min_likes]

    return filtered


st.title("Cyber YouTube Intelligence Engine")
st.caption("Agentic RAG system for cybersecurity discussion analysis")

df = load_data(DF_PATH)

if df.empty:
    st.warning("No final analysis file found. Run the backend pipeline first.")
    st.stop()

filtered_df = apply_filters(df)

tab1, tab2, tab3, tab4 = st.tabs([
    "Ask the system",
    "Analytics",
    "Evidence explorer",
    "Dataset preview"
])


with tab1:
    st.subheader("Ask a cybersecurity question")

    question = st.text_input(
        "Question",
        placeholder="Example: What cybersecurity certifications do students recommend?"
    )

    use_mlflow = st.checkbox("Log this query to MLflow", value=True)

    if st.button("Ask agentic RAG system"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Running LangGraph supervisor + RAG + Llama 3.2..."):
                start_time = time.time()

                result = ask_cyber_system(
                    question=question,
                    df_path=DF_PATH
                )

                total_latency = time.time() - start_time

            st.success("Answer generated")

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Route", result.get("route", "unknown"))
            col_b.metric("Evidence count", len(result.get("evidence", [])))
            col_c.metric("Latency", f"{total_latency:.2f}s")

            st.subheader("Answer")
            st.write(result.get("answer", ""))

            evidence = result.get("evidence", [])

            st.subheader("Evidence")
            if evidence:
                for i, item in enumerate(evidence, start=1):
                    metadata = item.get("metadata", item)

                    with st.expander(f"Source {i}"):
                        st.write(
                            metadata.get(
                                "original_text",
                                metadata.get("text", item.get("text", ""))
                            )
                        )

                        meta_cols = st.columns(4)
                        meta_cols[0].write(f"**Author:** {metadata.get('author', 'unknown')}")
                        meta_cols[1].write(f"**Sentiment:** {metadata.get('sentiment', 'unknown')}")
                        meta_cols[2].write(f"**Topic:** {metadata.get('topic_id', 'unknown')}")
                        meta_cols[3].write(f"**Score:** {item.get('hybrid_score', item.get('semantic_score', 'unknown'))}")

                        st.json(metadata)
            else:
                st.info("No evidence returned for this route.")

            if use_mlflow:
                setup_mlflow()
                eval_result = log_agent_run(
                    question=question,
                    agent_result=result,
                    df_path=DF_PATH,
                    llm_model="llama3.2:3b",
                    retrieval_strategy="hybrid",
                    total_latency=total_latency
                )

                st.subheader("MLflow evaluation")
                st.json(eval_result)


with tab2:
    st.subheader("Dataset analytics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total comments", len(df))
    col2.metric("Filtered comments", len(filtered_df))
    col3.metric("Columns", len(df.columns))

    if "video_id" in filtered_df.columns:
        col4.metric("Videos", filtered_df["video_id"].nunique())

    if "sentiment" in filtered_df.columns:
        st.write("Sentiment distribution")
        st.bar_chart(filtered_df["sentiment"].value_counts())

    if "emotion_label" in filtered_df.columns:
        st.write("Emotion distribution")
        st.bar_chart(filtered_df["emotion_label"].value_counts())

    if "topic_id" in filtered_df.columns:
        st.write("Top topic clusters")
        st.bar_chart(filtered_df["topic_id"].value_counts().head(15))

    if "updated_at" in filtered_df.columns and "sentiment" in filtered_df.columns:
        temp = filtered_df.copy()
        temp["updated_at"] = pd.to_datetime(temp["updated_at"], errors="coerce")
        temp = temp.dropna(subset=["updated_at"])
        temp["date"] = temp["updated_at"].dt.date

        sentiment_over_time = (
            temp.groupby(["date", "sentiment"])
            .size()
            .unstack(fill_value=0)
        )

        st.write("Sentiment over time")
        st.line_chart(sentiment_over_time)


with tab3:
    st.subheader("Evidence explorer")

    search_term = st.text_input("Search inside comments")

    explorer_df = filtered_df.copy()

    if search_term.strip() and "text" in explorer_df.columns:
        explorer_df = explorer_df[
            explorer_df["text"].astype(str).str.contains(
                search_term,
                case=False,
                na=False
            )
        ]

    columns = [
        col for col in [
            "text",
            "cyber_keywords",
            "all_extracted_terms",
            "sentiment",
            "emotion_label",
            "topic_id",
            "video_id",
            "like_count"
        ]
        if col in explorer_df.columns
    ]

    st.dataframe(
        explorer_df[columns].head(100),
        use_container_width=True
    )


with tab4:
    st.subheader("Filtered dataset preview")

    preview_columns = [
        col for col in [
            "text",
            "clean_text",
            "final_text",
            "sentiment",
            "sentiment_score",
            "irony_label",
            "emotion_label",
            "topic_id",
            "cyber_keywords",
            "video_id",
            "updated_at",
            "like_count"
        ]
        if col in filtered_df.columns
    ]

    st.dataframe(
        filtered_df[preview_columns].head(100),
        use_container_width=True
    )