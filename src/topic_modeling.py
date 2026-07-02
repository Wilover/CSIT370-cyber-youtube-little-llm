import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


def run_topic_modeling(df, text_column="final_text", min_topic_size=10):
    texts = df[text_column].fillna("").astype(str).tolist()

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        calculate_probabilities=True,
        verbose=True
    )

    topics, probabilities = topic_model.fit_transform(texts)

    df["topic_id"] = topics

    topic_info = topic_model.get_topic_info()

    return df, topic_model, topic_info

def run_topic_modeling_per_sentiment(df, text_column="final_text", sentiment_column="sentiment", min_topic_size=5):
    all_topic_results = []
    sentiments = df[sentiment_column].dropna().unique()

    for sentiment in sentiments:
        print(f"\nRunning BERTopic for sentiment: {sentiment}")
        subset = df[df[sentiment_column] == sentiment].copy()

        if len(subset) < min_topic_size:
            print(f"Skipping {sentiment}: not enough comments")
            continue

        texts = subset[text_column].fillna("").astype(str).tolist()

        embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        topic_model = BERTopic(
            embedding_model=embedding_model,
            min_topic_size=min_topic_size,
            calculate_probabilities=False,
            verbose=True
        )

        topics, _ = topic_model.fit_transform(texts)
        subset["sentiment_topic_id"] = topics
        topic_info = topic_model.get_topic_info()
        topic_info["sentiment"] = sentiment

        subset.to_csv(
            f"outputs/topic_results_{sentiment}.csv",
            index=False
        )

        topic_info.to_csv(
            f"outputs/topic_info_{sentiment}.csv",
            index=False
        )

        topic_model.save(
            f"outputs/bertopic_model_{sentiment}"
        )

        all_topic_results.append(topic_info)

    if all_topic_results:
        combined_topic_info = pd.concat(
            all_topic_results,
            ignore_index=True
        )
    else:
        combined_topic_info = pd.DataFrame()

    return combined_topic_info