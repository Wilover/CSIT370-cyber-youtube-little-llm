import pickle

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.embeddings import load_embedding_model


def load_faiss_store(
    index_path="vector_store/faiss.index",
    metadata_path="vector_store/metadata.pkl"
):
    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        store = pickle.load(f)

    return index, store["texts"], store["metadata"]


def semantic_retrieval(query, k=5):
    index, texts, metadata = load_faiss_store()

    model = load_embedding_model()
    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_vector, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        results.append(
            {
                "text": texts[idx],
                "metadata": metadata[idx],
                "semantic_score": float(score)
            }
        )

    return results


def lexical_retrieval(query, df, text_column="final_text", k=5):
    texts = df[text_column].fillna("").astype(str).tolist()

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    top_indices = scores.argsort()[::-1][:k]

    results = []

    for idx in top_indices:
        results.append(
            {
                "text": texts[idx],
                "metadata": df.iloc[idx].to_dict(),
                "lexical_score": float(scores[idx])
            }
        )

    return results


def metadata_filter(df, sentiment=None, topic_id=None, video_id=None):
    filtered = df.copy()

    if sentiment is not None:
        filtered = filtered[filtered["sentiment"] == sentiment]

    if topic_id is not None:
        filtered = filtered[filtered["topic_id"] == topic_id]

    if video_id is not None:
        filtered = filtered[filtered["video_id"] == video_id]

    return filtered


def hybrid_retrieval(query, df, k=5, alpha=0.7):
    semantic_results = semantic_retrieval(query, k=k * 2)
    lexical_results = lexical_retrieval(query, df, k=k * 2)

    combined = {}

    for item in semantic_results:
        key = item["metadata"].get("original_text", item["text"])
        combined[key] = {
            "text": item["text"],
            "metadata": item["metadata"],
            "semantic_score": item["semantic_score"],
            "lexical_score": 0.0
        }

    for item in lexical_results:
        key = item["metadata"].get("text", item["text"])

        if key not in combined:
            combined[key] = {
                "text": item["text"],
                "metadata": item["metadata"],
                "semantic_score": 0.0,
                "lexical_score": item["lexical_score"]
            }
        else:
            combined[key]["lexical_score"] = item["lexical_score"]

    ranked = []

    for item in combined.values():
        final_score = (
            alpha * item["semantic_score"]
            + (1 - alpha) * item["lexical_score"]
        )

        item["hybrid_score"] = final_score
        ranked.append(item)

    ranked = sorted(
        ranked,
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return ranked[:k]