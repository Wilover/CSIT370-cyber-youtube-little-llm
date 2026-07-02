import os
import pickle

import faiss
import numpy as np

from src.embeddings import load_embedding_model, create_embeddings


def build_metadata(row):
    return {
        "author": row.get("author", ""),
        "updated_at": row.get("updated_at", ""),
        "like_count": row.get("like_count", ""),
        "video_id": row.get("video_id", ""),
        "sentiment": row.get("sentiment", ""),
        "sentiment_score": row.get("sentiment_score", ""),
        "topic_id": row.get("topic_id", ""),
        "cyber_keywords": row.get("cyber_keywords", ""),
        "all_extracted_terms": row.get("all_extracted_terms", ""),
        "original_text": row.get("text", "")
    }


def build_faiss_index(df, text_column="final_text", index_path="vector_store/faiss.index", metadata_path="vector_store/metadata.pkl"):
    os.makedirs("vector_store", exist_ok=True)

    df = df.copy()
    df[text_column] = df[text_column].fillna("").astype(str)

    texts = df[text_column].tolist()

    metadata = [
        build_metadata(row)
        for _, row in df.iterrows()
    ]

    model = load_embedding_model()
    embeddings = create_embeddings(texts, model)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    # cosine similarity because embeddings are normalized
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, index_path)

    with open(metadata_path, "wb") as f:
        pickle.dump(
            {
                "texts": texts,
                "metadata": metadata
            },
            f
        )

    print("FAISS index saved:", index_path)
    print("Metadata saved:", metadata_path)
    print("Number of vectors:", index.ntotal)

    return index