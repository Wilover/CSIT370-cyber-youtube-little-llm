from sentence_transformers import SentenceTransformer


def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def create_embeddings(texts, model):
    return model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )