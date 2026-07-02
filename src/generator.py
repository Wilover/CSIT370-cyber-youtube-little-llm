from langchain_ollama import ChatOllama


def load_generator():
    return ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )


def build_context(retrieved_results):
    context_blocks = []

    for i, item in enumerate(retrieved_results, start=1):
        metadata = item.get("metadata", {})

        original_comment = metadata.get(
            "original_text",
            metadata.get("text", item.get("text", ""))
        )

        block = f"""
Source {i}
Comment: {original_comment}
Processed text: {item.get("text", "")}
Sentiment: {metadata.get("sentiment", "unknown")}
Emotion: {metadata.get("emotion_label", "unknown")}
Topic ID: {metadata.get("topic_id", "unknown")}
Video ID: {metadata.get("video_id", "unknown")}
Author: {metadata.get("author", "unknown")}
Like count: {metadata.get("like_count", "unknown")}
Retrieval score: {item.get("hybrid_score", item.get("semantic_score", "unknown"))}
"""
        context_blocks.append(block)

    return "\n".join(context_blocks)


def estimate_confidence(retrieved_results):
    if not retrieved_results:
        return 0.0

    scores = []

    for item in retrieved_results:
        score = item.get("hybrid_score", item.get("semantic_score", 0.0))
        try:
            scores.append(float(score))
        except Exception:
            scores.append(0.0)

    return round(sum(scores) / len(scores), 3)


def answer_question(question, retrieved_results):
    llm = load_generator()
    context = build_context(retrieved_results)
    confidence = estimate_confidence(retrieved_results)

    prompt = f"""
You are a cybersecurity teaching assistant.

Use ONLY the retrieved YouTube comments below.
Do not invent facts.
If the comments do not contain enough evidence, say:
"The dataset does not contain enough evidence to answer this question."

Student question:
{question}

Retrieved comments:
{context}

Write a short answer for a cybersecurity student.
Mention source numbers when possible.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content.strip(),
        "confidence": confidence,
        "context": context,
        "sources": retrieved_results
    }


def summarize_results(retrieved_results):
    llm = load_generator()
    context = build_context(retrieved_results)
    confidence = estimate_confidence(retrieved_results)

    prompt = f"""
You are a cybersecurity teaching assistant.

Use ONLY the retrieved YouTube comments below.
Summarize the main discussion patterns.
Mention repeated cybersecurity tools, risks, or concerns.
Do not add external information.

Retrieved comments:
{context}

Summary:
"""

    response = llm.invoke(prompt)

    return {
        "summary": response.content.strip(),
        "confidence": confidence,
        "context": context,
        "sources": retrieved_results
    }


def format_answer_with_evidence(result):
    print("\nANSWER:")
    print(result.get("answer", result.get("summary", "")))

    print("\nCONFIDENCE:")
    print(result.get("confidence", 0.0))

    print("\nEVIDENCE:")
    for i, item in enumerate(result.get("sources", []), start=1):
        metadata = item.get("metadata", {})

        print(f"\nSource {i}")
        print("Comment:", metadata.get("original_text", item.get("text", "")))
        print("Video ID:", metadata.get("video_id", "unknown"))
        print("Sentiment:", metadata.get("sentiment", "unknown"))
        print("Topic ID:", metadata.get("topic_id", "unknown"))
        print("Score:", item.get("hybrid_score", item.get("semantic_score", "unknown")))