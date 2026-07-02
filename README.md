# Cyber YouTube Intelligence Engine

An end-to-end NLP and Agentic RAG system for analysing cybersecurity YouTube comments and answering student questions using grounded evidence from real community discussions.

---

## About the Project

**Cyber YouTube Intelligence Engine** transforms noisy cybersecurity-related YouTube comments into structured, searchable, and explainable knowledge.

The system collects and processes YouTube comments, extracts cybersecurity entities and keywords, performs sentiment and emotion analysis, discovers discussion topics, builds a FAISS vector database, and uses an Agentic RAG workflow to answer user questions through a Streamlit dashboard.

The final system is designed as a student-facing cybersecurity intelligence assistant.

---

## Key Features

- YouTube comment preprocessing
- English-language filtering
- Emoji, mention, punctuation, and noisy text normalization
- Cybersecurity keyword extraction using a custom cyber lexicon
- Regex extraction for CVEs, IP addresses, and hashes
- spaCy-based NER
- Sentiment analysis
- Sarcasm / irony detection
- Emotion detection
- BERTopic topic modeling
- Topic modeling per sentiment
- FAISS vector database
- Hybrid retrieval
- Llama 3.2 answer generation through Ollama
- LangGraph supervisor-based agent orchestration
- MLflow monitoring and evaluation
- Streamlit dashboard for user interaction

---

## System Architecture

```text
YouTube Comments
        ↓
Preprocessing Layer
        ↓
NER + Cybersecurity Keyword Extraction
        ↓
Sentiment / Emotion / Sarcasm Analysis
        ↓
Topic Modeling
        ↓
FAISS Vector Store
        ↓
Hybrid Retrieval
        ↓
LangGraph Supervisor
        ↓
Agentic RAG
        ↓
Llama 3.2 Grounded Answer
        ↓
MLflow Monitoring
        ↓
Streamlit Dashboard
