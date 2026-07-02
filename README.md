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
```

---

# Technologies Used
```text
Python 3.10
pandas
spaCy
langdetect
emoji
Hugging Face Transformers
SentenceTransformers
BERTopic
FAISS
LangChain
LangGraph
Ollama
Llama 3.2
MLflow
Streamlit
```

---

## Installation
1. Clone the repository
```text
git clone https://github.com/Wilover/CSIT370-cyber-youtube-little-llm.git
cd CSIT370-cyber-youtube-little-llm
```
2. Create a virtual environment
```text
py -3.10 -m venv cyber_env
```
3. Activate the environment
```text
.\cyber_env\Scripts\Activate.ps1
```
If PowerShell blocks activation:
```text
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\cyber_env\Scripts\Activate.ps1
```
4. Install dependencies
```text
pip install -r requirements.txt
```
5. Download the spaCy model
```text
python -m spacy download en_core_web_sm
```
6. Install and prepare Ollama
```text
https://ollama.com
Then pull Llama 3.2:
ollama pull llama3.2:3b
```

---

# Running the Pipeline

## The project is controlled from run.py.

Edit the MODE variable:

MODE = "full_build"

Available modes:
```text
MODE = "preprocess"
MODE = "ner"
MODE = "sentiment"
MODE = "emotion"
MODE = "topic"
MODE = "index"
MODE = "agent_demo"
MODE = "full_build"
MODE = "full_demo"
```
Run:
```text
python -B run.py
Recommended first run
MODE = "full_build"
```
This runs:
```text
Preprocessing
→ NER
→ Sentiment
→ Emotion
→ Topic Modeling
→ FAISS Index
```
Then use:
```text
MODE = "full_demo"
```
to run the Agentic RAG demo.

---

# Running the Dashboard

The Streamlit dashboard is the main user interaction point.
```text
python -m streamlit run src/dashboard.py
```
The dashboard includes:
```text
Question box
Agentic RAG answer panel
Evidence viewer
Sentiment charts
Emotion charts
Topic distribution
Dataset preview
Optional MLflow logging
```
---

# Running MLflow
```text
mlflow ui --host 127.0.0.1 --port 5000 --workers 1
```
Open:
```text
http://127.0.0.1:5000
```
MLflow tracks:
        User question
        Agent route
        Retrieval scores
        Evidence count
        Answer length
        Faithfulness overlap
        Latency
        LLM model
        Retrieved evidence
        Generated answer

---

# Example Questions
```text
What cybersecurity certifications do students recommend?
What cybersecurity tools are students discussing?
What are students concerned about in AI security?
Which topics receive the most negative reactions?
Summarize the discussion about ethical hacking.
What do students say about prompt injection?
```
---

# Example Workflow
```text
User question:
"What cybersecurity certifications do students recommend?"

LangGraph supervisor:
routes query to rag_agent

Retriever:
finds relevant YouTube comments

Generator:
Llama 3.2 creates a grounded answer

Output:
answer + source comments + confidence + MLflow trace
```

---


# Evaluation and Monitoring
```text
The project uses MLflow to monitor the RAG and agentic workflow.

Evaluation includes:

Retrieval quality
Average retrieval score
Maximum retrieval score
Minimum retrieval score
Evidence count
Faithfulness overlap
Answer length
Generation latency
Total latency
Route tracking
Dataset summary logging
```

---

# Limitations
```text
The dataset is based on YouTube comments, which are noisy and informal.
Some cybersecurity terms may be missed if they are not included in the custom lexicon.
General-purpose NER may misclassify cybersecurity-specific entities.
Emotion models trained on social media may produce imperfect labels for technical comments.
The system answers only from retrieved dataset evidence and does not verify facts externally.
```
