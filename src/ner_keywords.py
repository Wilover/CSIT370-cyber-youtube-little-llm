import re
import pandas as pd
import spacy
from tqdm import tqdm
from pathlib import Path

tqdm.pandas()


CVE_PATTERN = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HASH_PATTERN = re.compile(r"\b[a-fA-F0-9]{32,64}\b")
BASE_DIR = Path(__file__).resolve().parent.parent

def load_cyber_terms():
    resource_path = BASE_DIR / "resources" / "cyber_terms.txt"
    with open(resource_path, "r", encoding="utf-8") as f:
        return {
            line.strip().lower()
            for line in f
            if line.strip()
        }


def extract_regex_entities(text):
    text = str(text)

    return {
        "cves": CVE_PATTERN.findall(text),
        "ips": IP_PATTERN.findall(text),
        "hashes": HASH_PATTERN.findall(text)
    }


def extract_cyber_terms(text, cyber_terms):
    text_lower = str(text).lower()

    found = []

    for term in cyber_terms:
        if term in text_lower:
            found.append(term)

    return sorted(set(found))


def extract_spacy_entities(text, nlp):
    doc = nlp(str(text))

    entities = []

    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_
        })

    return entities


def run_ner_keywords(df, text_column="clean_text"):
    nlp = spacy.load("en_core_web_sm")
    cyber_terms = load_cyber_terms()

    df["regex_entities"] = df[text_column].progress_apply(extract_regex_entities)

    df["cyber_keywords"] = df[text_column].progress_apply(
        lambda x: extract_cyber_terms(x, cyber_terms)
    )

    df["spacy_entities"] = df[text_column].progress_apply(
        lambda x: extract_spacy_entities(x, nlp)
    )

    df["all_extracted_terms"] = df.apply(
        lambda row: list(
            set(
                row["regex_entities"]["cves"]
                + row["regex_entities"]["ips"]
                + row["regex_entities"]["hashes"]
                + row["cyber_keywords"]
            )
        ),
        axis=1
    )

    return df