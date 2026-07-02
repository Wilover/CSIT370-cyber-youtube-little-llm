import re
from collections import Counter
import emoji

import pandas as pd
import spacy
from tqdm import tqdm
from langdetect import detect, LangDetectException

tqdm.pandas()


CYBER_KEEP_PATTERN = re.compile(
    r"^(cve-\d{4}-\d+|xss|csrf|sqli|sql|2fa|mfa|c2|sha-\d+|md5|dns|http|https|tcp|udp|ip|vpn|api|rce|dos|ddos|user_mention|multi_exclamation|multi_question)$",
    re.IGNORECASE
)


def detect_language(text):
    try:
        return detect(str(text))
    except LangDetectException:
        return "unknown"


def clean_text(text):
    text = str(text)
    text = emoji.demojize(text, language="en")
    text = re.sub(r"@\w+", " USER_MENTION ", text)
    text = text.replace("\u2028", " ")
    text = text.replace("\u2029", " ")
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"!{2,}", " MULTI_EXCLAMATION ", text)
    text = re.sub(r"\?{2,}", " MULTI_QUESTION ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keep_token(token):
    token_text = token.text.lower()

    if CYBER_KEEP_PATTERN.match(token_text):
        return True

    if token.is_stop or token.is_punct or token.is_space:
        return False

    if len(token_text) <= 2:
        return False

    return token.is_alpha


def get_lemmas(text, nlp):
    doc = nlp(str(text))
    lemmas = []

    for token in doc:
        if keep_token(token):
            if CYBER_KEEP_PATTERN.match(token.text):
                lemmas.append(token.text.lower())
            else:
                lemmas.append(token.lemma_.lower())

    return lemmas


def preprocess_data(df, text_column="text", top_n_zipf=50):
    print(">>> preprocess_data() STARTED <<<")

    nlp = spacy.load("en_core_web_sm")

    df = df.copy()

    df["clean_text"] = df[text_column].progress_apply(clean_text)

    df["language"] = df["clean_text"].progress_apply(detect_language)
    df["is_english"] = df["language"] == "en"

    print("\nLanguage distribution before filtering:")
    print(df["language"].value_counts().head(10))

    print("\nEnglish comments:", df["is_english"].sum())
    print("Non-English comments:", (~df["is_english"]).sum())

    # Keep only English comments for this project
    df = df[df["is_english"]].copy()

    print("Rows after keeping English only:", len(df))

    df["lemmas"] = df["clean_text"].progress_apply(lambda x: get_lemmas(x, nlp))
    df["lemma_text"] = df["lemmas"].apply(lambda x: " ".join(x))

    all_words = []
    for tokens in df["lemmas"]:
        all_words.extend(tokens)

    word_freq = Counter(all_words)

    zipf_df = pd.DataFrame(
        word_freq.most_common(),
        columns=["word", "frequency"]
    )

    zipf_df["rank"] = range(1, len(zipf_df) + 1)

    zipf_stopwords = set(zipf_df.head(top_n_zipf)["word"])

    df["final_tokens"] = df["lemmas"].apply(
        lambda tokens: [word for word in tokens if word not in zipf_stopwords]
    )

    df["final_text"] = df["final_tokens"].apply(lambda x: " ".join(x))

    return df, zipf_df, zipf_stopwords