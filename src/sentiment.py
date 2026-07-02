import pandas as pd
from tqdm import tqdm
from transformers import pipeline

tqdm.pandas()


def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )


def map_sentiment(label):
    label = label.lower()

    if "positive" in label:
        return "positive"
    elif "negative" in label:
        return "negative"
    else:
        return "neutral"


def predict_sentiment(text, sentiment_model):
    text = str(text)

    if len(text.strip()) == 0:
        return {
            "sentiment_label": "neutral",
            "sentiment_score": 0.0
        }

    result = sentiment_model(text[:512])[0]

    return {
        "sentiment_label": map_sentiment(result["label"]),
        "sentiment_score": result["score"]
    }


def run_sentiment_analysis(df, text_column="clean_text"):
    sentiment_model = load_sentiment_model()

    results = df[text_column].progress_apply(
        lambda x: predict_sentiment(x, sentiment_model)
    )

    df["sentiment"] = results.apply(lambda x: x["sentiment_label"])
    df["sentiment_score"] = results.apply(lambda x: x["sentiment_score"])

    return df