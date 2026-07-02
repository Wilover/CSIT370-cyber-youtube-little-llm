from tqdm import tqdm
from transformers import pipeline

tqdm.pandas()


def load_irony_model():
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-irony"
    )


def load_emotion_model():
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-emotion"
    )


def predict_irony(text, irony_model):
    text = str(text)

    if len(text.strip()) == 0:
        return {
            "irony_label": "non_irony",
            "irony_score": 0.0
        }

    result = irony_model(text[:512])[0]

    return {
        "irony_label": result["label"],
        "irony_score": result["score"]
    }


def predict_emotion(text, emotion_model):
    text = str(text)

    if len(text.strip()) == 0:
        return {
            "emotion_label": "neutral",
            "emotion_score": 0.0
        }

    result = emotion_model(text[:512])[0]

    return {
        "emotion_label": result["label"],
        "emotion_score": result["score"]
    }


def run_sarcasm_emotion_analysis(df, text_column="clean_text"):
    irony_model = load_irony_model()
    emotion_model = load_emotion_model()

    irony_results = df[text_column].progress_apply(
        lambda x: predict_irony(x, irony_model)
    )

    emotion_results = df[text_column].progress_apply(
        lambda x: predict_emotion(x, emotion_model)
    )

    df["irony_label"] = irony_results.apply(lambda x: x["irony_label"])
    df["irony_score"] = irony_results.apply(lambda x: x["irony_score"])

    df["emotion_label"] = emotion_results.apply(lambda x: x["emotion_label"])
    df["emotion_score"] = emotion_results.apply(lambda x: x["emotion_score"])

    return df