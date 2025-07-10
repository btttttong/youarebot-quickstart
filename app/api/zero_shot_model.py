# app/api/zero_shot_model.py
from transformers.pipelines import pipeline

zero_shot = pipeline('zero-shot-classification', model='facebook/bart-large-mnli', device=-1)

def classify_text(text: str, labels: list[str]) -> dict:
    return zero_shot(text, candidate_labels=labels)