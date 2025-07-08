# app/api/zero_shot_model.py
from transformers import pipeline

zero_shot = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

def classify_text(text: str, labels: list[str]) -> dict:
    return zero_shot(text, candidate_labels=labels)