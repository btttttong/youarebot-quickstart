import json
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split

# 1️⃣ Load and preprocess data
def load_train_data(json_file, csv_file):
    labels_df = pd.read_csv(csv_file)
    labels_df = labels_df[labels_df["participant_index"] == 0]
    labels_dict = dict(zip(labels_df["dialog_id"], labels_df["is_bot"]))

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    labels = []

    for dialog_id, messages in data.items():
        part_0 = " ".join([m["text"] for m in messages if m["participant_index"] == "0"])
        part_1 = " ".join([m["text"] for m in messages if m["participant_index"] == "1"])

        label_0 = int(labels_dict[dialog_id])
        label_1 = 1 - label_0

        texts.append(part_0)
        labels.append(label_0)

        texts.append(part_1)
        labels.append(label_1)

    df = pd.DataFrame({"text": texts, "label": labels})
    return df

df = load_train_data("app/data/train.json", "app/data/ytrain.csv")

# 2️⃣ Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["label"], test_size=0.1, random_state=42, stratify=df["label"]
)

train_df = Dataset.from_pandas(pd.DataFrame({"text": train_texts, "label": train_labels}))
val_df = Dataset.from_pandas(pd.DataFrame({"text": val_texts, "label": val_labels}))

# 3️⃣ Tokenization
model_name = "facebook/bart-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

train_df = train_df.map(tokenize, batched=True)
val_df = val_df.map(tokenize, batched=True)

# 4️⃣ Load model for classification (binary task)
model = AutoModelForSequenceClassification.from_pretrained(
    "facebook/bart-large-mnli",
    num_labels=2,
    ignore_mismatched_sizes=True
)

# 5️⃣ Training setup
args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10
)

# Optional metrics
import numpy as np
from evaluate import load as load_metric

accuracy_metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)
    return acc

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_df,
    eval_dataset=val_df,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 6️⃣ Train!
trainer.train()

# 7️⃣ Save final model
trainer.save_model("./finetuned-bart-large-mnli-bot-human")
tokenizer.save_pretrained("./finetuned-bart-large-mnli-bot-human")