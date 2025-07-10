from fastapi import FastAPI, HTTPException

from app.core.logging import app_logger
from app.models import GetMessageRequestModel, GetMessageResponseModel, IncomingMessage, Prediction
from random import random
from uuid import uuid4
from app.api.zero_shot_model import classify_text
import time
import requests
import uvicorn
import psycopg2

from app.config.database import insert_message, select_messages_by_dialog, init_db
from app.config.config import (DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME)

app = FastAPI()
LLAMA_URL = "http://llama:8000/v1/chat/completions"

@app.on_event("startup")
def on_startup():
    while True:
        try:
            init_db()
            app_logger.info("Database initialized and ready.")
            break
        except Exception as e:
            app_logger.warning(f"Waiting for PostgreSQL... {e}")
            time.sleep(2)


@app.post("/get_message", response_model=GetMessageResponseModel)
async def get_message(body: GetMessageRequestModel):
    app_logger.info(f"Received message dialog_id: {body.dialog_id}, last_msg_id: {body.last_message_id}")

    # ⏹️ Create IncomingMessage object from body
    input_msg = IncomingMessage(
        id=uuid4(),
        dialog_id=body.dialog_id,
        participant_index=0,  # assume 0 for user
        text=body.last_msg_text
    )

    # 1️⃣ Save user message
    insert_message(
        id=input_msg.id,
        text=input_msg.text,
        dialog_id=input_msg.dialog_id,
        participant_index=input_msg.participant_index
    )

    # 2️⃣ Retrieve dialog history
    conversation = select_messages_by_dialog(body.dialog_id)
    if not conversation:
        return GetMessageResponseModel(
            new_msg_text="❌ No dialog history found.",
            dialog_id=body.dialog_id
        )

    # 3️⃣ Format conversation as chat history for llama.cpp
    chat_history = []
    for msg in conversation:
        role = "assistant" if msg["participant_index"] == 1 else "user"
        chat_history.append({"role": role, "content": msg["text"]})

    # 4️⃣ Query llama.cpp API (แทน predict)
    response = requests.post(
        LLAMA_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy-key"
        },
        json={
            "model": "llama.cpp",
            "messages": chat_history
        }
    )

    if response.status_code != 200:
        reply_text = "❌ LLM inference failed."
    else:
        reply_text = response.json()["choices"][0]["message"]["content"]

    # 5️⃣ Save assistant reply
    bot_msg_id = uuid4()
    insert_message(
        id=bot_msg_id,
        text=reply_text,
        dialog_id=body.dialog_id,
        participant_index=1
    )

    # 6️⃣ Return result exactly how you did:
    return GetMessageResponseModel(
        new_msg_text=reply_text,
        dialog_id=body.dialog_id
    )

@app.post("/predict", response_model=Prediction)
def predict(msg: IncomingMessage) -> Prediction:
    candidate_labels = ["bot", "human"]
    result = classify_text(msg.text, candidate_labels)
    print(f"Model result for text '{msg.text}': {result}")

     # Compute actual probability from model
    prob_dict = dict(zip(result["labels"], result["scores"]))
    is_bot_probability = prob_dict.get("bot", 0.0)

    prediction_id = uuid4()

    return Prediction(
        id=prediction_id,
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=is_bot_probability
    )