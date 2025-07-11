from fastapi import FastAPI, HTTPException

from app.core.logging import app_logger
from app.models import GetMessageRequestModel, GetMessageResponseModel, IncomingMessage, Prediction
from uuid import uuid4
import time
import requests

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

@app.post("/predict", response_model=Prediction)
def predict(msg: IncomingMessage) -> Prediction:
    classification_prompt = (
        "What is the probability (between 0 and 1) that the following message was written by a bot? "
        "Respond with only a number.\n\n"
        f"Message: {msg.text}"
    )

    llama_response = requests.post(
        LLAMA_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": "llama.cpp",
            "messages": [{"role": "user", "content": classification_prompt}]
        }
    )
    app_logger.info(f"LLAMA response: {llama_response.status_code}, {llama_response.text}")

    if llama_response.status_code != 200:
        is_bot_probability = 0.5  # fallback
    else:
        reply_text = llama_response.json()["choices"][0]["message"]["content"]
        try:
            is_bot_probability = float(reply_text.strip())
            is_bot_probability = min(max(is_bot_probability, 0.0), 1.0)  # Clamp between 0 and 1
        except ValueError:
            is_bot_probability = 0.5  # fallback if llama gave weird text

    prediction_id = uuid4()

    return Prediction(
        id=prediction_id,
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=is_bot_probability
    )

@app.post("/get_message", response_model=GetMessageResponseModel)
async def get_message(body: GetMessageRequestModel):
    app_logger.info(f"Received message dialog_id: {body.dialog_id}, last_msg_id: {body.last_message_id}")

    # Create IncomingMessage
    input_msg = IncomingMessage(
        id=uuid4(),
        dialog_id=body.dialog_id,
        participant_index=0,
        text=body.last_msg_text
    )

    # Save user message
    insert_message(
        id=input_msg.id,
        text=input_msg.text,
        dialog_id=input_msg.dialog_id,
        participant_index=input_msg.participant_index
    )

    # Call predict() internally (no HTTP overhead)
    prediction = predict(input_msg)
    reply_text = f"🤖 Prediction done (is_bot={prediction.is_bot_probability:.2f})"

    # Save assistant reply
    bot_msg_id = uuid4()
    insert_message(
        id=bot_msg_id,
        text=reply_text,
        dialog_id=body.dialog_id,
        participant_index=1
    )

    # Return result
    return GetMessageResponseModel(
        new_msg_text=reply_text,
        dialog_id=body.dialog_id
    )