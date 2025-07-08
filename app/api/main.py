from fastapi import FastAPI, HTTPException

from app.core.logging import app_logger
from app.models import GetMessageRequestModel, GetMessageResponseModel, IncomingMessage, Prediction
from random import random
from uuid import uuid4
from app.api.zero_shot_model import classify_text

app = FastAPI()


@app.post("/get_message", response_model=GetMessageResponseModel)
async def get_message(body: GetMessageRequestModel):
    app_logger.info(f"Received message dialog_id: {body.dialog_id}, last_msg_id: {body.last_message_id}")

    #  zero-shot:  label bot/human
    candidate_labels = ["bot", "human"]
    result = classify_text(body.last_msg_text, candidate_labels)

    top_label = result["labels"][0]
    confidence = result["scores"][0]
    response_text = f"🤖 Prediction: {top_label.upper()} (confidence: {confidence:.2f})"

    return GetMessageResponseModel(
        new_msg_text=response_text,
        dialog_id=body.dialog_id
    )
    app_logger.info(
        f"Received message dialog_id: {body.dialog_id}, last_msg_id: {body.last_message_id}"
    )
    return GetMessageResponseModel(
        new_msg_text=body.last_msg_text, dialog_id=body.dialog_id
    )

@app.post("/predict", response_model=Prediction)
def predict(msg: IncomingMessage) -> Prediction:
    """
    Endpoint to save a message and get the probability
    that this message if from bot .

    Returns a `Prediction` object.
    """

    is_bot_probability = random()  # Simulate a probability for the sake of example
    prediction_id = uuid4()

    return Prediction(
        id=prediction_id,
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=is_bot_probability
    )