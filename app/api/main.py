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

    input_msg = IncomingMessage(
        id=uuid4(),
        dialog_id=body.dialog_id,
        participant_index=0,  # assume 0 for user
        text=body.last_msg_text
    )

    prediction = predict(input_msg)  # Reuse prediction logic

    response_text = f"🤖 BOT probability: {prediction.is_bot_probability:.2f}"

    return GetMessageResponseModel(
        new_msg_text=response_text,
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