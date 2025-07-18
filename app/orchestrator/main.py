from fastapi import FastAPI, HTTPException
import requests
import os
from app.models import IncomingMessage, Prediction, GetMessageRequestModel, GetMessageResponseModel
from uuid import uuid4

app = FastAPI()

# Service URLs from environment variables
CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://classifier:8000")
LLM_URL = os.getenv("LLM_URL", "http://llm:11434")

@app.get("/health")
async def health():
    return {"status": "healthy", "services": {"classifier": CLASSIFIER_URL, "llm": LLM_URL}}

@app.post("/predict", response_model=Prediction)
async def predict(msg: IncomingMessage) -> Prediction:
    """Forward prediction request to classifier service"""
    try:
        print(f"Forwarding to classifier: {CLASSIFIER_URL}/predict")
        print(f"Message data: {msg.model_dump()}")
        
        # Convert UUIDs to strings for JSON serialization
        msg_data = msg.model_dump()
        msg_data['id'] = str(msg_data['id'])
        msg_data['dialog_id'] = str(msg_data['dialog_id'])
        
        response = requests.post(
            f"{CLASSIFIER_URL}/predict",
            json=msg_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Classifier response: {response.status_code} - {response.text}")
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Classifier service error: {response.text}")
        
        return Prediction(**response.json())
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Unable to connect to classifier service: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/get_message", response_model=GetMessageResponseModel)
async def get_message(body: GetMessageRequestModel) -> GetMessageResponseModel:
    """Forward chat request to LLM service"""
    try:
        # Prepare the request for the LLM service
        llm_request = {
            "model": "llama.cpp",
            "messages": [{"role": "user", "content": body.last_msg_text}]
        }
        
        response = requests.post(
            f"{LLM_URL}/v1/chat/completions",
            json=llm_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"LLM service error: {response.text}")
        
        # Extract the response from the LLM
        llm_response = response.json()
        reply_text = llm_response["choices"][0]["message"]["content"]
        
        return GetMessageResponseModel(
            new_msg_text=reply_text,
            dialog_id=body.dialog_id
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Unable to connect to LLM service: {str(e)}")
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=500, detail=f"Invalid LLM response format: {str(e)}")