from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.pyfunc
import os
from app.models import IncomingMessage, Prediction
from uuid import uuid4

app = FastAPI()

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Global model variable
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        # Load the champion model from MLflow
        # This assumes there's a registered model named "bot_classifier"
        model = mlflow.pyfunc.load_model("models:/bot_classifier/Champion")
        print("Model loaded successfully from MLflow")
    except Exception as e:
        print(f"Warning: Could not load model from MLflow: {e}")
        # Fallback to a simple rule-based classifier
        model = None

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=Prediction)
async def predict(msg: IncomingMessage) -> Prediction:
    if model is None:
        # Fallback logic when model is not available
        # Simple heuristic: messages with lots of repetition or ALL CAPS might be bot-like
        text = msg.text.lower()
        is_bot_probability = 0.5
        
        if len(text) > 100:
            is_bot_probability += 0.2
        if text.count('http') > 0:
            is_bot_probability += 0.3
        if text.isupper() and len(text) > 20:
            is_bot_probability += 0.2
        
        is_bot_probability = min(is_bot_probability, 1.0)
    else:
        # Use the MLflow model for prediction
        try:
            # Assuming the model expects a pandas DataFrame with a 'text' column
            import pandas as pd
            df = pd.DataFrame({'text': [msg.text]})
            prediction_result = model.predict(df)
            
            # Extract probability (assuming the model returns probabilities)
            if hasattr(prediction_result, '__iter__') and not isinstance(prediction_result, str):
                is_bot_probability = float(prediction_result[0])
            else:
                is_bot_probability = float(prediction_result)
            
            # Ensure probability is between 0 and 1
            is_bot_probability = max(0.0, min(1.0, is_bot_probability))
        except Exception as e:
            print(f"Error during prediction: {e}")
            is_bot_probability = 0.5  # Fallback

    prediction_id = uuid4()
    
    return Prediction(
        id=prediction_id,
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=is_bot_probability
    )