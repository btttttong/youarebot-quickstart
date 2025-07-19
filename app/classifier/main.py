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
    global model, tokenizer_global
    model_path = "/models/bot_classifier"
    
    print(f"=== Classifier Service Startup ===")
    print(f"Looking for model at: {model_path}")
    
    # Check if the models directory exists
    models_dir = "/models"
    if os.path.exists(models_dir):
        print(f"Models directory exists: {models_dir}")
        print(f"Contents: {os.listdir(models_dir)}")
    else:
        print(f"Models directory does not exist: {models_dir}")
    
    try:
        if os.path.exists(model_path):
            print(f"Bot classifier directory found: {model_path}")
            print(f"Model files: {os.listdir(model_path)}")
            
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer_global = AutoTokenizer.from_pretrained(model_path)
            
            print(f"✅ Model loaded successfully from {model_path}")
            print(f"Model type: {type(model)}")
            print(f"Tokenizer type: {type(tokenizer_global)}")
        else:
            print(f"❌ Model path {model_path} does not exist")
            print("To train a model, run: python models/train_lora.py")
            model = None
            tokenizer_global = None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        model = None
        tokenizer_global = None
    
    print(f"Model loading complete. Model available: {model is not None}")

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
        # Use the transformers model for prediction
        try:
            import torch
            from torch.nn.functional import softmax
            
            # Tokenize the input
            inputs = tokenizer_global(msg.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = softmax(outputs.logits, dim=-1)
                is_bot_probability = probabilities[0][1].item()  # Probability of class 1 (bot)
            
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