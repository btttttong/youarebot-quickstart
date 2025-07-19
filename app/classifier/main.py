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
    local_model_path = "/models/bot_classifier"
    
    print(f"=== Classifier Service Startup ===")
    
    # Strategy 1: Try loading from MLflow first (proper ML lifecycle)
    print("🔄 Attempting to load from MLflow...")
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Get Champion model if available
        champion_versions = client.get_latest_versions("bot_classifier", stages=["Champion"])
        if champion_versions:
            version = champion_versions[0]
            print(f"Found Champion model version {version.version}")
            
            # Try loading with MLflow
            import mlflow.pyfunc
            model_uri = f"models:/bot_classifier/Champion"
            mlflow_model = mlflow.pyfunc.load_model(model_uri)
            
            print(f"✅ Model loaded from MLflow Champion: {model_uri}")
            model = mlflow_model
            tokenizer_global = None  # MLflow handles tokenization
            print(f"Model type: {type(model)}")
            return
            
    except Exception as e:
        print(f"❌ MLflow loading failed: {e}")
    
    # Strategy 2: Load from local directory (Docker volume mount)
    print(f"🔄 Attempting to load from local path: {local_model_path}")
    
    # Check if the models directory exists
    models_dir = "/models"
    if os.path.exists(models_dir):
        print(f"Models directory exists: {models_dir}")
        print(f"Contents: {os.listdir(models_dir)}")
    else:
        print(f"Models directory does not exist: {models_dir}")
    
    try:
        if os.path.exists(local_model_path):
            print(f"Bot classifier directory found: {local_model_path}")
            print(f"Model files: {os.listdir(local_model_path)}")
            
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
            tokenizer_global = AutoTokenizer.from_pretrained(local_model_path)
            
            print(f"✅ Model loaded successfully from {local_model_path}")
            print(f"Model type: {type(model)}")
            print(f"Tokenizer type: {type(tokenizer_global)}")
        else:
            print(f"❌ Model path {local_model_path} does not exist")
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
        # Use the loaded model for prediction
        try:
            # Check if it's an MLflow model or local transformers model
            if hasattr(model, 'predict'):
                # MLflow model - use MLflow interface
                import pandas as pd
                df = pd.DataFrame({'text': [msg.text]})
                prediction_result = model.predict(df)
                
                # Extract probability
                if hasattr(prediction_result, '__iter__') and not isinstance(prediction_result, str):
                    is_bot_probability = float(prediction_result[0])
                else:
                    is_bot_probability = float(prediction_result)
                    
                print(f"MLflow prediction: {is_bot_probability}")
                
            else:
                # Local transformers model - use transformers interface
                import torch
                from torch.nn.functional import softmax
                
                # Tokenize the input
                inputs = tokenizer_global(msg.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                
                # Get prediction
                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = softmax(outputs.logits, dim=-1)
                    is_bot_probability = probabilities[0][1].item()  # Probability of class 1 (bot)
                
                print(f"Transformers prediction: {is_bot_probability}")
            
            # Ensure probability is between 0 and 1
            is_bot_probability = max(0.0, min(1.0, is_bot_probability))
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            is_bot_probability = 0.5  # Fallback

    prediction_id = uuid4()
    
    return Prediction(
        id=prediction_id,
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=is_bot_probability
    )