from fastapi import FastAPI
from pydantic import BaseModel
import os
from uuid import uuid4

app = FastAPI(title="Bot Classifier", description="Simple bot classification service")

# Simple models for the required API
class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    is_bot_probability: float
    text: str

# Global model variable
model = None
tokenizer_global = None

@app.on_event("startup")
def load_model():
    global model, tokenizer_global
    local_model_path = "/models/bot_classifier"
    
    print(f"=== Simple Bot Classifier Startup ===")
    print(f"🔄 Loading model from: {local_model_path}")
    
    try:
        if os.path.exists(local_model_path):
            print(f"Model directory found: {local_model_path}")
            print(f"Model files: {os.listdir(local_model_path)}")
            
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
            tokenizer_global = AutoTokenizer.from_pretrained(local_model_path)
            
            print(f"✅ Model loaded successfully!")
            print(f"Model type: {type(model)}")
            print(f"Tokenizer type: {type(tokenizer_global)}")
        else:
            print(f"❌ Model path {local_model_path} does not exist")
            model = None
            tokenizer_global = None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        model = None
        tokenizer_global = None
    
    print(f"✅ Startup complete. Model available: {model is not None}")

@app.get("/")
async def root():
    return {
        "message": "Bot Classifier API", 
        "model_loaded": model is not None,
        "endpoints": {
            "predict": "POST /predict - Classify if text is from a bot",
            "health": "GET /health - Health check"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: TextInput) -> PredictionOutput:
    text = input_data.text
    
    if model is None:
        # Fallback logic when model is not available
        print(f"🤖 Using fallback logic for: {text}")
        
        # Simple heuristic: messages with certain patterns might be bot-like
        text_lower = text.lower()
        is_bot_probability = 0.5
        
        # Adjust probability based on simple rules
        if len(text) > 100:
            is_bot_probability += 0.2
        if 'http' in text_lower or 'www.' in text_lower:
            is_bot_probability += 0.3
        if text.isupper() and len(text) > 20:
            is_bot_probability += 0.2
        if any(word in text_lower for word in ['click', 'buy', 'offer', 'deal']):
            is_bot_probability += 0.1
        
        is_bot_probability = min(is_bot_probability, 1.0)
        
    else:
        # Use the loaded model for prediction
        try:
            print(f"🧠 Using trained model for: {text}")
            
            import torch
            from torch.nn.functional import softmax
            
            # Tokenize the input
            inputs = tokenizer_global(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = softmax(outputs.logits, dim=-1)
                is_bot_probability = probabilities[0][1].item()  # Probability of class 1 (bot)
            
            print(f"Model prediction: {is_bot_probability:.4f}")
            
            # Ensure probability is between 0 and 1
            is_bot_probability = max(0.0, min(1.0, is_bot_probability))
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            is_bot_probability = 0.5  # Fallback

    print(f"📊 Final prediction for '{text}': {is_bot_probability:.4f}")
    
    return PredictionOutput(
        is_bot_probability=is_bot_probability,
        text=text
    )