from fastapi import FastAPI
from pydantic import BaseModel
from uuid import UUID, uuid4
import os

app = FastAPI(title="Bot Classifier with Trained Model", description="Bot classification using trained RoBERTa LoRA model")

# Simple input model for youare.bot compatibility
class TextInput(BaseModel):
    text: str

# Full input model for complete API compatibility
class IncomingMessage(BaseModel):
    text: str
    dialog_id: UUID = None
    id: UUID = None
    participant_index: int = 0

# Output model - matches youare.bot PredictionClassifier format
class PredictionOutput(BaseModel):
    id: UUID
    message_id: UUID
    dialog_id: UUID
    participant_index: int
    is_bot_probability: float

# Global model variable
model = None
tokenizer_global = None

@app.on_event("startup")
def load_model():
    global model, tokenizer_global
    local_model_path = "/models/bot_classifier"
    
    print(f"=== Bot Classifier Startup ===")
    print(f"🔄 Loading trained model from: {local_model_path}")
    
    try:
        if os.path.exists(local_model_path):
            print(f"Model directory found: {local_model_path}")
            print(f"Model files: {os.listdir(local_model_path)}")
            
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
            tokenizer_global = AutoTokenizer.from_pretrained(local_model_path)
            
            print(f"✅ Trained RoBERTa LoRA model loaded successfully!")
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
    
    print(f"✅ Startup complete. Trained model available: {model is not None}")

@app.get("/")
async def root():
    return {
        "message": "Bot Classifier with Trained RoBERTa LoRA Model", 
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
    
    print(f"🤖 Processing: {text}")
    
    # Generate UUIDs for required fields
    message_id = uuid4()
    dialog_id = uuid4()
    participant_index = 0
    
    if model is None:
        # Fallback logic when model is not available
        print(f"⚠️ Using fallback heuristics for: {text}")
        
        text_lower = text.lower()
        is_bot_probability = 0.5
        
        # Adjust probability based on simple rules
        if len(text) > 100:
            is_bot_probability += 0.2
        if 'http' in text_lower or 'www.' in text_lower:
            is_bot_probability += 0.3
        if text.isupper() and len(text) > 20:
            is_bot_probability += 0.2
        if any(word in text_lower for word in ['click', 'buy', 'offer', 'deal', 'free', 'urgent']):
            is_bot_probability += 0.1
        if any(word in text_lower for word in ['hello', 'hi', 'thanks', 'please']):
            is_bot_probability -= 0.1
        
        is_bot_probability = max(0.0, min(1.0, is_bot_probability))
        
    else:
        # Use the trained RoBERTa LoRA model for prediction
        try:
            print(f"🧠 Using trained RoBERTa LoRA model for: {text}")
            
            import torch
            from torch.nn.functional import softmax
            
            # Tokenize the input
            inputs = tokenizer_global(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = softmax(outputs.logits, dim=-1)
                is_bot_probability = probabilities[0][1].item()  # Probability of class 1 (bot)
            
            print(f"🎯 Model prediction: {is_bot_probability:.4f}")
            
            # Ensure probability is between 0 and 1
            is_bot_probability = max(0.0, min(1.0, is_bot_probability))
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            is_bot_probability = 0.5  # Fallback

    print(f"📊 Final prediction for '{text}': {is_bot_probability:.4f}")
    
    # Generate prediction ID
    prediction_id = uuid4()
    
    return PredictionOutput(
        id=prediction_id,
        message_id=message_id,
        dialog_id=dialog_id,
        participant_index=participant_index,
        is_bot_probability=is_bot_probability
    )