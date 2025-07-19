from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Simple Bot Classifier Test", description="Lightweight bot classification service")

# Simple models for the required API
class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    is_bot_probability: float
    text: str

@app.get("/")
async def root():
    return {
        "message": "Simple Bot Classifier Test API", 
        "endpoints": {
            "predict": "POST /predict - Classify if text is from a bot",
            "health": "GET /health - Health check"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": False}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: TextInput) -> PredictionOutput:
    text = input_data.text
    
    print(f"🤖 Processing: {text}")
    
    # Simple heuristic logic (no ML model required)
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

    print(f"📊 Prediction for '{text}': {is_bot_probability:.4f}")
    
    return PredictionOutput(
        is_bot_probability=is_bot_probability,
        text=text
    )