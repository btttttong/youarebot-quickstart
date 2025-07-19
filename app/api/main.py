from fastapi import FastAPI, HTTPException
import os
import logging

from app.core.logging import app_logger
from app.models import GetMessageRequestModel, GetMessageResponseModel, IncomingMessage, Prediction, TextInput
from uuid import uuid4
import time

from app.config.database_wrapper import insert_message, select_messages_by_dialog, init_db
from app.config.config import USE_MEMORY_DB

app = FastAPI(
    title="Bot Classifier API", 
    description="Bot classification using trained RoBERTa LoRA model",
    version="1.0.0"
)

# Global model variables for trained RoBERTa LoRA model
model = None
tokenizer_global = None

def load_trained_model():
    """Load the trained RoBERTa LoRA model"""
    global model, tokenizer_global
    
    # Model path - works both locally and in Docker
    local_model_path = os.getenv("MODEL_PATH", "./models/bot_classifier")
    if not os.path.exists(local_model_path):
        local_model_path = "/models/bot_classifier"  # Docker path
    
    app_logger.info("=== Loading Trained RoBERTa LoRA Model ===")
    app_logger.info(f"🔄 Loading trained model from: {local_model_path}")
    
    try:
        if os.path.exists(local_model_path):
            app_logger.info(f"Model directory found: {local_model_path}")
            app_logger.info(f"Model files: {os.listdir(local_model_path)}")
            
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
            tokenizer_global = AutoTokenizer.from_pretrained(local_model_path)
            
            app_logger.info("✅ Trained RoBERTa LoRA model loaded successfully!")
            app_logger.info(f"Model type: {type(model)}")
            app_logger.info(f"Tokenizer type: {type(tokenizer_global)}")
        else:
            app_logger.warning(f"❌ Model path {local_model_path} does not exist")
            model = None
            tokenizer_global = None
    except Exception as e:
        app_logger.error(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        model = None
        tokenizer_global = None
    
    app_logger.info(f"✅ Model loading complete. Trained model available: {model is not None}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Bot Classifier API with Trained RoBERTa LoRA Model", 
        "model_loaded": model is not None,
        "endpoints": {
            "predict": "POST /predict - Simple text classification (youare.bot compatible)",
            "predict/full": "POST /predict/full - Full prediction with IncomingMessage format",
            "get_message": "POST /get_message - Process message and save to database",
            "health": "GET /health - Health check"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.on_event("startup")
def on_startup():
    # Load the trained model first
    load_trained_model()
    
    # Initialize database
    while True:
        try:
            init_db()
            app_logger.info("Database initialized and ready.")
            break
        except Exception as e:
            app_logger.warning(f"Waiting for database... {e}")
            time.sleep(2)

def predict_with_heuristics(text: str) -> float:
    """Fallback heuristic prediction when model is not available"""
    app_logger.info(f"⚠️ Using fallback heuristics for: {text}")
    
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
    
    return max(0.0, min(1.0, is_bot_probability))

def predict_with_model(text: str) -> float:
    """Predict using the trained RoBERTa LoRA model"""
    try:
        app_logger.info(f"🧠 Using trained RoBERTa LoRA model for: {text}")
        
        import torch
        from torch.nn.functional import softmax
        
        # Tokenize the input
        inputs = tokenizer_global(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = softmax(outputs.logits, dim=-1)
            is_bot_probability = probabilities[0][1].item()  # Probability of class 1 (bot)
        
        app_logger.info(f"🎯 Model prediction: {is_bot_probability:.4f}")
        
        # Ensure probability is between 0 and 1
        return max(0.0, min(1.0, is_bot_probability))
        
    except Exception as e:
        app_logger.error(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return 0.5  # Fallback

@app.post("/predict", response_model=Prediction)
def predict(input_data: TextInput) -> Prediction:
    """
    Main prediction endpoint - youare.bot compatible
    Takes just text and returns prediction with generated UUIDs
    """
    text = input_data.text
    app_logger.info(f"🤖 Processing: {text}")
    
    # Generate UUIDs for required fields (youare.bot compatible)
    prediction_id = uuid4()
    message_id = uuid4()
    dialog_id = uuid4()
    participant_index = 0
    
    # Get prediction using model or fallback heuristics
    if model is None:
        is_bot_probability = predict_with_heuristics(text)
    else:
        is_bot_probability = predict_with_model(text)

    app_logger.info(f"📊 Final prediction for '{text}': {is_bot_probability:.4f}")

    return Prediction(
        id=prediction_id,
        message_id=message_id,
        dialog_id=dialog_id,
        participant_index=participant_index,
        is_bot_probability=is_bot_probability
    )

@app.post("/predict/full", response_model=Prediction)
def predict_full(msg: IncomingMessage) -> Prediction:
    """
    Full prediction endpoint for complete IncomingMessage format
    Takes IncomingMessage with all fields and returns prediction
    """
    text = msg.text
    app_logger.info(f"🤖 Processing full message: {text}")
    
    # Get prediction using model or fallback heuristics
    if model is None:
        is_bot_probability = predict_with_heuristics(text)
    else:
        is_bot_probability = predict_with_model(text)

    app_logger.info(f"📊 Final prediction for '{text}': {is_bot_probability:.4f}")
    
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

    # Call predict_full() internally (no HTTP overhead)
    prediction = predict_full(input_msg)
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