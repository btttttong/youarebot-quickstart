#!/usr/bin/env python3
"""Register the existing trained model to MLflow registry."""

import mlflow
import mlflow.transformers
from mlflow.tracking import MlflowClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5001")

def register_existing_model():
    """Register the existing trained model from ../models/bot_classifier"""
    
    model_path = "../models/bot_classifier"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return False
    
    try:
        # Load the existing model
        print(f"Loading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("✅ Model loaded successfully")
        
        # Start MLflow run and log the model
        with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("bot_classifier_training").experiment_id) as run:
            print("Logging model to MLflow...")
            
            # Log the model
            mlflow.transformers.log_model(
                transformers_model={"model": model, "tokenizer": tokenizer},
                artifact_path="model",
                registered_model_name="bot_classifier"
            )
            print("✅ Model logged to MLflow")
            
            # Get the client to promote to Champion
            client = MlflowClient()
            
            # Find the latest version of the registered model
            model_versions = client.search_model_versions("name='bot_classifier'")
            if model_versions:
                latest_version = max(model_versions, key=lambda x: int(x.version))
                
                # Promote to Champion stage
                client.transition_model_version_stage(
                    name="bot_classifier",
                    version=latest_version.version,
                    stage="Champion"
                )
                print(f"✅ Model version {latest_version.version} promoted to Champion stage")
                return True
            else:
                print("❌ No model versions found after registration")
                return False
                
    except Exception as e:
        print(f"❌ Model registration failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Registering Existing Model to MLflow ===")
    
    if register_existing_model():
        print("\n✅ Model registration complete!")
        print("The classifier should now be able to load from MLflow registry.")
    else:
        print("\n❌ Model registration failed!")
        print("The classifier will continue using local model fallback.")