#!/usr/bin/env python3
"""Test script to verify MLflow model registration and fix the 404 error."""

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5001")

def test_mlflow_connection():
    """Test basic MLflow connectivity."""
    try:
        client = MlflowClient()
        experiments = client.search_experiments()
        print(f"✅ MLflow connection successful. Found {len(experiments)} experiments.")
        return True
    except Exception as e:
        print(f"❌ MLflow connection failed: {e}")
        return False

def test_model_registry():
    """Test model registry access."""
    try:
        client = MlflowClient()
        models = client.search_registered_models()
        print(f"✅ Model registry accessible. Found {len(models)} registered models.")
        
        # Look for bot_classifier specifically
        bot_classifier = None
        for model in models:
            if model.name == "bot_classifier":
                bot_classifier = model
                break
        
        if bot_classifier:
            print(f"✅ Found bot_classifier model")
            
            # Check for Champion stage
            versions = client.search_model_versions(f"name='{bot_classifier.name}'")
            champion_versions = [v for v in versions if v.current_stage == "Champion"]
            
            if champion_versions:
                print(f"✅ Found {len(champion_versions)} Champion versions")
                return True
            else:
                print("⚠️  No Champion versions found")
                return False
        else:
            print("⚠️  bot_classifier model not found in registry")
            return False
            
    except Exception as e:
        print(f"❌ Model registry access failed: {e}")
        return False

def test_model_loading():
    """Test loading model from registry."""
    try:
        model = mlflow.pyfunc.load_model("models:/bot_classifier/Champion")
        print("✅ Successfully loaded Champion model from registry")
        return True
    except Exception as e:
        print(f"❌ Failed to load Champion model: {e}")
        return False

if __name__ == "__main__":
    print("=== MLflow Testing ===")
    
    print("\n1. Testing MLflow connection...")
    if not test_mlflow_connection():
        exit(1)
    
    print("\n2. Testing model registry...")
    if not test_model_registry():
        print("   Model registry issues detected - this explains the 404 error")
    
    print("\n3. Testing model loading...")
    if not test_model_loading():
        print("   Model loading failed - classifier will fall back to local models")
    
    print("\n=== Test Complete ===")