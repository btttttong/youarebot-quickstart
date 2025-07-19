#!/usr/bin/env python3
"""
Test script to verify the complete model training and deployment flow
"""

import os
import requests
import time
from uuid import uuid4

def check_model_exists():
    """Check if the trained model exists"""
    model_path = "./models/bot_classifier"
    if os.path.exists(model_path):
        files = os.listdir(model_path)
        print(f"✅ Model found at {model_path}")
        print(f"   Files: {files}")
        return True
    else:
        print(f"❌ No model found at {model_path}")
        print("   Run: python models/train_lora.py")
        return False

def test_classifier_service():
    """Test the classifier service directly"""
    classifier_url = "http://localhost:8001"
    
    print(f"\n=== Testing Classifier Service ===")
    
    # Test health endpoint
    try:
        response = requests.get(f"{classifier_url}/health", timeout=5)
        health_data = response.json()
        print(f"Classifier health: {health_data}")
        
        if not health_data.get("model_loaded", False):
            print("❌ Model not loaded in classifier service")
            return False
            
    except Exception as e:
        print(f"❌ Cannot connect to classifier service: {e}")
        return False
    
    # Test prediction endpoint
    try:
        test_message = {
            "id": str(uuid4()),
            "dialog_id": str(uuid4()),
            "participant_index": 0,
            "text": "Hello, this is a test message for bot classification!"
        }
        
        response = requests.post(
            f"{classifier_url}/predict",
            json=test_message,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful: {result['is_bot_probability']:.4f}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction request failed: {e}")
        return False

def test_orchestrator_service():
    """Test the orchestrator service"""
    orchestrator_url = "http://localhost:8000"
    
    print(f"\n=== Testing Orchestrator Service ===")
    
    # Test health endpoint
    try:
        response = requests.get(f"{orchestrator_url}/health", timeout=5)
        health_data = response.json()
        print(f"Orchestrator health: {health_data}")
        
    except Exception as e:
        print(f"❌ Cannot connect to orchestrator service: {e}")
        return False
    
    # Test prediction through orchestrator
    try:
        test_message = {
            "id": str(uuid4()),
            "dialog_id": str(uuid4()),
            "participant_index": 0,
            "text": "This is a test message through the orchestrator!"
        }
        
        response = requests.post(
            f"{orchestrator_url}/predict",
            json=test_message,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Orchestrator prediction successful: {result['is_bot_probability']:.4f}")
            return True
        else:
            print(f"❌ Orchestrator prediction failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Orchestrator prediction request failed: {e}")
        return False

def test_web_interface():
    """Test the web interface"""
    web_url = "http://localhost:8501"
    
    print(f"\n=== Testing Web Interface ===")
    
    try:
        response = requests.get(web_url, timeout=5)
        if response.status_code == 200:
            print(f"✅ Web interface accessible at {web_url}")
            return True
        else:
            print(f"❌ Web interface error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Cannot connect to web interface: {e}")
        return False

def main():
    print("=== Complete Flow Test ===")
    print("Testing model training -> deployment -> inference flow")
    
    # Check if model exists
    model_exists = check_model_exists()
    
    if not model_exists:
        print("\n❌ No trained model found!")
        print("Please run: python models/train_lora.py")
        return
    
    print("\n⏳ Waiting for services to start...")
    time.sleep(2)
    
    # Test each service
    classifier_ok = test_classifier_service()
    orchestrator_ok = test_orchestrator_service()
    web_ok = test_web_interface()
    
    print(f"\n=== Flow Test Results ===")
    print(f"Model exists: {'✅' if model_exists else '❌'}")
    print(f"Classifier service: {'✅' if classifier_ok else '❌'}")
    print(f"Orchestrator service: {'✅' if orchestrator_ok else '❌'}")
    print(f"Web interface: {'✅' if web_ok else '❌'}")
    
    if all([model_exists, classifier_ok, orchestrator_ok, web_ok]):
        print("\n🎉 Complete flow working successfully!")
        print("You can now use the web interface for bot classification!")
    else:
        print("\n❌ Some components are not working properly.")
        print("Check the service logs for more details.")

if __name__ == "__main__":
    main()