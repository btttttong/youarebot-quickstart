#!/usr/bin/env python3
"""
Test script to verify the microservices setup
"""

import requests
import json
import time
from uuid import uuid4

# Service URLs
ORCHESTRATOR_URL = "http://localhost:8000"
CLASSIFIER_URL = "http://localhost:8001"
LLM_URL = "http://localhost:11434"

def test_health_endpoints():
    """Test health endpoints of all services"""
    print("Testing health endpoints...")
    
    services = [
        ("Orchestrator", f"{ORCHESTRATOR_URL}/health"),
        ("Classifier", f"{CLASSIFIER_URL}/health"),
        ("LLM", f"{LLM_URL}/health")
    ]
    
    for name, url in services:
        try:
            response = requests.get(url, timeout=5)
            print(f"✓ {name} health check: {response.status_code} - {response.json()}")
        except Exception as e:
            print(f"✗ {name} health check failed: {e}")

def test_predict_endpoint():
    """Test the /predict endpoint"""
    print("\nTesting /predict endpoint...")
    
    test_message = {
        "id": str(uuid4()),
        "dialog_id": str(uuid4()),
        "participant_index": 0,
        "text": "Hello, this is a test message to check if the prediction works!"
    }
    
    try:
        response = requests.post(
            f"{ORCHESTRATOR_URL}/predict",
            json=test_message,
            headers={"Content-Type": "application/json"}
        )
        print(f"✓ /predict endpoint: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"  Bot probability: {result['is_bot_probability']}")
            print(f"  Full response: {result}")
        else:
            print(f"  Error: {response.text}")
    except Exception as e:
        print(f"✗ /predict endpoint failed: {e}")

def test_get_message_endpoint():
    """Test the /get_message endpoint"""
    print("\nTesting /get_message endpoint...")
    
    test_request = {
        "dialog_id": str(uuid4()),
        "last_message_id": str(uuid4()),
        "last_msg_text": "What is the weather like today?"
    }
    
    try:
        response = requests.post(
            f"{ORCHESTRATOR_URL}/get_message",
            json=test_request,
            headers={"Content-Type": "application/json"}
        )
        print(f"✓ /get_message endpoint: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"  LLM response: {result['new_msg_text']}")
        else:
            print(f"  Error: {response.text}")
    except Exception as e:
        print(f"✗ /get_message endpoint failed: {e}")

if __name__ == "__main__":
    print("=== Microservices Setup Test ===")
    print("Make sure to run 'docker compose up --build' first!")
    print()
    
    # Wait a bit for services to start
    print("Waiting 5 seconds for services to initialize...")
    time.sleep(5)
    
    test_health_endpoints()
    test_predict_endpoint()
    test_get_message_endpoint()
    
    print("\n=== Test Complete ===")
    print("If all tests pass, your microservices setup is working correctly!")