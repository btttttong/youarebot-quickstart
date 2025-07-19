#!/usr/bin/env python3
"""Simple fix for MLflow by using a direct registration approach."""

import mlflow
from mlflow.tracking import MlflowClient
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5001")

def register_model_manually():
    """Register model using the client API directly."""
    try:
        client = MlflowClient()
        
        # Create registered model first
        try:
            registered_model = client.create_registered_model("bot_classifier")
            print(f"✅ Created registered model: bot_classifier")
        except Exception as e:
            if "already exists" in str(e):
                print("ℹ️  Registered model bot_classifier already exists")
            else:
                print(f"⚠️  Error creating registered model: {e}")
        
        # Check if we have any runs with models
        experiment = client.get_experiment_by_name("bot_classifier_training")
        if experiment:
            runs = client.search_runs([experiment.experiment_id])
            
            model_runs = []
            for run in runs:
                # Check if run has model artifact
                artifacts = client.list_artifacts(run.info.run_id)
                for artifact in artifacts:
                    if artifact.path == "model":
                        model_runs.append(run)
                        break
            
            if model_runs:
                # Use the latest run with model
                latest_run = max(model_runs, key=lambda x: x.info.start_time)
                print(f"Found model in run: {latest_run.info.run_id}")
                
                # Create model version
                try:
                    model_version = client.create_model_version(
                        name="bot_classifier",
                        source=f"runs:/{latest_run.info.run_id}/model",
                        run_id=latest_run.info.run_id
                    )
                    
                    print(f"✅ Created model version {model_version.version}")
                    
                    # Promote to Champion
                    client.transition_model_version_stage(
                        name="bot_classifier",
                        version=model_version.version,
                        stage="Champion"
                    )
                    print(f"✅ Promoted version {model_version.version} to Champion")
                    return True
                    
                except Exception as e:
                    print(f"❌ Failed to create model version: {e}")
                    return False
            else:
                print("⚠️  No runs with model artifacts found")
                return False
        else:
            print("⚠️  Experiment 'bot_classifier_training' not found")
            return False
            
    except Exception as e:
        print(f"❌ Model registration failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Manual Model Registration ===")
    
    if register_model_manually():
        print("\n✅ Model registration successful!")
    else:
        print("\n❌ Model registration failed!")