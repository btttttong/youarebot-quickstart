#!/bin/bash

# Deploy supakavadee-r-harbour-ml-solution system to Fly.io
# This script deploys all services: database, MLflow, classifier, LLM, orchestrator, and web interface

set -e

echo "🚀 Deploying supakavadee-r-harbour-ml-solution to Fly.io"

# Check if flyctl is installed
if ! command -v fly &> /dev/null; then
    echo "❌ flyctl is not installed. Please install it first:"
    echo "   https://fly.io/docs/hands-on/install-flyctl/"
    exit 1
fi

# Check if user is logged in
if ! fly auth whoami &> /dev/null; then
    echo "❌ Not logged in to Fly.io. Please run 'fly auth login' first"
    exit 1
fi

echo "✅ flyctl is installed and authenticated"

# 1. Skip database setup - using in-memory storage
echo "🧠 Using in-memory storage (no database needed)"

# 2. Create volumes for persistent storage
echo "💾 Creating volumes..."
if ! fly volumes list -a supakavadee-r-mlflow | grep -q "supakavadee_mlflow"; then
    fly volumes create supakavadee_mlflow --region dfw --size 5 -a supakavadee-r-mlflow || true
fi

if ! fly volumes list -a supakavadee-r-llm | grep -q "supakavadee_models"; then
    fly volumes create supakavadee_models --region dfw --size 10 -a supakavadee-r-llm || true
fi

# 3. Deploy MLflow service
echo "📈 Deploying MLflow service..."
fly deploy --config fly-mlflow.toml --app supakavadee-r-mlflow

# 4. Deploy classifier service
echo "🤖 Deploying classifier service..."
fly deploy --config fly-classifier.toml --app supakavadee-r-classifier

# 5. Deploy LLM service
echo "🧠 Deploying LLM service..."
fly deploy --config fly-llm.toml --app supakavadee-r-llm

# 6. Set memory database environment variable
echo "🧠 Configuring in-memory database..."
fly secrets set USE_MEMORY_DB="true" -a supakavadee-r-harbour-ml-solution
fly secrets set USE_MEMORY_DB="true" -a supakavadee-r-web
echo "✅ In-memory database configured"

# 7. Set environment variables for services
echo "⚙️  Setting environment variables..."

# Get service URLs
MLFLOW_URL="https://supakavadee-r-mlflow.fly.dev"
CLASSIFIER_URL="https://supakavadee-r-classifier.fly.dev"
LLM_URL="https://supakavadee-r-llm.fly.dev"

# Set secrets for orchestrator
fly secrets set CLASSIFIER_URL="$CLASSIFIER_URL" -a supakavadee-r-harbour-ml-solution
fly secrets set LLM_URL="$LLM_URL" -a supakavadee-r-harbour-ml-solution
fly secrets set MLFLOW_TRACKING_URI="$MLFLOW_URL" -a supakavadee-r-classifier

# Set secrets for web app
fly secrets set ORCHESTRATOR_URL="https://supakavadee-r-harbour-ml-solution.fly.dev" -a supakavadee-r-web

# 8. Deploy orchestrator service
echo "🎯 Deploying orchestrator service..."
fly deploy --config fly-orchestrator.toml --app supakavadee-r-harbour-ml-solution

# 9. Deploy web interface
echo "🌐 Deploying web interface..."
fly deploy --config fly-web.toml --app supakavadee-r-web

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "📝 Service URLs:"
echo "   Web Interface: https://supakavadee-r-web.fly.dev"
echo "   API:          https://supakavadee-r-harbour-ml-solution.fly.dev"
echo "   Classifier:   https://supakavadee-r-classifier.fly.dev"
echo "   MLflow:       https://supakavadee-r-mlflow.fly.dev"
echo "   LLM:          https://supakavadee-r-llm.fly.dev"
echo ""
echo "🔧 Next steps:"
echo "   1. Check service health: fly status -a <app-name>"
echo "   2. View logs: fly logs -a <app-name>"
echo "   3. Monitor apps: fly dashboard"
echo ""
echo "⚠️  Notes:"
echo "   - Using in-memory storage: data will be lost on restart"
echo "   - The LLM service requires models to be uploaded to the volume"
echo "   - Use 'fly ssh console -a supakavadee-r-llm' to upload model files"