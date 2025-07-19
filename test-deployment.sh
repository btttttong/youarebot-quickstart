#!/bin/bash

# Test deployment readiness for YouAreBot

set -e

echo "🧪 Testing deployment readiness..."

# Check if required files exist
echo "📁 Checking required files..."
required_files=(
    "fly.toml"
    "fly-classifier.toml" 
    "fly-web.toml"
    "fly-mlflow.toml"
    "Dockerfile.orchestrator"
    "Dockerfile.classifier"
    "Dockerfile.streamlit"
    "Dockerfile.mlflow"
    "requirements.txt"
    "app/"
    "models/bot_classifier/"
)

for file in "${required_files[@]}"; do
    if [ ! -e "$file" ]; then
        echo "❌ Missing: $file"
        exit 1
    else
        echo "✅ Found: $file"
    fi
done

# Check if trained model exists
if [ ! -f "models/bot_classifier/config.json" ]; then
    echo "❌ Trained model not found in models/bot_classifier/"
    echo "   Please run the training script first"
    exit 1
else
    echo "✅ Trained model found"
fi

# Test Docker builds locally (optional)
if command -v docker &> /dev/null; then
    echo "🐳 Testing Docker builds..."
    
    # Test classifier build
    if docker build -f Dockerfile.classifier -t test-classifier . >/dev/null 2>&1; then
        echo "✅ Classifier Docker build successful"
        docker rmi test-classifier >/dev/null 2>&1
    else
        echo "❌ Classifier Docker build failed"
        exit 1
    fi
    
    # Test orchestrator build  
    if docker build -f Dockerfile.orchestrator -t test-orchestrator . >/dev/null 2>&1; then
        echo "✅ Orchestrator Docker build successful"
        docker rmi test-orchestrator >/dev/null 2>&1
    else
        echo "❌ Orchestrator Docker build failed"
        exit 1
    fi
    
    # Test web build
    if docker build -f Dockerfile.streamlit -t test-web . >/dev/null 2>&1; then
        echo "✅ Web Docker build successful"
        docker rmi test-web >/dev/null 2>&1
    else
        echo "❌ Web Docker build failed"
        exit 1
    fi
    
    # Test MLflow build
    if docker build -f Dockerfile.mlflow -t test-mlflow . >/dev/null 2>&1; then
        echo "✅ MLflow Docker build successful"
        docker rmi test-mlflow >/dev/null 2>&1
    else
        echo "❌ MLflow Docker build failed"
        exit 1
    fi
else
    echo "⚠️  Docker not found - skipping build tests"
fi

# Check flyctl
if command -v fly &> /dev/null; then
    echo "✅ flyctl is installed"
    
    if fly auth whoami &> /dev/null; then
        echo "✅ Authenticated with Fly.io"
    else
        echo "⚠️  Not logged in to Fly.io - run 'fly auth login'"
    fi
else
    echo "❌ flyctl not installed - install from https://fly.io/docs/hands-on/install-flyctl/"
    exit 1
fi

echo ""
echo "🧠 Testing memory database..."
if python test_memory_db.py >/dev/null 2>&1; then
    echo "✅ Memory database test passed"
else
    echo "❌ Memory database test failed"
    exit 1
fi

echo ""
echo "🎉 Deployment readiness check complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Login to Fly.io: fly auth login"
echo "   2. Run deployment: ./deploy-to-fly.sh"
echo "   3. Or deploy manually using README-deployment.md"
echo ""
echo "💰 Cost savings:"
echo "   - No PostgreSQL database costs"
echo "   - Auto-scaling reduces compute costs"
echo "   - In-memory storage (data not persistent)"