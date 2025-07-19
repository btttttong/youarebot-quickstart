#!/bin/bash

# Test readiness for simple single-service deployment

set -e

echo "🧪 Testing Simple Deployment Readiness..."

# Check required files
echo "📁 Checking required files..."
required_files=(
    "fly.toml"
    "Dockerfile"
    "simple_main.py"
    "requirements.txt"
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

# Check fly.toml configuration
echo "⚙️  Checking fly.toml configuration..."
if grep -q 'dockerfile = "Dockerfile"' fly.toml; then
    echo "✅ fly.toml points to correct Dockerfile"
else
    echo "❌ fly.toml does not point to Dockerfile"
    exit 1
fi

if grep -q 'app = "supakavadee-r-harbour-ml-solution"' fly.toml; then
    echo "✅ App name is correct"
else
    echo "❌ App name is incorrect"
    exit 1
fi

# Check simple_main.py
echo "🤖 Checking simple_main.py..."
if grep -q '/predict' simple_main.py; then
    echo "✅ /predict endpoint found"
else
    echo "❌ /predict endpoint not found"
    exit 1
fi

if grep -q 'class TextInput' simple_main.py; then
    echo "✅ TextInput model found"
else
    echo "❌ TextInput model not found"
    exit 1
fi

# Check if flyctl is available
echo "🛠️  Checking flyctl..."
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
echo "🎉 Simple deployment readiness check complete!"
echo ""
echo "📋 To deploy:"
echo "   1. fly deploy"
echo "   2. Test: curl -X POST https://supakavadee-r-harbour-ml-solution.fly.dev/predict -H 'Content-Type: application/json' -d '{\"text\":\"Hello\"}'"
echo "   3. Register at: https://youare.bot"
echo ""
echo "🌐 Your endpoint: https://supakavadee-r-harbour-ml-solution.fly.dev/predict"