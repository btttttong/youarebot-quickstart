# Simple Single-Service Deployment

## ✅ Ready for Single fly.toml Deployment

Your simplified bot classifier is ready for deployment with just one `fly.toml` file.

### 🏗️ What Was Created

**Single Service Architecture:**
- `simple_main.py` - Simple FastAPI service with `/predict` endpoint
- `Dockerfile` - Single container with trained model included
- `fly.toml` - Single service deployment configuration

**API Interface:**
```bash
# Request format (exactly as required)
curl -X POST https://<app>.fly.dev/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello"}'

# Response format
{
  "is_bot_probability": 0.5,
  "text": "Hello"
}
```

### 🚀 Deploy to Fly.io

```bash
# Deploy your app
fly deploy

# Check status
fly status

# View logs
fly logs

# Get your URL
echo "Your app URL: https://supakavadee-r-harbour-ml-solution.fly.dev"
```

### 🧪 Test the Deployed Service

```bash
# Test with simple text
curl -X POST https://supakavadee-r-harbour-ml-solution.fly.dev/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello"}'

# Test with bot-like text
curl -X POST https://supakavadee-r-harbour-ml-solution.fly.dev/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"CLICK HERE FOR DEALS!"}'

# Health check
curl https://supakavadee-r-harbour-ml-solution.fly.dev/health
```

### 📋 Register on youare.bot

1. Go to https://youare.bot
2. Register your endpoint: `https://supakavadee-r-harbour-ml-solution.fly.dev/predict`
3. Test with their interface

### 🎯 Key Features

- ✅ **Single service** - No complex multi-service setup
- ✅ **Correct API** - Accepts `{"text":"Hello"}` as required
- ✅ **Trained model** - Includes your fine-tuned model in the container
- ✅ **Fallback logic** - Works even if model loading fails
- ✅ **Cost optimized** - Auto-scaling, in-memory storage
- ✅ **Health checks** - Built-in monitoring

### 🔧 Configuration

**App Name:** `supakavadee-r-harbour-ml-solution`
**Region:** `dfw` (Dallas)
**Resources:** 2 CPU, 2GB RAM (sufficient for model)
**URL:** https://supakavadee-r-harbour-ml-solution.fly.dev

All set for deployment! 🎉