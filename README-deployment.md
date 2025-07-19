# YouAreBot Fly.io Deployment Guide

This guide walks you through deploying the YouAreBot classification system to Fly.io.

## Prerequisites

1. Install [flyctl](https://fly.io/docs/hands-on/install-flyctl/)
2. Sign up for a [Fly.io account](https://fly.io/app/sign-up)
3. Login: `fly auth login`

## Deployment Architecture

The system deploys as multiple Fly.io applications:

- **supakavadee-r-harbour-ml-solution**: Main orchestrator API (entry point)
- **supakavadee-r-classifier**: Bot classification service  
- **supakavadee-r-web**: Streamlit web interface
- **supakavadee-r-mlflow**: MLflow model tracking
- **supakavadee-r-llm**: LLM service (Llama.cpp)

**Note**: This deployment uses **in-memory storage** instead of PostgreSQL to save costs. Data will be lost when services restart.

## Step-by-Step Deployment

### 1. Deploy Classifier Service

```bash
# Deploy classifier (includes trained models)
fly deploy --config fly-classifier.toml --app supakavadee-r-classifier
```

### 2. Deploy MLflow Service

```bash
# Create volume for MLflow data
fly volumes create supakavadee_mlflow --region dfw --size 5 -a supakavadee-r-mlflow

# Deploy MLflow
fly deploy --config fly-mlflow.toml --app supakavadee-r-mlflow
```

### 3. Deploy LLM Service (Optional)

```bash
# Create volume for models
fly volumes create supakavadee_models --region dfw --size 10 -a supakavadee-r-llm

# Deploy LLM service
fly deploy --config fly-llm.toml --app supakavadee-r-llm

# Upload model file to volume
fly ssh console -a supakavadee-r-llm
# Then upload your qwen2.5-0.5b-instruct-q4_k_m.gguf file to /models/
```

### 4. Configure Environment Variables

```bash
# Configure in-memory database
fly secrets set USE_MEMORY_DB="true" -a supakavadee-r-harbour-ml-solution
fly secrets set USE_MEMORY_DB="true" -a supakavadee-r-web

# Set service URLs
fly secrets set CLASSIFIER_URL="https://supakavadee-r-classifier.fly.dev" -a supakavadee-r-harbour-ml-solution
fly secrets set LLM_URL="https://supakavadee-r-llm.fly.dev" -a supakavadee-r-harbour-ml-solution
fly secrets set MLFLOW_TRACKING_URI="https://supakavadee-r-mlflow.fly.dev" -a supakavadee-r-classifier
```

### 5. Deploy Main Orchestrator

```bash
# Deploy main API
fly deploy --app supakavadee-r-harbour-ml-solution
```

### 6. Deploy Web Interface

```bash
# Set orchestrator URL for web app
fly secrets set ORCHESTRATOR_URL="https://supakavadee-r-harbour-ml-solution.fly.dev" -a supakavadee-r-web

# Deploy web interface
fly deploy --config fly-web.toml --app supakavadee-r-web
```

## Quick Deployment Script

Alternatively, use the automated deployment script:

```bash
./deploy-to-fly.sh
```

## Service URLs

After deployment, your services will be available at:

- **Web Interface**: https://supakavadee-r-web.fly.dev
- **Main API**: https://supakavadee-r-harbour-ml-solution.fly.dev
- **Classifier API**: https://supakavadee-r-classifier.fly.dev
- **MLflow UI**: https://supakavadee-r-mlflow.fly.dev
- **LLM API**: https://supakavadee-r-llm.fly.dev

## Monitoring and Troubleshooting

```bash
# Check application status
fly status -a supakavadee-r-harbour-ml-solution

# View logs
fly logs -a supakavadee-r-harbour-ml-solution

# SSH into application
fly ssh console -a supakavadee-r-harbour-ml-solution

# Scale application
fly scale count 2 -a supakavadee-r-harbour-ml-solution

# Update secrets
fly secrets set KEY=value -a supakavadee-r-harbour-ml-solution
```

## Cost Optimization

- **No database costs** - uses in-memory storage instead of PostgreSQL
- Services use `auto_stop_machines = true` to reduce costs
- Start with minimal machine sizes (can be scaled up later)
- LLM service is optional and can be omitted to reduce costs

## Notes

1. **In-memory storage**: Data is not persistent and will be lost on service restart
2. The classifier service includes the trained models in the Docker image
3. MLflow uses SQLite backend for simplicity
4. The system gracefully falls back to local models if MLflow is unavailable
5. For production with persistent data, consider upgrading to PostgreSQL