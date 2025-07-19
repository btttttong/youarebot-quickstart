FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install basic Python dependencies first
RUN pip install --no-cache-dir fastapi==0.116.1 uvicorn==0.35.0 pydantic==2.11.7

# Install CPU-only PyTorch for faster builds and smaller image
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2+cpu

# Install transformers and other ML dependencies
RUN pip install --no-cache-dir transformers==4.41.2

# Copy requirements and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trained model
COPY models/bot_classifier/ /models/bot_classifier/

# Copy the application
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV MODEL_PATH="/models/bot_classifier"
ENV USE_MEMORY_DB="true"

# Run the bot classifier service
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]