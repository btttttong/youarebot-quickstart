import os

# Database configuration
USE_MEMORY_DB = os.getenv("USE_MEMORY_DB", "false").lower() == "true"

# Database connection parameters from env vars (or defaults)
DB_USER = os.getenv("DB_USER", "student")
DB_PASSWORD = os.getenv("DB_PASSWORD", "student_pass")
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "chat_db")

DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Database type selection
if USE_MEMORY_DB:
    print("🧠 Using in-memory database (no persistence)")
else:
    print("🐘 Using PostgreSQL database")

# Model config (for classification model, not LLM)
MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
CANDIDATE_LABELS = ["bot", "human"]