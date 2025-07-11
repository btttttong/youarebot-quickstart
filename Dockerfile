FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN ls -alR /app

CMD ["bash", "-c", "PYTHONPATH=/app uvicorn app.api.main:app --host 0.0.0.0 --port 8000 & PYTHONPATH=/app streamlit run web/streamlit_app.py --server.port=8501 --server.address=0.0.0.0 && wait"]
