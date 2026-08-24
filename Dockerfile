# Multi-stage Dockerfile for Industrial Prescriptive Maintenance Application
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Run data ingestion and model training during build
RUN python train_pipeline.py

# Expose Streamlit and FastAPI ports
EXPOSE 8501
EXPOSE 8000

ENV PORT=8501
ENV API_PORT=8000

# Default command: launch Streamlit interface
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
