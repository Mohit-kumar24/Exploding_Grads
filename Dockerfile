# KingdomForge Dockerfile
# Base image: standard lightweight Python 3.12
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APPEND_FILE=inputs/append.txt \
    MODE=full \
    TASK_ID=task_3 \
    SEED=42

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries=5 -r requirements.txt

# Copy project code
COPY . .

# Run harness
CMD ["python", "-m", "src.harness.runner"]
