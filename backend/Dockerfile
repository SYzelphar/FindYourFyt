# FindYourFit API. Build from the repo root:  docker compose build
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn==23.0.0
COPY backend/app.py ./
COPY backend/recsys ./recsys

# Model bundle, images and the event log are mounted as volumes (see docker-compose.yml).
ENV FYF_MODELS_DIR=/app/models \
    FYF_IMAGES_DIR=/data/images \
    FYF_EVENTS_DIR=/data/events \
    HF_HOME=/hf-cache \
    PYTHONUNBUFFERED=1

EXPOSE 5000
# One worker: sessions live in process memory. Threads handle concurrency.
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--timeout", "120", "--bind", "0.0.0.0:5000", "app:build_app()"]
