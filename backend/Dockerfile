FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime libs needed by Pillow/TensorFlow image handling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY backend /app/backend
COPY model /app/model

ENV ECOSORT_MODEL_PATH=/app/model/ecosort.keras \
    ECOSORT_LABELS_PATH=/app/model/labels.json \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    # Disable AVX-512 oneDNN kernels — fixes SIGILL on CPUs without AVX-512
    TF_ENABLE_ONEDNN_OPTS=0 \
    # Suppress TF startup noise in logs
    TF_CPP_MIN_LOG_LEVEL=2

RUN mkdir -p /app/.cache/matplotlib

WORKDIR /app/backend
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
