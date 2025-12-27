# Multi-stage build for smaller image
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ADD after line 12 (after apt-get install):

# Install Google Cloud SDK for GCS access
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && \
    apt-get install -y google-cloud-sdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=$PATH:/usr/local/gcloud/google-cloud-sdk/bin

# Copy only requirements first for better caching
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Final stage
FROM python:3.10-slim

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Set Python path
ENV PYTHONPATH=/app:/app/src
ENV PYTHONUNBUFFERED=1

# Create necessary directories
RUN mkdir -p /app/logs /app/checkpoints /app/certs /app/datasets /app/data

# Copy application structure correctly
COPY pyproject.toml /app/
COPY src/flower_benchmarks/ /app/src/flower_benchmarks/
COPY yolov5/ /app/yolov5/
COPY download_coco_n_images.py /app/

# Install the Flower app in editable mode
RUN pip install --no-cache-dir -e .

# Optimize by removing unnecessary files
RUN find /app -type f -name "*.pyc" -delete && \
    find /app -type d -name "__pycache__" -delete && \
    find /app -type f -name "*.md" -delete && \
    find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Expose ports for Flower components and Prometheus metrics
EXPOSE 9091 9092 9093

# Health check (will be overridden by docker-compose)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import os; exit(0 if os.path.exists('/app/.healthy') else 1)" || exit 1

# Default command (will be overridden by docker-compose)
CMD ["flower-superlink", "--insecure"]