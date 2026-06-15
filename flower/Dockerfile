# Multi-stage build for smaller image
FROM python:3.10-slim AS builder

# Install build dependencies
# libgomp1 is kept for general numerical library OpenMP support
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Google Cloud SDK for GCS access
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl gnupg apt-transport-https ca-certificates && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends google-cloud-sdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and wheel, but pin setuptools <72.
# pkg_resources (needed by yolov5) was removed from setuptools in 72+.
# --force-reinstall ensures pip does not skip the install despite dist-info existing.
RUN pip install --no-cache-dir --upgrade pip wheel
RUN pip install --no-cache-dir --force-reinstall "setuptools>=65.0.0,<72.0.0"

# Copy only requirements first for better caching
COPY requirements.txt /tmp/

# Install Python dependencies with proper compiler flags for FLAML/XGBoost
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Final stage
FROM python:3.10-slim

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
# libgomp1 is kept for general numerical library OpenMP support
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

# Patch yolov5/utils/general.py: pkg_resources was removed in setuptools >=72.
# Replace the bare import with a try/except that falls back to `packaging`,
# which is always available (it is a pip hard dependency and already in the venv).
RUN sed -i 's/import pkg_resources as pkg/try:\n    import pkg_resources as pkg\nexcept ImportError:\n    from packaging import version as _pv\n    class pkg:\n        parse_version = staticmethod(_pv.parse)/' /app/yolov5/utils/general.py

RUN mkdir -p /root/.flwr
COPY flwr-config.toml /root/.flwr/config.toml

# REMOVED: COPY download_coco_n_images.py /app/
# (Dataset download handled by GCS bucket + deployment script)

# Install the Flower app in editable mode
RUN pip install --no-cache-dir -e .

# Optimize by removing unnecessary files
RUN find /app -type f -name "*.pyc" -delete && \
    find /app -type d -name "__pycache__" -delete && \
    find /app -type f -name "*.md" -delete && \
    find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Re-pin setuptools <72 AFTER pip install -e . and cleanup.
# hatchling (the build backend) or pip's resolver can silently upgrade setuptools
# during editable install; this step re-locks the version so it survives.
RUN pip install --no-cache-dir --force-reinstall "setuptools>=65.0.0,<72.0.0"

# Expose ports for Flower components
EXPOSE 9091 9092 9093

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import os; exit(0 if os.path.exists('/app/.healthy') else 1)" || exit 1

# Default command (will be overridden by docker-compose)
CMD ["flower-superlink", "--insecure"]