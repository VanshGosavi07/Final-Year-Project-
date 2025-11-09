# Medical AI System - Production Dockerfile for Google Cloud Run
# =============================================================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt gunicorn && \
    rm -rf ~/.cache/pip

# Copy only necessary application files (excluding large model files initially)
COPY main.py .
COPY breast_cancer_predictor.py .
COPY lung_cancer_predictor.py .
COPY form_validators.py .

# Copy templates and static files
COPY templates/ ./templates/
COPY static/ ./static/

# Copy RAG Data
COPY "RAG Data/" ./RAG\ Data/

# Copy ML Models (these are large files)
COPY Modal/ ./Modal/

# Create necessary directories
RUN mkdir -p logs static/uploads instance

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV FLASK_DEBUG=False
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8080

# Run with Gunicorn - optimized for Cloud Run
CMD exec gunicorn --bind :$PORT --workers 2 --threads 4 --timeout 300 --worker-class gthread --worker-tmp-dir /dev/shm main:app
