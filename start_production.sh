#!/bin/bash
# =============================================================================
# Medical AI System - Production Startup Script
# =============================================================================
# Usage: ./start_production.sh
# This script starts the Flask application using Gunicorn WSGI server

echo "🚀 Starting Medical AI System in Production Mode..."
echo "=================================================="

# Activate virtual environment
source venv/bin/activate

# Set environment variables for production
export FLASK_ENV=production
export FLASK_DEBUG=False

# Start Gunicorn with optimal settings
echo "📦 Starting Gunicorn server..."
gunicorn \
    --workers 4 \
    --worker-class sync \
    --bind 0.0.0.0:5000 \
    --timeout 120 \
    --keep-alive 5 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log \
    --log-level info \
    --capture-output \
    --enable-stdio-inheritance \
    main:app

echo "✅ Server started successfully!"
echo "🌐 Access at: http://34.170.30.162:5000"
