#!/bin/bash
# Quick test script for Google Cloud deployment

echo "🧪 Testing application startup..."
echo "================================"

# Run app with timeout to test startup only
timeout 15s python main.py &
PID=$!

# Wait a bit for startup
sleep 10

# Check if process is still running
if kill -0 $PID 2>/dev/null; then
    echo "✓ Application started successfully!"
    echo "  Process is running (PID: $PID)"
    kill $PID 2>/dev/null
    echo ""
    echo "✅ Startup test PASSED"
    echo ""
    echo "🚀 Ready to deploy!"
    echo "Run: gcloud app deploy"
    exit 0
else
    echo "❌ Application failed to start"
    echo ""
    echo "Check the error messages above"
    exit 1
fi
