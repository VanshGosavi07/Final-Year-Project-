#!/bin/bash
# Deploy Medical AI System to Google Cloud Run with Secrets
# =============================================================================

set -e

echo "🚀 Deploying Medical AI System to Google Cloud Run..."

# Configuration
SERVICE_NAME="medical-ai-system"
REGION="asia-southeast1"
PROJECT_ID=$(gcloud config get-value project)

echo "📦 Service: $SERVICE_NAME"
echo "🌏 Region: $REGION"
echo "🔑 Project: $PROJECT_ID"
echo ""

# Build and deploy with secrets from Secret Manager
gcloud run deploy $SERVICE_NAME \
    --source . \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --timeout 300 \
    --port 8080 \
    --set-env-vars FLASK_ENV=production,FLASK_DEBUG=False,PYTHONUNBUFFERED=1 \
    --set-secrets="SECRET_KEY=SECRET_KEY:latest,\
SQLALCHEMY_DATABASE_URI=SQLALCHEMY_DATABASE_URI:latest,\
CLOUDINARY_CLOUD_NAME=CLOUDINARY_CLOUD_NAME:latest,\
CLOUDINARY_API_KEY=CLOUDINARY_API_KEY:latest,\
CLOUDINARY_API_SECRET=CLOUDINARY_API_SECRET:latest,\
GROQ_API_KEY=GROQ_API_KEY:latest,\
BREAST_CANCER_MODEL_PATH=BREAST_CANCER_MODEL_PATH:latest,\
LUNG_CANCER_MODEL_PATH=LUNG_CANCER_MODEL_PATH:latest" \
    --max-instances 10 \
    --min-instances 0 \
    --concurrency 80

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Getting service URL..."
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    🎉 DEPLOYMENT SUCCESSFUL! 🎉                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "🔗 Your Medical AI System is now live at:"
echo "   $SERVICE_URL"
echo ""
echo "🔒 HTTPS: ✅ Enabled (automatic)"
echo "📊 Status: Check at $SERVICE_URL"
echo ""
echo "🧪 Test your deployment:"
echo "   curl $SERVICE_URL"
echo ""
