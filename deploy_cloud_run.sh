#!/bin/bash
# =============================================================================
# Deploy Medical AI System to Google Cloud Run
# =============================================================================

echo "🚀 Deploying Medical AI System to Google Cloud Run..."
echo "======================================================"

# Set your project ID
PROJECT_ID="sublime-seat-451203-j4"
SERVICE_NAME="medical-ai-system"
REGION="asia-southeast1"

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "📦 Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Build and deploy to Cloud Run
echo "🔨 Building and deploying application..."
gcloud run deploy $SERVICE_NAME \
    --source . \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --timeout 300 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars "FLASK_ENV=production,FLASK_DEBUG=False" \
    --set-secrets "SECRET_KEY=medical-ai-secret:latest,\
SQLALCHEMY_DATABASE_URI=medical-ai-db-uri:latest,\
CLOUDINARY_CLOUD_NAME=medical-ai-cloudinary-name:latest,\
CLOUDINARY_API_KEY=medical-ai-cloudinary-key:latest,\
CLOUDINARY_API_SECRET=medical-ai-cloudinary-secret:latest,\
GROQ_API_KEY=medical-ai-groq-key:latest"

echo ""
echo "✅ Deployment complete!"
echo "🌐 Your application is now available at the URL shown above"
echo "🔒 It automatically has HTTPS enabled!"
