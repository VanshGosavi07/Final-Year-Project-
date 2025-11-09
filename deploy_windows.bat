@echo off
REM =============================================================================
REM Deploy Medical AI System to Google Cloud Run from Windows
REM =============================================================================

echo 🚀 Deploying Medical AI System to Google Cloud Run...
echo ======================================================

set PROJECT_ID=sublime-seat-451203-j4
set SERVICE_NAME=medical-ai-system
set REGION=asia-southeast1

gcloud config set project %PROJECT_ID%

echo 📦 Enabling required APIs...
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

echo 🔨 Building and deploying...
gcloud run deploy %SERVICE_NAME% ^
    --source . ^
    --platform managed ^
    --region %REGION% ^
    --allow-unauthenticated ^
    --memory 4Gi ^
    --cpu 2 ^
    --timeout 300 ^
    --min-instances 0 ^
    --max-instances 10 ^
    --port 8080

echo.
echo ✅ Deployment complete!
echo 🌐 Access your HTTPS app at the URL shown above
pause
