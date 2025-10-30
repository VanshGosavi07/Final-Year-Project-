# Google Cloud Deployment Guide - Medical AI System

## Prerequisites
1. Google Cloud account with billing enabled
2. Git installed locally
3. Your project pushed to GitHub with Git LFS

## Step 1: Install Google Cloud CLI

### Windows
Download and install from: https://cloud.google.com/sdk/docs/install

Or use PowerShell:
```powershell
(New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")
& $env:Temp\GoogleCloudSDKInstaller.exe
```

After installation, restart your terminal.

## Step 2: Initialize Google Cloud

```bash
# Login to Google Cloud
gcloud auth login

# Set your project ID (replace with your project name)
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable sqladmin.googleapis.com
```

## Step 3: Create Google Cloud Project (via Web Console)

1. Go to: https://console.cloud.google.com/
2. Click "Select a project" → "NEW PROJECT"
3. Enter project name: `medical-ai-system`
4. Click "CREATE"
5. Note your PROJECT_ID (e.g., `medical-ai-system-123456`)

## Step 4: Set Environment Variables in Google Cloud

Create a `.env.yaml` file with your secrets (DO NOT commit this file):

```yaml
# .env.yaml
SECRET_KEY: "your_production_secret_key_here"
SQLALCHEMY_DATABASE_URI: "your_neon_postgresql_connection_string"
CLOUDINARY_CLOUD_NAME: "your_cloudinary_cloud_name"
CLOUDINARY_API_KEY: "your_cloudinary_api_key"
CLOUDINARY_API_SECRET: "your_cloudinary_api_secret"
GROQ_API_KEY: "your_groq_api_key"
FLASK_ENV: "production"
FLASK_DEBUG: "False"
```

## Step 5: Deploy to Google Cloud Run (Recommended)

### Option A: Deploy from GitHub Repository

```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Deploy from GitHub (this will clone and build automatically)
gcloud run deploy medical-ai-system \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars-file .env.yaml \
  --max-instances 10 \
  --min-instances 1

# Or deploy using Dockerfile
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/medical-ai-system
gcloud run deploy medical-ai-system \
  --image gcr.io/YOUR_PROJECT_ID/medical-ai-system \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars-file .env.yaml \
  --max-instances 10 \
  --min-instances 1
```

### Option B: Deploy to App Engine

```bash
# Deploy to App Engine
gcloud app deploy app.yaml --project YOUR_PROJECT_ID

# Set environment variables
gcloud app deploy --set-env-vars-file=.env.yaml
```

## Step 6: Configure Git LFS for Cloud Build

Since you're using Git LFS for .keras files, add this to your deployment:

```bash
# Enable Git LFS in Cloud Build
gcloud builds submit --config=cloudbuild.yaml
```

Create `cloudbuild.yaml`:
```yaml
steps:
  - name: 'gcr.io/cloud-builders/git'
    args: ['lfs', 'install']
  
  - name: 'gcr.io/cloud-builders/git'
    args: ['clone', 'https://github.com/VanshGosavi07/Final-Year-Project-.git', '.']
  
  - name: 'gcr.io/cloud-builders/git'
    args: ['lfs', 'pull']
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/medical-ai-system', '.']
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/medical-ai-system']

images:
  - 'gcr.io/$PROJECT_ID/medical-ai-system'

timeout: '1200s'
```

## Step 7: Monitor Your Deployment

```bash
# View logs
gcloud run services logs read medical-ai-system --region=us-central1

# Get service URL
gcloud run services describe medical-ai-system --region=us-central1 --format='value(status.url)'
```

## Step 8: Set Up Domain (Optional)

```bash
# Map custom domain
gcloud run domain-mappings create --service medical-ai-system --domain yourdomain.com --region us-central1
```

## Quick Commands Summary

```bash
# 1. Install gcloud CLI
# 2. Login
gcloud auth login

# 3. Create project (via console or):
gcloud projects create medical-ai-system-123456

# 4. Set project
gcloud config set project medical-ai-system-123456

# 5. Enable APIs
gcloud services enable cloudbuild.googleapis.com run.googleapis.com

# 6. Deploy
gcloud run deploy medical-ai-system \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --min-instances 1 \
  --max-instances 10

# 7. Get URL
gcloud run services describe medical-ai-system --region=us-central1 --format='value(status.url)'
```

## Important Notes

1. **Git LFS Files**: Your .keras model files will be pulled via Git LFS during build
2. **Memory**: Set to 4GB minimum for AI models (--memory 4Gi)
3. **Timeout**: Set to 300s for model loading (--timeout 300)
4. **Cost**: Cloud Run charges based on usage; keep min-instances=1 to reduce cold starts
5. **Database**: Your Neon PostgreSQL will work as-is (already cloud-based)
6. **Cloudinary**: Your image storage will work as-is (already cloud-based)

## Troubleshooting

If deployment fails:
```bash
# Check build logs
gcloud builds log --region=us-central1

# Check service logs
gcloud run services logs read medical-ai-system --region=us-central1 --limit=50

# Describe service
gcloud run services describe medical-ai-system --region=us-central1
```

## Expected Deployment Time
- First deployment: 5-10 minutes (includes Docker build + model downloads via Git LFS)
- Subsequent deployments: 3-5 minutes

## Post-Deployment Checklist
- [ ] Verify service URL is accessible
- [ ] Test user registration/login
- [ ] Test image upload and diagnosis
- [ ] Test chat functionality
- [ ] Monitor logs for errors
- [ ] Check database connections
- [ ] Verify Cloudinary uploads work
