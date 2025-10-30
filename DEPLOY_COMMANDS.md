# Google Cloud App Engine Deployment (WITHOUT Docker)

## Step 1: Install Google Cloud CLI (if not installed)

Download from: https://cloud.google.com/sdk/docs/install

## Step 2: Create Project on Google Cloud Console

1. Go to: https://console.cloud.google.com/
2. Click "Select a project" → "NEW PROJECT"
3. Enter project name: `medical-ai-system`
4. Click "CREATE"
5. Note your PROJECT_ID (e.g., `medical-ai-system-123456`)

## Step 3: Initial Setup

```cmd
:: Login to Google Cloud
gcloud auth login

:: Set your project (replace YOUR_PROJECT_ID with your actual project ID)
gcloud config set project YOUR_PROJECT_ID

:: Enable required services for App Engine
gcloud services enable appengine.googleapis.com
gcloud services enable cloudbuild.googleapis.com

:: Initialize App Engine (choose region when prompted - e.g., us-central)
gcloud app create --region=us-central
```

## Step 4: Clone Your Repository on Local Machine

```cmd
:: Navigate to where you want to clone
cd D:\Projects

:: Clone your repository
git clone https://github.com/VanshGosavi07/Final-Year-Project-.git
cd Final-Year-Project-

:: Pull Git LFS files (.keras models)
git lfs pull
```

## Step 5: Update .env.yaml with Your Credentials

Edit `.env.yaml` file with your actual values:
- SECRET_KEY (generate a strong key for production)
- GROQ_API_KEY (your actual Groq API key)
- CLOUDINARY credentials (if not already correct)
- Update any other placeholder values

**Important**: `.env.yaml` is in `.gitignore` and won't be committed to GitHub.

## Step 6: Deploy to App Engine (WITHOUT Docker)

```cmd
:: Make sure you're in the project directory
cd "D:\Project\BTech Major Project"

:: Deploy with environment variables
gcloud app deploy app.yaml --set-env-vars-file=.env.yaml

:: When prompted:
:: - Confirm the service details (Y)
:: - Wait for deployment (5-10 minutes)
```

## Step 7: Get Your Live URL

```cmd
:: Open your app in browser
gcloud app browse

:: Or get the URL
gcloud app describe --format="value(defaultHostname)"
```

## Step 8: View Logs

```cmd
:: View real-time logs
gcloud app logs tail

:: View recent logs
gcloud app logs read --limit=50
```

## 🚀 QUICK DEPLOYMENT COMMANDS (Copy-Paste Ready)

```cmd
:: Step 1: Login
gcloud auth login

:: Step 2: Set project (replace YOUR_PROJECT_ID with your actual project ID)
gcloud config set project YOUR_PROJECT_ID

:: Step 3: Enable App Engine
gcloud services enable appengine.googleapis.com cloudbuild.googleapis.com

:: Step 4: Create App Engine app (choose region: us-central)
gcloud app create --region=us-central

:: Step 5: Navigate to your project
cd "D:\Project\BTech Major Project"

:: Step 6: Pull Git LFS files (models)
git lfs pull

:: Step 7: Deploy to App Engine (WITH environment variables)
gcloud app deploy app.yaml --set-env-vars-file=.env.yaml

:: Step 8: Open your app
gcloud app browse
```

## Important Notes

1. **Git LFS Models**: The `.keras` files will be automatically pulled via Git LFS during Cloud Build
2. **First Deployment**: May take 10-15 minutes (includes building Docker image, pulling LFS files)
3. **Memory**: Set to 4GB for AI models
4. **Region**: Using us-central1 (change if needed)
5. **Cost**: Cloud Run charges per usage; min-instances=1 reduces cold starts

## Troubleshooting

```cmd
:: Check build status
gcloud builds list --limit=5

:: View build logs
gcloud builds log <BUILD_ID>

:: Check service status
gcloud run services describe medical-ai-system --region=us-central1

:: View recent logs
gcloud run services logs read medical-ai-system --region=us-central1 --limit=100
```
