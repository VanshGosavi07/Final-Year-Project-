# 🔐 Secure Deployment Instructions

## Overview
We've updated the deployment to use **Google Cloud Secret Manager** instead of committing sensitive `.env` files to GitHub. This is the secure, production-ready approach.

## Steps to Deploy

### 1️⃣ Pull Latest Code on Your VM

```bash
cd ~/flask_app/Final-Year-Project-
git pull origin main
```

### 2️⃣ Set Up Secrets (One-Time Only)

Make the scripts executable:
```bash
chmod +x setup_secrets.sh deploy_with_secrets.sh
```

Upload your secrets to Google Cloud Secret Manager:
```bash
./setup_secrets.sh
```

This will:
- Read your local `.env` file
- Create secrets in Google Cloud Secret Manager for each variable
- Grant Cloud Run permission to access these secrets
- Keep your credentials secure (never uploaded to GitHub)

### 3️⃣ Deploy to Cloud Run

```bash
./deploy_with_secrets.sh
```

This will:
- Build your Docker container from GitHub source
- Deploy to Cloud Run
- Automatically inject secrets as environment variables
- Generate your HTTPS URL

### 4️⃣ Get Your HTTPS URL

After deployment completes, you'll see:
```
🔗 Your Medical AI System is now live at:
   https://medical-ai-system-xxxxx-uc.a.run.app
```

## 🎯 Why This Approach?

✅ **Secure**: Credentials never stored in GitHub  
✅ **Best Practice**: Industry-standard secret management  
✅ **Auditable**: Track who accesses secrets  
✅ **Automatic HTTPS**: Cloud Run provides SSL certificates  
✅ **Scalable**: Secrets managed centrally  

## 🔍 Verify Secrets

Check your secrets in Cloud Console:
```bash
gcloud secrets list
```

View a specific secret version:
```bash
gcloud secrets versions access latest --secret="SECRET_KEY"
```

## 📝 Notes

- Your `.env` file stays **only** on your VM
- Secrets are encrypted at rest in Google Cloud
- Each deployment uses the latest version of secrets
- You can update secrets without redeploying:
  ```bash
  echo "new_value" | gcloud secrets versions add SECRET_NAME --data-file=-
  ```

## 🆘 Troubleshooting

If secrets are not accessible:
```bash
# Grant permissions again
PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format="value(projectNumber)")
gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

## 🚀 You're Ready!

Run these commands on your VM to deploy securely with HTTPS! 🎉
