# Google Cloud Deployment Guide

## 🎯 Problem Solved: Segmentation Fault Fix

The segmentation fault was caused by loading heavy ML models (TensorFlow + PyTorch) at startup in memory-constrained Google Cloud Shell.

### ✅ Solution Implemented: Lazy Loading

All heavy components now load **on-demand** (only when actually used):
- ✓ Breast Cancer Predictor
- ✓ Lung Cancer Predictor  
- ✓ Document Processors (RAG)
- ✓ Embedding Models

## 📋 Deployment Steps

### 1. Test Locally on Google Cloud Shell

```bash
# Make test script executable
chmod +x test-startup.sh

# Run startup test
./test-startup.sh
```

If the test passes, proceed to deployment.

### 2. Deploy to Google App Engine

```bash
# Deploy your application
gcloud app deploy

# When prompted:
# - Confirm the project
# - Confirm the region
# - Type 'Y' to proceed
```

### 3. View Your Deployed Application

```bash
# Open in browser
gcloud app browse

# Or get the URL
gcloud app describe --format="value(defaultHostname)"
```

### 4. Monitor Logs

```bash
# View real-time logs
gcloud app logs tail -s default

# View specific log level
gcloud app logs tail -s default --level=error
```

## 🔧 Troubleshooting

### If deployment fails with memory issues:

1. **Increase instance class** in `app.yaml`:
```yaml
instance_class: F4_1G  # Current
# Change to:
instance_class: F4  # 512MB + 1.2GHz
# or
instance_class: F4_HIGHMEM  # 1GB + 1.2GHz
```

2. **Reduce workers** in `app.yaml`:
```yaml
entrypoint: gunicorn -b :$PORT main:app --timeout 300 --workers 1
# Reduced from 2 to 1 worker
```

### If models fail to load:

Check if model files are included in deployment:
```bash
ls -lh Modal/Breast\ Cancer/breast_cancer.keras
ls -lh Modal/Lung\ Cancer/Lung\ Cancer.keras
```

If missing, they will be loaded on first prediction request (expect slower first response).

### Memory optimization tips:

1. **Use external model storage** (recommended for production):
   - Upload models to Google Cloud Storage
   - Load from GCS instead of bundling

2. **Add to `.gcloudignore`** (already configured):
   - Screenshots/
   - __pycache__/
   - *.pyc

## 🎉 Success Indicators

✅ App starts without errors
✅ Homepage loads successfully  
✅ First prediction triggers model loading (expect 10-30 seconds)
✅ Subsequent predictions are fast

## 📊 Performance Notes

- **First request**: Slow (models loading)
- **Subsequent requests**: Fast (models cached in memory)
- **Cold start**: ~30-60 seconds (Google Cloud spins down idle instances)
- **Warm requests**: <2 seconds

## 🔐 Security Checklist

- ✅ Environment variables set in `.env.yaml`
- ✅ `.env` not committed to git (in `.gitignore`)
- ✅ Cloudinary API keys configured
- ✅ Groq API key configured
- ✅ Database connection secured (SSL mode)

## 💰 Cost Optimization

Free tier includes:
- 28 instance hours per day
- 5 GB outbound traffic per month
- Shared memcache

Monitor usage:
```bash
gcloud app versions list
gcloud app services list
```

## 🆘 Need Help?

Check application health:
```bash
curl https://YOUR_APP_ID.appspot.com/
```

View detailed logs:
```bash
gcloud app logs read --service=default --limit=50
```

---

**Ready to deploy?** Run: `gcloud app deploy`
