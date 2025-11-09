#!/bin/bash
# =============================================================================
# PRE-DEPLOYMENT MODEL VERIFICATION AND FIX
# =============================================================================

set -e

echo "🔍 Medical AI System - Pre-Deployment Verification"
echo "====================================================================="
echo ""

# Step 1: Verify we're in the project directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: Not in project directory. Please cd to Final-Year-Project-"
    exit 1
fi

PROJECT_DIR=$(pwd)
echo "✓ Project directory: $PROJECT_DIR"
echo ""

# Step 2: Verify model files exist locally
echo "STEP 1: Checking Model Files on VM..."
echo "---------------------------------------------------------------------"

BREAST_MODEL="Modal/Breast Cancer/breast_cancer.keras"
LUNG_MODEL="Modal/Lung Cancer/Lung Cancer.keras"

if [ -f "$BREAST_MODEL" ]; then
    SIZE=$(du -h "$BREAST_MODEL" | cut -f1)
    echo "✅ Breast Cancer Model: $BREAST_MODEL ($SIZE)"
else
    echo "❌ Breast Cancer Model NOT FOUND: $BREAST_MODEL"
    exit 1
fi

if [ -f "$LUNG_MODEL" ]; then
    SIZE=$(du -h "$LUNG_MODEL" | cut -f1)
    echo "✅ Lung Cancer Model: $LUNG_MODEL ($SIZE)"
else
    echo "❌ Lung Cancer Model NOT FOUND: $LUNG_MODEL"
    exit 1
fi

echo ""

# Step 3: Test model loading with Python
echo "STEP 2: Testing Model Loading with Python..."
echo "---------------------------------------------------------------------"

python3 test_models_loading.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Model loading test FAILED!"
    echo "   Please fix the errors above before deploying."
    exit 1
fi

echo ""

# Step 4: Verify Dockerfile includes models
echo "STEP 3: Verifying Dockerfile Configuration..."
echo "---------------------------------------------------------------------"

if grep -q "COPY Modal/ ./Modal/" Dockerfile; then
    echo "✅ Dockerfile includes: COPY Modal/ ./Modal/"
else
    echo "❌ Dockerfile missing Modal copy command!"
    exit 1
fi

if grep -q 'COPY \["RAG Data/", "./RAG Data/"\]' Dockerfile; then
    echo "✅ Dockerfile includes: COPY RAG Data"
else
    echo "⚠️  Warning: RAG Data copy might have issues"
fi

echo ""

# Step 5: Check .dockerignore doesn't exclude models
echo "STEP 4: Checking .dockerignore..."
echo "---------------------------------------------------------------------"

if grep -q "Modal/" .dockerignore 2>/dev/null; then
    echo "⚠️  WARNING: .dockerignore might exclude Modal folder!"
else
    echo "✅ .dockerignore does not exclude Modal folder"
fi

if grep -q "*.keras" .dockerignore 2>/dev/null; then
    echo "❌ ERROR: .dockerignore excludes .keras files!"
    exit 1
else
    echo "✅ .dockerignore does not exclude .keras files"
fi

echo ""

# Step 6: Verify latest code is committed
echo "STEP 5: Checking Git Status..."
echo "---------------------------------------------------------------------"

if git diff --quiet breast_cancer_predictor.py lung_cancer_predictor.py; then
    echo "✅ Model predictor files are committed"
else
    echo "⚠️  WARNING: Uncommitted changes in predictor files"
    echo "   Run: git add . && git commit -m 'Update predictors' && git push"
fi

LATEST_COMMIT=$(git log -1 --oneline)
echo "Latest commit: $LATEST_COMMIT"

echo ""

# Step 7: Display deployment instructions
echo "====================================================================="
echo "✅ PRE-DEPLOYMENT VERIFICATION COMPLETE!"
echo "====================================================================="
echo ""
echo "All checks passed. Ready to deploy!"
echo ""
echo "🚀 DEPLOYMENT COMMANDS:"
echo ""
echo "Option 1: Deploy with secrets (recommended)"
echo "  bash deploy_with_secrets.sh"
echo ""
echo "Option 2: Quick deploy (no secrets)"
echo "  bash deploy_simple.sh"
echo ""
echo "After deployment, verify models load by checking logs:"
echo "  gcloud logging tail \"resource.type=cloud_run_revision AND resource.labels.service_name=medical-ai-system\""
echo ""
echo "Then test by uploading an image at:"
echo "  https://medical-ai-system-424bnofprq-as.a.run.app/form"
echo ""
