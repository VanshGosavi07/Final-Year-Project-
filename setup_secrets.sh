#!/bin/bash
# Setup Google Cloud Secrets for Medical AI System
# =============================================================================

set -e

echo "🔐 Setting up Google Cloud Secrets..."

# Read .env file and create secrets
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    exit 1
fi

# Function to create or update secret
create_or_update_secret() {
    local secret_name=$1
    local secret_value=$2
    
    if gcloud secrets describe "$secret_name" &>/dev/null; then
        echo "✏️  Updating existing secret: $secret_name"
        echo -n "$secret_value" | gcloud secrets versions add "$secret_name" --data-file=-
    else
        echo "✨ Creating new secret: $secret_name"
        echo -n "$secret_value" | gcloud secrets create "$secret_name" --data-file=-
    fi
}

# Parse .env file and create secrets
while IFS='=' read -r key value; do
    # Skip comments and empty lines
    [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
    
    # Remove quotes and whitespace
    key=$(echo "$key" | xargs)
    value=$(echo "$value" | xargs | sed "s/^['\"]//;s/['\"]$//")
    
    if [ -n "$key" ] && [ -n "$value" ]; then
        echo "Processing: $key"
        create_or_update_secret "$key" "$value"
    fi
done < .env

echo ""
echo "✅ All secrets have been created/updated!"
echo ""
echo "🔑 Grant Cloud Run access to secrets..."

# Get the project number
PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format="value(projectNumber)")
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

# Grant access to all secrets
gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/secretmanager.secretAccessor" \
    --condition=None

echo ""
echo "✅ Secrets setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Run: ./deploy_with_secrets.sh"
echo "   2. Your secrets are now securely stored in Google Cloud Secret Manager"
echo "   3. The .env file will NOT be uploaded to GitHub or Cloud Build"
