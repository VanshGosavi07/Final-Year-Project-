#!/bin/bash
# =============================================================================
# GOOGLE CLOUD DEPLOYMENT - RUN THIS SCRIPT
# =============================================================================

set -e  # Exit on any error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     Medical AI System - Google Cloud Deployment Script         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Step 1: Check Python
echo -e "${BLUE}[1/5]${NC} Checking Python version..."
python --version
echo ""

# Step 2: Test application startup
echo -e "${BLUE}[2/5]${NC} Testing application startup (15 seconds)..."
timeout 15s python main.py > /tmp/startup_test.log 2>&1 &
TEST_PID=$!
sleep 10

if kill -0 $TEST_PID 2>/dev/null; then
    echo -e "${GREEN}✓ Application starts successfully!${NC}"
    kill $TEST_PID 2>/dev/null || true
else
    echo -e "${RED}✗ Application failed to start${NC}"
    echo "Error log:"
    tail -20 /tmp/startup_test.log
    exit 1
fi
echo ""

# Step 3: Check environment variables
echo -e "${BLUE}[3/5]${NC} Checking environment configuration..."
if [ ! -f .env ]; then
    echo -e "${RED}✗ .env file not found!${NC}"
    echo "Please create .env file with required variables"
    exit 1
fi

if grep -q "YOUR_GROQ_API_KEY_HERE" .env 2>/dev/null; then
    echo -e "${RED}✗ GROQ_API_KEY not configured!${NC}"
    echo "Please edit .env and set your Groq API key"
    exit 1
fi

echo -e "${GREEN}✓ Environment variables configured${NC}"
echo ""

# Step 4: Verify app.yaml
echo -e "${BLUE}[4/5]${NC} Checking deployment configuration..."
if [ ! -f app.yaml ]; then
    echo -e "${RED}✗ app.yaml not found!${NC}"
    exit 1
fi
echo -e "${GREEN}✓ app.yaml found${NC}"
echo ""

# Step 5: Deploy
echo -e "${BLUE}[5/5]${NC} Ready to deploy!"
echo ""
echo -e "${YELLOW}════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}                   DEPLOYMENT READY                          ${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "To deploy your application, run:"
echo ""
echo -e "${GREEN}  gcloud app deploy${NC}"
echo ""
echo "This will:"
echo "  • Upload your application code"
echo "  • Install dependencies from requirements.txt"
echo "  • Deploy to Google App Engine"
echo "  • Provide you with a public URL"
echo ""
echo "After deployment, view your app:"
echo -e "${GREEN}  gcloud app browse${NC}"
echo ""
echo "View logs:"
echo -e "${GREEN}  gcloud app logs tail -s default${NC}"
echo ""
read -p "Do you want to deploy now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo ""
    echo -e "${BLUE}Starting deployment...${NC}"
    gcloud app deploy --quiet
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║            DEPLOYMENT SUCCESSFUL! 🎉                    ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Your application is now live!"
    echo ""
    echo "Open in browser:"
    gcloud app browse
else
    echo ""
    echo "Deployment cancelled. Run 'gcloud app deploy' when ready."
fi
