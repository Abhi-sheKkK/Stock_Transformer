#!/bin/bash
# Build deployable zip package for AI Financial Intelligence System
# Excludes virtual environments, cache folders, local logs, and git files.

set -e

ZIP_FILE="Stock_Transformer_deploy.zip"
echo "📦 Packaging Stock Transformer for deployment..."

# Remove old zip if it exists
if [ -f "$ZIP_FILE" ]; then
    echo "  Removing existing $ZIP_FILE..."
    rm "$ZIP_FILE"
fi

# Check if model weight file exists
if [ ! -f "best_model.pth" ]; then
    echo "⚠️ WARNING: 'best_model.pth' not found in workspace root."
    echo "   Ensure you copy it to the deployment directory before starting the app."
fi

# Run zip command
zip -r "$ZIP_FILE" \
    api \
    src \
    frontend \
    models \
    requirements.txt \
    .env.example \
    README.md \
    best_model.pth \
    main.py \
    predict.py \
    -x "**/__pycache__/*" \
    -x "**/.DS_Store" \
    -x "stock_env/*" \
    -x "venv/*" \
    -x ".git/*" \
    -x ".cache/*" \
    -x "runs/*" \
    -x "scratch/*" \
    -x "data/*" \
    -x "results_global/*"

echo "==========================================="
echo "✅ Deployment package created: $ZIP_FILE"
echo "==========================================="
echo "Deployment Instructions:"
echo "1. Upload and extract $ZIP_FILE on your server."
echo "2. Copy .env.example to .env and configure keys."
echo "3. Run: pip install -r requirements.txt"
echo "4. Start backend: uvicorn api.main:app --host 0.0.0.0 --port 8000"
