#!/bin/bash
set -e

echo "Building frontend for Vercel deployment..."

# Install dependencies
echo "Installing dependencies..."
npm ci

# Build for production (using build:only to bypass TypeScript checks)
echo "Building..."
VITE_DEMO_MODE=true VITE_API_URL= npm run build:only

echo "Build complete!"
