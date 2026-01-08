#!/bin/bash
# Vercel Build Script
# Creates .env file with environment variables for Vite build

cat > .env.production << EOF
VITE_DEMO_MODE=${VITE_DEMO_MODE:-true}
VITE_API_URL=${VITE_API_URL:-}
VITE_TABLEAU_EMBED_URL=${VITE_TABLEAU_EMBED_URL:-}
EOF

echo "Environment variables set for build:"
cat .env.production

npm run build:prod

