#!/bin/bash
# Vercel Build Script
# Forces demo mode by creating .env.production file

# Force demo mode regardless of Vercel env vars
cat > .env.production << 'EOF'
VITE_DEMO_MODE=true
VITE_API_URL=
VITE_TABLEAU_EMBED_URL=
EOF

echo "✅ Forcing DEMO MODE for production build"
echo "Environment file contents:"
cat .env.production
echo ""

# Verify the file was created
if [ -f .env.production ]; then
  echo "✅ .env.production file created successfully"
else
  echo "❌ ERROR: .env.production file not created!"
  exit 1
fi

npm run build:prod

