#!/bin/bash
# Vercel Build Script
# Forces demo mode by creating .env.production file

echo "📍 Current directory: $(pwd)"
echo "📁 Files in current directory:"
ls -la | grep -E "\.env|package"

# Force demo mode regardless of Vercel env vars
cat > .env.production << 'EOF'
VITE_DEMO_MODE=true
VITE_API_URL=
VITE_TABLEAU_EMBED_URL=
EOF

echo ""
echo "✅ Created .env.production file:"
cat .env.production
echo ""

# Verify the file was created
if [ -f .env.production ]; then
  echo "✅ .env.production file exists"
else
  echo "❌ ERROR: .env.production file not created!"
  exit 1
fi

echo "🔨 Starting build..."
npm run build:only

echo ""
echo "🔍 Checking for localhost in built files..."
if grep -r "localhost:8000" dist/assets/*.js 2>/dev/null; then
  echo "❌ WARNING: localhost still found in bundle!"
  grep -c "localhost:8000" dist/assets/*.js
else
  echo "✅ SUCCESS: No localhost references in bundle!"
fi

