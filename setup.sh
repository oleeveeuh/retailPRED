#!/bin/bash

# RetailPRED Setup Script
# This script helps you set up the development environment

set -e

echo "================================"
echo "RetailPRED Setup Script"
echo "================================"
echo ""

# Check Python
echo "📦 Checking Python installation..."
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Python 3.11 is required."
    exit 1
fi
PYTHON_VERSION=$(python3.11 --version | cut -d' ' -f2)
echo "✅ Python $PYTHON_VERSION found"

# Check Node.js
echo "📦 Checking Node.js installation..."
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18 or higher."
    exit 1
fi
NODE_VERSION=$(node --version)
echo "✅ Node $NODE_VERSION found"

# Python setup
echo ""
echo "🔧 Setting up Python environment..."
if [ ! -d ".venv-v2" ]; then
    echo "Creating Python virtual environment..."
    python3.11 -m venv .venv-v2
fi

echo "Activating virtual environment..."
source .venv-v2/bin/activate

echo "Installing Python dependencies..."
python -m pip install pip==25.3
python -m pip install -r requirements-dev.lock

echo "Running database migration..."
python -m backend.db.migrations --schema-path data/db/schema.sql --db-path data/retailpred.db

# Frontend setup
echo ""
echo "🔧 Setting up frontend..."
cd frontend

if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
fi

echo "Installing Node dependencies..."
NPM_CONFIG_CACHE=/tmp/retailpred-npm-cache npm ci

cd ..

# Root setup
echo ""
echo "🔧 Setting up root dependencies..."
NPM_CONFIG_CACHE=/tmp/retailpred-npm-cache npm ci

echo ""
echo "================================"
echo "✅ Setup Complete!"
echo "================================"
echo ""
echo "To start the development servers:"
echo ""
echo "  From project root:"
echo "  $ npm run dev"
echo ""
echo "  Or separately:"
echo "  $ npm run dev:backend  # Terminal 1"
echo "  $ npm run dev:frontend # Terminal 2"
echo ""
echo "Backend will run on: http://localhost:8000"
echo "Frontend will run on: http://localhost:5173"
echo "API docs at: http://localhost:8000/docs"
echo ""
echo "Happy forecasting! 🚀"
