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
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python $PYTHON_VERSION found"

# Check Node.js
echo "📦 Checking Node.js installation..."
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18 or higher."
    exit 1
fi
NODE_VERSION=$(node --version)
echo "✅ Node $NODE_VERSION found"

# Backend setup
echo ""
echo "🔧 Setting up backend..."
cd backend

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install -q -r requirements.txt

echo "Running database migration..."
python -m db.migrations apply ../data/db/schema.sql

cd ..

# Frontend setup
echo ""
echo "🔧 Setting up frontend..."
cd frontend

if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
fi

echo "Installing Node dependencies..."
npm install

cd ..

# Root setup
echo ""
echo "🔧 Setting up root dependencies..."
npm install

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
