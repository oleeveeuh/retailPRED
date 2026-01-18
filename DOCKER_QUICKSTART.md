# 🐳 RetailPRED Docker Quick Start

## Quick Reference

### Build and Run (Backend Only)

```bash
# Build image
docker build -t retailpred-api:latest .

# Run container
docker run -d \
  --name retailpred-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/backend/ml/models:/app/backend/ml/models \
  retailpred-api:latest

# Check logs
docker logs -f retailpred-api

# Stop container
docker stop retailpred-api
```

### Build and Run (Full Stack with Docker Compose)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Rebuild after changes
docker-compose up -d --build
```

## Health Check

```bash
curl http://localhost:8000/api/health
```

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Common Commands

```bash
# List running containers
docker ps

# Shell into container
docker exec -it retailpred-api /bin/bash

# View container stats
docker stats retailpred-api

# Remove container
docker rm -f retailpred-api

# Remove image
docker rmi retailpred-api:latest
```

## Full Deployment Guide

See [DEPLOYMENT.md](./DEPLOYMENT.md) for:
- ☁️ Cloud deployment (AWS, GCP, Azure, Heroku)
- 🔐 Security best practices
- 📊 Monitoring and logging
- 🚦 Performance optimization
- 🔄 CI/CD integration

---

**Built full-stack ML application with React/TypeScript frontend and FastAPI backend, deployed via automated CI/CD with Docker containerization for portable deployment across cloud platforms.**
