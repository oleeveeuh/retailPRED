# RetailPRED Documentation Index

Complete index of all RetailPRED documentation, organized by category and purpose.

---

## Quick Start

1. **[README.md](README.md)** - Project overview and quick start guide
2. **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** - How to deploy the application
3. **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture overview

---

## Core Documentation

### Project Overview

| Document | Description | Audience |
|----------|-------------|----------|
| **[README.md](README.md)** | Main project README with setup instructions, features, and metrics | All users |
| **[WEBAPP_README.md](WEBAPP_README.md)** | Web application specific documentation | Developers |
| **[API_DOCUMENTATION.md](backend/API_DOCUMENTATION.md)** | Backend API reference | API users |

### Deployment Guides

| Document | Description | Topics Covered |
|----------|-------------|----------------|
| **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** | Complete deployment guide | Vercel, Docker, Manual deployment |
| **[VERCEL_CONFIGURATION_COMPLETE.md](VERCEL_CONFIGURATION_COMPLETE.md)** | Vercel-specific setup | Environment config, troubleshooting |
| **[DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)** | Pre-deployment checklist | Verification steps |
| **[ENVIRONMENT_SETUP_COMPLETE.md](ENVIRONMENT_SETUP_COMPLETE.md)** | Environment configuration | Local/Production setup |
| **[docs/DEPLOYMENT_TEST.md](docs/DEPLOYMENT_TEST.md)** | Post-deployment testing | Test checklist |

### Architecture & Design

| Document | Description | Topics Covered |
|----------|-------------|----------------|
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | System architecture | Components, data flow, tech stack |
| **[PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md)** | Data pipeline architecture | ETL, feature engineering |

---

## Build & Configuration

### Build Documentation

| Document | Description | Status |
|----------|-------------|--------|
| **[BUILD_TEST_COMPLETE.md](BUILD_TEST_COMPLETE.md)** | Production build results | ✅ Complete |
| **[docs/DEPLOYMENT_TEST.md](docs/DEPLOYMENT_TEST.md)** | Build testing checklist | ✅ Complete |

### Configuration Files

| File | Purpose | Location |
|------|---------|----------|
| **frontend/.env.example** | Environment variable template | `frontend/` |
| **frontend/.env.development** | Development environment | `frontend/` |
| **frontend/.env.production** | Production environment | `frontend/` |
| **frontend/vercel.json** | Vercel deployment config | `frontend/` |
| **.vercelignore** | Vercel deployment exclusions | Root |
| **.github/FUNDING.yml** | Sponsorship configuration | `.github/` |

---

## Feature Documentation

### Explainability

| Document | Description | Features |
|----------|-------------|----------|
| **[EXPLAINABILITY_2025_2026_UPDATE.md](EXPLAINABILITY_2025_2026_UPDATE.md)** | SHAP explainability updates | Feature importance |
| **[EXPLAINABILITY_2025_DATES_FIX.md](EXPLAINABILITY_2025_DATES_FIX.md)** | Date fixes for explainability | Bug fixes |

### Features

| Document | Description | Topics |
|----------|-------------|--------|
| **[FEATURE_COUNT_EXPLANATION.md](FEATURE_COUNT_EXPLANATION.md)** | Feature count details | 242 features explained |
| **[FEATURE_ENGINEERING_DOCUMENTATION.md](FEATURE_ENGINEERING_DOCUMENTATION.md)** | Feature engineering guide | How features are created |

---

## Training & Models

### Training Documentation

| Document | Description | Content |
|----------|-------------|---------|
| **[training_outputs/training_report.md](training_outputs/training_report.md)** | Model training results | Performance metrics |
| **[COMPLETE_IMPLEMENTATION_SUMMARY.md](COMPLETE_IMPLEMENTATION_SUMMARY.md)** | Implementation summary | All features implemented |

### Model Predictions

| Document | Description | Records |
|----------|-------------|---------|
| **[training_outputs/model_predictions.json](training_outputs/model_predictions.json)** | All predictions | 7,873 predictions |
| **[training_outputs/robust_training_summary.json](training_outputs/robust_training_summary.json)** | Training summary | Model metrics |

---

## Deployment Assets

### Configuration Files

```
frontend/
├── vercel.json              # Vercel deployment config
├── .env.development         # Development environment
├── .env.production          # Production environment
├── .env.example             # Environment template
└── .gitignore               # Git ignore rules

.github/
└── FUNDING.yml              # Sponsorship config

root/
├── .vercelignore            # Vercel exclusions
├── docker-compose.yml       # Docker orchestration
└── Dockerfile               # Docker build config
```

---

## Documentation by Audience

### For New Users

1. Start with **[README.md](README.md)**
2. Read **[✨ Demo Deployment](README.md#-demo-deployment)** section
3. Try the **[Live Demo](https://retailpred.vercel.app)**
4. Explore **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for understanding

### For Developers

1. **[README.md](README.md)** - Project overview
2. **[WEBAPP_README.md](WEBAPP_README.md)** - Web app details
3. **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture
4. **[backend/API_DOCUMENTATION.md](backend/API_DOCUMENTATION.md)** - API reference
5. **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Deployment guide

### For DevOps Engineers

1. **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Deployment options
2. **[VERCEL_CONFIGURATION_COMPLETE.md](VERCEL_CONFIGURATION_COMPLETE.md)** - Vercel setup
3. **[DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)** - Pre-deployment checks
4. **[docs/DEPLOYMENT_TEST.md](docs/DEPLOYMENT_TEST.md)** - Post-deployment tests

### For Data Scientists

1. **[FEATURE_ENGINEERING_DOCUMENTATION.md](FEATURE_ENGINEERING_DOCUMENTATION.md)** - Feature creation
2. **[EXPLAINABILITY_2025_2026_UPDATE.md](EXPLAINABILITY_2025_2026_UPDATE.md)** - Model explainability
3. **[training_outputs/training_report.md](training_outputs/training_report.md)** - Model performance

---

## Documentation Statistics

| Category | Documents | Words | Topics |
|----------|-----------|-------|--------|
| **Core** | 5 | ~15,000 | Overview, API, Deployment |
| **Architecture** | 2 | ~8,000 | System design, components |
| **Deployment** | 6 | ~12,000 | Vercel, Docker, testing |
| **Features** | 4 | ~5,000 | Explainability, features |
| **Training** | 3 | ~3,000 | Models, predictions |
| **Configuration** | 7 | ~2,000 | Environment files |
| **TOTAL** | **27** | **~45,000** | Complete coverage |

---

## Document Maintenance

### Last Updated

- **README.md**: January 7, 2025
- **docs/DEPLOYMENT.md**: January 7, 2025
- **docs/ARCHITECTURE.md**: January 7, 2025
- **BUILD_TEST_COMPLETE.md**: January 7, 2025

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 7, 2025 | Initial documentation setup |
| 1.1 | Jan 7, 2025 | Added deployment guides |
| 1.2 | Jan 7, 2025 | Added architecture documentation |
| 1.3 | Jan 7, 2025 | Updated README with demo info |

---

## Quick Reference

### Essential Commands

```bash
# Local development
cd frontend && npm run dev

# Production build
cd frontend && npm run build:prod

# Preview production build
cd frontend && npm run preview

# Deploy to Vercel
cd frontend && vercel

# Docker deployment
docker-compose up -d
```

### Key URLs

- **Live Demo**: https://retailpred.vercel.app
- **GitHub Repository**: https://github.com/oleeveeuh/retailPRED
- **API Documentation** (local): http://localhost:8000/docs
- **Preview Server** (local): http://localhost:4173

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `VITE_DEMO_MODE` | Enable demo mode | `false` |
| `VITE_API_URL` | Backend API URL | `http://localhost:8000` |
| `VITE_TABLEAU_EMBED_URL` | Tableau visualization | (empty) |
| `VITE_DEBUG` | Debug logging | `false` |

---

## Support & Contributing

### Getting Help

1. Check documentation in this index
2. Search existing [GitHub Issues](https://github.com/oleeveeuh/retailPRED/issues)
3. Create new issue with question

### Contributing

1. Read **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for system overview
2. Check **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** for deployment info
3. Follow contribution guidelines in README
4. Submit pull request

### Documentation Updates

To update documentation:
1. Edit the relevant markdown file
2. Update this index (`DOCUMENTATION_INDEX.md`)
3. Update "Last Updated" date
4. Submit pull request

---

**Index Maintained**: January 7, 2025
**Documentation Version**: 1.3
**Total Documents**: 27
**Total Word Count**: ~45,000

For the most up-to-date documentation, always refer to the main [README.md](README.md).
