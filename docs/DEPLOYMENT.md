# Deployment Guide

## Quick Start (Local Docker)

```bash
# Build and run
docker-compose up --build

# Access at http://localhost:8000
```

## Deployment Options

### Option 1: Render (Recommended - Free Tier)

1. **Create Render Account**: https://render.com
2. **Connect GitHub**: Link your repository
3. **Create Web Service**:
   - Select repository
   - Environment: Docker
   - Instance Type: Free
4. **Set Environment Variables**:
   ```
   ENVIRONMENT=production
   LOG_LEVEL=WARNING
   SECRET_KEY=<generate-secure-key>
   ```
5. **Deploy**: Render auto-deploys on push to `main`

### Option 2: Railway

1. **Create Railway Account**: https://railway.app
2. **New Project** → Deploy from GitHub
3. **Auto-detects Dockerfile**
4. **Set Variables** in Railway dashboard

### Option 3: Manual VPS

```bash
# On your VPS
git clone <your-repo>
cd ai-nutrition

# Build and run with Docker
docker-compose -f docker-compose.yml up -d

# Or run directly
pip install -r requirements.txt
python src/main.py
```

## Health Check

After deployment, verify:
```bash
curl https://your-app.render.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "4.0.0",
  "environment": "production",
  "services": {...}
}
```

## Monitoring

- **Logs**: Check Render/Railway dashboard
- **Errors**: Monitor `/logs/` directory
- **Health**: `/health` endpoint

## Troubleshooting

| Issue | Solution |
|-------|----------|
| App not starting | Check logs, verify PORT env |
| 500 errors | Check application logs |
| Slow cold starts | Normal for free tier (sleeps after idle) |
