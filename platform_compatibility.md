# Platform Compatibility Guide

## ✅ Supported Platforms

The universal Dockerfile works across all major container platforms:

### **Cloud Platforms**
| Platform | Status | Notes |
|----------|--------|-------|
| **Fly.io** | ✅ Full Support | Optimized startup script, persistent volumes |
| **Railway** | ✅ Full Support | Auto-detects Railway environment |
| **Render** | ✅ Full Support | Handles zero-timeout requirements |
| **Heroku** | ✅ Full Support | Uses WEB_CONCURRENCY, Gunicorn optimization |
| **Google Cloud Run** | ✅ Full Support | Serverless-friendly timeout handling |
| **AWS ECS/Fargate** | ✅ Full Support | Standard container deployment |
| **Azure Container Instances** | ✅ Full Support | Compatible with Azure's container service |
| **DigitalOcean App Platform** | ✅ Full Support | Standard Docker deployment |

### **Self-Hosted Options**
| Platform | Status | Notes |
|----------|--------|-------|
| **Docker** | ✅ Full Support | Direct docker run/docker-compose |
| **Kubernetes** | ✅ Full Support | Standard K8s deployment |
| **Podman** | ✅ Full Support | Docker alternative |
| **Local Development** | ✅ Full Support | Works with any Docker-compatible system |

## 🔧 Platform-Specific Configurations

### **Fly.io**
```toml
# fly.toml
[env]
  PLATFORM = "flyio"
  
[[mounts]]
  source = "arabic_tts_models"
  destination = "/app/models"
```

### **Railway**
```json
// railway.json
{
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "restartPolicyType": "ON_FAILURE"
  }
}
```

### **Render**
```yaml
# render.yaml
services:
- type: web
  name: arabic-msp-tts
  env: docker
  plan: standard
  envVars:
  - key: PLATFORM
    value: render
```

### **Heroku**
```bash
# Set environment variables
heroku config:set PLATFORM=heroku
heroku config:set WEB_CONCURRENCY=1

# Use container stack
heroku stack:set container
```

### **Google Cloud Run**
```bash
# Deploy command
gcloud run deploy arabic-msp-tts \
  --source . \
  --platform managed \
  --set-env-vars PLATFORM=gcp \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600
```

### **AWS ECS**
```json
{
  "family": "arabic-msp-tts",
  "memory": "4096",
  "cpu": "2048",
  "environment": [
    {
      "name": "PLATFORM",
      "value": "aws"
    }
  ]
}
```

## 🚀 Deployment Commands by Platform

### **Fly.io**
```bash
flyctl launch --dockerfile Dockerfile
flyctl deploy
```

### **Railway**
```bash
railway login
railway link
railway up
```

### **Render**
```bash
# Connect GitHub repo or use CLI
render deploy
```

### **Heroku**
```bash
heroku create arabic-msp-tts
heroku stack:set container
git push heroku main
```

### **Google Cloud Run**
```bash
gcloud run deploy --source .
```

### **Docker (Local)**
```bash
docker build -t arabic-msp-tts .
docker run -p 8000:8000 arabic-msp-tts
```

### **Docker Compose**
```yaml
version: '3.8'
services:
  arabic-tts:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PLATFORM=docker
    volumes:
      - ./models:/app/models
```

## ⚙️ Platform-Specific Optimizations

### **Memory Requirements**
| Platform | Recommended RAM | CPU |
|----------|----------------|-----|
| Fly.io | 4GB | 2 vCPU |
| Railway | 4GB | 2 vCPU |
| Render | 4GB | 2 vCPU |
| Heroku | Performance-M+ | 2.5 vCPU |
| Cloud Run | 4GB | 2 vCPU |
| Local Docker | 4GB+ | 2+ cores |

### **Storage Considerations**
- **Fly.io**: Use persistent volumes for models
- **Railway**: Models downloaded on each deployment
- **Render**: Persistent disk available
- **Heroku**: Ephemeral filesystem (models re-download)
- **Cloud Run**: Stateless (consider Cloud Storage)

## 🔒 Security & Best Practices

### **Universal Security Features**
- ✅ Non-root user execution
- ✅ Minimal base image (Python slim)
- ✅ No hardcoded secrets
- ✅ Health check endpoints
- ✅ Proper signal handling

### **Environment Variables**
```bash
# Universal environment variables
PORT=8000                    # Server port
PYTHONUNBUFFERED=1          # Python output buffering
WEB_CONCURRENCY=1           # Number of workers
PLATFORM=<platform-name>    # Platform detection
```

## 📊 Cost Comparison

| Platform | Monthly Cost (4GB RAM) | Free Tier | Notes |
|----------|----------------------|-----------|-------|
| **Fly.io** | $30-50 | $5 credit | Auto-sleep saves costs |
| **Railway** | $20-40 | $5 credit | Usage-based pricing |
| **Render** | $25-45 | Free tier available | Always-on pricing |
| **Heroku** | $50-100 | No free tier | Performance dynos required |
| **Cloud Run** | $15-30 | Generous free tier | Pay per request |
| **Docker (VPS)** | $20-40 | N/A | Full control, more setup |

## 🐛 Troubleshooting by Platform

### **Common Issues**

**Model Download Failures**
```bash
# All platforms - check logs
docker logs <container-id>

# Platform-specific log commands
flyctl logs           # Fly.io
railway logs          # Railway  
heroku logs --tail    # Heroku
```

**Memory Issues**
```bash
# Increase memory allocation per platform
flyctl scale memory 8192        # Fly.io
railway memory 8192             # Railway
heroku ps:resize web=performance-l  # Heroku
```

**Port Binding Issues**
- Ensure `PORT` environment variable is set correctly
- Each platform may use different port assignment methods

## 🔄 Migration Between Platforms

The universal Dockerfile makes migration easy:

1. **Export environment variables** from current platform
2. **Update platform-specific configs** (if any)
3. **Deploy using new platform's method**
4. **Update DNS/domains** to point to new deployment

### **Migration Checklist**
- [ ] Environment variables exported
- [ ] Model files backed up (if using persistent storage)
- [ ] DNS records updated
- [ ] SSL certificates configured
- [ ] Monitoring/logging reconfigured
- [ ] Old platform resources cleaned up

## 📚 Additional Resources

- **Docker Documentation**: https://docs.docker.com/
- **Platform-specific guides**: Check each platform's documentation
- **Container best practices**: https://12factor.net/
- **Security guidelines**: OWASP Container Security Guide

---

*The universal Dockerfile ensures your Arabic MSP-TTS API can run anywhere containers are supported!*