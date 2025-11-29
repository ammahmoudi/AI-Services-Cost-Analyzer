# Docker Deployment Guide

This guide provides detailed information about deploying and managing the AI Services Cost Analyzer using Docker.

## Architecture

The Docker deployment consists of three services:

```
┌─────────────────────────────────────────┐
│          Docker Compose Stack           │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────────────────────────────┐  │
│  │         Flask App (app)          │  │
│  │  - Python 3.11                   │  │
│  │  - Playwright + Chromium         │  │
│  │  - Port: 5000                    │  │
│  └───────────┬──────────────────────┘  │
│              │                          │
│              ├──────────────┐           │
│              ▼              ▼           │
│  ┌───────────────────┐  ┌──────────┐   │
│  │  PostgreSQL (DB)  │  │ pgAdmin  │   │
│  │  - Version: 15    │  │ (optional)│  │
│  │  - Port: 5432     │  │ Port: 5050│  │
│  └───────────────────┘  └──────────┘   │
│                                         │
└─────────────────────────────────────────┘
```

## Services

### 1. PostgreSQL (`postgres`)

**Image**: `postgres:15-alpine`

**Purpose**: Primary database for storing AI models, sources, cache, and configuration

**Configuration**:
```yaml
environment:
  POSTGRES_USER: postgres
  POSTGRES_PASSWORD: postgres
  POSTGRES_DB: ai_costs
ports:
  - "5432:5432"
volumes:
  - postgres_data:/var/lib/postgresql/data
```

**Health Check**: Checks `pg_isready` every 5 seconds

### 2. Flask Application (`app`)

**Image**: Built from `Dockerfile`

**Purpose**: Web application and extraction engine

**Configuration**:
```yaml
environment:
  DATABASE_URL: postgresql://postgres:postgres@postgres:5432/ai_costs
  CACHE_BACKEND: database
  FLASK_SECRET_KEY: <your-secret-key>
ports:
  - "5000:5000"
volumes:
  - ./cache:/app/cache
depends_on:
  postgres:
    condition: service_healthy
```

**Startup Process**:
1. Wait for PostgreSQL health check
2. Run `init_db()` to create tables and seed default sources
3. Start Flask application on port 5000

### 3. pgAdmin (`pgadmin`)

**Image**: `dpage/pgadmin4:latest`

**Purpose**: Optional database administration interface

**Configuration**:
```yaml
environment:
  PGADMIN_DEFAULT_EMAIL: admin@admin.com
  PGADMIN_DEFAULT_PASSWORD: admin
ports:
  - "5050:80"
profiles:
  - admin
```

**Note**: Only starts when using `--profile admin` flag

## Dockerfile Details

### Base Image
```dockerfile
FROM python:3.11-slim
```

### Dependencies Installed
- PostgreSQL client libraries
- Python packages from `requirements.txt`
- Playwright with Chromium browser (~200MB)

### Working Directory
```
/app
```

### Exposed Port
```
5000
```

### Startup Command
```bash
python -c "from ai_cost_manager.database import init_db; init_db()" && python app.py
```

## Quick Start

### 1. First-Time Setup

```bash
# Clone repository
git clone <repository-url>
cd ai-costs

# Create environment file
cp .env.example .env

# Edit .env with your configuration
# At minimum, set FLASK_SECRET_KEY
nano .env
```

### 2. Build and Start

```bash
# Build images
docker compose build

# Start services (without pgAdmin)
docker compose up -d

# Or with pgAdmin
docker compose --profile admin up -d
```

### 3. Verify Services

```bash
# Check service status
docker compose ps

# Should show:
# NAME                STATUS              PORTS
# ai-costs-postgres   running (healthy)   0.0.0.0:5432->5432/tcp
# ai-costs-app        running             0.0.0.0:5000->5000/tcp
```

### 4. Access Application

- **Web UI**: http://localhost:5000
- **pgAdmin** (if started): http://localhost:5050

### 5. View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f app
docker compose logs -f postgres
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Flask Configuration
FLASK_SECRET_KEY=<generate-with-secrets.token_hex(32)>

# Database Configuration
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/ai_costs

# Cache Configuration
CACHE_BACKEND=database
CACHE_DIR=cache

# OpenRouter (for LLM extraction)
OPENROUTER_API_KEY=your_key_here
```

### Generate Secure Secret Key

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### Custom PostgreSQL Credentials

Edit `docker-compose.yml`:

```yaml
services:
  postgres:
    environment:
      POSTGRES_USER: myuser
      POSTGRES_PASSWORD: secure_password_here
      POSTGRES_DB: ai_costs
  
  app:
    environment:
      DATABASE_URL: postgresql://myuser:secure_password_here@postgres:5432/ai_costs
```

## Management Commands

### Start/Stop Services

```bash
# Start all services
docker compose up -d

# Stop all services (keeps data)
docker compose stop

# Stop and remove containers (keeps data)
docker compose down

# Stop and remove containers + volumes (DELETES DATA)
docker compose down -v
```

### Rebuild After Code Changes

```bash
# Rebuild and restart
docker compose up -d --build
```

### Access Container Shell

```bash
# Flask app
docker compose exec app bash

# PostgreSQL
docker compose exec postgres psql -U postgres -d ai_costs
```

### View Resource Usage

```bash
docker compose stats
```

## Data Persistence

### Volumes

The stack creates two named volumes:

1. **postgres_data**: PostgreSQL database files
2. **pgadmin_data**: pgAdmin configuration (if using admin profile)

### Backup Database

```bash
# Create backup
docker compose exec postgres pg_dump -U postgres ai_costs > backup.sql

# Restore backup
docker compose exec -T postgres psql -U postgres ai_costs < backup.sql
```

### Export Cache (if using file backend)

```bash
# Cache is mounted at ./cache
# Just copy the directory
cp -r cache cache_backup
```

## Networking

### Internal Network

All services communicate via a Docker network named `ai-costs-network`.

**Container Hostnames**:
- `postgres` - PostgreSQL server
- `app` - Flask application
- `pgadmin` - pgAdmin (if running)

### Port Mapping

| Service | Internal Port | External Port |
|---------|--------------|---------------|
| Flask App | 5000 | 5000 |
| PostgreSQL | 5432 | 5432 |
| pgAdmin | 80 | 5050 |

### Accessing from Host

```bash
# Test Flask app
curl http://localhost:5000

# Connect to PostgreSQL
psql -h localhost -p 5432 -U postgres -d ai_costs
```

## Troubleshooting

### App Container Fails to Start

**Check logs**:
```bash
docker compose logs app
```

**Common issues**:
- PostgreSQL not ready: Wait for health check
- Missing `.env` file: Copy from `.env.example`
- Port 5000 in use: Change port mapping in `docker-compose.yml`

### Database Connection Error

**Verify PostgreSQL is running**:
```bash
docker compose ps postgres
# Should show "healthy"
```

**Check connection from app container**:
```bash
docker compose exec app python -c "from ai_cost_manager.database import engine; print(engine.url)"
```

**Verify credentials**:
```bash
docker compose exec postgres psql -U postgres -c "\l"
```

### Playwright Browser Not Found

**Reinstall browser** (should be automatic):
```bash
docker compose exec app playwright install chromium
```

**Rebuild image**:
```bash
docker compose up -d --build
```

### pgAdmin Can't Connect to Database

**In pgAdmin, use these connection settings**:
- **Host**: `postgres` (not `localhost`)
- **Port**: `5432`
- **Username**: `postgres`
- **Password**: `postgres`
- **Database**: `ai_costs`

### Out of Disk Space

**Check Docker disk usage**:
```bash
docker system df
```

**Clean up**:
```bash
# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Full cleanup (careful!)
docker system prune -a --volumes
```

## Production Deployment

### Security Hardening

1. **Change default passwords**:
```yaml
environment:
  POSTGRES_PASSWORD: <strong-password>
  PGADMIN_DEFAULT_PASSWORD: <strong-password>
```

2. **Don't expose PostgreSQL**:
```yaml
ports:
  # - "5432:5432"  # Comment out
```

3. **Use secrets management**:
```bash
# Use Docker secrets or external secret manager
docker secret create postgres_password password.txt
```

4. **Enable HTTPS**: Use reverse proxy (nginx, traefik)

### Resource Limits

Add to `docker-compose.yml`:

```yaml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
  
  postgres:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
```

### Health Monitoring

```bash
# Install Docker health check plugin
docker plugin install grafana/docker-integration-gcplogs

# Monitor with Prometheus/Grafana
# Add exporters to docker-compose.yml
```

## Development vs Production

### Development Setup

```yaml
services:
  app:
    build:
      context: .
      target: development  # Add multi-stage build
    volumes:
      - ./:/app  # Live code reload
    environment:
      FLASK_ENV: development
      FLASK_DEBUG: 1
```

### Production Setup

```yaml
services:
  app:
    build:
      context: .
      target: production
    restart: always
    environment:
      FLASK_ENV: production
      FLASK_DEBUG: 0
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Push

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: docker compose build
      
      - name: Run tests
        run: docker compose run app pytest
      
      - name: Push to registry
        run: |
          docker tag ai-costs-app:latest registry.example.com/ai-costs:latest
          docker push registry.example.com/ai-costs:latest
```

## Scaling

### Horizontal Scaling (Multiple App Instances)

```yaml
services:
  app:
    deploy:
      replicas: 3
```

### Load Balancer

Add nginx or traefik service:

```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - app
```

## Monitoring

### Container Logs

```bash
# Follow all logs
docker compose logs -f

# Last 100 lines
docker compose logs --tail=100

# Since timestamp
docker compose logs --since 2024-01-01T00:00:00
```

### Health Checks

```bash
# Check all services
docker compose ps

# Detailed health status
docker inspect ai-costs-app | grep Health -A 10
```

### Performance Metrics

```bash
# Real-time stats
docker compose stats

# Resource usage
docker compose top
```

## Maintenance

### Update Images

```bash
# Pull latest base images
docker compose pull

# Rebuild with latest
docker compose up -d --build
```

### Database Maintenance

```bash
# Vacuum database
docker compose exec postgres psql -U postgres -d ai_costs -c "VACUUM ANALYZE;"

# Check database size
docker compose exec postgres psql -U postgres -c "\l+ ai_costs"
```

### Log Rotation

Configure in `docker-compose.yml`:

```yaml
services:
  app:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [PostgreSQL Docker Hub](https://hub.docker.com/_/postgres)
- [Playwright Documentation](https://playwright.dev/python/)
- [Flask Deployment](https://flask.palletsprojects.com/en/3.0.x/deploying/)
