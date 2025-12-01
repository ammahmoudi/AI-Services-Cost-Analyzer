# Production Deployment Guide

## 🚀 How Deployments Work

### Automatic Migration System

With the new setup, migrations run **automatically** on container startup:

```
┌─────────────────────────────────────────────────┐
│ 1. Developer pushes code                        │
│    git push origin main                         │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│ 2. Production pulls & rebuilds                  │
│    docker compose pull                          │
│    docker compose up --build -d                 │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│ 3. Container starts                             │
│    ├─> entrypoint.py runs                       │
│    ├─> Scans migrations/ directory              │
│    ├─> Runs all *.py files                      │
│    └─> migrations/add_auth_credentials.py       │
│        migrations/add_model_matching.py         │
│        migrations/add_timestamp_columns.py      │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│ 4. Migrations execute                           │
│    ├─> Check if columns exist                   │
│    ├─> Add if missing                           │
│    ├─> Skip if already applied ✅               │
│    └─> Safe to run multiple times              │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│ 5. Application starts                           │
│    └─> gunicorn with 4 workers                  │
│    └─> Listens on 0.0.0.0:5000                  │
│    └─> Database schema up-to-date ✅            │
└─────────────────────────────────────────────────┘
```

## 📋 Step-by-Step Deployment

### First Time Setup

```bash
# 1. Clone repository
git clone https://github.com/ammahmoudi/AI-Services-Cost-Analyzer.git
cd AI-Services-Cost-Analyzer

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings

# 3. Start services
docker compose up -d

# 4. Check logs
docker compose logs -f app
```

**Expected output:**
```
🐳 AI Cost Manager - Container Startup
============================================================
🔧 Running database migrations...
Found 3 migration(s)
  Running add_timestamp_columns.py... ✅
  Running add_auth_credentials.py... ✅
  Running add_model_matching.py... ✅
✅ Migrations complete

🚀 Starting application...
[2025-12-01 12:00:00] [INFO] Starting gunicorn 21.2.0
[2025-12-01 12:00:00] [INFO] Listening at: http://0.0.0.0:5000
[2025-12-01 12:00:00] [INFO] Using worker: sync
[2025-12-01 12:00:00] [INFO] Booting worker with pid: 10
```

### Regular Updates (Push New Code)

```bash
# 1. Developer commits changes
git add .
git commit -m "Added model matching feature"
git push origin main

# 2. On production server
cd AI-Services-Cost-Analyzer
git pull origin main

# 3. Rebuild and restart
docker compose up --build -d

# 4. Migrations run automatically!
# Check logs:
docker compose logs -f app
```

**What happens:**
- ✅ New code pulled
- ✅ Docker image rebuilt
- ✅ Container restarted
- ✅ Migrations run automatically
- ✅ App starts with updated schema

## 🔄 Zero-Downtime Deployments

For production with no downtime:

```bash
# Build new image
docker compose build app

# Start new container alongside old one
docker compose up -d --no-deps --scale app=2 app

# Wait for health check
sleep 10

# Remove old container
docker compose up -d --no-deps --scale app=1 app
```

Or use **blue-green deployment** with a load balancer.

## 🗃️ Database Persistence

### What Persists

```
docker compose down           # ✅ Data safe (volume persists)
docker compose restart        # ✅ Data safe
docker compose up --build     # ✅ Data safe
git pull && docker compose up # ✅ Data safe
```

### What Deletes Data

```
docker compose down -v        # ❌ Deletes volume!
docker volume rm postgres_data # ❌ Deletes data!
```

### Backup Before Updates

```bash
# Backup PostgreSQL
docker exec ai-costs-db pg_dump -U ai_costs_user ai_costs > backup_$(date +%Y%m%d).sql

# Backup SQLite
cp ai_costs.db ai_costs.db.backup_$(date +%Y%m%d)
```

### Restore from Backup

```bash
# PostgreSQL
docker exec -i ai-costs-db psql -U ai_costs_user ai_costs < backup_20251201.sql

# SQLite
cp ai_costs.db.backup_20251201 ai_costs.db
```

## 🔧 Migration Best Practices

### 1. Idempotent Migrations

All migrations check if changes already exist:

```python
# Good ✅
try:
    conn.execute(text("ALTER TABLE auth_settings ADD COLUMN username VARCHAR(255)"))
    conn.commit()
except Exception as e:
    # Column already exists, safe to continue
    pass
```

### 2. Naming Convention

```
migrations/
├── add_timestamp_columns.py      # Descriptive name
├── add_auth_credentials.py       # What it does
└── add_model_matching.py         # Feature name
```

### 3. Testing Migrations

```bash
# Test locally first
python migrations/add_auth_credentials.py

# Check database
sqlite3 ai_costs.db ".schema auth_settings"
```

### 4. Rollback Strategy

Create downgrade functions:

```python
def upgrade():
    """Add columns"""
    # ... add columns

def downgrade():
    """Remove columns"""
    # ... remove columns
```

## 🚨 Troubleshooting

### Migration Fails

```bash
# View logs
docker compose logs app

# Common issues:
# 1. Syntax error → Fix migration file
# 2. Permission denied → Check volume permissions
# 3. Database locked → Restart container
```

### Manual Migration

If automatic migration fails:

```bash
# Enter container
docker exec -it ai-costs-app bash

# Run manually
python migrations/add_auth_credentials.py

# Exit and restart
exit
docker compose restart app
```

### Check Applied Migrations

```bash
# PostgreSQL
docker exec -it ai-costs-db psql -U ai_costs_user ai_costs -c "\d auth_settings"

# SQLite
sqlite3 ai_costs.db ".schema auth_settings"
```

## 📊 Monitoring Deployments

### Health Check

Add to `docker-compose.yml`:

```yaml
services:
  app:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Logs

```bash
# Follow logs
docker compose logs -f app

# Last 100 lines
docker compose logs --tail=100 app

# With timestamps
docker compose logs -f -t app
```

### Database Size

```bash
# PostgreSQL
docker exec ai-costs-db psql -U ai_costs_user ai_costs -c "
  SELECT pg_size_pretty(pg_database_size('ai_costs'));"

# SQLite
ls -lh ai_costs.db
```

## 🔐 Security Checklist

- [ ] Change `FLASK_SECRET_KEY` in production
- [ ] Change PostgreSQL password
- [ ] Use HTTPS (reverse proxy with nginx/traefik)
- [ ] Backup database regularly
- [ ] Monitor disk space
- [ ] Review logs for errors
- [ ] Encrypt passwords in auth_settings

## 📦 CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to production
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.HOST }}
          username: ${{ secrets.USERNAME }}
          key: ${{ secrets.SSH_KEY }}
          script: |
            cd AI-Services-Cost-Analyzer
            git pull origin main
            docker compose up --build -d
            docker compose logs --tail=50 app
```

## 📝 Deployment Checklist

Before each deployment:

- [ ] Test migrations locally
- [ ] Backup database
- [ ] Review code changes
- [ ] Check disk space
- [ ] Update .env if needed

After deployment:

- [ ] Check logs for errors
- [ ] Verify migrations ran
- [ ] Test critical features
- [ ] Monitor for 5-10 minutes

## 🎯 Summary

**Automatic migrations mean:**
- ✅ No manual intervention needed
- ✅ Schema always up-to-date
- ✅ Safe to run multiple times
- ✅ Logs show what happened
- ✅ Deployment is just: `git pull && docker compose up --build -d`

**Your data is safe because:**
- ✅ Database in persistent volume
- ✅ Volume survives container restarts
- ✅ .gitignore excludes database files
- ✅ Migrations are additive (don't delete data)

**Production workflow:**
```bash
# Developer
git push origin main

# Production (automatic)
git pull && docker compose up --build -d
# ↓
# Migrations run automatically
# ↓
# App starts with new schema
# ↓
# Done! ✅
```
