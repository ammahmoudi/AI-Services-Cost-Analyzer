# PRODUCTION DEPLOYMENT GUIDE

## Problem: Database Resets on Each Push

If you're experiencing database resets with each deployment, it's because you're using the **internal Docker PostgreSQL container**, which doesn't persist data across deployments in most CI/CD environments.

## Solution: Use External Managed Database

### Option 1: Railway (Recommended)

1. **Create PostgreSQL Database** in Railway:
   - Go to your Railway project
   - Click "New" → "Database" → "Add PostgreSQL"
   - Railway will automatically provide `DATABASE_URL`

2. **Deploy Application**:
   - Your app will automatically use Railway's `DATABASE_URL`
   - No configuration needed!

3. **Verify**:
   ```bash
   # Check logs
   railway logs
   
   # Should see: "Using PostgreSQL database"
   ```

### Option 2: Render

1. **Create PostgreSQL Database**:
   - Dashboard → "New" → "PostgreSQL"
   - Note the connection string

2. **Configure Web Service**:
   - Environment Variables:
     ```
     DATABASE_URL=<your-render-postgres-url>
     FLASK_SECRET_KEY=<generate-random-key>
     ```

3. **Deploy**:
   - Render will use the external database
   - Data persists across deployments

### Option 3: DigitalOcean App Platform

1. **Create Managed PostgreSQL**:
   - Create a new Managed Database (PostgreSQL)
   - Get connection details

2. **Create App**:
   - Deploy from GitHub
   - Add environment variables:
     ```
     DATABASE_URL=<your-do-postgres-url>
     FLASK_SECRET_KEY=<generate-random-key>
     ```

3. **Link Database**:
   - In App settings, add database as a component
   - DigitalOcean handles connection automatically

### Option 4: Docker with External Database

1. **Set up external PostgreSQL** (AWS RDS, Azure, etc.)

2. **Configure Docker**:
   ```bash
   # Create .env file
   DATABASE_URL=postgresql://user:password@external-host:5432/dbname
   FLASK_SECRET_KEY=your-secret-key-here
   ```

3. **Deploy with environment variables**:
   ```bash
   docker compose up -d app
   # Note: postgres service won't start (it's in 'local' profile)
   ```

## Local Development

For local development, use the included PostgreSQL container:

```bash
# Start both app and local database
docker compose --profile local up -d

# Or just the app (if you have external DB)
docker compose up -d app
```

## Verifying Your Setup

Check which database is being used:

```bash
# View logs
docker compose logs app | grep -i database

# Should see one of:
# - "Using PostgreSQL database" (external)
# - Connected to postgres:5432 (local)
```

## Migration

If you need to migrate data from local to external database:

```bash
# 1. Backup local database
docker exec ai-costs-db pg_dump -U ai_costs_user ai_costs > backup.sql

# 2. Restore to external database
psql <your-external-database-url> < backup.sql
```

## Troubleshooting

**Database still resets?**
- ✅ Verify `DATABASE_URL` environment variable is set
- ✅ Check it points to external database, not `postgres:5432`
- ✅ Ensure external database exists and is accessible
- ✅ Check logs for connection errors

**Can't connect to database?**
- Check firewall rules allow connections from your deployment
- Verify SSL requirements (add `?sslmode=require` to DATABASE_URL if needed)
- Test connection: `psql <your-database-url>`
