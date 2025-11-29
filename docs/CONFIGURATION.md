# Configuration Guide

## Environment Variables

Configure the AI Cost Manager using environment variables in a `.env` file.

### Database Configuration

#### SQLite (Default)
```bash
DATABASE_URL=sqlite:///ai_costs.db
```
- Simple file-based database
- No setup required
- Perfect for single-user or development use
- Data stored in `ai_costs.db` file

#### PostgreSQL
```bash
DATABASE_URL=postgresql://username:password@localhost:5432/database_name
```
- Production-ready database
- Better performance for large datasets
- Supports multiple concurrent users
- Requires PostgreSQL server installation

**PostgreSQL Setup Example:**
```bash
# Install PostgreSQL
# Create database
createdb ai_costs

# Set environment variable
DATABASE_URL=postgresql://myuser:mypassword@localhost:5432/ai_costs
```

### Cache Backend

Choose where to store cached data (raw API responses, schemas, playground data, LLM extractions):

#### Database Backend (Default)
```bash
CACHE_BACKEND=database
```
- ✅ Better performance with database indexing
- ✅ No file system clutter
- ✅ Automatic cleanup when models are deleted
- ✅ Better for deployment (no file permissions issues)
- ✅ Transactional safety

#### File Backend
```bash
CACHE_BACKEND=file
CACHE_DIR=cache
```
- ✅ Easy to inspect cache data (JSON files)
- ✅ Can manually edit cached responses for testing
- ✅ Lower database size
- ⚠️ Creates many files in filesystem
- ⚠️ Manual cleanup needed

### Flask Configuration

```bash
FLASK_SECRET_KEY=your-secret-key-here
```
- Used for session management and security
- Generate a secure random key: `python -c "import secrets; print(secrets.token_hex(32))"`

## Complete Example

Create a `.env` file in the project root:

```bash
# PostgreSQL production setup
DATABASE_URL=postgresql://ai_costs_user:secure_password@localhost:5432/ai_costs
CACHE_BACKEND=database
FLASK_SECRET_KEY=8f7d4c9a2b1e5f3d7c6a9b8e4f2d1c3a5b7e9f4d2c6a8b1e3f5d7c9a2b4e6f8

# OR SQLite development setup
# DATABASE_URL=sqlite:///ai_costs.db
# CACHE_BACKEND=file
# CACHE_DIR=cache
# FLASK_SECRET_KEY=dev-secret-key-change-in-production
```

## Switching Backends

### Migrate Cache from File to Database

1. Stop the application
2. Update `.env`:
   ```bash
   CACHE_BACKEND=database
   ```
3. Restart the application
4. Re-fetch models to populate database cache
5. (Optional) Delete old cache files: `rm -rf cache/`

### Migrate Cache from Database to File

1. Stop the application
2. Update `.env`:
   ```bash
   CACHE_BACKEND=file
   CACHE_DIR=cache
   ```
3. Restart the application
4. Re-fetch models to populate file cache
5. (Optional) Clear database cache via UI: Cache Management → Clear All Cache

### Switch from SQLite to PostgreSQL

1. **Backup your data** (export models, settings)
2. Create PostgreSQL database
3. Update `DATABASE_URL` in `.env`
4. Run migrations: `python -c "from ai_cost_manager.database import init_db; init_db()"`
5. Re-add sources and re-fetch models

**Note:** No automatic migration tool is provided. Plan to re-fetch data when switching databases.

## Performance Considerations

### SQLite
- Good for: Single user, <10,000 models, development
- Limitations: Single writer, limited concurrency

### PostgreSQL
- Good for: Multiple users, >10,000 models, production
- Requirements: Separate PostgreSQL server
- Benefits: Better concurrency, advanced features

### Cache Backend

**Database cache** is recommended for:
- Production deployments
- Large number of models (>1000)
- Frequent cache access
- Clean filesystem

**File cache** is recommended for:
- Development/debugging
- Manual cache inspection
- Limited database resources
- Testing with modified responses

## Troubleshooting

### PostgreSQL Connection Issues
```bash
# Test connection
psql "postgresql://user:pass@localhost:5432/db_name"

# Common fixes:
# 1. Check PostgreSQL is running: systemctl status postgresql
# 2. Verify credentials in .env
# 3. Ensure database exists: createdb ai_costs
# 4. Check firewall/network access
```

### Cache Issues
```bash
# Clear all cache via Python
python -c "from ai_cost_manager.cache import cache_manager; cache_manager.clear_all_cache()"

# Check current backend
python -c "from ai_cost_manager.cache import CACHE_BACKEND; print(f'Using: {CACHE_BACKEND}')"
```

### Database Locked (SQLite)
- Ensure only one Flask instance is running
- Close all database connections before restart
- Consider switching to PostgreSQL for multi-user scenarios
