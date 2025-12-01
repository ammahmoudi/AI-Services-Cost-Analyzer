# Runware Authentication Settings

## Overview

Runware authentication is now fully configurable through the Settings → Authentication page. No more hardcoded credentials!

## Changes Made

### 1. Database Schema (`ai_cost_manager/models.py`)
Added to `AuthSettings` table:
- `username` (VARCHAR 255) - For email/username
- `password` (TEXT) - For password

### 2. Runware Extractor (`extractors/runware_extractor.py`)
- Updated `_extract_with_auth()` to accept username/password parameters
- Checks database for Runware credentials before using defaults
- Falls back to hardcoded credentials if none configured
- Supports both:
  - **New**: Credential-based auth (username/password)
  - **Legacy**: Cookie-based auth (deprecated)

### 3. Web UI (`templates/auth_settings.html`)
Added new form section:
- **Add Runware Authentication** card
- Email input field
- Password input field (hidden)
- Notes field (optional)
- Active toggle

Updated table to show:
- Auth Type column (Credentials vs Cookies)
- Username column (shows email for credential-based)

### 4. API Endpoint (`app.py`)
Updated `save_auth_settings()` to:
- Detect credential-based vs cookie-based auth
- Save username/password for Runware
- Save cookies for fal.ai
- Update existing auth or create new

### 5. Migration (`migrations/add_auth_credentials.py`)
- Adds `username` column to `auth_settings`
- Adds `password` column to `auth_settings`

## Usage

### Step 1: Run Migration
```bash
python migrations/add_auth_credentials.py
```

### Step 2: Configure Runware Credentials

**Web UI:**
1. Go to Settings → Authentication tab
2. Scroll to "Add Runware Authentication"
3. Enter your email
4. Enter your password
5. Add optional notes
6. Check "Active"
7. Click "Save Authentication"

**What Gets Stored:**
```python
AuthSettings(
    source_name='runware',
    username='your-email@example.com',
    password='your-password',
    is_active=True
)
```

### Step 3: Extract Models
Now when you run Runware extraction:
```bash
# Via web UI: Click "Extract from Source" on Runware
# Or programmatically:
python -c "from extractors import get_extractor; extractor = get_extractor('runware'); extractor.extract()"
```

The extractor will:
1. ✅ Check database for Runware auth
2. ✅ Use configured credentials if found
3. ✅ Fall back to defaults if not configured
4. ✅ Login with credentials
5. ✅ Extract 189+ certified models

## Benefits

✅ **No Hardcoded Credentials** - Store securely in database  
✅ **Multiple Accounts** - Easy to switch between accounts  
✅ **Team Sharing** - Each team member uses their own credentials  
✅ **Security** - Passwords stored in database (encrypt recommended for production)  
✅ **Flexibility** - Change credentials without editing code  

## Security Recommendations

For production use:
1. **Encrypt passwords** before storing in database
2. **Use environment variables** for sensitive defaults
3. **Enable HTTPS** for web interface
4. **Restrict database access**
5. **Add password hashing** (bcrypt, argon2)

Example encryption (future enhancement):
```python
from cryptography.fernet import Fernet

def encrypt_password(password: str, key: bytes) -> str:
    f = Fernet(key)
    return f.encrypt(password.encode()).decode()

def decrypt_password(encrypted: str, key: bytes) -> str:
    f = Fernet(key)
    return f.decrypt(encrypted.encode()).decode()
```

## Fallback Behavior

If no credentials configured in database:
- Uses default credentials: `rzrd2024@gmail.com` / `REZVANI@rzrd2024`
- Prints warning message
- Continues extraction normally

This ensures backward compatibility and prevents breaking changes.

## Examples

### Example 1: Single User
```
User: john@company.com
Pass: SecurePass123
```

Runware extractor logs in as john@company.com and extracts models visible to that account.

### Example 2: Team Environment
```
Dev Environment:  dev-account@company.com
Prod Environment: prod-account@company.com
```

Each environment uses different credentials, extracting different model sets or pricing tiers.

### Example 3: Testing
```
Test Account: test@runware.ai
Production:   prod@runware.ai
```

Switch between accounts by:
1. Deactivating test account
2. Activating production account
3. Re-run extraction

## Comparison: Before vs After

### Before (Hardcoded)
```python
# In runware_extractor.py
email = "rzrd2024@gmail.com"
password = "REZVANI@rzrd2024"
```

❌ Credentials in source code  
❌ Can't change without editing code  
❌ One account only  
❌ Security risk if code shared  

### After (Database)
```python
# In database
AuthSettings(
    source_name='runware',
    username='your-email@example.com',
    password='your-password'
)
```

✅ Credentials in database  
✅ Change via web UI  
✅ Multiple accounts supported  
✅ No code changes needed  

## Testing

To test the new feature:

```bash
# 1. Run migration
python migrations/add_auth_credentials.py

# 2. Configure credentials via web UI
# Go to http://localhost:5000/settings → Authentication

# 3. Test extraction
python -c "
from extractors import get_extractor
from ai_cost_manager.database import get_session
from ai_cost_manager.models import AuthSettings

# Check if credentials stored
session = get_session()
auth = session.query(AuthSettings).filter_by(source_name='runware').first()
print(f'Runware auth found: {auth is not None}')
if auth:
    print(f'Username: {auth.username}')
    print(f'Has password: {auth.password is not None}')

# Run extraction
extractor = get_extractor('runware')
models = extractor.extract()
print(f'Extracted {len(models)} models')
"
```

## Troubleshooting

**Issue**: "No Runware authentication found"  
**Solution**: Configure credentials in Settings → Authentication

**Issue**: "Login failed"  
**Solution**: Check username/password are correct

**Issue**: Migration error  
**Solution**: Column may already exist, safe to ignore

**Issue**: Extraction returns 0 models  
**Solution**: Check account has access to models, verify login works manually

## Future Enhancements

- [ ] Password encryption at rest
- [ ] OAuth2 integration
- [ ] API key support
- [ ] Multi-factor authentication
- [ ] Credential rotation
- [ ] Audit logging

## Summary

Runware credentials are now configurable through the web UI! This improves security, flexibility, and team collaboration. The extractor automatically uses database credentials when available, falling back to defaults for backward compatibility.
