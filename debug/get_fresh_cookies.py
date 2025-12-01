"""
Get fresh Runware cookies - Run this when your authentication expires
"""

import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                  GET FRESH RUNWARE COOKIES                           ║
╚══════════════════════════════════════════════════════════════════════╝

Your authentication tokens expire after ~1 hour. Follow these steps:

1. Open Chrome/Edge/Firefox
2. Go to: https://my.runware.ai/models/all
3. Log in if needed
4. Press F12 to open Developer Tools
5. Go to "Application" tab (Chrome/Edge) or "Storage" tab (Firefox)
6. Click "Cookies" → "https://my.runware.ai"
7. Find these 3 cookies and copy their VALUES:

   ┌─────────────────────────────────────────────────────────────┐
   │ BEARER                                                      │
   │ RUNWARE_APP-AUTH_TOKEN                                      │
   │ RUNWARE_APP-AUTH_REFRESH_TOKEN                             │
   └─────────────────────────────────────────────────────────────┘

8. Format them like this (one line):
   BEARER=<value>; RUNWARE_APP-AUTH_TOKEN=<value>; RUNWARE_APP-AUTH_REFRESH_TOKEN=<value>

9. Paste the cookie string below and press Enter:

""")

cookie_string = input("Cookie string: ").strip()

if not cookie_string or '=' not in cookie_string:
    print("\n❌ Invalid cookie string! Please try again.")
    sys.exit(1)

# Parse and save
from ai_cost_manager.database import get_session, init_db
from ai_cost_manager.models import AuthSettings
import json

def parse_cookies(cookie_str):
    """Parse cookie string"""
    cookies = []
    for part in cookie_str.split(';'):
        part = part.strip()
        if '=' in part:
            key, value = part.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            if key and value:
                domain = 'my.runware.ai' if 'RUNWARE_APP' in key else '.runware.ai'
                cookies.append({
                    'name': key,
                    'value': value,
                    'domain': domain,
                    'path': '/',
                    'secure': True,
                    'httpOnly': False,
                    'sameSite': 'Lax'
                })
    return cookies

try:
    cookies = parse_cookies(cookie_string)
    
    if len(cookies) < 2:
        print(f"\n⚠️  Only found {len(cookies)} cookies. Make sure you have BEARER and RUNWARE_APP-AUTH_TOKEN")
    
    init_db()
    session = get_session()
    
    existing = session.query(AuthSettings).filter_by(source_name='runware').first()
    
    if existing:
        # Use update() to avoid SQLAlchemy type issues
        session.query(AuthSettings).filter_by(source_name='runware').update({
            'cookies': json.dumps(cookies),
            'is_active': True
        })
        print(f"\n✅ Updated authentication with {len(cookies)} cookies")
    else:
        auth = AuthSettings(
            source_name='runware',
            cookies=json.dumps(cookies),
            is_active=True
        )
        session.add(auth)
        print(f"\n✅ Created authentication with {len(cookies)} cookies")
    
    session.commit()
    session.close()
    
    print("\n✅ Authentication saved successfully!")
    print("\nNow run: python test_runware.py")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
