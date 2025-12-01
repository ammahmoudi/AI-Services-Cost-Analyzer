"""
Helper script to set up Runware authentication cookies
"""

from ai_cost_manager.database import get_session, init_db
from ai_cost_manager.models import AuthSettings
import json

# Your cookie string from browser
COOKIE_STRING = """
BEARER=eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJpYXQiOjE3NjQ1NzUwOTMsImV4cCI6MTc2NDU3ODY5Mywicm9sZXMiOlsiSVNfQVVUSEVOVElDQVRFRF9GVUxMWSJdLCJ1c2VybmFtZSI6InJ6cmQyMDI0QGdtYWlsLmNvbSIsInVzZXJJZCI6IjFmYzJjOTY5LWY4NjUtNDE5ZS1iNDczLThkNTkwMWE4MzVlNiJ9.lCMNE7jcz9QeuSoXWAZd24JGLvpGkPrxUa40jYKEf0pFA5tv90OpslR7rsxHos0mDc4840pPDjNuPiSK1ramExtITjcJ4RNnPU-NmUDhOXREwhuDmldV23bg7yXguG24CkcgzL5tKPvTOrueyAgLxQpFs_JYb9e4tfjQOJh0mlIpbWAxAkGzbPDnVXiteeOAUbi7rwBKx1PioRoLfpoJr977hDh32wfInRDIguuEZ8M3iMFQ5rphevYMeMfTY0QdKwgUML2q6q4J5GvmjOcpMfesMfk21Rg0OvEHmtViG8T0_y08DqXUUKHyVVUEGEZe3RVvV5fRV80c_CbR9uz86MXOU6uOPkZRmq-avdDLp4iECdABbUqgwmmbChHsu6y3xUZJ8epZGahXxRBCou1pS1a3cGeXScLYp113YBt1dkaHyxBKyszPz4J6an5rr7cPKdTISmGp-ZaIKHBkeNQvi5CRlSvvv6atCNiqDNqL29q_F1EKauif8NDg9sMm_W2PxN3agaIJhfX5KGhkYWSDMrfrbcTFVOBLz3LwhggXaH5itChEG3IwskT6uyVRfQc4LoSCxhdMdZeGPsqe58eDN0Xo5a2PnOnqTEG6KtvJA5vFCYR1oq7Z93Vd3BGWcDcuNgbbs_RC9mv_AMEwJrP8CdzHKq8llMcSSZwXqxJT7cg; RUNWARE_APP-AUTH_TOKEN=eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJpYXQiOjE3NjQ1NzUwOTMsImV4cCI6MTc2NDU3ODY5Mywicm9sZXMiOlsiSVNfQVVUSEVOVElDQVRFRF9GVUxMWSJdLCJ1c2VybmFtZSI6InJ6cmQyMDI0QGdtYWlsLmNvbSIsInVzZXJJZCI6IjFmYzJjOTY5LWY4NjUtNDE5ZS1iNDczLThkNTkwMWE4MzVlNiJ9.lCMNE7jcz9QeuSoXWAZd24JGLvpGkPrxUa40jYKEf0pFA5tv90OpslR7rsxHos0mDc4840pPDjNuPiSK1ramExtITjcJ4RNnPU-NmUDhOXREwhuDmldV23bg7yXguG24CkcgzL5tKPvTOrueyAgLxQpFs_JYb9e4tfjQOJh0mlIpbWAxAkGzbPDnVXiteeOAUbi7rwBKx1PioRoLfpoJr977hDh32wfInRDIguuEZ8M3iMFQ5rphevYMeMfTY0QdKwgUML2q6q4J5GvmjOcpMfesMfk21Rg0OvEHmtViG8T0_y08DqXUUKHyVVUEGEZe3RVvV5fRV80c_CbR9uz86MXOU6uOPkZRmq-avdDLp4iECdABbUqgwmmbChHsu6y3xUZJ8epZGahXxRBCou1pS1a3cGeXScLYp113YBt1dkaHyxBKyszPz4J6an5rr7cPKdTISmGp-ZaIKHBkeNQvi5CRlSvvv6atCNiqDNqL29q_F1EKauif8NDg9sMm_W2PxN3agaIJhfX5KGhkYWSDMrfrbcTFVOBLz3LwhggXaH5itChEG3IwskT6uyVRfQc4LoSCxhdMdZeGPsqe58eDN0Xo5a2PnOnqTEG6KtvJA5vFCYR1oq7Z93Vd3BGWcDcuNgbbs_RC9mv_AMEwJrP8CdzHKq8llMcSSZwXqxJT7cg; RUNWARE_APP-AUTH_REFRESH_TOKEN=02f7697ab36e09e90de8e974e15a9d94cdaaf6452cd628840a8e520b5846e760d4c58ec4bd31f67ac27fe401092fe9d23b44385f3fbd46b3cd4d5c2fbbbcd06e
"""

def parse_cookie_string(cookie_str):
    """Parse cookie string into structured format for Playwright"""
    cookies = []
    
    for line in cookie_str.strip().split(';'):
        line = line.strip()
        if not line or '=' not in line:
            continue
            
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        if key and value:
            # Determine correct domain based on cookie name
            if 'RUNWARE_APP' in key:
                domain = 'my.runware.ai'
            else:
                domain = '.runware.ai'
            
            cookies.append({
                'name': key,
                'value': value,
                'domain': domain,
                'path': '/',
                'secure': True,
                'httpOnly': False,  # Changed to False for browser cookies
                'sameSite': 'Lax'
            })
    
    return cookies

def setup_auth():
    """Set up Runware authentication in database"""
    
    if 'PASTE YOUR COOKIES HERE' in COOKIE_STRING:
        print("❌ Please edit this script and paste your cookies first!")
        print("\nTo get cookies:")
        print("1. Open https://my.runware.ai in your browser")
        print("2. Open Developer Tools (F12)")
        print("3. Go to Application/Storage → Cookies → https://my.runware.ai")
        print("4. Copy all cookie values in format: name1=value1; name2=value2; ...")
        print("5. Paste them into this script where it says 'PASTE YOUR COOKIES HERE'")
        return
    
    init_db()
    session = get_session()
    
    try:
        # Parse cookies
        cookies = parse_cookie_string(COOKIE_STRING)
        
        if not cookies:
            print("❌ No valid cookies found!")
            return
        
        print(f"✅ Parsed {len(cookies)} cookies")
        
        # Check if auth already exists
        existing = session.query(AuthSettings).filter_by(source_name='runware').first()
        
        if existing:
            print(f"Found existing Runware auth (active={existing.is_active})")
            existing.cookies = json.dumps(cookies)
            existing.is_active = True
            print("✅ Updated existing authentication")
        else:
            # Create new auth
            auth = AuthSettings(
                source_name='runware',
                cookies=json.dumps(cookies),
                is_active=True
            )
            session.add(auth)
            print("✅ Created new authentication")
        
        session.commit()
        print("\n✅ Runware authentication configured successfully!")
        print("You can now run: python test_runware.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == '__main__':
    setup_auth()
