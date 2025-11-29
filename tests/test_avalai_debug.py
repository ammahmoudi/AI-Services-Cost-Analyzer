"""
Debug AvalAI HTML Fetching

Test to see what HTML structure we're getting from the pricing page.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup


def test_fetch_html():
    """Fetch and analyze the HTML structure"""
    url = "https://docs.avalai.ir/fa/pricing.md"
    
    print(f"Fetching {url}...")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until='networkidle', timeout=60000)
        html = page.content()
        browser.close()
    
    print(f"✓ Fetched {len(html)} bytes of HTML\n")
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    
    # Find all tables
    tables = soup.find_all('table')
    print(f"Found {len(tables)} tables\n")
    
    # Analyze first table
    if tables:
        table = tables[0]
        
        # Get preceding heading
        prev_heading = table.find_previous(['h1', 'h2', 'h3', 'h4'])
        if prev_heading:
            print(f"First table heading: {prev_heading.get_text(strip=True)}")
        
        # Get headers
        header_row = table.find('thead')
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
            print(f"Headers: {headers}")
        
        # Get first data row
        tbody = table.find('tbody')
        if tbody:
            first_row = tbody.find('tr')
            if first_row:
                cells = [td.get_text(strip=True) for td in first_row.find_all('td')]
                print(f"First row ({len(cells)} cells): {cells[:3]}...")
                
                # Check for code tags in first cell
                first_cell = first_row.find('td')
                if first_cell:
                    code_tags = first_cell.find_all('code')
                    print(f"Code tags in first cell: {len(code_tags)}")
                    if code_tags:
                        print(f"First code tag: {code_tags[0].get_text(strip=True)}")
    
    # Save HTML sample for inspection
    sample_path = Path(__file__).parent / "avalai_html_sample.html"
    with open(sample_path, 'w', encoding='utf-8') as f:
        # Save first 50,000 chars
        f.write(html[:50000])
    print(f"\n✓ Saved HTML sample to {sample_path}")


if __name__ == '__main__':
    test_fetch_html()
