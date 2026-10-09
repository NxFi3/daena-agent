import requests, json, os
from pathlib import Path

def fetch_page(page):
    url = f'https://divar.ir/s/tehran/buy-residential?page={page}'
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    # extract JSON from script tag
    import re
    m = re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, re.S)
    if not m:
        return []
    data = json.loads(m.group(1))
    listings = data.get('props',{}).get('pageProps',{}).get('data',{}).get('items',[])
    return listings

def main():
    all=[]
    page=1
    while True:
        items=fetch_page(page)
        if not items:
            break
        all.extend(items)
        print(f'Page {page} fetched {len(items)} items, total {len(all)}')
        page+=1
    out=Path('divar_listings.json')
    out.write_text(json.dumps(all,ensure_ascii=False,indent=2))
    print('Saved',out)

if __name__=='__main__':
    main()
