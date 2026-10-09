import json, re, sys
from pathlib import Path
from urllib.request import urlopen

url='https://divar.ir/s/tehran/buy-residential?page=1'
# fetch page
import requests
r=requests.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=10)
text=r.text
m=re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', text, re.S)
if not m:
    print('no script')
    sys.exit(1)
js=json.loads(m.group(1))
items=js.get('props',{}).get('pageProps',{}).get('data',{}).get('items',[])
print('found',len(items))
Path('divar_page1_items.json').write_text(json.dumps(items,ensure_ascii=False,indent=2))
print('wrote')
