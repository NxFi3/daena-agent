import json, re, sys
from pathlib import Path
url='https://divar.ir/s/tehran/buy-residential?page=1'
# use requests
import requests
r=requests.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=10)
text=r.text
# find script tag
m=re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', text, re.S)
if not m:
    print('no script')
    sys.exit(1)
data=json.loads(m.group(1))
items=data.get('props',{}).get('pageProps',{}).get('data',{}).get('items',[])
print('items',len(items))
# write to json
Path('divar_page1.json').write_text(json.dumps(items,ensure_ascii=False,indent=2))
print('wrote')
