import requests, re, json
url='https://divar.ir/s/tehran/buy-residential?page=1'
r=requests.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=10)
print('status',r.status_code)
m=re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, re.S)
print('found',bool(m))
if m:
    data=json.loads(m.group(1))
    items=data.get('props',{}).get('pageProps',{}).get('data',{}).get('items',[])
    print('items',len(items))
    print(items[0].keys() if items else None)
else:
    print('no script')
