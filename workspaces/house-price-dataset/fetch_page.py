import requests, json, re
url='https://divar.ir/s/tehran/buy-residential?page=1'
headers={'User-Agent':'Mozilla/5.0'}
r=requests.get(url,headers=headers,timeout=10)
print('status',r.status_code)
print('content-type',r.headers.get('content-type'))
print('len',len(r.text))
open('raw_page1.txt','w',encoding='utf8').write(r.text)
print('saved raw_page1.txt')
# find script tag
m=re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, re.S)
print('found script', bool(m))
if m:
    try:
        data=json.loads(m.group(1))
        items=data.get('props',{}).get('pageProps',{}).get('data',{}).get('items',[])
        print('items count',len(items))
    except Exception as e:
        print('json error',e)
else:
    print('no script tag')
