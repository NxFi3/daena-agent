import requests, re, json
url='https://divar.ir/s/tehran/buy-residential?page=1'
headers={'User-Agent':'Mozilla/5.0'}
r=requests.get(url,headers=headers,timeout=10)
print('status',r.status_code)
print('content-type',r.headers.get('content-type'))
print('len',len(r.text))
# save first 2000 chars
open('first2000.txt','w',encoding='utf-8').write(r.text[:2000])
print('saved first2000.txt')
# search for __NEXT_DATA__
m=re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, re.S)
print('found',bool(m))
if m:
    try:
        data=json.loads(m.group(1))
        print('items',len(data.get('props',{}).get('pageProps',{}).get('data',{}).get('items',[])))
    except Exception as e:
        print('json error',e)
else:
    print('no script tag')
