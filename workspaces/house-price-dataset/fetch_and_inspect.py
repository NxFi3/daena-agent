import requests, json, re
url='https://divar.ir/s/tehran/buy-residential?page=1'
headers={'User-Agent':'Mozilla/5.0'}
resp=requests.get(url,headers=headers,timeout=10)
print('status',resp.status_code)
print('content-type',resp.headers.get('content-type'))
print('len',len(resp.text))
# save first 2000 chars
open('raw_page.txt','w',encoding='utf-8').write(resp.text[:2000])
print('saved raw_page.txt')
# find script tags
scripts=re.findall(r'<script[^>]*>(.*?)</script>',resp.text,re.S)
print('found',len(scripts),'scripts')
for i,s in enumerate(scripts[:5]):
    if s.strip().startswith('{'):
        try:
            data=json.loads(s.strip())
            print('script',i,'keys',list(data.keys()))
        except Exception as e:
            pass
# look for price field
for i,s in enumerate(scripts):
    if 'price' in s:
        try:
            data=json.loads(s.strip())
            print('price script',i,'keys',list(data.keys()))
        except Exception:
            pass
print('done')
