import requests, json, re
url='https://divar.ir/s/tehran/buy-residential?page=1'
headers={'User-Agent':'Mozilla/5.0'}
r=requests.get(url,headers=headers,timeout=10)
print('status',r.status_code)
print('len',len(r.text))
# find all script tags
scripts=re.findall(r'<script[^>]*>(.*?)</script>',r.text,re.S)
print('found',len(scripts),'scripts')
for i,s in enumerate(scripts[:10]):
    if s.strip().startswith('{'):
        try:
            data=json.loads(s.strip())
            print('script',i,'keys',list(data.keys()))
        except Exception as e:
            pass
# try to find any JSON that contains 'price'
for i,s in enumerate(scripts):
    if 'price' in s:
        try:
            data=json.loads(s.strip())
            print('price script',i,'keys',list(data.keys()))
        except Exception:
            pass
print('done')
