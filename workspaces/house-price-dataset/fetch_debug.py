import requests, json, re, sys
url='https://divar.ir/s/tehran/buy-residential?page=1'
headers={'User-Agent':'Mozilla/5.0'}
r=requests.get(url,headers=headers,timeout=10)
print('status',r.status_code)
print('content-type',r.headers.get('content-type'))
print('len',len(r.text))
open('raw_page1.txt','w',encoding='utf8').write(r.text)
print('saved raw_page1.txt')
# try to find listings pattern
pattern=r'\bTitle:.*?\n'
print('sample',re.findall(pattern,r.text)[:5])
