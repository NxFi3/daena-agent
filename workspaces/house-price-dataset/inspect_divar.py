import requests, json, re
url='https://divar.ir/s/tehran/buy-residential?page=1'
headers={'User-Agent':'Mozilla/5.0'}
resp=requests.get(url,headers=headers,timeout=10)
print('status',resp.status_code)
print('content-type',resp.headers.get('content-type'))
print('len',len(resp.text))
# find script tags with id="__NEXT_DATA__"
import re
m=re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', resp.text, re.S)
print('script found',bool(m))
if m:
    try:
        data=json.loads(m.group(1))
        print('keys',list(data.keys()))
        props=data.get('props',{})
        print('props keys',list(props.keys()))
        pageProps=props.get('pageProps',{})
        print('pageProps keys',list(pageProps.keys()))
        data_section=pageProps.get('data',{})
        print('data keys',list(data_section.keys()))
        items=data_section.get('items',[])
        print('items count',len(items))
    except Exception as e:
        print('json error',e)
else:
    # try embedded_json extractor logic: find first <script> with JSON
    m2=re.search(r'<script[^>]*>(\{.*?\})</script>', resp.text, re.S)
    if m2:
        try:
            data=json.loads(m2.group(1))
            print('embedded json found')
        except Exception as e:
            print('embedded json parse error',e)
        with open('raw_embedded.json','w',encoding='utf-8') as f:
            f.write(m2.group(1))
    else:
        print('no script tags found')
print('done')