import json, re, sys
from pathlib import Path

html_path = Path('page1.html')
html = html_path.read_text(encoding='utf-8')
# find window.__PRELOADED_STATE__ = {...};
match = re.search(r'window\.__PRELOADED_STATE__\s*=\s*(\{.*?\});', html, re.S)
if not match:
    print('No PRELOADED_STATE found')
    sys.exit(1)
state_json = match.group(1)
state = json.loads(state_json)
# navigate to listWidgets
widgets = state.get('nb', {}).get('listWidgets', [])
posts = []
for w in widgets:
    if w.get('data', {}).get('widgetType') == 'POST_ROW':
        posts.append(w['data'])
print('Found', len(posts), 'posts')
# write to output
Path('posts.json').write_text(json.dumps(posts, ensure_ascii=False, indent=2), encoding='utf-8')
print('Wrote posts.json')
