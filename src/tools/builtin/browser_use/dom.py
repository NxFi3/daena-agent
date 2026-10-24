"""DOM observation only. Actions use Playwright's native actionability checks."""

COLLECT = r"""(limit) => {
  const nodes = [], records = [];
  const tidy = s => (s || '').replace(/\s+/g, ' ').trim().slice(0, 220);
  const visible = e => {
    const r = e.getBoundingClientRect(), s = getComputedStyle(e);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
  };
  const role = e => e.getAttribute('role') || ({BUTTON:'button', A:'link',
    TEXTAREA:'textbox', SELECT:'combobox', SUMMARY:'button', OPTION:'option'}[e.tagName]) ||
    (e.tagName === 'INPUT' ? ({checkbox:'checkbox', radio:'radio', range:'slider',
      button:'button', submit:'button', reset:'button', file:'file'}[e.type] || 'textbox') :
      e.isContentEditable ? 'textbox' : 'control');
  const name = e => {
    const labelled = (e.getAttribute('aria-labelledby') || '').split(/\s+/)
      .map(id => e.getRootNode().getElementById?.(id)?.textContent || '').join(' ').trim();
    return tidy(labelled || e.getAttribute('aria-label') ||
      Array.from(e.labels || []).map(l=>l.innerText).join(' ') ||
      e.getAttribute('alt') || e.innerText || e.getAttribute('placeholder') ||
      e.getAttribute('title') || ((e.type === 'submit' || e.type === 'button') ? e.value : ''));
  };
  let total = 0;
  function walk(root) {
    for (const e of root.querySelectorAll('*')) {
      if (e.shadowRoot) walk(e.shadowRoot);
      if (!e.matches('a[href],button,input,textarea,select,summary,[role],[contenteditable=true],[tabindex]')) continue;
      const shown = visible(e);
      if (!shown && !(e.tagName === 'INPUT' && e.type === 'file')) continue;
      total++;
      if (nodes.length >= limit) continue;
      nodes.push(e);
      const record = {role:role(e), name:name(e), tag:e.tagName.toLowerCase(),
        visible:shown, disabled:!!e.disabled || e.getAttribute('aria-disabled') === 'true',
        focused:e.getRootNode().activeElement === e};
      if (shown) {
        const r = e.getBoundingClientRect();
        const x = Math.max(0, r.left) + (Math.min(innerWidth, r.right) - Math.max(0, r.left)) / 2;
        const y = Math.max(0, r.top) + (Math.min(innerHeight, r.bottom) - Math.max(0, r.top)) / 2;
        const root = e.getRootNode();
        const hit = root.elementFromPoint?.(x, y);
        record.in_viewport = r.bottom > 0 && r.right > 0 && r.top < innerHeight && r.left < innerWidth;
        if (record.in_viewport && hit) record.occluded = hit !== e && !e.contains(hit);
      }
      if (e.matches('input,textarea,select')) record.value = e.type === 'password' ? '[redacted]' : tidy(e.value);
      if (e.matches('input[type=checkbox],input[type=radio]')) record.checked = e.checked;
      if (e.placeholder) record.placeholder = tidy(e.placeholder);
      if (e.href) record.href = e.href.slice(0, 500);
      if (e.tagName === 'SELECT') record.options = Array.from(e.options).slice(0,40).map(o=>({value:o.value,label:tidy(o.label),selected:o.selected}));
      records.push(record);
    }
  }
  walk(document);
  return {nodes, records, total, text:(document.body?.innerText || '').slice(0, 6000),
    ready_state:document.readyState};
}"""
