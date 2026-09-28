'use strict';

/* ---------- helpers ---------- */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v; // only ever given pre-escaped or static markup
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

const escapeHtml = (s) =>
  s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const AR_DIGITS = '٠١٢٣٤٥٦٧٨٩';
const toWestern = (s) => Number(String(s).replace(/[٠-٩]/g, (d) => AR_DIGITS.indexOf(d)));
// Same pattern as rag/llm/citations.py, including gpt-oss's full-width brackets.
const CITE_RE = /[[【［]\s*Art(?:icle)?s?\.?\s*([0-9٠-٩]+)\s*[\]】］]/gi;

function isArabic(s) {
  const letters = [...s].filter((ch) => /\p{L}/u.test(ch));
  if (!letters.length) return false;
  return letters.filter((ch) => /[؀-ۿ]/.test(ch)).length / letters.length >= 0.5;
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), 4000);
}

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

const ICON = {
  up: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 10v11H4V10h3zm0 0 4-8a3 3 0 0 1 3 3v4h5.5a2 2 0 0 1 2 2.3l-1.4 8A2 2 0 0 1 18.1 21H7"/></svg>',
  down: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><g transform="rotate(180 12 12)"><path d="M7 10v11H4V10h3zm0 0 4-8a3 3 0 0 1 3 3v4h5.5a2 2 0 0 1 2 2.3l-1.4 8A2 2 0 0 1 18.1 21H7"/></g></svg>',
};

/* ---------- state ---------- */

const state = {
  config: null,
  articles: [],
  byNumber: new Map(),
  articleSource: null,
  articleNote: '',
  selected: new Set(),
  stages: [],
  stageById: new Map(),
  backend: 'bedrock',
  feedback: [],
  feedbackFilter: 'all',
  retrieve: false, // true once the index exists: each question searches it for its context
  topK: 5,
};

const EXAMPLES = [
  { tag: 'Arabic', q: 'ما حكم هبة الأموال المستقبلة؟', note: 'Expected: void, Article 492.' },
  { tag: 'English', q: 'How does the Civil Code define a partnership?', note: 'Expected: Article 505.' },
  { tag: 'Reasoning', q: 'Can a husband take back a gift he gave to his wife?', note: 'Expected: no, Article 502(d).' },
  { tag: 'Arabic · reasoning', q: 'هل يجوز للواهب الرجوع في الهبة إذا رُزق بولد بعدها؟', note: 'Expected: yes, an acceptable excuse under Article 501(c).' },
  { tag: 'Repealed', q: 'What does Article 60 say?', note: 'Expected: says Article 60 was repealed. Adds Art. 60 to the context.', ensure: [60] },
  { tag: 'Out of scope', q: 'What is the penalty for theft?', note: 'Expected: declines, since no article covers it.' },
];

const FEEDBACK_TAGS = [
  'Wrong conclusion', 'Missing citation', 'Irrelevant citation', 'Wrong language',
  'Should have declined', 'Declined wrongly', 'Too long', 'Formatting',
];

const STATE_LABEL = { working: 'Working', offline: 'Offline', planned: 'Planned' };
const GROUP_ORDER = ['Data', 'Retrieval', 'Generation', 'Serving', 'Observability', 'Evaluation', 'Optimization'];
const VIEWS = ['ask', 'compare', 'status', 'corpus', 'retrieval', 'evaluation', 'traces', 'feedback'];

const stageState = (id) => state.stageById.get(id)?.state ?? 'pending';
const stageDetail = (id) => state.stageById.get(id)?.detail ?? 'Checking…';

/* ---------- routing ---------- */

function showView(name) {
  if (!VIEWS.includes(name)) name = 'ask';
  closeDrawer();
  for (const v of VIEWS) $(`#view-${v}`).hidden = v !== name;
  $$('#nav a').forEach((a) => a.classList.toggle('active', a.dataset.view === name));
  const slot = $(`#view-${name} .context-slot`);
  if (slot) slot.append($('#context-picker'));
  if (name === 'status') refreshStatus();
  if (name === 'feedback') refreshFeedback();
}

/* ---------- rich text: minimal markdown + citation chips ---------- */

function inline(s, cites) {
  s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/`([^`]+)`/g, '<code>$1</code>');
  return s.replace(CITE_RE, (_, n) => {
    const num = toWestern(n);
    const cls = !cites ? '' : cites.invalid.includes(num) ? 'bad' : cites.valid.includes(num) ? 'ok' : '';
    return `<button type="button" class="cite ${cls}" data-article="${num}" dir="ltr">Art.&nbsp;${num}</button>`;
  });
}

function renderRich(text, cites) {
  const lines = escapeHtml(text).split('\n');
  let html = '';
  let list = null;
  let table = null;
  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
  const closeTable = () => {
    if (!table) return;
    const [head, ...rows] = table;
    const cells = (r) => r.replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
    html += '<table><thead><tr>' + cells(head).map((c) => `<th>${inline(c, cites)}</th>`).join('') + '</tr></thead><tbody>';
    for (const r of rows) html += '<tr>' + cells(r).map((c) => `<td>${inline(c, cites)}</td>`).join('') + '</tr>';
    html += '</tbody></table>';
    table = null;
  };
  for (const raw of lines) {
    const line = raw.trimEnd();
    let m;
    if (/^\s*\|.*\|\s*$/.test(line)) {
      closeList();
      if (/^\s*\|[\s:|-]+\|\s*$/.test(line)) continue; // header separator row
      (table ??= []).push(line.trim());
      continue;
    }
    closeTable();
    if (!line.trim()) { closeList(); continue; }
    if ((m = line.match(/^\s*[-*•]\s+(.*)$/))) {
      if (list !== 'ul') { closeList(); html += '<ul dir="auto">'; list = 'ul'; }
      html += `<li>${inline(m[1], cites)}</li>`;
      continue;
    }
    if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) {
      if (list !== 'ol') { closeList(); html += '<ol dir="auto">'; list = 'ol'; }
      html += `<li>${inline(m[1], cites)}</li>`;
      continue;
    }
    closeList();
    if ((m = line.match(/^#{1,4}\s+(.*)$/))) { html += `<p class="md-h" dir="auto">${inline(m[1], cites)}</p>`; continue; }
    html += `<p dir="auto">${inline(line, cites)}</p>`;
  }
  closeList();
  closeTable();
  return html;
}

/* ---------- context picker ---------- */

const articlePreview = (a) => (a.is_repealed ? 'Repealed article' : (a.text_en || a.text_ar || '').split('\n')[0]);
const topicGroup = (a) => (a.topic || 'Other').split(':')[0].trim();
const selectedNumbers = () => state.articles.map((a) => a.article_number).filter((n) => state.selected.has(n));

const LARGE_CORPUS = 60; // above this: nothing preselected, search instead of scrolling
const CONTEXT_WARN = 25; // more selected articles than this: slow, and may exceed vLLM's 8K window
const LIST_CAP = 200;
const matchesQuery = (a, q) => `${a.article_number} ${a.topic || ''} ${a.text_en || ''} ${a.text_ar || ''}`.toLowerCase().includes(q);
const listHint = (text) => h('div', { class: 'ctx-item' }, h('span'), h('span'), h('span', { class: 'ctx-topic' }, text));

function contextPool(q) {
  if (q) {
    const hits = state.articles.filter((a) => matchesQuery(a, q));
    return hits.sort((a, b) => (String(b.article_number) === q) - (String(a.article_number) === q));
  }
  return state.articles.length > LARGE_CORPUS ? state.articles.filter((a) => state.selected.has(a.article_number)) : state.articles;
}

function renderContextPicker() {
  const q = $('#ctx-filter').value.trim().toLowerCase();
  const list = $('#ctx-list');
  list.replaceChildren();
  const pool = contextPool(q);
  for (const a of pool.slice(0, LIST_CAP)) {
    const id = `ctx-${a.article_number}`;
    const cb = h('input', { type: 'checkbox', id, checked: state.selected.has(a.article_number) });
    cb.addEventListener('change', () => {
      if (cb.checked) state.selected.add(a.article_number);
      else state.selected.delete(a.article_number);
      updateCtxCount();
    });
    list.append(
      h('label', { class: 'ctx-item', for: id },
        cb,
        h('span', { class: 'ctx-num' }, `Art. ${a.article_number}`),
        h('span', { class: 'ctx-text' },
          h('div', { class: 'ctx-topic' }, a.topic || ''),
          h('div', { class: 'ctx-preview', dir: 'auto' }, articlePreview(a)))),
    );
  }
  if (pool.length > LIST_CAP) list.append(listHint(`${pool.length - LIST_CAP} more match: refine the search.`));
  if (!pool.length) {
    list.append(listHint(q ? 'No articles match.'
      : `Search ${state.articles.length.toLocaleString()} articles by number or word, then tick the ones the model may use.`));
  }
  updateCtxCount();
}

function renderCtxMode() {
  const canRetrieve = state.articleSource === 'corpus';
  const seg = (on, label, sub, onclick, disabled) => h('button', {
    class: 'seg', type: 'button', role: 'radio', 'aria-checked': String(on), disabled, onclick,
  }, h('span', { class: 'seg-top' }, label), h('span', { class: 'seg-model' }, sub));
  const k = h('select', { class: 'topk', 'aria-label': 'Articles to retrieve',
    onchange: (e) => { state.topK = Number(e.target.value); renderCtxMode(); } },
  [3, 5, 8, 10].map((n) => h('option', { value: n, selected: n === state.topK }, `top ${n}`)));
  $('#ctx-mode').replaceChildren(
    seg(state.retrieve, 'Retrieve from the index', canRetrieve ? 'search 1,149 articles per question' : 'build the corpus + index first',
      () => { state.retrieve = true; renderCtxMode(); }, !canRetrieve),
    seg(!state.retrieve, 'Pick by hand', 'choose the articles yourself', () => { state.retrieve = false; renderCtxMode(); }),
  );
  $('#ctx-auto').hidden = !state.retrieve;
  $('#ctx-manual').hidden = state.retrieve;
  $('#ctx-quick').hidden = state.retrieve;
  $('#ctx-auto').replaceChildren(
    h('div', { class: 'auto-row' }, h('strong', {}, 'Each question searches the index'), k),
    h('div', {}, 'The articles it finds (with scores) appear in the answer card under "Context sent". '
      + 'Questions that name an article ("المادة 801", "Article 60") look it up directly.'));
  updateCtxCount();
}

function updateCtxCount() {
  if (state.retrieve) {
    $('#ctx-count').textContent = `automatic · top ${state.topK} of ${state.articles.length.toLocaleString()}`;
    $('#ctx-count').classList.remove('warn-text');
    return;
  }
  const n = state.selected.size;
  const el = $('#ctx-count');
  el.textContent = `${n} of ${state.articles.length.toLocaleString()} selected`
    + (n > CONTEXT_WARN ? " · large context: slower, may exceed vLLM's 8K window" : '');
  el.classList.toggle('warn-text', n > CONTEXT_WARN);
}

function renderCtxQuick() {
  const set = (nums) => { state.selected = new Set(nums); renderContextPicker(); };
  const live = state.articles.filter((a) => !a.is_repealed);
  const chips = [h('button', { class: 'chip', type: 'button', onclick: () => set([]) }, 'None')];
  if (state.articles.length <= LARGE_CORPUS) {
    chips.unshift(h('button', { class: 'chip', type: 'button', onclick: () => set(live.map((a) => a.article_number)) }, 'All'));
    const groups = [...new Set(live.map(topicGroup))];
    if (groups.length <= 12) {
      chips.push(...groups.map((g) => h('button', { class: 'chip', type: 'button', onclick: () => set(live.filter((a) => topicGroup(a) === g).map((a) => a.article_number)) }, g)));
    }
  } else {
    chips.push(h('button', {
      class: 'chip', type: 'button', title: 'Adds up to 20 live articles matching the search box',
      onclick: () => {
        const q = $('#ctx-filter').value.trim().toLowerCase();
        if (!q) { toast('Type a search first: an article number or a word.'); return; }
        const add = contextPool(q).filter((a) => !a.is_repealed).slice(0, 20).map((a) => a.article_number);
        set([...state.selected, ...add]);
      },
    }, 'Add matches (up to 20)'));
  }
  $('#ctx-quick').replaceChildren(...chips);
}

/* ---------- backends and examples ---------- */

function renderBackends() {
  if (!state.config) return;
  const seg = $('#ask-backend');
  seg.replaceChildren();
  for (const [key, b] of Object.entries(state.config.backends)) {
    const st = stageState(`llm_${key}`);
    seg.append(
      h('button', {
        class: 'seg', type: 'button', role: 'radio', 'aria-checked': String(state.backend === key),
        title: stageDetail(`llm_${key}`), onclick: () => { state.backend = key; renderBackends(); },
      },
        h('span', { class: 'seg-top' }, h('span', { class: `dot ${st}` }), b.label, st === 'offline' ? h('span', { class: 'badge offline' }, 'offline') : null),
        h('span', { class: 'seg-model' }, b.model)),
    );
  }
  const pills = $('#backend-pills');
  pills.replaceChildren();
  for (const [key, b] of Object.entries(state.config.backends)) {
    const st = stageState(`llm_${key}`);
    const label = st === 'working' ? 'online' : st === 'offline' ? 'offline' : 'checking…';
    pills.append(h('div', { class: 'backend-pill', title: stageDetail(`llm_${key}`) }, h('span', { class: `dot ${st}` }), h('strong', {}, b.label), label));
  }
}

function renderExamples() {
  for (const box of $$('.examples')) {
    const target = $(`#${box.dataset.for}`);
    box.replaceChildren(
      ...EXAMPLES.map((ex) =>
        h('button', {
          class: 'example', type: 'button', title: ex.note,
          onclick: () => {
            target.value = ex.q;
            target.focus();
            const missing = (ex.ensure || []).filter((n) => !state.selected.has(n));
            if (missing.length) {
              missing.forEach((n) => state.selected.add(n));
              renderContextPicker();
              toast(`Added ${missing.map((n) => `Art. ${n}`).join(', ')} to the context.`);
            }
          },
        }, h('span', { class: 'ex-tag' }, ex.tag), h('span', { class: 'ex-q', dir: 'auto' }, ex.q))),
    );
  }
}

/* ---------- answer cards ---------- */

function citationCheck(c) {
  if (c.invalid.length) {
    return h('div', { class: 'check bad' },
      `✗ Cites ${c.invalid.map((n) => `Art. ${n}`).join(', ')}, which ${c.invalid.length === 1 ? 'was' : 'were'} not in the context: hallucinated citation.`);
  }
  if (c.cited.length === 1) return h('div', { class: 'check ok' }, '✓ The citation comes from the context.');
  if (c.cited.length) return h('div', { class: 'check ok' }, `✓ All ${c.cited.length} citations come from the context.`);
  return h('div', { class: 'check neutral' }, "No citations. Expected only when the articles don't answer the question.");
}

function renderMetrics(node, m) {
  node.replaceChildren(
    h('span', {}, 'Latency ', h('b', {}, `${m.latency_s.toFixed(2)} s`)),
    h('span', {}, 'Tokens ', h('b', {}, `${m.input_tokens} in → ${m.output_tokens} out`)),
    h('span', {}, 'Stop ', h('b', {}, m.stop_reason || '—')),
  );
}

function feedbackBox(result) {
  let rating = null;
  const tags = new Set();
  const upBtn = h('button', { class: 'fb-btn up', type: 'button', 'aria-pressed': 'false', html: `${ICON.up}<span>Right</span>` });
  const downBtn = h('button', { class: 'fb-btn down', type: 'button', 'aria-pressed': 'false', html: `${ICON.down}<span>Wrong</span>` });
  const tagRow = h('div', { class: 'fb-tags' },
    FEEDBACK_TAGS.map((t) => {
      const chip = h('button', { class: 'chip', type: 'button' }, t);
      chip.onclick = () => { if (tags.has(t)) tags.delete(t); else tags.add(t); chip.classList.toggle('active'); };
      return chip;
    }));
  const comment = h('textarea', { dir: 'auto', rows: 2, placeholder: 'Optional: what was right or wrong? e.g. correct article, wrong conclusion' });
  const save = h('button', { class: 'btn primary small', type: 'button' }, 'Save feedback');
  const saved = h('span', { class: 'fb-saved', hidden: true }, '✓ Saved to the feedback log');
  const more = h('div', { class: 'fb-more', hidden: true }, tagRow, comment, h('div', { class: 'fb-row' }, save, saved));

  const choose = (r) => {
    rating = r;
    upBtn.setAttribute('aria-pressed', String(r === 'up'));
    downBtn.setAttribute('aria-pressed', String(r === 'down'));
    more.hidden = false;
    tagRow.hidden = r === 'up';
  };
  upBtn.onclick = () => choose('up');
  downBtn.onclick = () => choose('down');
  save.onclick = async () => {
    const d = result.done;
    save.disabled = true;
    try {
      await api('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rating,
          tags: rating === 'down' ? [...tags] : [],
          comment: comment.value.trim(),
          question: result.question,
          backend: result.backend,
          model: d.metrics.model,
          answer: result.text,
          article_numbers: result.articleNumbers,
          cited: d.citations.cited,
          invalid: d.citations.invalid,
          latency_s: d.metrics.latency_s,
          view: result.view,
        }),
      });
      saved.hidden = false;
      [upBtn, downBtn, comment, ...tagRow.children].forEach((n) => (n.disabled = true));
      refreshFeedback();
    } catch (e) {
      save.disabled = false;
      toast(`Could not save feedback: ${e.message}`);
    }
  };
  return h('div', { class: 'feedback' }, h('div', { class: 'fb-row' }, h('span', { class: 'fb-label' }, 'Was this answer right?'), upBtn, downBtn), more);
}

function createAnswerCard({ backend, question, articleNumbers, view, showQuestion = true, retrieving = false }) {
  const b = state.config.backends[backend];
  const contextDetails = h('details', { class: 'context-sent', open: retrieving });
  const badge = h('span', { class: 'badge neutral' }, 'Waiting');
  const body = h('div', { class: 'answer-body' }, h('div', { class: 'thinking' }, h('span', { class: 'spinner' }), 'Waiting for the first tokens…'));
  const checkSlot = h('div');
  const metrics = h('div', { class: 'metrics', hidden: true });
  const feedbackSlot = h('div');
  const card = h('article', { class: 'answer-card', dataset: { context: articleNumbers.join(',') } },
    h('div', { class: 'answer-head' },
      h('div', { class: 'answer-who' }, h('strong', {}, b.label), h('span', { class: 'answer-model' }, b.model)),
      badge),
    showQuestion ? h('div', { class: 'answer-q', dir: 'auto' }, question) : null,
    body, checkSlot, metrics, feedbackSlot,
    contextDetails);

  function renderContext(numbers, hits, meta) {
    contextDetails.replaceChildren(
      h('summary', {}, retrieving && !hits ? 'Searching the index…' : `Context sent: ${plural(numbers.length, 'article')}`
        + (meta ? ` · retrieved in ${meta.latency_s.toFixed(2)} s on ${meta.device}` : '')),
      h('div', { class: 'ctx-chips' },
        numbers.length
          ? (hits || numbers.map((n) => ({ article_number: n }))).map((x) => h('button', {
            class: 'cite', type: 'button', dataset: { article: x.article_number },
            title: x.by_number ? 'named in the question' : x.score != null ? `similarity ${x.score}` : '',
          }, `Art. ${x.article_number}`, x.by_number ? ' · by number' : x.score != null ? ` · ${x.score.toFixed(2)}` : ''))
          : h('span', { class: 'card-sub' }, retrieving && !hits ? '' : 'None: the model saw no articles.')));
  }

  renderContext(articleNumbers, null, null);
  let text = '';
  const result = { backend, question, articleNumbers, view, text: '', done: null };
  const setBadge = (cls, label) => { badge.className = `badge ${cls}`; badge.textContent = label; };
  const renderText = (cites, streaming) => {
    body.className = `answer-body${isArabic(text) ? ' is-ar' : ''}`;
    body.innerHTML = renderRich(text, cites);
    if (streaming) (body.lastElementChild || body).classList.add('caret');
  };
  return {
    card,
    result,
    finished: false,
    start() { setBadge('partial', 'Streaming'); },
    retrieved(evt) {
      const numbers = evt.hits.filter((x) => x.in_corpus).map((x) => x.article_number);
      result.articleNumbers = numbers;
      card.dataset.context = numbers.join(',');
      renderContext(numbers, evt.hits.filter((x) => x.in_corpus), evt);
    },
    delta(chunk) { text += chunk; renderText(null, true); },
    done(evt) {
      this.finished = true;
      text = evt.text;
      result.text = text;
      result.done = evt;
      renderText(evt.citations, false);
      setBadge('working', 'Done');
      checkSlot.replaceChildren(citationCheck(evt.citations));
      renderMetrics(metrics, evt.metrics);
      metrics.hidden = false;
      feedbackSlot.replaceChildren(feedbackBox(result));
    },
    error(msg) {
      this.finished = true;
      setBadge('offline', 'Error');
      if (text) renderText(null, false); else body.replaceChildren();
      checkSlot.replaceChildren(h('div', { class: 'error-box' }, msg));
    },
    stopped() {
      this.finished = true;
      setBadge('neutral', 'Stopped');
      if (text) renderText(null, false); else body.replaceChildren(h('div', { class: 'thinking' }, 'Stopped before any output.'));
    },
  };
}

async function streamAsk(payload, ui, signal) {
  let res;
  try {
    res = await fetch('/api/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal });
  } catch (e) {
    if (e.name === 'AbortError') ui.stopped(); else ui.error(`Could not reach the console server: ${e.message}`);
    return;
  }
  if (!res.ok) { ui.error(`Request failed (${res.status}): ${await res.text()}`); return; }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i);
        buf = buf.slice(i + 1);
        if (!line.trim()) continue;
        const evt = JSON.parse(line);
        if (evt.type === 'retrieved') ui.retrieved(evt);
        else if (evt.type === 'start') ui.start(evt);
        else if (evt.type === 'delta') ui.delta(evt.text);
        else if (evt.type === 'done') ui.done(evt);
        else if (evt.type === 'error') ui.error(evt.message);
      }
    }
  } catch (e) {
    if (e.name === 'AbortError') ui.stopped(); else ui.error(`Stream interrupted: ${e.message}`);
    return;
  }
  if (!ui.finished) ui.error('The stream ended without a final result.');
}

function contextPayload() {
  return state.retrieve ? { retrieve: true, top_k: state.topK, article_numbers: [] } : { article_numbers: selectedNumbers() };
}

function generationSettings() {
  return { temperature: Number($('#temp').value), max_tokens: Number($('#max-tokens').value) };
}

function setRunning(prefix, on) {
  $(`#${prefix}-run`).disabled = on;
  $(`#${prefix}-stop`).hidden = !on;
}

/* ---------- Ask ---------- */

let askCtrl = null;

async function runAsk() {
  if (askCtrl) return;
  const question = $('#ask-q').value.trim();
  if (!question) { $('#ask-q').focus(); toast('Write a question first.'); return; }
  const payload = { question, backend: state.backend, ...contextPayload(), ...generationSettings() };
  const ui = createAnswerCard({ backend: payload.backend, question, articleNumbers: payload.article_numbers,
    view: 'ask', retrieving: state.retrieve });
  const results = $('#ask-results');
  results.querySelector('.empty')?.remove();
  results.prepend(ui.card);
  askCtrl = new AbortController();
  setRunning('ask', true);
  await streamAsk(payload, ui, askCtrl.signal);
  askCtrl = null;
  setRunning('ask', false);
}

/* ---------- Compare ---------- */

let cmpCtrl = null;

async function runCompare() {
  if (cmpCtrl) return;
  const question = $('#cmp-q').value.trim();
  if (!question) { $('#cmp-q').focus(); toast('Write a question first.'); return; }
  const ctx = contextPayload();
  const nums = ctx.article_numbers;
  const cols = $('#cmp-cols');
  cols.replaceChildren();
  $('#cmp-summary').replaceChildren();
  cmpCtrl = new AbortController();
  setRunning('cmp', true);
  const uis = Object.keys(state.config.backends).map((backend) => {
    const ui = createAnswerCard({ backend, question, articleNumbers: nums, view: 'compare', showQuestion: false,
      retrieving: state.retrieve });
    cols.append(ui.card);
    return ui;
  });
  await Promise.all(uis.map((ui) =>
    streamAsk({ question, backend: ui.result.backend, ...ctx, ...generationSettings() }, ui, cmpCtrl.signal)));
  cmpCtrl = null;
  setRunning('cmp', false);
  renderCompareSummary(uis);
}

function renderCompareSummary(uis) {
  const done = uis.filter((u) => u.result.done);
  if (done.length < 2) return;
  const m = (u) => u.result.done.metrics;
  const c = (u) => u.result.done.citations;
  const fastest = Math.min(...done.map((u) => m(u).latency_s));
  const row = (label, cells) => h('tr', {}, h('th', {}, label), cells);
  const checkCell = (u) => {
    const cc = c(u);
    if (cc.invalid.length) return h('span', { class: 'badge offline' }, `${cc.invalid.length} not in context`);
    return cc.cited.length ? h('span', { class: 'badge working' }, 'passed') : h('span', { class: 'badge neutral' }, 'no citations');
  };
  $('#cmp-summary').replaceChildren(
    h('div', { class: 'card', style: 'margin-top:16px' },
      h('div', { class: 'card-title' }, 'Summary'),
      h('table', { class: 'summary-table' },
        h('thead', {}, h('tr', {}, h('th', {}, ''), done.map((u) => h('th', {}, state.config.backends[u.result.backend].label)))),
        h('tbody', {},
          row('Latency', done.map((u) => h('td', { class: m(u).latency_s === fastest ? 'win' : '' }, `${m(u).latency_s.toFixed(2)} s`))),
          row('Tokens in → out', done.map((u) => h('td', {}, `${m(u).input_tokens} → ${m(u).output_tokens}`))),
          row('Citations', done.map((u) => h('td', {}, c(u).cited.length ? c(u).cited.map((n) => `Art. ${n}`).join(', ') : 'none'))),
          row('Citation check', done.map((u) => h('td', {}, checkCell(u)))),
          row('Stop reason', done.map((u) => h('td', {}, m(u).stop_reason || '—')))))),
  );
}

/* ---------- article drawer ---------- */

function openArticle(num, contextNums) {
  const a = state.byNumber.get(num);
  $('#drawer-title').textContent = `Article ${num}`;
  const body = $('#drawer-body');
  body.replaceChildren();
  if (contextNums) {
    body.append(contextNums.includes(num)
      ? h('div', { class: 'check ok' }, '✓ This article was in the context sent to the model.')
      : h('div', { class: 'check bad' }, '✗ This article was not in the context sent to the model.'));
  }
  if (!a) {
    body.append(h('p', { class: 'card-sub' }, state.articleSource === 'sample'
      ? 'Not among the loaded articles. The full corpus is not built yet.' : 'No such article in the corpus.'));
  } else {
    const crumbs = breadcrumb(a);
    if (crumbs) body.append(crumbs);
    else if (a.topic) body.append(h('div', {}, h('span', { class: 'badge neutral' }, a.topic)));
    if (a.source_pages) body.append(h('div', { class: 'card-sub' }, `Source PDF page${a.source_pages.length > 1 ? 's' : ''} ${a.source_pages.join(', ')}`));
    if (a.is_repealed) body.append(h('div', { class: 'notice partial' }, a.repeal_note || 'This article was repealed; it has no text.'));
    if (a.text_ar) body.append(h('div', {}, h('div', { class: 'drawer-section-label' }, 'العربية'), h('div', { class: 'drawer-text is-ar', dir: 'rtl' }, a.text_ar)));
    if (a.text_en) body.append(h('div', {}, h('div', { class: 'drawer-section-label' }, 'English'), h('div', { class: 'drawer-text' }, a.text_en)));
  }
  $('#drawer').hidden = false;
  $('#drawer-backdrop').hidden = false;
  $('#drawer-close').focus();
}

function closeDrawer() {
  $('#drawer').hidden = true;
  $('#drawer-backdrop').hidden = true;
}

/* ---------- status ---------- */

async function refreshStatus() {
  try {
    const { stages } = await api('/api/status');
    state.stages = stages;
    state.stageById = new Map(stages.map((s) => [s.id, s]));
  } catch (e) {
    toast(`Status check failed: ${e.message}`);
    return;
  }
  renderStatus();
  renderBackends();
  renderNavTags();
  renderPlanned();
}

function renderStatus() {
  const counts = { working: 0, offline: 0, planned: 0 };
  state.stages.forEach((s) => counts[s.state]++);
  const total = state.stages.length;
  $('#nav-status-meta').textContent = `${counts.working}/${total}`;
  $('#status-summary').replaceChildren(
    h('div', { class: 'status-summary' },
      h('div', { class: 'progress', title: `${counts.working} of ${total} stages working` },
        ['working', 'offline', 'planned'].map((k) => h('span', { class: k, style: `width:${(counts[k] / total) * 100}%` }))),
      h('div', { class: 'legend' },
        ['working', 'offline', 'planned'].map((k) => h('span', {}, h('span', { class: `dot ${k}` }), `${counts[k]} ${k}`)))),
  );
  const groups = $('#status-groups');
  groups.replaceChildren();
  for (const g of GROUP_ORDER) {
    const items = state.stages.filter((s) => s.group === g);
    if (!items.length) continue;
    groups.append(
      h('div', { class: 'status-group' },
        h('h2', {}, g),
        items.map((s) =>
          h('div', { class: 'stage' },
            h('div', {}, h('span', { class: `badge ${s.state}` }, STATE_LABEL[s.state])),
            h('div', {},
              h('div', { class: 'stage-title' }, s.title),
              h('div', { class: 'stage-detail' }, s.detail),
              s.next_step ? h('div', { class: 'stage-next' }, s.next_step) : null)))),
    );
  }
}

function renderNavTags() {
  for (const a of $$('#nav a[data-stage]')) {
    const tag = $('.nav-tag', a);
    const s = state.stageById.get(a.dataset.stage);
    if (a.dataset.view === 'corpus' && state.articleSource === 'sample') {
      tag.className = 'nav-tag tag partial';
      tag.textContent = 'sample';
    } else if (!s || s.state === 'working') {
      tag.className = 'nav-tag';
      tag.textContent = '';
    } else {
      tag.className = `nav-tag tag ${s.state}`;
      tag.textContent = s.state;
    }
  }
}

/* ---------- corpus ---------- */

const LEVEL_LABELS = [['part', 'Part'], ['book', 'Book'], ['chapter', 'Chapter'], ['section', 'Section'], ['topic', 'Topic']];

function breadcrumb(a) {
  const items = [];
  for (const [key, label] of LEVEL_LABELS) {
    const n = a[`${key}_number`];
    const en = a[`${key}_title_en`];
    const ar = a[`${key}_title_ar`];
    if (n == null && !en) continue;
    items.push(h('span', { class: 'crumb' },
      h('span', { class: 'crumb-en' }, `${label}${n != null ? ` ${n}` : ''}${en ? `: ${en}` : ''}`),
      ar ? h('span', { class: 'crumb-ar', dir: 'rtl' }, ar) : null));
  }
  if (a.subtopic_title_en) {
    items.push(h('span', { class: 'crumb' }, h('span', { class: 'crumb-en' }, a.subtopic_title_en),
      a.subtopic_title_ar ? h('span', { class: 'crumb-ar', dir: 'rtl' }, a.subtopic_title_ar) : null));
  }
  return items.length ? h('div', { class: 'crumbs' }, items) : null;
}

function articleBody(a) {
  return h('div', { class: 'article-detail' },
    breadcrumb(a),
    a.is_repealed ? h('div', { class: 'notice partial' }, [a.repeal_note, a.repeal_note_ar].filter(Boolean).join(' · ') || 'Repealed.') : null,
    h('div', { class: 'article-body' },
      h('div', { class: 'col is-ar', dir: 'rtl' }, h('div', { class: 'col-label' }, 'العربية'), a.text_ar || '—'),
      h('div', { class: 'col' }, h('div', { class: 'col-label' }, 'English'), a.text_en || '—')));
}

function pickRandom(n) {
  const live = state.articles.filter((a) => !a.is_repealed).map((a) => a.article_number);
  for (let i = live.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [live[i], live[j]] = [live[j], live[i]];
  }
  return live.slice(0, n).sort((x, y) => x - y);
}

async function renderCorpus() {
  const banner = $('#corpus-banner');
  if (state.articleSource === 'sample') {
    banner.replaceChildren(h('div', { class: 'notice partial', style: 'margin-bottom:14px' },
      h('strong', {}, `Sample only: ${plural(state.articles.length, 'article')}. `),
      'The structured corpus is not built yet: run python -m rag.corpus.build. ',
      'These were hand-cleaned from the source PDF, pages 64–66, so the rest of the console can be tested meanwhile.'));
    renderCorpusList();
    return;
  }
  let report = null;
  try { report = await api('/api/corpus/report'); } catch { /* the list still works without it */ }
  const c = report?.counts || {};
  const warnings = report?.warnings || [];
  const stat = (label, value) => h('div', { class: 'stat' }, h('div', { class: 'stat-label' }, label), h('div', { class: 'stat-val' }, value));
  banner.replaceChildren(
    h('div', { class: 'stat-row' },
      stat('Articles', (c.records ?? state.articles.length).toLocaleString()),
      stat('Live / repealed', `${(c.live ?? '—').toLocaleString()} / ${c.repealed ?? '—'}`),
      stat('Hierarchy', `${c.parts ?? '—'}·${c.books ?? '—'}·${c.chapters ?? '—'}·${c.sections ?? '—'}`),
      stat('Build warnings', String(warnings.length))),
    h('div', { class: 'corpus-actions' },
      h('button', { class: 'btn small', type: 'button', onclick: () => { state.corpusSample = pickRandom(20); renderCorpusList(); } },
        'Random 20 (eyeball check)'),
      h('button', { class: 'btn ghost small', type: 'button', onclick: () => { state.corpusSample = null; renderCorpusList(); } }, 'Show all'),
      h('span', { class: 'card-sub' }, report?.generated_at ? `Built ${new Date(report.generated_at).toLocaleString()} · parts·books·chapters·sections` : '')),
    warnings.length ? h('details', { class: 'card warnings-panel' },
      h('summary', {}, h('strong', {}, `${plural(warnings.length, 'build warning')} to review`),
        h('span', { class: 'card-sub' }, ' — from python -m rag.corpus.build; click an article to read it')),
      h('div', { class: 'warn-list' }, warnings.map((w) => h('div', { class: 'warn-item' },
        h('span', { class: 'badge partial' }, w.code),
        w.article ? h('button', { class: 'cite', type: 'button', dataset: { article: w.article } }, `Art. ${w.article}`) : null,
        w.page ? h('span', { class: 'card-sub' }, `p. ${w.page}`) : null,
        h('span', { class: 'warn-msg', dir: 'auto' }, w.message))))) : null,
  );
  renderCorpusList();
}

function renderCorpusList() {
  const q = $('#corpus-filter').value.trim().toLowerCase();
  const list = $('#corpus-list');
  list.replaceChildren();
  const sample = state.corpusSample;
  const pool = sample ? sample.map((n) => state.byNumber.get(n)).filter(Boolean) : state.articles;
  let shown = 0;
  let hidden = 0;
  for (const a of pool) {
    if (q && !matchesQuery(a, q)) continue;
    if (shown >= 300) { hidden += 1; continue; }
    shown += 1;
    const details = h('details', { class: 'article', open: Boolean(sample) },
      h('summary', {},
        h('span', { class: 'article-num' }, `Art. ${a.article_number}`),
        h('span', { class: 'article-preview', dir: 'auto' }, articlePreview(a)),
        h('span', { class: `badge ${a.is_repealed ? 'partial' : 'neutral'}` }, a.is_repealed ? 'Repealed' : a.topic || '')));
    if (sample) details.append(articleBody(a));
    else details.addEventListener('toggle', () => { if (details.open && details.children.length === 1) details.append(articleBody(a)); });
    list.append(details);
  }
  if (sample) list.prepend(h('div', { class: 'notice planned' }, `Random sample of 20 live articles: compare each with the PDF page shown in its header.`));
  if (hidden) list.append(h('div', { class: 'empty' }, `${hidden.toLocaleString()} more articles match: refine the search.`));
  if (!shown) list.append(h('div', { class: 'empty' }, 'No articles match.'));
}

/* ---------- planned views ---------- */

function mockRetrieval() {
  return h('div', {},
    h('input', { type: 'search', class: 'search', value: 'هل يجوز للزوج الرجوع في هبته لزوجته؟', disabled: true }),
    [0.9, 0.82, 0.74, 0.55, 0.3].map((s) =>
      h('div', { class: 'mock-row' },
        h('span', { class: 'article-num' }, 'Art. ···'),
        h('div', {}, h('div', { class: 'skel', style: 'width:85%' }), h('div', { class: 'skel', style: 'width:55%;margin-top:6px' })),
        h('div', { class: 'bar', title: 'similarity score' }, h('span', { style: `width:${s * 100}%` })))));
}

function mockEvaluation() {
  const metrics = ['Faithfulness', 'Answer relevancy', 'Context precision', 'Context recall'];
  return h('div', {},
    h('div', { class: 'score-grid' }, metrics.map((m) => h('div', { class: 'score' }, h('div', { class: 'score-name' }, m), h('div', { class: 'score-val' }, '—')))),
    h('table', { class: 'summary-table' },
      h('thead', {}, h('tr', {}, h('th', {}, 'Run'), metrics.map((m) => h('th', {}, m)))),
      h('tbody', {},
        ['Bedrock · gpt-oss-120b', 'vLLM · official AWQ', 'vLLM · own AWQ (Arabic calibration)'].map((r) =>
          h('tr', {}, h('td', {}, r), metrics.map(() => h('td', {}, '—')))))));
}

function mockTraces() {
  const line = (name, type, depth, a, b) =>
    h('div', { class: 'trace-line' },
      h('span', { style: `padding-inline-start:${depth * 22}px` }, depth ? '└ ' : '', h('span', { class: 'trace-type' }, type), name),
      h('span', { class: 'card-sub' }, a),
      h('span', { class: 'card-sub' }, b));
  return h('div', { class: 'trace-tree' },
    line('ask', 'trace', 0, 'question, answer', '— s'),
    line('retrieve-articles', 'retriever', 1, 'top-k articles', '— s'),
    line('generate-answer', 'generation', 1, '— → — tokens', '— s'),
    line('check-citations', 'span', 1, 'valid / invalid', '— s'));
}

const PLANNED = {
  retrieval: {
    stage: 'retrieval',
    title: 'Retrieval inspector',
    lede: 'See which articles the retriever returns for a question, with scores, before the model answers.',
    will: [
      'Type a question and see the top-k articles with similarity scores',
      'Check that the Arabic and English versions of a question retrieve the same articles',
      'Find missing or irrelevant context behind a bad answer',
    ],
    needs: ['The structured corpus (one record per article)', 'A multilingual embedding model and a Chroma index', 'TASKS.md: Corpus, then Retrieval'],
    mock: mockRetrieval,
  },
  evaluation: {
    stage: 'ragas',
    title: 'Evaluation scorecard',
    lede: 'RAGAS scores on a fixed question set, per backend and per experiment.',
    will: [
      'Faithfulness, answer relevancy, context precision and recall on 50+ questions',
      'Bedrock, the official AWQ build and our own AWQ build side by side',
      'Drill into the questions that fail',
    ],
    needs: ['Retrieval (the scores need retrieved contexts)', 'A 50+ question evaluation set in Arabic and English', 'TASKS.md: RAGAS, MLflow'],
    mock: mockEvaluation,
  },
  traces: {
    stage: 'tracing',
    title: 'Traces',
    lede: 'Every request traced in Langfuse: retrieval, generation, tokens and cost.',
    will: [
      'Open the full trace behind any answer from Ask or Compare',
      'See retrieval and generation as separate, timed steps',
      'Track token cost per backend over time',
    ],
    needs: ['Langfuse SDK wired into retrieval and generation', 'Retrieval, for the retriever step', 'TASKS.md: Langfuse tracing'],
    mock: mockTraces,
  },
};

async function runSearch() {
  const question = $('#ret-q').value.trim();
  if (!question) { $('#ret-q').focus(); return; }
  const out = $('#ret-results');
  out.replaceChildren(h('div', { class: 'thinking' }, h('span', { class: 'spinner' }), 'Searching… (the first search loads the embedding model)'));
  let res;
  try {
    res = await api('/api/search', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, k: Number($('#ret-k').value) }) });
  } catch (e) {
    out.replaceChildren(h('div', { class: 'error-box' }, e.message));
    return;
  }
  const best = Math.max(...res.hits.map((x) => x.score ?? 0), 0.01);
  out.replaceChildren(
    h('div', { class: 'card-sub', style: 'margin-bottom:8px' }, `${plural(res.hits.length, 'article')} in ${res.latency_s.toFixed(2)} s · embedding on ${res.device}`),
    ...res.hits.map((x, i) => {
      const a = state.byNumber.get(x.article_number) || {};
      return h('div', { class: 'ret-hit' },
        h('div', { class: 'ret-rank' }, `${i + 1}`),
        h('div', { class: 'ret-main' },
          h('div', { class: 'ret-head' },
            h('button', { class: 'cite', type: 'button', dataset: { article: x.article_number } }, `Art. ${x.article_number}`),
            x.by_number ? h('span', { class: 'badge working' }, 'named in the question')
              : h('span', { class: 'ret-score' }, h('span', { class: 'bar' }, h('span', { style: `width:${(x.score / best) * 100}%` })), x.score.toFixed(3)),
            a.is_repealed ? h('span', { class: 'badge partial' }, 'Repealed') : null,
            h('span', { class: 'card-sub' }, a.topic || '')),
          h('div', { class: 'ret-text is-ar', dir: 'rtl' }, (a.text_ar || a.repeal_note_ar || '').slice(0, 220)),
          h('div', { class: 'ret-text' }, (a.text_en || a.repeal_note || '').slice(0, 260))));
    }),
  );
}

function renderRetrievalView() {
  $('#view-retrieval').replaceChildren(
    h('header', { class: 'view-head' },
      h('h1', {}, 'Retrieval inspector'),
      h('p', { class: 'lede' }, 'Search the index the way Ask does: the question is embedded with Qwen3-Embedding-0.6B and '
        + 'compared with one bilingual chunk per article. Scores are cosine similarity; articles named by number come first.')),
    h('div', { class: 'card' },
      h('textarea', { id: 'ret-q', dir: 'auto', rows: 2, placeholder: 'اكتب سؤالك هنا… or ask in English',
        onkeydown: (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); runSearch(); } } }),
      h('div', { class: 'actions' },
        h('button', { class: 'btn primary', type: 'button', onclick: runSearch }, 'Search ', h('kbd', {}, 'Ctrl ↵')),
        h('select', { id: 'ret-k', class: 'topk' }, [5, 10, 20].map((n) => h('option', { value: n, selected: n === 10 }, `top ${n}`))))),
    h('div', { id: 'ret-results', class: 'ret-results' }),
  );
}

function renderPlanned() {
  for (const [view, p] of Object.entries(PLANNED)) {
    if (view === 'retrieval' && state.articleSource === 'corpus') continue; // the real inspector replaces it
    const s = state.stageById.get(p.stage);
    const live = s?.state === 'working';
    const extra = view === 'traces' && state.config?.langfuse_url
      ? h('p', { class: 'stage-detail', style: 'margin-top:10px' }, 'Langfuse project: ', h('a', { href: state.config.langfuse_url, target: '_blank', rel: 'noopener' }, state.config.langfuse_url))
      : null;
    $(`#view-${view}`).replaceChildren(
      h('header', { class: 'view-head' },
        h('h1', {}, p.title, ' ', h('span', { class: `badge ${live ? 'working' : 'planned'}` }, live ? 'Stage live, view not built yet' : 'Planned')),
        h('p', { class: 'lede' }, p.lede)),
      h('div', { class: 'planned-hero' },
        h('div', { class: 'card' }, h('div', { class: 'card-title' }, "What you'll be able to test here"), h('ul', {}, p.will.map((x) => h('li', {}, x)))),
        h('div', { class: 'card' },
          h('div', { class: 'card-title' }, 'Needed first'),
          h('ul', {}, p.needs.map((x) => h('li', {}, x))),
          s ? h('p', { class: 'stage-detail', style: 'margin-top:10px' }, `Status: ${s.detail}`) : null,
          extra)),
      h('div', { class: 'preview-label' }, 'Planned layout'),
      h('div', { class: 'card mock' }, p.mock()),
    );
  }
}

/* ---------- feedback log ---------- */

async function refreshFeedback() {
  try {
    state.feedback = (await api('/api/feedback')).entries;
  } catch (e) {
    toast(`Could not load feedback: ${e.message}`);
    return;
  }
  renderFeedback();
}

function renderFeedback() {
  const all = state.feedback;
  $('#nav-feedback-meta').textContent = all.length ? String(all.length) : '';
  $('#feedback-download').hidden = !all.length;
  const up = all.filter((e) => e.rating === 'up').length;
  const pct = (a, b) => (b ? `${Math.round((a / b) * 100)}%` : '—');
  const perBackend = {};
  for (const e of all) {
    perBackend[e.backend] ??= { up: 0, n: 0 };
    perBackend[e.backend].n++;
    if (e.rating === 'up') perBackend[e.backend].up++;
  }
  const label = (k) => state.config?.backends[k]?.label ?? k;
  $('#feedback-stats').replaceChildren(
    h('div', { class: 'stat-row' },
      h('div', { class: 'stat' }, h('div', { class: 'stat-label' }, 'Ratings'), h('div', { class: 'stat-val' }, all.length)),
      h('div', { class: 'stat' }, h('div', { class: 'stat-label' }, 'Rated right'), h('div', { class: 'stat-val' }, pct(up, all.length)), h('div', { class: 'stat-sub' }, `${up} of ${all.length}`)),
      h('div', { class: 'stat' }, h('div', { class: 'stat-label' }, 'Hallucinated citations'), h('div', { class: 'stat-val' }, all.filter((e) => e.invalid?.length).length), h('div', { class: 'stat-sub' }, 'answers citing articles outside the context')),
      h('div', { class: 'stat' }, h('div', { class: 'stat-label' }, 'Rated right, by backend'),
        Object.keys(perBackend).length
          ? Object.entries(perBackend).map(([k, v]) => h('div', { class: 'stat-sub', style: 'font-size:14px;color:var(--text)' }, `${label(k)}: ${pct(v.up, v.n)} (${v.n})`))
          : h('div', { class: 'stat-val' }, '—'))),
  );
  const shown = all.filter((e) => state.feedbackFilter === 'all' || e.rating === state.feedbackFilter);
  const list = $('#feedback-list');
  list.replaceChildren();
  if (!shown.length) {
    list.append(h('div', { class: 'empty' },
      h('div', { class: 'empty-title' }, all.length ? 'Nothing matches this filter' : 'No feedback yet'),
      h('p', {}, 'Rate answers in Ask or Compare; every rating lands here with the question, answer, context and citations.')));
    return;
  }
  for (const e of shown) {
    list.append(
      h('details', { class: 'fb-entry' },
        h('summary', {},
          h('span', { class: 'fb-rating', title: e.rating === 'up' ? 'Right' : 'Wrong' }, e.rating === 'up' ? '👍' : '👎'),
          h('span', { class: 'fb-q', dir: 'auto' }, e.question),
          h('span', { class: 'fb-meta' }, `${label(e.backend)} · ${new Date(e.ts).toLocaleString()}`)),
        h('div', { class: 'fb-detail' },
          e.tags?.length ? h('div', { class: 'fb-tag-list' }, e.tags.map((t) => h('span', { class: 'badge offline' }, t))) : null,
          e.comment ? h('div', { dir: 'auto' }, h('strong', {}, 'Comment: '), e.comment) : null,
          h('div', { class: 'answer-text', dir: 'auto' }, e.answer),
          h('div', { class: 'card-sub' },
            `Model ${e.model} · ${e.view} view · context ${e.article_numbers.length ? e.article_numbers.map((n) => `Art. ${n}`).join(', ') : 'none'}`,
            ` · cited ${e.cited.length ? e.cited.join(', ') : 'none'}`,
            e.invalid.length ? ` · not in context: ${e.invalid.join(', ')}` : '',
            e.latency_s != null ? ` · ${e.latency_s.toFixed(2)} s` : ''))),
    );
  }
}

/* ---------- wiring ---------- */

function bindEvents() {
  window.addEventListener('hashchange', () => showView(location.hash.slice(1)));
  $('#temp').addEventListener('input', (e) => ($('#temp-out').textContent = Number(e.target.value).toFixed(1)));
  $('#ask-run').addEventListener('click', runAsk);
  $('#ask-stop').addEventListener('click', () => askCtrl?.abort());
  $('#cmp-run').addEventListener('click', runCompare);
  $('#cmp-stop').addEventListener('click', () => cmpCtrl?.abort());
  const submitOnCtrlEnter = (fn) => (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); fn(); } };
  $('#ask-q').addEventListener('keydown', submitOnCtrlEnter(runAsk));
  $('#cmp-q').addEventListener('keydown', submitOnCtrlEnter(runCompare));
  $('#ask-clear').addEventListener('click', () => { $('#ask-results').innerHTML = $('#ask-results').dataset.empty; });
  $('#ctx-filter').addEventListener('input', renderContextPicker);
  $('#corpus-filter').addEventListener('input', renderCorpusList);
  $('#status-refresh').addEventListener('click', refreshStatus);
  $('#feedback-refresh').addEventListener('click', refreshFeedback);
  $('#feedback-filter').addEventListener('click', (e) => {
    const chip = e.target.closest('[data-filter]');
    if (!chip) return;
    state.feedbackFilter = chip.dataset.filter;
    $$('#feedback-filter .chip').forEach((c) => c.classList.toggle('active', c === chip));
    renderFeedback();
  });
  document.addEventListener('click', (e) => {
    const chip = e.target.closest('.cite[data-article]');
    if (!chip) return;
    const card = chip.closest('.answer-card');
    const ctx = card ? (card.dataset.context ? card.dataset.context.split(',').map(Number) : []) : null;
    openArticle(Number(chip.dataset.article), ctx);
  });
  $('#drawer-close').addEventListener('click', closeDrawer);
  $('#drawer-backdrop').addEventListener('click', closeDrawer);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeDrawer(); });
}

async function init() {
  $('#ask-results').dataset.empty = $('#ask-results').innerHTML;
  bindEvents();
  try {
    const [config, arts] = await Promise.all([api('/api/config'), api('/api/articles')]);
    state.config = config;
    state.backend = config.default_backend;
    $('#temp').value = config.temperature;
    $('#temp-out').textContent = Number(config.temperature).toFixed(1);
    $('#max-tokens').value = config.max_tokens;
    state.articles = arts.articles;
    state.byNumber = new Map(arts.articles.map((a) => [a.article_number, a]));
    state.articleSource = arts.source;
    state.articleNote = arts.note;
    state.selected = new Set(arts.articles.length > LARGE_CORPUS ? []
      : arts.articles.filter((a) => !a.is_repealed).map((a) => a.article_number));
  } catch (e) {
    toast(`Could not load the console: ${e.message}`);
    return;
  }
  $('#ctx-source').textContent = state.articleSource === 'sample'
    ? `Source: sample fixture, ${plural(state.articles.length, 'article')} from PDF pp. 64–66`
    : `Source: ${state.articleNote}`;
  renderCtxQuick();
  renderContextPicker();
  renderBackends();
  renderExamples();
  state.retrieve = state.articleSource === 'corpus';
  renderCtxMode();
  if (state.articleSource === 'corpus') renderRetrievalView();
  renderCorpus();
  renderPlanned();
  renderNavTags();
  showView(location.hash.slice(1) || 'ask');
  refreshStatus();
  setInterval(refreshStatus, 30000);
  refreshFeedback();
}

init();
