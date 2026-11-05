/* UltiSim evolution dashboard (v1). Plain JS, no build step, no external dependencies.
 * Reads the run files described in viz/RUN_FORMAT.md from /runs/<run>/ and the run list
 * from /api/runs (viz/server.py). Polls every 4 s.
 */
'use strict';

(function () {
  // ---------------------------------------------------------------------------
  // Constants & helpers
  // ---------------------------------------------------------------------------
  const POLL_MS = 4000;
  const RUNS_EVERY = 5;          // refresh the run list every 5th poll (~20 s)
  const STALE_S = 90;            // "running" but status.json not rewritten for this long -> stalled
  const N_CAT = 8;               // categorical slots in dashboard.css (--cat-1..8)
  const SVGNS = 'http://www.w3.org/2000/svg';

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
  const num = (v) => (isNum(v) ? v : null);
  const get = (o, ...keys) => { for (const k of keys) { if (o == null || typeof o !== 'object') return undefined; o = o[k]; } return o; };
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const bust = (path) => path + (path.includes('?') ? '&' : '?') + '_=' + Date.now();
  const humanKey = (k) => String(k).replace(/_/g, ' ');

  const fmtRate = (v) => (isNum(v) ? v.toFixed(3) : '–');
  const fmtPct = (v, d = 1) => (isNum(v) ? (v * 100).toFixed(d) + '%' : '–');
  function fmtBig(n) {
    if (!isNum(n)) return '–';
    if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(2) + ' B';
    if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + ' M';
    return Math.round(n).toLocaleString('en-US');
  }
  function fmtVal(v) {
    if (v === null || v === undefined) return '–';
    if (typeof v === 'boolean') return v ? 'yes' : 'no';
    if (!isNum(v)) return typeof v === 'object' ? JSON.stringify(v) : String(v);
    if (Number.isInteger(v)) return String(v);
    const a = Math.abs(v);
    return v.toFixed(a >= 100 ? 1 : a >= 10 ? 2 : 3);
  }
  function fmtDur(s) {
    if (!isNum(s) || s < 0) return '–';
    s = Math.round(s);
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    if (h) return `${h}h ${String(m).padStart(2, '0')}m`;
    if (m) return `${m}m ${String(sec).padStart(2, '0')}s`;
    return `${sec}s`;
  }
  function fmtAgo(epochS) {
    if (!isNum(epochS) || epochS <= 0) return '';
    const d = Date.now() / 1000 - epochS;
    if (d < 60) return 'just now';
    if (d < 3600) return `${Math.round(d / 60)} min ago`;
    if (d < 86400) return `${Math.round(d / 3600)} h ago`;
    return new Date(epochS * 1000).toLocaleDateString();
  }
  // status.updated is a naive local ISO time written on this machine
  function parseLocal(iso) {
    if (typeof iso !== 'string') return null;
    const t = Date.parse(iso);
    return Number.isFinite(t) ? t : null;
  }
  function prettyScheme(s) {
    const [a, ...rest] = String(s).split('_');
    const head = a.charAt(0).toUpperCase() + a.slice(1);
    return rest.length ? `${head} · ${rest.join(' ')}` : head;
  }
  function deltaHtml(cur, prev, lowerIsBetter) {
    if (!isNum(cur) || !isNum(prev)) return '';
    const d = cur - prev;
    if (Math.abs(d) < 0.0005) return '<span class="delta flat">= 0.000</span>';
    const better = lowerIsBetter ? d < 0 : d > 0;
    return `<span class="delta ${better ? 'good' : 'bad'}">${d > 0 ? '▲ +' : '▼ −'}${Math.abs(d).toFixed(3)}</span>`;
  }

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  const S = {
    runs: [],
    run: null,           // selected run name
    meta: null,
    status: null,
    archive: null,
    hist: [],            // parsed history lines, one per gen, sorted by gen
    histChars: 0,        // length of the complete-lines prefix already parsed
    histHead: '',        // first bytes of the file, to notice a rewrite
    histBad: 0,          // unparsable complete lines (skipped)
    histByGen: new Map(),
    replay: false,       // latest_best.json exists
    replaySrc: null,
    lastPollOk: null,
    lastPollAt: 0,
    error: null,
    pollN: 0,
    generation: 0,       // bumps on run switch so stale responses are dropped
  };
  let pollTimer = null;
  let polling = false;
  let pollAgain = false;
  const hover = { A: null, B: null };   // hovered gen per chart (survives re-render)

  const runBase = (name) => '/runs/' + encodeURIComponent(name) + '/';

  async function fetchText(url) {
    const res = await fetch(bust(url), { cache: 'no-store' });
    if (res.status === 404) return null;
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    return res.text();
  }
  async function fetchJson(url) {
    const t = await fetchText(url);
    if (t === null) return { missing: true, value: null };
    try { return { missing: false, value: JSON.parse(t) }; } catch (err) { return { missing: false, value: undefined, bad: true }; }
  }

  // history.jsonl: append-only. Only whole lines count; the text after the last '\n' may be
  // a line that is still being written, so it is left for the next poll.
  function ingestHistory(text) {
    if (text === null) {
      const had = S.hist.length > 0;
      resetHistory();
      return had;
    }
    const end = text.lastIndexOf('\n');
    const complete = end >= 0 ? text.slice(0, end + 1) : '';
    const head = complete.slice(0, 256);
    if (complete.length < S.histChars || (S.histHead && !complete.startsWith(S.histHead.slice(0, Math.min(S.histHead.length, complete.length))))) {
      resetHistory();   // file truncated or rewritten (e.g. a fresh run reusing the name)
    }
    if (complete.length === S.histChars) return false;
    const fresh = complete.slice(S.histChars);
    S.histChars = complete.length;
    S.histHead = head;
    let changed = false;
    for (const raw of fresh.split('\n')) {
      const line = raw.trim();
      if (!line) continue;
      let obj;
      try { obj = JSON.parse(line); } catch (_) { S.histBad++; continue; }
      if (!obj || typeof obj !== 'object' || !isNum(obj.gen)) { S.histBad++; continue; }
      S.histByGen.set(obj.gen, obj);   // a resumed run may repeat a gen; the newest line wins
      changed = true;
    }
    if (changed) S.hist = Array.from(S.histByGen.values()).sort((a, b) => a.gen - b.gen);
    return changed;
  }
  function resetHistory() {
    S.hist = []; S.histChars = 0; S.histHead = ''; S.histBad = 0; S.histByGen = new Map();
  }

  // ---------------------------------------------------------------------------
  // Families: fixed order from meta.json -> fixed colours
  // ---------------------------------------------------------------------------
  function familyOrder() {
    const order = [];
    const add = (f) => { if (typeof f === 'string' && !order.includes(f)) order.push(f); };
    (Array.isArray(get(S.meta, 'families')) ? S.meta.families : []).forEach(add);
    for (const h of S.hist) {
      if (h.families && typeof h.families === 'object') Object.keys(h.families).forEach(add);
      if (h.family_best && typeof h.family_best === 'object') Object.keys(h.family_best).forEach(add);
    }
    if (get(S.archive, 'families')) Object.keys(S.archive.families).forEach(add);
    return order;
  }
  function famColor(f, order) {
    const i = (order || familyOrder()).indexOf(f);
    return i >= 0 && i < N_CAT ? `var(--cat-${i + 1})` : 'var(--cat-rest)';
  }

  // ---------------------------------------------------------------------------
  // Derived numbers
  // ---------------------------------------------------------------------------
  const last = () => (S.hist.length ? S.hist[S.hist.length - 1] : null);
  const prev = () => (S.hist.length > 1 ? S.hist[S.hist.length - 2] : null);
  function planned() {
    return num(get(S.status, 'generations_planned')) ?? num(get(S.meta, 'generations_planned'));
  }
  function doneGens() {
    const l = last();
    return l ? l.gen + 1 : 0;
  }
  function secsPerGen() {
    const xs = S.hist.slice(-5).map((h) => num(h.gen_s)).filter((v) => v !== null);
    if (xs.length) return xs.reduce((a, b) => a + b, 0) / xs.length;
    const l = last();
    return l && isNum(l.elapsed_s) && l.gen >= 0 ? l.elapsed_s / (l.gen + 1) : null;
  }
  function statusAgeS() {
    const t = parseLocal(get(S.status, 'updated'));
    return t === null ? null : (Date.now() - t) / 1000;
  }
  function runState() {
    const st = get(S.status, 'state');
    const p = planned();
    if (st === 'running') {
      const age = statusAgeS();
      return age !== null && age > STALE_S ? 'stalled' : 'live';
    }
    if (st === 'finished' || st === 'stopped' || st === 'error') return st;
    if (!S.hist.length && !S.meta) return 'waiting';
    if (p && doneGens() >= p) return 'finished';
    return 'unknown';
  }
  function elapsedNow() {
    const st = get(S.status, 'state');
    let e = num(get(S.status, 'elapsed_s'));
    const l = last();
    if (e === null && l) e = num(l.elapsed_s);
    if (e !== null && st === 'running') {
      const age = statusAgeS();
      if (age !== null && age >= 0 && age <= STALE_S) e += age;
    }
    return e;
  }
  function allTimeBest() {
    let best = null;
    const fb = get(last(), 'family_best') || {};
    for (const [f, v] of Object.entries(fb)) if (isNum(v) && (!best || v > best.v)) best = { f, v };
    const ar = get(S.archive, 'families') || {};
    for (const [f, e] of Object.entries(ar)) if (isNum(get(e, 'score_rate')) && (!best || e.score_rate > best.v)) best = { f, v: e.score_rate };
    return best;
  }

  // ---------------------------------------------------------------------------
  // Top bar, progress, KPIs
  // ---------------------------------------------------------------------------
  function setStatus(msg, isError) {
    const el = $('status');
    el.textContent = msg || '';
    el.title = msg || '';
    el.classList.toggle('error', !!isError);
  }

  const BADGE_TEXT = { live: 'live', finished: 'finished', stopped: 'stopped', error: 'error', stalled: 'stalled', waiting: 'waiting', unknown: 'no status' };
  function renderHeader() {
    const rs = S.run ? runState() : 'waiting';
    const b = $('stateBadge');
    b.className = 'badge ' + rs;
    b.textContent = BADGE_TEXT[rs] || rs;
    const age = statusAgeS();
    b.title = rs === 'stalled' ? `status.json says running but was last updated ${fmtDur(age)} ago`
      : rs === 'error' ? (get(S.status, 'message') || 'run reported an error')
      : get(S.status, 'updated') ? `status updated ${S.status.updated}` : '';

    const p = planned();
    const done = doneGens();
    const pct = p ? clamp(done / p, 0, 1) * 100 : 0;
    $('progFill').style.width = pct.toFixed(1) + '%';
    $('progress').classList.toggle('done', !!p && done >= p);
    $('progress').setAttribute('aria-valuenow', String(Math.round(pct)));
    $('progText').textContent = S.run
      ? `${done}${p ? ' / ' + p : ''} generations${p ? ` · ${Math.round(pct)}%` : ''}`
      : 'No run selected';
    renderTiming();
  }

  function renderTiming() {
    const parts = [];
    const st = get(S.status, 'state');
    const phase = get(S.status, 'phase');
    const sgen = num(get(S.status, 'gen'));
    if (st === 'running' && phase) parts.push(`${esc(phase)}${sgen !== null ? ` gen <b>${sgen}</b>` : ''}`);
    const e = elapsedNow();
    if (e !== null) parts.push(`elapsed <b>${fmtDur(e)}</b>`);
    const spg = secsPerGen();
    if (spg !== null) parts.push(`<b>${spg.toFixed(1)}</b> s/generation`);
    const p = planned();
    const rs = runState();
    if (spg !== null && p && (rs === 'live') && doneGens() < p) parts.push(`ETA <b>${fmtDur((p - doneGens()) * spg)}</b>`);
    const msg = get(S.status, 'message');
    if (msg) parts.push(esc(msg));
    if (S.lastPollAt) parts.push(`checked ${new Date(S.lastPollAt).toLocaleTimeString()}`);
    const html = parts.join(' &middot; ');
    const el = $('timing');
    if (el.innerHTML !== html) el.innerHTML = html;
  }

  function renderKpis() {
    const l = last(), pv = prev();
    if (!l) {
      for (const id of ['kGen', 'kPoss', 'kOff', 'kDef', 'kFam']) $(id).innerHTML = '–';
      for (const id of ['kPossSub', 'kOffSub', 'kDefSub', 'kFamSub']) $(id).innerHTML = '&nbsp;';
      const p = planned();
      $('kGenSub').innerHTML = S.run ? (p ? `0 of ${p} complete` : 'waiting for the first generation') : '&nbsp;';
      return;
    }
    const p = planned();
    $('kGen').innerHTML = `${l.gen} <small>last completed</small>`;
    const sgen = num(get(S.status, 'gen'));
    const running = get(S.status, 'state') === 'running';
    $('kGenSub').innerHTML = `${doneGens()}${p ? ` of ${p}` : ''} complete` +
      (running && sgen !== null && sgen > l.gen ? ` · working on gen ${sgen}` : '');

    $('kPoss').textContent = fmtBig(num(l.possessions));
    const perGen = pv && isNum(l.possessions) && isNum(pv.possessions) ? l.possessions - pv.possessions : null;
    $('kPossSub').innerHTML = perGen !== null ? `+${fmtBig(perGen)} last generation` : '&nbsp;';

    const ob = num(get(l, 'off', 'best'));
    $('kOff').innerHTML = `${fmtRate(ob)} ${deltaHtml(ob, num(get(pv, 'off', 'best')), false)}`;
    const atb = allTimeBest();
    $('kOffSub').innerHTML = atb ? `all-time best ${fmtRate(atb.v)} (${esc(atb.f)}) · mean ${fmtRate(num(get(l, 'off', 'mean')))}` : `mean ${fmtRate(num(get(l, 'off', 'mean')))}`;

    const db = num(get(l, 'def', 'best'));
    $('kDef').innerHTML = `${fmtRate(db)} ${deltaHtml(db, num(get(pv, 'def', 'best')), true)}`;
    $('kDefSub').innerHTML = `lower is better · mean ${fmtRate(num(get(l, 'def', 'mean')))}`;

    const fam = get(l, 'champion', 'family');
    if (typeof fam === 'string') {
      $('kFam').innerHTML = `<span class="fam-dot" style="background:${famColor(fam)}"></span>${esc(fam)}`;
      let streak = 0;
      for (let i = S.hist.length - 1; i >= 0 && get(S.hist[i], 'champion', 'family') === fam; i--) streak++;
      $('kFamSub').innerHTML = `score ${fmtRate(num(get(l, 'champion', 'score_rate')))} · champion family ${streak} gen${streak === 1 ? '' : 's'} running`;
    } else {
      $('kFam').textContent = '–';
      $('kFamSub').innerHTML = '&nbsp;';
    }
  }

  // ---------------------------------------------------------------------------
  // SVG chart helpers
  // ---------------------------------------------------------------------------
  function svgEl(tag, attrs, parent) {
    const e = document.createElementNS(SVGNS, tag);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function linePath(pts) {
    // pts: [x, y|null]; nulls break the line
    let d = '', pen = false;
    for (const [x, y] of pts) {
      if (y === null || !isNum(y)) { pen = false; continue; }
      d += (pen ? 'L' : 'M') + x.toFixed(1) + ',' + y.toFixed(1);
      pen = true;
    }
    return d;
  }
  function niceTicks(lo, hi, maxN) {
    const span = Math.max(1, hi - lo);
    const raw = span / Math.max(1, maxN);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = Math.max(1, [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) || 10 * mag);
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v));
    return out;
  }
  // spread direct labels vertically so they don't overlap
  function spreadLabels(items, minGap, top, bottom) {
    items.sort((a, b) => a.y - b.y);
    for (let i = 1; i < items.length; i++) if (items[i].y - items[i - 1].y < minGap) items[i].y = items[i - 1].y + minGap;
    const over = items.length ? items[items.length - 1].y - bottom : 0;
    if (over > 0) for (const it of items) it.y -= over;
    for (let i = items.length - 2; i >= 0; i--) if (items[i + 1].y - items[i].y < minGap) items[i].y = items[i + 1].y - minGap;
    for (const it of items) it.y = Math.max(top, it.y);
    return items;
  }

  // Generic frame: x = generation, y = [0,1]. Returns helpers.
  function frame(host, height, margin, yTicks, yFmt) {
    host.innerHTML = '';
    const W = Math.max(320, host.clientWidth || 600);
    const H = height;
    const m = margin;
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img' }, host);
    const g0 = S.hist.length ? S.hist[0].gen : 0;
    const g1 = Math.max(S.hist.length ? last().gen : 0, g0 + 1);
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const x = (g) => m.l + ((g - g0) / (g1 - g0)) * pw;
    const y = (v) => m.t + (1 - v) * ph;
    const grid = svgEl('g', { class: 'grid' }, svg);
    for (const t of yTicks) {
      svgEl('line', { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t) }, grid);
      const tx = svgEl('text', { x: m.l - 6, y: y(t) + 3.5, 'text-anchor': 'end' }, svg);
      tx.textContent = yFmt(t);
    }
    svgEl('line', { class: 'axis-line', x1: m.l, x2: W - m.r, y1: y(0), y2: y(0) }, svg);
    const xt = niceTicks(g0, g1, Math.floor(pw / 70));
    for (const t of xt) {
      if (t < g0 || t > g1) continue;
      svgEl('line', { class: 'axis-line', x1: x(t), x2: x(t), y1: y(0), y2: y(0) + 4 }, svg);
      const tx = svgEl('text', { x: x(t), y: y(0) + 16, 'text-anchor': 'middle' }, svg);
      tx.textContent = String(t);
    }
    const xl = svgEl('text', { x: W - m.r + 10, y: y(0) + 16, 'text-anchor': 'start' }, svg);
    xl.textContent = 'generation';
    return { svg, W, H, m, x, y, g0, g1, pw, ph };
  }

  function emptyChart(host, height, msg) {
    host.innerHTML = '';
    const W = Math.max(320, host.clientWidth || 600);
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${height}`, height }, host);
    svgEl('rect', { x: 0.5, y: 0.5, width: W - 1, height: height - 1, rx: 8, fill: 'none', stroke: 'var(--border)', 'stroke-dasharray': '4 4' }, svg);
    const t = svgEl('text', { x: W / 2, y: height / 2, 'text-anchor': 'middle', class: 'empty-chart' }, svg);
    t.textContent = msg;
  }

  // Hover layer: crosshair + tooltip, nearest generation to the pointer.
  function attachHover(key, host, f, render) {
    const tip = document.createElement('div');
    tip.className = 'tip hidden';
    host.appendChild(tip);
    const cross = svgEl('line', { class: 'crosshair', y1: f.m.t, y2: f.H - f.m.b, visibility: 'hidden' }, f.svg);
    const dots = svgEl('g', {}, f.svg);
    const hit = svgEl('rect', { x: f.m.l - 8, y: 0, width: f.pw + 16, height: f.H, fill: 'transparent' }, f.svg);
    const show = (gen) => {
      const h = S.histByGen.get(gen);
      if (!h) return hide();
      hover[key] = gen;
      const cx = f.x(gen);
      cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); cross.setAttribute('visibility', 'visible');
      while (dots.firstChild) dots.removeChild(dots.firstChild);
      const { html, points } = render(h);
      for (const p of points) if (isNum(p.v)) svgEl('circle', { class: 'hover-dot', cx, cy: f.y(p.v), r: 4.5, fill: p.color }, dots);
      tip.innerHTML = html;
      tip.classList.remove('hidden');
      const tw = tip.offsetWidth || 180;
      const left = cx + 14 + tw > f.W ? cx - 14 - tw : cx + 14;
      tip.style.left = Math.max(0, left) + 'px';
      tip.style.top = f.m.t + 'px';
    };
    const hide = () => {
      hover[key] = null;
      cross.setAttribute('visibility', 'hidden');
      while (dots.firstChild) dots.removeChild(dots.firstChild);
      tip.classList.add('hidden');
    };
    const nearest = (evt) => {
      const r = f.svg.getBoundingClientRect();
      const sx = r.width ? f.W / r.width : 1;
      const px = (evt.clientX - r.left) * sx;
      const g = Math.round(f.g0 + ((px - f.m.l) / f.pw) * (f.g1 - f.g0));
      // snap to the closest gen that actually exists
      let best = null;
      for (const h of S.hist) if (best === null || Math.abs(h.gen - g) < Math.abs(best - g)) best = h.gen;
      return best;
    };
    hit.addEventListener('mousemove', (e) => { const g = nearest(e); if (g !== null) show(g); });
    hit.addEventListener('mouseleave', hide);
    if (hover[key] !== null && S.histByGen.has(hover[key])) show(hover[key]);
  }

  // ---------------------------------------------------------------------------
  // Chart A: offense best / median / mean (+ worst..best band), defense best conceded
  // ---------------------------------------------------------------------------
  const SERIES_A = [
    { key: 'ob', label: 'offense best', short: 'best', color: 'var(--off)', width: 2.25, dash: null, get: (h) => num(get(h, 'off', 'best')) },
    { key: 'om', label: 'offense median', short: 'median', color: 'var(--off)', width: 1.75, dash: '6 4', get: (h) => num(get(h, 'off', 'median')) },
    { key: 'oa', label: 'offense mean', short: 'mean', color: 'var(--off)', width: 1.75, dash: '1.5 3.5', get: (h) => num(get(h, 'off', 'mean')) },
    { key: 'db', label: 'defense best (conceded)', short: 'D conceded', color: 'var(--def)', width: 2.25, dash: null, get: (h) => num(get(h, 'def', 'best')) },
  ];
  function lineSwatch(s) {
    return `<svg width="22" height="10" aria-hidden="true"><line x1="1" y1="5" x2="21" y2="5" stroke="${s.color}" stroke-width="${s.width}" stroke-linecap="round"${s.dash ? ` stroke-dasharray="${s.dash}"` : ''}/></svg>`;
  }
  function renderLegendA() {
    const html = SERIES_A.map((s) => `<span>${lineSwatch(s)}${esc(s.label)}</span>`).join('') +
      '<span><span class="sq" style="background:var(--off);opacity:.14"></span>offense worst–best range</span>';
    if ($('legendA').innerHTML !== html) $('legendA').innerHTML = html;
  }

  function renderChartA() {
    renderLegendA();
    const host = $('chartA');
    const H = 280;
    if (!S.hist.length) return emptyChart(host, H, S.run ? 'Waiting for the first generation…' : 'No run selected');
    const f = frame(host, H, { l: 38, r: 96, t: 10, b: 30 }, [0, 0.2, 0.4, 0.6, 0.8, 1], (t) => t.toFixed(1));
    const xs = S.hist.map((h) => f.x(h.gen));

    // worst..best band
    const band = [];
    S.hist.forEach((h, i) => { const hi = num(get(h, 'off', 'best')), lo = num(get(h, 'off', 'worst')); if (hi !== null && lo !== null) band.push([xs[i], f.y(hi), f.y(lo)]); });
    if (band.length > 1) {
      const d = 'M' + band.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join('L') +
        'L' + band.slice().reverse().map((p) => `${p[0].toFixed(1)},${p[2].toFixed(1)}`).join('L') + 'Z';
      svgEl('path', { d, fill: 'var(--off)', 'fill-opacity': 0.09, stroke: 'none' }, f.svg);
    }

    const labels = [];
    for (const s of SERIES_A) {
      const pts = S.hist.map((h, i) => [xs[i], s.get(h) === null ? null : f.y(s.get(h))]);
      const d = linePath(pts);
      if (d) {
        const attrs = { d, fill: 'none', stroke: s.color, 'stroke-width': s.width, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' };
        if (s.dash) attrs['stroke-dasharray'] = s.dash;
        svgEl('path', attrs, f.svg);
      }
      if (S.hist.length <= 2) {   // too few points for a visible line: show markers
        pts.forEach(([px, py]) => { if (py !== null) svgEl('circle', { cx: px, cy: py, r: 4, fill: s.color, stroke: 'var(--panel)', 'stroke-width': 2 }, f.svg); });
      }
      const lv = s.get(last());
      if (lv !== null) labels.push({ y: f.y(lv) + 4, text: `${s.short} ${fmtRate(lv)}`, color: s.color });
    }
    spreadLabels(labels, 13, f.m.t + 4, f.H - f.m.b);
    const lx = f.x(last().gen) + 8;
    for (const lb of labels) {
      const t = svgEl('text', { x: lx, y: lb.y, class: 'dl' }, f.svg);
      t.textContent = lb.text;
    }

    attachHover('A', host, f, (h) => {
      const rows = [
        ['offense best', get(h, 'off', 'best'), 'var(--off)'],
        ['median', get(h, 'off', 'median'), 'var(--off)'],
        ['mean', get(h, 'off', 'mean'), 'var(--off)'],
        ['worst', get(h, 'off', 'worst'), null],
      ];
      const drows = [['defense best', get(h, 'def', 'best'), 'var(--def)'], ['defense mean', get(h, 'def', 'mean'), null]];
      const tr = (r, cls) => `<tr${cls ? ` class="${cls}"` : ''}><td>${r[2] ? `<span class="sw" style="background:${r[2]}"></span>` : '<span class="sw"></span>'}${esc(r[0])}</td><td class="v">${fmtRate(num(r[1]))}</td></tr>`;
      const champ = get(h, 'champion', 'family');
      return {
        html: `<div class="t-head">Generation ${h.gen}${champ ? ` <span class="muted">· champion ${esc(champ)}</span>` : ''}</div><table>` +
          rows.map((r) => tr(r)).join('') + drows.map((r, i) => tr(r, i === 0 ? 'sep' : '')).join('') + '</table>',
        points: [
          { v: num(get(h, 'off', 'best')), color: 'var(--off)' },
          { v: num(get(h, 'off', 'median')), color: 'var(--off)' },
          { v: num(get(h, 'off', 'mean')), color: 'var(--off)' },
          { v: num(get(h, 'def', 'best')), color: 'var(--def)' },
        ],
      };
    });
  }

  // ---------------------------------------------------------------------------
  // Chart B: stacked share of each structure family
  // ---------------------------------------------------------------------------
  function sharesOf(h, order) {
    const fams = h && typeof h.families === 'object' && h.families ? h.families : {};
    const counts = order.map((f) => Math.max(0, num(fams[f]) ?? 0));
    const total = counts.reduce((a, b) => a + b, 0);
    return { counts, total, shares: counts.map((c) => (total > 0 ? c / total : 0)) };
  }

  function renderChartB() {
    const order = familyOrder();
    const lh = order.map((f) => `<span><span class="sq" style="background:${famColor(f, order)}"></span>${esc(f)}</span>`).join('');
    if ($('legendB').innerHTML !== lh) $('legendB').innerHTML = lh;
    const host = $('chartB');
    const H = 250;
    if (!S.hist.length || !order.length) return emptyChart(host, H, S.run ? 'Waiting for the first generation…' : 'No run selected');
    const f = frame(host, H, { l: 38, r: 96, t: 8, b: 30 }, [0, 0.25, 0.5, 0.75, 1], (t) => Math.round(t * 100) + '%');
    const data = S.hist.map((h) => sharesOf(h, order));
    const xs = S.hist.map((h) => f.x(h.gen));
    const n = S.hist.length;
    const cum = data.map(() => 0);
    const bands = [];
    order.forEach((fam, k) => {
      const lo = cum.slice();
      data.forEach((d, i) => { cum[i] += d.shares[k]; });
      bands.push({ fam, k, lo, hi: cum.slice() });
    });
    const color = (fam) => famColor(fam, order);
    if (n === 1) {
      // a single generation: one stacked bar
      const bw = Math.min(40, f.pw / 3), x0 = xs[0] - bw / 2;
      for (const b of bands) {
        const h = (b.hi[0] - b.lo[0]) * f.ph;
        if (h <= 0) continue;
        svgEl('rect', { x: x0, y: f.y(b.hi[0]), width: bw, height: h, fill: color(b.fam), stroke: 'var(--panel)', 'stroke-width': 1.5 }, f.svg);
      }
    } else {
      for (const b of bands) {
        if (b.hi.every((v, i) => v - b.lo[i] <= 0)) continue;
        const up = xs.map((x, i) => `${x.toFixed(1)},${f.y(b.hi[i]).toFixed(1)}`);
        const dn = xs.map((x, i) => `${x.toFixed(1)},${f.y(b.lo[i]).toFixed(1)}`).reverse();
        svgEl('path', { d: 'M' + up.join('L') + 'L' + dn.join('L') + 'Z', fill: color(b.fam), stroke: 'var(--panel)', 'stroke-width': 1.25, 'stroke-linejoin': 'round' }, f.svg);
      }
    }
    // direct labels at the right edge for bands big enough to hold one
    const labels = [];
    for (const b of bands) {
      const share = b.hi[n - 1] - b.lo[n - 1];
      if (share >= 0.06) labels.push({ y: f.y((b.hi[n - 1] + b.lo[n - 1]) / 2) + 4, text: `${b.fam} ${Math.round(share * 100)}%` });
    }
    spreadLabels(labels, 13, f.m.t + 4, f.H - f.m.b);
    const lx = (n === 1 ? xs[0] + Math.min(40, f.pw / 3) / 2 : xs[n - 1]) + 8;
    for (const lb of labels) {
      const t = svgEl('text', { x: lx, y: lb.y, class: 'dl' }, f.svg);
      t.textContent = lb.text;
    }

    attachHover('B', host, f, (h) => {
      const d = sharesOf(h, order);
      const rows = order.map((fam, k) => ({ fam, c: d.counts[k], s: d.shares[k] })).reverse();   // top of the stack first
      const champ = get(h, 'champion', 'family');
      return {
        html: `<div class="t-head">Generation ${h.gen} <span class="muted">· ${d.total} offenses</span></div><table>` +
          rows.map((r) => `<tr${r.c ? '' : ' class="muted"'}><td><span class="sw" style="background:${color(r.fam)}"></span>${esc(r.fam)}${r.fam === champ ? ' <span class="muted">(champion)</span>' : ''}</td><td class="v">${r.c}</td><td class="m">${Math.round(r.s * 100)}%</td></tr>`).join('') +
          '</table>',
        points: [],
      };
    });
  }

  // ---------------------------------------------------------------------------
  // Panel C: best score rate per family (history.family_best merged with archive.json)
  // ---------------------------------------------------------------------------
  function renderFamilyBest() {
    const box = $('famBest');
    const order = familyOrder();
    const l = last();
    if (!order.length || (!l && !get(S.archive, 'families'))) { box.className = 'muted'; box.innerHTML = S.run ? 'No generations yet.' : ''; return; }
    const fb = get(l, 'family_best') || {};
    const ar = get(S.archive, 'families') || {};
    const champ = get(l, 'champion', 'family');
    const rows = order.map((f) => {
      const a = ar[f];
      let v = num(fb[f]);
      const av = num(get(a, 'score_rate'));
      if (av !== null && (v === null || av > v)) v = av;
      return { f, v, gen: num(get(a, 'gen')), a };
    });
    rows.sort((p, q) => (q.v ?? -1) - (p.v ?? -1) || order.indexOf(p.f) - order.indexOf(q.f));
    const maxV = Math.max(0.1, ...rows.map((r) => r.v ?? 0));
    const top = Math.min(1, Math.ceil(maxV * 10 - 1e-9) / 10);
    const html = '<div class="bars">' + rows.map((r) => {
      const title = r.a ? `${r.f}: ${fmtRate(r.v)} (found gen ${r.gen ?? '?'})` + (r.a.descriptors ? '\n' + Object.entries(r.a.descriptors).map(([k, v]) => `${humanKey(k)}: ${fmtVal(v)}`).join('\n') : '') : `${r.f}: ${r.v === null ? 'never seen' : fmtRate(r.v)}`;
      if (r.v === null) {
        return `<div class="bar-row null" title="${esc(title)}"><span class="b-name"><span class="sw" style="background:var(--null)"></span>${esc(r.f)}</span><span class="b-track"></span><span class="b-val">not seen</span></div>`;
      }
      return `<div class="bar-row" title="${esc(title)}"><span class="b-name"><span class="sw" style="background:${famColor(r.f, order)}"></span>${esc(r.f)}${r.f === champ ? '<span class="tag">champ</span>' : ''}</span>` +
        `<span class="b-track"><span class="b-fill" style="display:block;width:${(clamp(r.v / top, 0, 1) * 100).toFixed(1)}%;background:${famColor(r.f, order)}"></span></span>` +
        `<span class="b-val">${fmtRate(r.v)}${r.gen !== null ? `<small>g${r.gen}</small>` : ''}</span></div>`;
    }).join('') + `</div><div class="bar-scale"><span></span><span><span>0</span><span>${top.toFixed(1)}</span></span><span></span></div>` +
      '<div class="note">Best score rate ever evaluated per structure family; g = generation it was found. Grey = no member seen yet.</div>';
    box.className = '';
    if (box.innerHTML !== html) box.innerHTML = html;
  }

  // ---------------------------------------------------------------------------
  // Panel D: defense scheme mix + defense champion
  // ---------------------------------------------------------------------------
  function renderDefense() {
    const l = last(), pv = prev();
    const box = $('defMix');
    const mix = get(l, 'def_schemes');
    if (!mix || typeof mix !== 'object') {
      box.className = 'muted'; box.innerHTML = l ? 'No defense scheme data in this run.' : (S.run ? 'No generations yet.' : '');
      $('defMixNote').textContent = '';
    } else {
      const order = [];
      (Array.isArray(get(S.meta, 'def_schemes')) ? S.meta.def_schemes : []).forEach((s) => { if (!order.includes(s)) order.push(s); });
      Object.keys(mix).forEach((s) => { if (!order.includes(s)) order.push(s); });
      const total = order.reduce((a, s) => a + Math.max(0, num(mix[s]) ?? 0), 0);
      const pmix = get(pv, 'def_schemes') || {};
      const rows = order.map((s) => ({ s, c: Math.max(0, num(mix[s]) ?? 0), d: pv ? (num(mix[s]) ?? 0) - (num(pmix[s]) ?? 0) : null }));
      rows.sort((a, b) => b.c - a.c || order.indexOf(a.s) - order.indexOf(b.s));
      const maxC = Math.max(1, ...rows.map((r) => r.c));
      const dchamp = get(l, 'def_champion');
      const champKey = dchamp ? [dchamp.scheme, dchamp.force].filter(Boolean).join('_') : null;
      $('defMixNote').textContent = `${total} defenses, generation ${l.gen}`;
      const html = '<div class="bars">' + rows.map((r) => {
        const dl = r.d ? `<small>${r.d > 0 ? '+' : '−'}${Math.abs(r.d)}</small>` : '';
        const isChamp = champKey && (r.s === champKey || r.s === dchamp.scheme);
        return `<div class="bar-row${r.c ? '' : ' null'}" title="${esc(r.s)}: ${r.c} of ${total}${r.d !== null ? ` (${r.d >= 0 ? '+' : ''}${r.d} vs previous gen)` : ''}">` +
          `<span class="b-name">${esc(prettyScheme(r.s))}${isChamp ? '<span class="tag">champ</span>' : ''}</span>` +
          `<span class="b-track"><span class="b-fill" style="display:block;width:${(r.c / maxC * 100).toFixed(1)}%;background:var(--def)"></span></span>` +
          `<span class="b-val">${r.c} <small>${total ? Math.round(r.c / total * 100) : 0}%</small>${dl}</span></div>`;
      }).join('') + '</div><div class="note">Count in the defense population; small numbers are the change vs the previous generation.</div>';
      box.className = '';
      if (box.innerHTML !== html) box.innerHTML = html;
    }

    const dbox = $('defChamp');
    const dc = get(l, 'def_champion');
    if (!dc || typeof dc !== 'object') { dbox.className = 'muted'; dbox.innerHTML = l ? 'No defense champion in this run.' : (S.run ? 'No generations yet.' : ''); return; }
    const pdc = get(pv, 'def_champion');
    const name = [dc.scheme, dc.force].filter(Boolean).map(esc).join(' · ');
    const html = `<div class="champ-head"><span class="fam-dot" style="background:var(--def)"></span><div><div class="c-name">${name || 'defense'}</div>` +
      `<div class="c-sub">generation ${l.gen} · lowest score rate conceded</div></div>` +
      `<div class="c-rate"><b>${fmtRate(num(dc.conceded))}</b><span>conceded ${deltaHtml(num(dc.conceded), num(get(pdc, 'conceded')), true)}</span></div></div>` +
      `<div class="gene-tables one">${kvTable('genes', dc.genes, get(pdc, 'genes'))}</div>`;
    dbox.className = '';
    if (dbox.innerHTML !== html) dbox.innerHTML = html;
  }

  // compact key/value table; values that changed vs the previous champion are tinted
  function kvTable(caption, obj, prevObj) {
    if (!obj || typeof obj !== 'object' || !Object.keys(obj).length) return `<table class="gtable"><caption>${esc(caption)}</caption><tr><td>–</td><td></td></tr></table>`;
    const rows = Object.entries(obj).map(([k, v]) => {
      const pvv = prevObj && typeof prevObj === 'object' ? prevObj[k] : undefined;
      const chg = pvv !== undefined && fmtVal(pvv) !== fmtVal(v);
      return `<tr><td>${esc(humanKey(k))}</td><td${chg ? ` class="chg" title="was ${esc(fmtVal(pvv))}"` : ''}>${esc(fmtVal(v))}</td></tr>`;
    }).join('');
    return `<table class="gtable"><caption>${esc(caption)}</caption>${rows}</table>`;
  }

  // ---------------------------------------------------------------------------
  // Champion card
  // ---------------------------------------------------------------------------
  function renderChampion() {
    const box = $('champ');
    const l = last(), pv = prev();
    const c = get(l, 'champion');
    if (!c || typeof c !== 'object') { box.className = 'muted'; box.innerHTML = S.run ? 'No generations yet.' : ''; $('champNote').textContent = ''; return; }
    const pc = get(pv, 'champion');
    const same = pc && pc.family === c.family;
    $('champNote').textContent = `generation ${l.gen}`;
    const html = `<div class="champ-head"><span class="fam-dot" style="background:${famColor(c.family)}"></span>` +
      `<div><div class="c-name">${esc(c.family ?? '?')}</div><div class="c-sub">best offense of generation ${l.gen}${pc ? (same ? ' · same family as before' : ` · was ${esc(pc.family)}`) : ''}</div></div>` +
      `<div class="c-rate"><b>${fmtRate(num(c.score_rate))}</b><span>score rate ${deltaHtml(num(c.score_rate), num(get(pc, 'score_rate')), false)}</span></div></div>` +
      `<div class="gene-tables">${kvTable('descriptors', c.descriptors, same ? get(pc, 'descriptors') : null)}${kvTable('genes', c.genes, same ? get(pc, 'genes') : null)}</div>` +
      (same ? '<div class="note">Tinted values changed since the previous generation’s champion.</div>' : '');
    box.className = '';
    if (box.innerHTML !== html) box.innerHTML = html;
  }

  // ---------------------------------------------------------------------------
  // Play stats with sparklines
  // ---------------------------------------------------------------------------
  const STAT_FMT = { completion: (v) => fmtPct(v), huck_share: (v) => fmtPct(v), break_share: (v) => fmtPct(v), throws_per_poss: (v) => fmtVal(v), games: (v) => fmtBig(v) };
  function spark(values) {
    const pts = values.map((v, i) => [i, v]).filter((p) => isNum(p[1]));
    if (pts.length < 2) return '<svg viewBox="0 0 100 22" preserveAspectRatio="none"></svg>';
    const lo = Math.min(...pts.map((p) => p[1])), hi = Math.max(...pts.map((p) => p[1]));
    const n = values.length - 1 || 1;
    const sx = (i) => 2 + (i / n) * 96, sy = (v) => (hi > lo ? 19 - ((v - lo) / (hi - lo)) * 16 : 11);
    const d = pts.map((p, i) => (i ? 'L' : 'M') + sx(p[0]).toFixed(1) + ',' + sy(p[1]).toFixed(1)).join('');
    const lp = pts[pts.length - 1];
    return `<svg viewBox="0 0 100 22" preserveAspectRatio="none" aria-hidden="true"><path d="${d}" vector-effect="non-scaling-stroke"/><circle cx="${sx(lp[0]).toFixed(1)}" cy="${sy(lp[1]).toFixed(1)}" r="1.8"/></svg>`;
  }
  function renderPlayStats() {
    const box = $('playStats');
    const l = last();
    const st = get(l, 'stats');
    if (!st || typeof st !== 'object') { box.className = 'muted'; box.innerHTML = l ? 'No play stats in this run.' : (S.run ? 'No generations yet.' : ''); return; }
    const keys = Object.keys(st);
    const html = '<div class="stat-rows">' + keys.map((k) => {
      const fmt = STAT_FMT[k] || fmtVal;
      const series = S.hist.map((h) => num(get(h, 'stats', k)));
      return `<span class="s-name">${esc(humanKey(k))}</span>${k === 'games' ? '<span></span>' : spark(series)}<span class="s-val">${esc(fmt(st[k]))}</span>`;
    }).join('') + '</div><div class="note">Sparklines: every generation so far (each scaled to its own range).</div>';
    box.className = '';
    if (box.innerHTML !== html) box.innerHTML = html;
  }

  // ---------------------------------------------------------------------------
  // Embedded replay
  // ---------------------------------------------------------------------------
  function viewerUrl(name) {
    return 'index.html?replay=' + encodeURIComponent(runBase(name) + 'latest_best.json') + '&live=1';
  }
  function renderReplay() {
    const box = $('replayBox');
    const link = $('replayLink');
    if (!S.run || !S.replay) {
      S.replaySrc = null;
      link.classList.add('hidden');
      if (!box.querySelector('.placeholder')) {
        box.innerHTML = '<div class="placeholder"><b>No champion replay yet</b>The first champion replay appears after a few generations.</div>';
      }
      return;
    }
    const src = viewerUrl(S.run);
    link.href = src;
    link.classList.remove('hidden');
    if (S.replaySrc === src && box.querySelector('iframe')) return;   // the viewer's own live mode reloads it
    S.replaySrc = src;
    box.innerHTML = '';
    const fr = document.createElement('iframe');
    fr.className = 'replay-frame';
    fr.title = 'Champion replay';
    fr.src = src;
    box.appendChild(fr);
  }

  // ---------------------------------------------------------------------------
  // Banner (empty / error states)
  // ---------------------------------------------------------------------------
  function renderBanner() {
    const b = $('banner');
    let html = '';
    if (S.error && !S.runs.length && !S.run) {
      html = `Could not reach <code>/api/runs</code> (${esc(S.error)}). Start the server with <code>python viz/server.py</code>.`;
    } else if (!S.run) {
      html = 'No evolution runs found under <code>runs/</code>. Start one with <code>ultisim/evolve.py</code>, or create a fake one with <code>python viz/make_sample_run.py</code>, then press &#x21bb;.';
    } else if (!S.meta && !S.hist.length && !S.status) {
      html = `Waiting for <code>runs/${esc(S.run)}/</code> to write its first files…`;
    } else if (!S.hist.length) {
      html = 'Run started &mdash; waiting for the first generation to finish.';
    } else if (get(S.status, 'state') === 'error') {
      html = `The run reported an error${get(S.status, 'message') ? `: ${esc(S.status.message)}` : '.'}`;
    }
    b.classList.toggle('hidden', !html);
    if (b.innerHTML !== html) b.innerHTML = html;
  }

  function renderCharts() { renderChartA(); renderChartB(); }
  function renderPanels() { renderFamilyBest(); renderDefense(); renderChampion(); renderPlayStats(); }
  function renderAll() {
    renderHeader(); renderKpis(); renderBanner(); renderCharts(); renderPanels(); renderReplay();
    document.title = S.run ? `${S.run} · UltiSim Evolution` : 'UltiSim Evolution';
  }

  // ---------------------------------------------------------------------------
  // Run list & polling
  // ---------------------------------------------------------------------------
  async function fetchRuns() {
    const res = await fetch(bust('/api/runs'), { cache: 'no-store' });
    if (!res.ok) throw new Error(`/api/runs: HTTP ${res.status}`);
    const list = await res.json();
    S.runs = Array.isArray(list) ? list.filter((r) => r && typeof r.name === 'string') : [];
    return S.runs;
  }

  function fillSelect() {
    const sel = $('runSelect');
    const opts = S.runs.map((r) => {
      const bits = [r.state || (r.kind ? r.kind : 'no status')];
      if (isNum(r.gen)) bits.push(`gen ${r.gen}${isNum(r.planned) ? '/' + r.planned : ''}`);
      const ago = fmtAgo(r.mtime);
      if (ago) bits.push(ago);
      return { value: r.name, label: `${r.name}  (${bits.join(' · ')})` };
    });
    if (S.run && !opts.some((o) => o.value === S.run)) opts.unshift({ value: S.run, label: `${S.run}  (not found yet)` });
    const key = JSON.stringify(opts) + '|' + S.run;
    if (sel.dataset.key === key) return;
    sel.dataset.key = key;
    sel.innerHTML = '';
    if (!opts.length) {
      const o = document.createElement('option');
      o.value = ''; o.textContent = 'No runs in runs/';
      sel.appendChild(o);
      return;
    }
    for (const o of opts) {
      const e = document.createElement('option');
      e.value = o.value; e.textContent = o.label;
      if (o.value === S.run) e.selected = true;
      sel.appendChild(e);
    }
    sel.value = S.run || '';
  }

  function selectRun(name, pushUrl) {
    if (name === S.run) return;
    switchRun(name, pushUrl);
    renderAll();
    poll();
  }
  function switchRun(name, pushUrl) {
    S.run = name || null;
    S.generation++;
    S.meta = null; S.status = null; S.archive = null; S.replay = false; S.replaySrc = null;
    resetHistory();
    hover.A = hover.B = null;
    if (pushUrl) {
      try {
        const url = new URL(window.location.href);
        if (S.run) url.searchParams.set('run', S.run); else url.searchParams.delete('run');
        window.history.replaceState(null, '', url.toString());
      } catch (_) { /* non-browser */ }
    }
    fillSelect();
  }

  async function poll() {
    if (polling) { pollAgain = true; return; }
    polling = true;
    pollAgain = false;
    try {
      if (S.pollN % RUNS_EVERY === 0) {
        try { await fetchRuns(); S.error = null; fillSelect(); } catch (err) { S.error = err.message; }
        if (!S.run && S.runs.length) switchRun(S.runs[0].name, true);   // first run appeared
      }
      S.pollN++;
      if (!S.run) { renderAll(); return; }
      const myGen = S.generation;
      const base = runBase(S.run);
      const [metaR, statusR, histT, archR, rep] = await Promise.all([
        S.meta ? Promise.resolve(null) : fetchJson(base + 'meta.json'),
        fetchJson(base + 'status.json'),
        fetchText(base + 'history.jsonl'),
        fetchJson(base + 'archive.json'),
        S.replay ? Promise.resolve(true) : fetch(bust(base + 'latest_best.json'), { method: 'HEAD', cache: 'no-store' }).then((r) => r.ok).catch(() => false),
      ]);
      if (myGen !== S.generation) return;   // the user switched runs meanwhile

      let chartsDirty = false, panelsDirty = false;
      if (metaR && metaR.value && typeof metaR.value === 'object') { S.meta = metaR.value; chartsDirty = panelsDirty = true; }
      // a JSON parse failure (file caught mid-write) keeps the previous value
      if (statusR.missing) S.status = null; else if (statusR.value && typeof statusR.value === 'object') S.status = statusR.value;
      if (ingestHistory(histT)) chartsDirty = panelsDirty = true;
      if (!archR.missing && archR.value && typeof archR.value === 'object') {
        const k = JSON.stringify(archR.value);
        if (k !== S._archKey) { S._archKey = k; S.archive = archR.value; panelsDirty = true; }
      }
      if (rep && !S.replay) S.replay = true;

      S.lastPollAt = Date.now();
      S.lastPollOk = true;
      setStatus(S.histBad ? `${S.histBad} unreadable history line(s) skipped` : '', false);
      renderHeader(); renderKpis(); renderBanner();
      if (chartsDirty) renderCharts();
      if (panelsDirty) renderPanels();
      renderReplay();
      document.title = `${S.run} · UltiSim Evolution`;
    } catch (err) {
      S.lastPollOk = false;
      setStatus(`poll failed: ${err.message} (will retry)`, true);
    } finally {
      polling = false;
      if (pollAgain) setTimeout(poll, 0);   // a run switch happened mid-poll
    }
  }

  function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => { if (!document.hidden) poll(); }, POLL_MS);
  }

  // ---------------------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------------------
  async function boot() {
    $('runSelect').addEventListener('change', (e) => selectRun(e.target.value, true));
    $('refreshRuns').addEventListener('click', async () => {
      try { await fetchRuns(); S.error = null; fillSelect(); setStatus('run list refreshed'); } catch (err) { setStatus(err.message, true); }
      if (!S.run && S.runs.length) selectRun(S.runs[0].name, true);
    });
    document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });
    let rt = null;
    window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(renderCharts, 120); });
    setInterval(renderTiming, 1000);

    let want = null;
    try { want = new URLSearchParams(window.location.search).get('run'); } catch (_) { /* ignore */ }
    try { await fetchRuns(); S.pollN = 1; } catch (err) { S.error = err.message; setStatus(`Run list unavailable: ${err.message}`, true); }
    if (!want && S.runs.length) want = S.runs[0].name;   // newest first
    renderAll();
    if (want) selectRun(want, false); else { fillSelect(); renderAll(); }
    startPolling();
  }

  // exposed for debugging / tests
  window.__dash = { S, poll, selectRun, ingestHistory, renderAll };
  boot();
})();
