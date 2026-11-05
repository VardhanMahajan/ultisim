/* UltiSim replay viewer (v1). Plain JS, no build step.
 * Reads the JSON contract in viz/REPLAY_FORMAT.md ("ultisim-replay", version 1).
 */
'use strict';

(function () {
  // ---------------------------------------------------------------------------
  // Constants
  // ---------------------------------------------------------------------------
  const SPEEDS = [0.25, 0.5, 1, 2, 4, 8];
  const TRAIL_S = 1.5;            // seconds of motion trail per player
  const TRAIL_SELECTED_S = 4.0;   // longer trail for the selected player
  const MARGIN = 3.0;             // metres of out-of-bounds shown around the field
  const MARK_RADIUS = 4.0;        // metres, marker wedge radius
  const TELEPORT_M = 5.0;         // a jump this big between frames isn't interpolated
  const FLAG_S = 1.4;             // seconds an on-field event flag stays visible
  const POLL_MS = 3000;

  const STATE_NAMES = [
    'set', 'cutting', 'clearing', 'chasing disc', 'on the ground',
    'marking', 'holding disc', 'defending', 'zone',
  ];
  const DISC_STATES = ['held', 'in the air', 'on the ground'];
  const TURNOVER_TYPES = new Set(['block', 'interception', 'drop', 'throwaway', 'out_of_bounds',
    'stall', 'mark_block', 'turnover']);
  const FLAG_LABELS = {
    score: 'SCORE', block: 'D!', interception: 'INTERCEPTION', drop: 'DROP', throwaway: 'THROWAWAY',
    out_of_bounds: 'OUT', stall: 'STALL', mark_block: 'HAND BLOCK', layout: 'LAYOUT', turnover: 'TURN',
    timeout: 'TIMEOUT', pull: 'PULL',
  };
  const TYPE_ABBR = { backhand: 'BH', forehand: 'FH', hammer: 'HAM' };

  const $ = (id) => document.getElementById(id);
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  // ---------------------------------------------------------------------------
  // DOM
  // ---------------------------------------------------------------------------
  const canvas = $('field');
  const ctx = canvas.getContext('2d');
  const wrap = $('canvasWrap');
  const scrub = $('scrub');
  const el = {
    select: $('replaySelect'), refresh: $('refreshList'), live: $('liveToggle'), liveStatus: $('liveStatus'),
    status: $('status'), title: $('title'), teams: $('teams'), banner: $('resultBanner'), empty: $('emptyMsg'),
    marks: $('eventMarks'), play: $('btnPlay'), back: $('btnBack'), fwd: $('btnFwd'), restart: $('btnRestart'),
    speeds: $('speeds'), clock: $('clock'), frameNo: $('frameNo'), hud: $('hud'), playerPanel: $('playerPanel'),
    closePlayer: $('closePlayer'), ticker: $('ticker'), eventCount: $('eventCount'), legend: $('legend'),
  };

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  const view = {
    R: null,              // normalized replay
    path: null,           // URL path of the loaded replay
    sig: null,            // size:mtime signature used by live mode
    t: 0,                 // playback time (s)
    playing: false,
    speed: 1,
    selected: -1,         // selected player index
    hover: -1,
    scrubbing: false,
    list: [],             // last /api/replays result
    pos: [],              // last drawn player positions in CSS px [[x, y], ...]
    tickerKey: '',
    panelKey: '',
  };
  let theme = null;
  let geom = { cssW: 0, cssH: 0, scale: 1, dpr: 1 };
  let fieldCache = null;
  let dirty = true;
  let liveTimer = null;
  let polling = false;

  // ---------------------------------------------------------------------------
  // Theme (CSS variables -> canvas colours)
  // ---------------------------------------------------------------------------
  function hexToRgb(h) {
    h = String(h).trim();
    if (h[0] === '#') h = h.slice(1);
    if (h.length === 3) h = h.split('').map((c) => c + c).join('');
    const n = parseInt(h, 16);
    if (Number.isNaN(n) || h.length !== 6) return [128, 128, 128];
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const rgba = (rgb, a) => `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`;
  const mix = (rgb, to, k) => rgb.map((c, i) => Math.round(c + (to[i] - c) * k));

  function readTheme() {
    const cs = getComputedStyle(document.documentElement);
    const v = (name, fb) => (cs.getPropertyValue(name) || fb).trim() || fb;
    const team = [hexToRgb(v('--team0', '#2563eb')), hexToRgb(v('--team1', '#ea580c'))];
    theme = {
      team,
      teamBright: team.map((c) => mix(c, [255, 255, 255], 0.38)),
      teamCss: team.map((c) => rgba(c, 1)),
      grass: v('--grass', '#4c9a4d'), grass2: v('--grass-2', '#469047'), oob: v('--oob', '#3c7c3e'),
      line: hexToRgb(v('--line', '#ffffff')),
      holder: hexToRgb(v('--holder', '#fde047')),
      disc: v('--disc', '#ffffff'),
      arc: {
        backhand: hexToRgb(v('--arc-backhand', '#ffffff')),
        forehand: hexToRgb(v('--arc-forehand', '#fde047')),
        hammer: hexToRgb(v('--arc-hammer', '#f0abfc')),
      },
      text: v('--field-text', '#ffffff'),
      font: v('--font', 'system-ui, sans-serif'),
    };
  }

  // ---------------------------------------------------------------------------
  // Replay normalization
  // ---------------------------------------------------------------------------
  function normalize(data) {
    if (!data || typeof data !== 'object') throw new Error('not a JSON object');
    if (data.format !== 'ultisim-replay') throw new Error(`format is "${data.format}", expected "ultisim-replay"`);
    const frames = Array.isArray(data.frames) ? data.frames : [];
    if (!frames.length) throw new Error('replay has no frames');
    const meta = Object.assign({ title: 'Untitled replay', mode: 'possession', dt: 0.1, teams: [] }, data.meta || {});
    const dt = Number(meta.dt) > 0 ? Number(meta.dt) : 0.1;
    const field = Object.assign({ length: 100, width: 37, endzone: 18 }, data.field || {});
    const nP = Math.floor((frames[0].p || []).length / 2);

    // Player k in "p" / "st" is the player with id k (fall back to array order).
    const byId = new Map((data.players || []).map((p) => [p.id, p]));
    const players = [];
    for (let k = 0; k < nP; k++) {
      const src = byId.get(k) || (data.players || [])[k] || {};
      players.push({
        id: src.id != null ? src.id : k,
        team: src.team != null ? src.team : (k < nP / 2 ? 0 : 1),
        num: src.num != null ? src.num : k,
        stats: src.stats || {},
        raw: src,
      });
    }
    const teams = [0, 1].map((i) => Object.assign({ name: `Team ${i}`, strategy: '' }, (meta.teams || [])[i] || {}));
    const throws = (Array.isArray(data.throws) ? data.throws : []).slice().sort((a, b) => a.t0 - b.t0);
    const events = (Array.isArray(data.events) ? data.events : []).slice().sort((a, b) => a.t - b.t);
    const possessions = (Array.isArray(data.possessions) ? data.possessions : []).slice().sort((a, b) => a.t0 - b.t0);

    // Per-player summaries from throws/events (for the side panel).
    const summary = players.map(() => ({ throws: 0, completions: 0, catches: 0, scores: 0, assists: 0, ds: 0, turns: 0, yards: 0 }));
    for (const th of throws) {
      const s = summary[th.thrower];
      if (s) {
        s.throws++;
        if (th.outcome === 'catch' || th.outcome === 'score') {
          s.completions++;
          if (th.path && th.path.length === 3) s.yards += th.path[2][0] - th.path[0][0];
        }
        if (th.outcome === 'score') s.assists++;
      }
    }
    for (const e of events) {
      const s = summary[e.player];
      if (!s) continue;
      if (e.type === 'catch') s.catches++;
      else if (e.type === 'score') s.scores++;
      else if (e.type === 'block' || e.type === 'interception' || e.type === 'mark_block') s.ds++;
      else if (e.type === 'drop' || e.type === 'throwaway' || e.type === 'out_of_bounds' || e.type === 'stall') s.turns++;
    }

    return {
      raw: data, meta, dt, field, nP, players, teams, frames, throws, events, possessions,
      result: data.result || null, summary,
      nF: frames.length, duration: (frames.length - 1) * dt, version: data.version,
    };
  }

  // ---------------------------------------------------------------------------
  // Geometry
  // ---------------------------------------------------------------------------
  function fieldDims() {
    const f = view.R ? view.R.field : { length: 100, width: 37, endzone: 18 };
    return { L: +f.length || 100, W: +f.width || 37, EZ: +f.endzone || 18 };
  }
  const X = (x) => (x + MARGIN) * geom.scale;
  const Y = (y) => (fieldDims().W + MARGIN - y) * geom.scale;

  function resize() {
    const { L, W } = fieldDims();
    const aspect = (L + 2 * MARGIN) / (W + 2 * MARGIN);
    const wrapW = wrap.clientWidth || 800;
    const maxH = Math.max(240, window.innerHeight - 210);
    const cssW = Math.floor(Math.min(wrapW, maxH * aspect));
    const cssH = Math.floor(cssW / aspect);
    const dpr = window.devicePixelRatio || 1;
    if (cssW === geom.cssW && cssH === geom.cssH && dpr === geom.dpr && fieldCache) return;
    geom = { cssW, cssH, dpr, scale: cssW / (L + 2 * MARGIN) };
    canvas.style.width = cssW + 'px';
    canvas.style.height = cssH + 'px';
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    buildFieldCache();
    dirty = true;
  }

  function buildFieldCache() {
    fieldCache = document.createElement('canvas');
    fieldCache.width = canvas.width;
    fieldCache.height = canvas.height;
    const c = fieldCache.getContext('2d');
    c.setTransform(geom.dpr, 0, 0, geom.dpr, 0, 0);
    drawField(c);
  }

  function drawField(c) {
    const { L, W, EZ } = fieldDims();
    const s = geom.scale;
    c.fillStyle = theme.oob;
    c.fillRect(0, 0, geom.cssW, geom.cssH);

    // mowing stripes, 5 m bands
    for (let x = 0, i = 0; x < L; x += 5, i++) {
      c.fillStyle = i % 2 ? theme.grass2 : theme.grass;
      const x1 = Math.min(L, x + 5);
      c.fillRect(X(x), Y(W), X(x1) - X(x), Y(0) - Y(W));
    }

    // end zones tinted by the team that scores in them (team 0 attacks x = L, team 1 attacks x = 0)
    const ez = [[L - EZ, L, 0], [0, EZ, 1]];
    for (const [x0, x1, team] of ez) {
      c.fillStyle = rgba(theme.team[team], 0.16);
      c.fillRect(X(x0), Y(W), X(x1) - X(x0), Y(0) - Y(W));
    }

    // lines
    const lw = Math.max(1.5, 0.12 * s);
    c.strokeStyle = rgba(theme.line, 0.92);
    c.lineWidth = lw;
    c.strokeRect(X(0), Y(W), X(L) - X(0), Y(0) - Y(W));
    c.beginPath();
    c.moveTo(X(EZ), Y(0)); c.lineTo(X(EZ), Y(W));
    c.moveTo(X(L - EZ), Y(0)); c.lineTo(X(L - EZ), Y(W));
    c.stroke();

    // brick marks: 20 m from each goal line, centred (WFDF)
    c.lineWidth = Math.max(1.2, 0.09 * s);
    const bk = 0.6;
    for (const bx of [EZ + 20, L - EZ - 20]) {
      if (bx <= EZ || bx >= L - EZ) continue;
      c.beginPath();
      c.moveTo(X(bx - bk), Y(W / 2 - bk)); c.lineTo(X(bx + bk), Y(W / 2 + bk));
      c.moveTo(X(bx - bk), Y(W / 2 + bk)); c.lineTo(X(bx + bk), Y(W / 2 - bk));
      c.stroke();
    }
    // midfield tick marks on the sidelines
    c.beginPath();
    c.moveTo(X(L / 2), Y(0)); c.lineTo(X(L / 2), Y(0.8));
    c.moveTo(X(L / 2), Y(W)); c.lineTo(X(L / 2), Y(W - 0.8));
    c.stroke();

    // distance labels along the bottom margin
    if (s >= 7) {
      c.fillStyle = rgba(theme.line, 0.55);
      c.font = `${Math.round(clamp(s * 0.95, 9, 12))}px ${theme.font}`;
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      for (let x = 0; x <= L + 1e-6; x += 10) {
        c.fillText(String(x), X(x), Y(-MARGIN / 2));
      }
    }

    // end-zone labels
    if (view.R) {
      const fs = Math.round(clamp(s * 1.3, 10, 17));
      c.font = `600 ${fs}px ${theme.font}`;
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      for (const [x0, x1, team] of ez) {
        const name = view.R.teams[team].name;
        c.save();
        c.translate(X((x0 + x1) / 2), Y(W / 2));
        c.rotate(team === 0 ? Math.PI / 2 : -Math.PI / 2);
        c.fillStyle = rgba(theme.line, 0.38);
        c.fillText(`${name} score here`, 0, 0);
        c.restore();
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Sampling / interpolation
  // ---------------------------------------------------------------------------
  function sampleAt(t) {
    const R = view.R;
    const f = clamp(t / R.dt, 0, R.nF - 1);
    const i0 = Math.min(R.nF - 1, Math.floor(f + 1e-6));
    const i1 = Math.min(i0 + 1, R.nF - 1);
    return { i0, i1, a: i1 === i0 ? 0 : clamp(f - i0, 0, 1), fr: R.frames[i0], fr1: R.frames[i1] };
  }

  function lerpPair(a0x, a0y, a1x, a1y, a) {
    if (Math.abs(a1x - a0x) + Math.abs(a1y - a0y) > TELEPORT_M) return a < 0.5 ? [a0x, a0y] : [a1x, a1y];
    return [a0x + (a1x - a0x) * a, a0y + (a1y - a0y) * a];
  }

  function playerXY(S, k) {
    const p0 = S.fr.p, p1 = S.fr1.p;
    return lerpPair(p0[2 * k], p0[2 * k + 1], p1[2 * k], p1[2 * k + 1], S.a);
  }

  function discXYZ(S) {
    const d0 = S.fr.d || [0, 0, 0], d1 = S.fr1.d || d0;
    const [x, y] = lerpPair(d0[0], d0[1], d1[0], d1[1], S.a);
    const z0 = d0[2] || 0, z1 = d1[2] != null ? d1[2] : z0;
    return [x, y, z0 + (z1 - z0) * S.a];
  }

  function activeThrow(t) {
    const th = view.R.throws;
    let best = null;
    for (let i = 0; i < th.length; i++) {
      if (th[i].t0 - 1e-6 <= t && t <= th[i].t1 + 1e-6) best = th[i];
      if (th[i].t0 > t) break;
    }
    return best;
  }

  function possessionAt(t) {
    const ps = view.R.possessions;
    let k = -1;
    for (let i = 0; i < ps.length; i++) if (ps[i].t0 - 1e-6 <= t) k = i;
    return k;
  }

  function bezierAt(P, u) {
    const a = (1 - u) * (1 - u), b = 2 * (1 - u) * u, c = u * u;
    return [a * P[0][0] + b * P[1][0] + c * P[2][0], a * P[0][1] + b * P[1][1] + c * P[2][1]];
  }

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------
  function playerRadius() { return clamp(0.62 * geom.scale, 7, 13); }

  function render() {
    ctx.setTransform(geom.dpr, 0, 0, geom.dpr, 0, 0);
    if (fieldCache) {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.drawImage(fieldCache, 0, 0);
      ctx.setTransform(geom.dpr, 0, 0, geom.dpr, 0, 0);
    }
    const R = view.R;
    if (!R) { updateHud(null); return; }

    const t = view.t;
    const S = sampleAt(t);
    const fr = S.fr;
    const nP = R.nP;
    const offense = fr.o != null ? fr.o : 0;
    const holder = fr.h != null ? fr.h : -1;
    const pos = new Array(nP);
    for (let k = 0; k < nP; k++) pos[k] = playerXY(S, k);
    view.pos = pos.map(([x, y]) => [X(x), Y(y)]);
    const rp = playerRadius();

    // marker wedge (under everything else)
    if (fr.m && holder >= 0 && holder < nP) drawMarkWedge(pos[holder], fr.m, 1 - offense);

    // motion trails
    drawTrails(S, pos);

    // throw arc
    const disc = discXYZ(S);
    if (fr.ds === 1) {
      const th = activeThrow(t);
      if (th) drawThrowArc(th, t);
    }

    // players (holder last so it sits on top)
    const order = [];
    for (let k = 0; k < nP; k++) if (k !== holder) order.push(k);
    if (holder >= 0 && holder < nP) order.push(holder);
    for (const k of order) {
      const st = fr.st ? fr.st[k] : 0;
      drawPlayer(k, pos[k], st, R.players[k].team === offense, k === holder, rp);
    }
    if (view.selected >= 0 && view.selected < nP) drawSelection(pos[view.selected], rp);
    if (view.hover >= 0 && view.hover !== view.selected && view.hover < nP) drawHover(pos[view.hover], rp);

    // stall count next to the holder
    if (holder >= 0 && holder < nP && fr.s != null) drawStall(pos[holder], fr.s, rp);

    // disc
    drawDisc(disc, fr.ds, holder >= 0);

    // event flags
    drawFlags(t, pos, disc);

    updateHud(S);
  }

  function drawMarkWedge(hp, m, defTeam) {
    const a = +m[0] || 0, w = Math.abs(+m[1] || 0);
    const cx = X(hp[0]), cy = Y(hp[1]);
    const r = MARK_RADIUS * geom.scale;
    // Field angles are CCW with y up; the canvas y axis points down, so negate.
    const start = -(a + w) * Math.PI / 180, end = -(a - w) * Math.PI / 180;
    const col = theme.team[defTeam];
    const g = ctx.createRadialGradient(cx, cy, r * 0.15, cx, cy, r);
    g.addColorStop(0, rgba(col, 0.42));
    g.addColorStop(1, rgba(col, 0.12));
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, r, start, end, false);
    ctx.closePath();
    ctx.fillStyle = g;
    ctx.fill();
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = rgba(col, 0.75);
    ctx.stroke();
  }

  function drawTrails(S, pos) {
    const R = view.R;
    const fr = S.fr;
    for (let k = 0; k < R.nP; k++) {
      const st = fr.st ? fr.st[k] : 0;
      const sel = k === view.selected;
      const n = Math.max(1, Math.round((sel ? TRAIL_SELECTED_S : TRAIL_S) / R.dt));
      const j0 = Math.max(0, S.i0 - n);
      if (S.i0 - j0 < 1 && S.a === 0) continue;
      const team = R.players[k].team;
      const fast = st === 1 || st === 3;
      const col = fast ? theme.teamBright[team] : theme.team[team];
      ctx.lineWidth = sel ? 3 : fast ? 2.6 : 1.6;
      ctx.lineCap = 'round';
      let px = R.frames[j0].p[2 * k], py = R.frames[j0].p[2 * k + 1];
      const total = S.i0 - j0 + S.a;
      for (let j = j0 + 1; j <= S.i0 + 1; j++) {
        let qx, qy;
        if (j <= S.i0) { qx = R.frames[j].p[2 * k]; qy = R.frames[j].p[2 * k + 1]; }
        else { qx = pos[k][0]; qy = pos[k][1]; }
        if (Math.abs(qx - px) + Math.abs(qy - py) <= TELEPORT_M) {
          const age = (j - j0) / Math.max(total, 1e-6);  // 0 = oldest, 1 = now
          ctx.strokeStyle = rgba(col, (fast ? 0.1 : 0.05) + age * (fast ? 0.75 : 0.45));
          ctx.beginPath();
          ctx.moveTo(X(px), Y(py));
          ctx.lineTo(X(qx), Y(qy));
          ctx.stroke();
        }
        px = qx; py = qy;
      }
    }
  }

  function drawThrowArc(th, t) {
    const P = th.path;
    if (!Array.isArray(P) || P.length !== 3) return;
    const col = theme.arc[th.type] || theme.arc.backhand;
    const u = clamp((t - th.t0) / Math.max(th.t1 - th.t0, 1e-6), 0, 1);
    const huck = !!th.huck;

    ctx.save();
    ctx.lineCap = 'round';
    // full planned path, faint
    ctx.setLineDash(huck ? [7, 6] : []);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = rgba(col, 0.35);
    ctx.beginPath();
    ctx.moveTo(X(P[0][0]), Y(P[0][1]));
    ctx.quadraticCurveTo(X(P[1][0]), Y(P[1][1]), X(P[2][0]), Y(P[2][1]));
    ctx.stroke();

    // remaining portion (sub-curve u..1), stronger
    const B = bezierAt(P, u);
    const C = [(1 - u) * P[1][0] + u * P[2][0], (1 - u) * P[1][1] + u * P[2][1]];
    ctx.lineWidth = huck ? 2.6 : 2.2;
    ctx.strokeStyle = rgba(col, 0.9);
    ctx.shadowColor = 'rgba(0,0,0,0.35)';
    ctx.shadowBlur = 3;
    ctx.beginPath();
    ctx.moveTo(X(B[0]), Y(B[1]));
    ctx.quadraticCurveTo(X(C[0]), Y(C[1]), X(P[2][0]), Y(P[2][1]));
    ctx.stroke();
    ctx.restore();

    // landing target
    const lx = X(P[2][0]), ly = Y(P[2][1]);
    const r = clamp(0.9 * geom.scale, 6, 12);
    ctx.save();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = rgba(col, 0.9);
    ctx.beginPath();
    ctx.arc(lx, ly, r, 0, Math.PI * 2);
    ctx.moveTo(lx - r * 0.5, ly); ctx.lineTo(lx + r * 0.5, ly);
    ctx.moveTo(lx, ly - r * 0.5); ctx.lineTo(lx, ly + r * 0.5);
    ctx.stroke();
    // label
    const tags = [TYPE_ABBR[th.type] || th.type];
    if (huck) tags.push('huck');
    if (th.break) tags.push('break');
    const label = tags.join(' · ');
    ctx.font = `600 ${Math.round(clamp(geom.scale * 1.05, 10, 13))}px ${theme.font}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(0,0,0,0.55)';
    ctx.strokeText(label, lx, ly - r - 3);
    ctx.fillStyle = rgba(col, 1);
    ctx.fillText(label, lx, ly - r - 3);
    ctx.restore();
  }

  function drawPlayer(k, p, st, isOff, isHolder, rp) {
    const R = view.R;
    const pl = R.players[k];
    const team = pl.team;
    const x = X(p[0]), y = Y(p[1]);
    const base = theme.team[team];
    let alpha = 1;
    let fill = rgba(base, 1);
    if (st === 1 || st === 3) fill = rgba(theme.teamBright[team], 1);
    if (st === 2) alpha = 0.82;
    if (st === 4) alpha = 0.5;

    ctx.save();
    ctx.globalAlpha = alpha;

    if (isHolder) {
      ctx.beginPath();
      ctx.arc(x, y, rp + 5, 0, Math.PI * 2);
      ctx.fillStyle = rgba(theme.holder, 0.35);
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = rgba(theme.holder, 1);
      ctx.stroke();
    }

    // drop shadow
    ctx.beginPath();
    ctx.arc(x + 1, y + 1.5, rp, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0,0,0,0.25)';
    ctx.fill();

    ctx.beginPath();
    ctx.arc(x, y, rp, 0, Math.PI * 2);
    let textCol = '#ffffff';
    if (st === 5) {
      // marker: outlined
      ctx.fillStyle = 'rgba(255,255,255,0.88)';
      ctx.fill();
      ctx.lineWidth = Math.max(2.5, rp * 0.32);
      ctx.strokeStyle = rgba(base, 1);
      ctx.stroke();
      textCol = rgba(mix(base, [0, 0, 0], 0.25), 1);
    } else {
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.lineWidth = isOff ? 2 : 1.4;
      ctx.strokeStyle = isOff ? 'rgba(255,255,255,0.95)' : 'rgba(0,0,0,0.55)';
      ctx.stroke();
    }

    // jersey number
    const num = String(pl.num);
    ctx.fillStyle = textCol;
    ctx.font = `700 ${Math.round(rp * (num.length > 1 ? 0.98 : 1.1))}px ${theme.font}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(num, x, y + 0.5);

    if (st === 4) {
      // on the ground: X
      ctx.globalAlpha = 0.95;
      ctx.lineWidth = 2.2;
      ctx.strokeStyle = 'rgba(20,20,20,0.85)';
      const r = rp * 0.95;
      ctx.beginPath();
      ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r);
      ctx.moveTo(x - r, y + r); ctx.lineTo(x + r, y - r);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawSelection(p, rp) {
    const x = X(p[0]), y = Y(p[1]);
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(x, y, rp + 8, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  function drawHover(p, rp) {
    ctx.save();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(255,255,255,0.6)';
    ctx.beginPath();
    ctx.arc(X(p[0]), Y(p[1]), rp + 6, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  function drawStall(p, s, rp) {
    const x = X(p[0]), y = Y(p[1]);
    const frac = clamp(s / 10, 0, 1);
    const hot = s >= 7;
    const col = hot ? '#ef4444' : '#facc15';
    ctx.save();
    // progress ring
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(0,0,0,0.3)';
    ctx.beginPath();
    ctx.arc(x, y, rp + 9, 0, Math.PI * 2);
    ctx.stroke();
    ctx.strokeStyle = col;
    ctx.beginPath();
    ctx.arc(x, y, rp + 9, -Math.PI / 2, -Math.PI / 2 + frac * Math.PI * 2);
    ctx.stroke();
    // pill
    const label = `stall ${Math.floor(s + 1e-6)}`;
    const fs = Math.round(clamp(geom.scale * 1.0, 10, 13));
    ctx.font = `700 ${fs}px ${theme.font}`;
    const tw = ctx.measureText(label).width;
    const pw = tw + 12, ph = fs + 6;
    let py = y - rp - 14 - ph;
    if (py < 2) py = y + rp + 14;
    const px = clamp(x - pw / 2, 2, geom.cssW - pw - 2);
    roundRect(px, py, pw, ph, ph / 2);
    ctx.fillStyle = hot ? 'rgba(127,29,29,0.92)' : 'rgba(20,24,30,0.82)';
    ctx.fill();
    ctx.fillStyle = col;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, px + pw / 2, py + ph / 2 + 0.5);
    ctx.restore();
  }

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function drawDisc(d, ds, held) {
    const z = clamp(+d[2] || 0, 0, 15);
    const x = X(d[0]), y = Y(d[1]);
    const base = clamp(0.33 * geom.scale, 3.2, 5.5);
    const r = base * (1 + z / 6);
    ctx.save();
    if (ds !== 2) {
      // ground shadow, offset more the higher the disc is
      const off = Math.min(z, 10) * 0.22 * geom.scale;
      ctx.beginPath();
      ctx.ellipse(x + off * 0.6, y + off, base * 1.05, base * 0.8, 0, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0,0,0,${(0.38 * (1 - z / 25)).toFixed(3)})`;
      ctx.fill();
    }
    if (z > 1.8) {
      ctx.shadowColor = 'rgba(0,0,0,0.45)';
      ctx.shadowBlur = Math.min(12, z * 1.5);
    }
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = ds === 2 ? '#d4d4d4' : theme.disc;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = 'rgba(0,0,0,0.6)';
    ctx.stroke();
    // inner rim
    ctx.beginPath();
    ctx.arc(x, y, r * 0.6, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(0,0,0,0.18)';
    ctx.lineWidth = 1;
    ctx.stroke();
    if (ds === 2 && !held) {
      ctx.strokeStyle = 'rgba(0,0,0,0.7)';
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r);
      ctx.moveTo(x - r, y + r); ctx.lineTo(x + r, y - r);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawFlags(t, pos, disc) {
    const R = view.R;
    const flags = [];
    for (const e of R.events) {
      if (e.t > t) break;
      if (t - e.t > FLAG_S) continue;
      const label = FLAG_LABELS[e.type];
      if (!label) continue;
      const k = e.player;
      const at = (k != null && k >= 0 && k < R.nP) ? pos[k] : [disc[0], disc[1]];
      const kind = e.type === 'score' ? 'good' : TURNOVER_TYPES.has(e.type) ? 'bad' : 'info';
      flags.push({ label, at, age: (t - e.t) / FLAG_S, kind });
    }
    for (const th of R.throws) {
      if (th.t0 > t) break;
      if (t - th.t0 > FLAG_S) continue;
      const tags = [];
      if (th.huck) tags.push('HUCK');
      if (th.break) tags.push('BREAK');
      if (!tags.length) continue;
      const k = th.thrower;
      const at = (k != null && k >= 0 && k < R.nP) ? pos[k] : th.path[0];
      flags.push({ label: tags.join(' + '), at, age: (t - th.t0) / FLAG_S, kind: 'info' });
    }
    if (!flags.length) return;
    const rp = playerRadius();
    const fs = Math.round(clamp(geom.scale * 1.15, 11, 15));
    ctx.save();
    ctx.font = `800 ${fs}px ${theme.font}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const used = [];
    for (const f of flags) {
      const x = X(f.at[0]);
      let y = Y(f.at[1]) - rp - 18 - f.age * 14;
      for (const u of used) if (Math.abs(u[0] - x) < 60 && Math.abs(u[1] - y) < fs + 4) y = u[1] - fs - 6;
      used.push([x, y]);
      const tw = ctx.measureText(f.label).width;
      ctx.globalAlpha = clamp(1.25 - f.age, 0, 1);
      const bg = f.kind === 'good' ? 'rgba(21,128,61,0.92)' : f.kind === 'bad' ? 'rgba(185,28,28,0.92)' : 'rgba(30,41,59,0.88)';
      roundRect(x - tw / 2 - 7, y - fs / 2 - 4, tw + 14, fs + 8, 5);
      ctx.fillStyle = bg;
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.fillText(f.label, x, y + 0.5);
    }
    ctx.restore();
  }

  // ---------------------------------------------------------------------------
  // HUD / panels
  // ---------------------------------------------------------------------------
  function teamName(i) { return view.R ? view.R.teams[i].name : `Team ${i}`; }
  function swatch(team) { return `<span class="swatch" style="background:${theme.teamCss[team]}"></span>`; }
  function playerLabel(k) {
    const R = view.R;
    if (!R || k == null || k < 0 || k >= R.nP) return '-';
    return `${swatch(R.players[k].team)}#${esc(R.players[k].num)}`;
  }
  function fmt(v) {
    if (typeof v === 'number') return Number.isInteger(v) ? String(v) : String(+v.toFixed(Math.abs(v) >= 10 ? 1 : 3));
    if (v == null) return '-';
    if (typeof v === 'object') return esc(JSON.stringify(v));
    return esc(v);
  }

  let hudKey = '';
  function updateHud(S) {
    const R = view.R;
    if (!R || !S) {
      el.hud.innerHTML = '<dt>Status</dt><dd class="muted">no replay</dd>';
      el.clock.textContent = '0.0 / 0.0 s';
      el.frameNo.textContent = '';
      hudKey = '';
      return;
    }
    const t = view.t;
    const fr = S.fr;
    el.clock.textContent = `${t.toFixed(1)} / ${R.duration.toFixed(1)} s`;
    el.frameNo.textContent = `frame ${S.i0} / ${R.nF - 1}`;
    if (!view.scrubbing) scrub.value = String(S.i0);

    const pk = possessionAt(t);
    const poss = pk >= 0 ? R.possessions[pk] : null;
    const off = fr.o != null ? fr.o : (poss ? poss.offense : 0);
    let thrown = 0, completed = 0;
    for (const th of R.throws) {
      if (th.t0 <= t + 1e-6) thrown++;
      if (th.t1 <= t + 1e-6 && (th.outcome === 'catch' || th.outcome === 'score')) completed++;
    }
    let disc = DISC_STATES[fr.ds] || String(fr.ds);
    if (fr.ds === 0 && fr.h >= 0) disc = `held by ${playerLabel(fr.h)}`;
    if (fr.ds === 1) {
      const th = activeThrow(t);
      if (th) disc = `in the air: ${esc(th.type)}${th.huck ? ' huck' : ''}${th.break ? ' (break)' : ''} to ${playerLabel(th.receiver)}`;
    }
    const s = fr.s;
    const stall = s == null ? '<span class="muted">not counting</span>'
      : `<span class="pill${s >= 7 ? ' hot' : ''}">${Math.floor(s + 1e-6)}</span><span class="stall-bar"><i style="width:${clamp(s * 10, 0, 100)}%"></i></span>`;
    const dir = off === 0 ? '→' : '←';
    const mark = fr.m ? `${fmt(+fr.m[0])}° ± ${fmt(+fr.m[1])}°` : '<span class="muted">none</span>';
    const html =
      `<dt>Time</dt><dd>${t.toFixed(1)} s</dd>` +
      `<dt>Offense</dt><dd>${swatch(off)}${esc(teamName(off))} <span class="muted">attacking ${dir}</span></dd>` +
      `<dt>Possession</dt><dd>${poss ? `${pk + 1} / ${R.possessions.length}${poss.force ? ` &middot; force <b>${esc(poss.force)}</b>` : ''}` : '<span class="muted">-</span>'}</dd>` +
      `<dt>Disc</dt><dd>${disc}</dd>` +
      `<dt>Stall</dt><dd>${stall}</dd>` +
      `<dt>Mark</dt><dd>${mark}</dd>` +
      `<dt>Throws</dt><dd>${thrown} thrown &middot; ${completed} complete</dd>` +
      `<dt>Mode</dt><dd>${esc(R.meta.mode || '-')}${R.meta.seed != null ? ` &middot; seed ${esc(R.meta.seed)}` : ''}</dd>`;
    if (html !== hudKey) { el.hud.innerHTML = html; hudKey = html; }

    // scoreboard chips (mark the offense)
    const chips = [0, 1].map((i) => {
      const tm = R.teams[i];
      const isO = i === off;
      return `<span class="team-chip${isO ? ' offense' : ''}" style="${isO ? `color:${theme.teamCss[i]}` : ''}">` +
        `<span class="sw" style="background:${theme.teamCss[i]}"></span>` +
        `<span style="color:var(--text)">${esc(tm.name)}</span>` +
        (tm.strategy ? `<span class="strat">${esc(tm.strategy)}</span>` : '') +
        (isO ? `<span class="o-tag" style="background:${theme.teamCss[i]}">O ${i === 0 ? '→' : '←'}</span>` : '') +
        `</span>`;
    }).join('');
    if (el.teams.dataset.key !== chips) { el.teams.innerHTML = chips; el.teams.dataset.key = chips; }

    updateTicker(t);
    updatePlayerPanel(S);
    updateBanner(t);
  }

  function evClass(type) {
    if (type === 'score') return 'ev-score';
    if (TURNOVER_TYPES.has(type)) return 'ev-bad';
    if (type === 'layout' || type === 'pull') return 'ev-special';
    if (type === 'throw' || type === 'fake' || type === 'timeout') return 'ev-quiet';
    return '';
  }

  function updateTicker(t) {
    const R = view.R;
    let n = 0;
    while (n < R.events.length && R.events[n].t <= t + 1e-6) n++;
    const key = `${view.path}|${n}`;
    if (key === view.tickerKey) return;
    view.tickerKey = key;
    el.eventCount.textContent = `${n} / ${R.events.length}`;
    if (!n) {
      el.ticker.innerHTML = `<li class="empty">${R.events.length ? 'No events yet.' : 'This replay has no events.'}</li>`;
      return;
    }
    const items = [];
    for (let i = n - 1; i >= Math.max(0, n - 40); i--) {
      const e = R.events[i];
      items.push(`<li data-t="${e.t}" class="${i === n - 1 ? 'newest' : ''}" title="${esc(e.type)}">` +
        `<span class="t">${(+e.t).toFixed(1)}s</span><span class="${evClass(e.type)}">${esc(e.text || e.type)}</span></li>`);
    }
    el.ticker.innerHTML = items.join('');
    el.ticker.scrollTop = 0;
  }

  function updatePlayerPanel(S) {
    const R = view.R;
    const k = view.selected;
    el.closePlayer.classList.toggle('hidden', k < 0);
    if (k < 0 || k >= R.nP) {
      const key = 'none';
      if (view.panelKey !== key) {
        el.playerPanel.innerHTML = '<span class="muted">Click a player on the field to see their stats.</span>';
        view.panelKey = key;
      }
      return;
    }
    const pl = R.players[k];
    const fr = S.fr;
    const st = fr.st ? fr.st[k] : null;
    const p = playerXY(S, k);
    // speed from neighbouring frames
    let sp = 0;
    if (R.nF > 1) {
      const j = Math.max(1, S.i1);
      const pa = R.frames[j - 1].p, pb = R.frames[j].p;
      sp = Math.hypot(pb[2 * k] - pa[2 * k], pb[2 * k + 1] - pa[2 * k + 1]) / R.dt;
    }
    const role = pl.team === fr.o ? 'offense' : 'defense';
    const liveHtml =
      `<dl class="kv">` +
      `<dt>State</dt><dd>${st != null ? `${esc(STATE_NAMES[st] || 'code ' + st)} <span class="muted">(${st})</span>` : '-'}</dd>` +
      `<dt>Position</dt><dd>x ${p[0].toFixed(1)} m, y ${p[1].toFixed(1)} m</dd>` +
      `<dt>Speed</dt><dd>${sp > TELEPORT_M / R.dt ? '-' : sp.toFixed(1) + ' m/s'}</dd>` +
      `</dl>`;

    const staticKey = `${view.path}|${k}|${role}`;
    if (view.panelKey !== staticKey) {
      view.panelKey = staticKey;
      const stats = pl.stats || {};
      const keys = Object.keys(stats);
      const rows = keys.map((key) => {
        const v = stats[key];
        const bar = typeof v === 'number' && v >= 0 && v <= 1 && !Number.isInteger(v)
          ? `<span class="bar" style="width:${Math.round(v * 40)}px"></span>` : '';
        return `<tr><td>${esc(key.replace(/_/g, ' '))}</td><td>${bar}${fmt(v)}</td></tr>`;
      }).join('');
      const sm = R.summary[k];
      const summaryRows = [
        ['throws', `${sm.throws}${sm.throws ? ` (${sm.completions} complete)` : ''}`],
        ['catches', sm.catches], ['goals', sm.scores], ['assists', sm.assists],
        ['blocks', sm.ds], ['turnovers', sm.turns],
        ['gain (completed throws)', `${sm.yards.toFixed(0)} m`],
      ].filter(([, v]) => v !== 0 && v !== '0 m' && v !== '0')
        .map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join('');
      el.playerPanel.innerHTML =
        `<div class="player-head"><div class="jersey" style="background:${theme.teamCss[pl.team]}">${esc(pl.num)}</div>` +
        `<div><div class="name">#${esc(pl.num)} &middot; ${esc(teamName(pl.team))}</div>` +
        `<div class="sub">id ${esc(pl.id)} &middot; team ${esc(pl.team)} &middot; on ${role}</div></div></div>` +
        `<div id="playerLive"></div>` +
        `<div class="subhead">Stats</div>` +
        (rows ? `<table class="stats">${rows}</table>` : '<div class="muted">No stats in this replay.</div>') +
        `<div class="subhead">This replay</div>` +
        (summaryRows ? `<table class="stats">${summaryRows}</table>` : '<div class="muted">No throws or catches.</div>');
    }
    const live = document.getElementById('playerLive');
    if (live && live.dataset.key !== liveHtml) { live.innerHTML = liveHtml; live.dataset.key = liveHtml; }
  }

  function updateBanner(t) {
    const R = view.R;
    const atEnd = t >= R.duration - 1e-6;
    if (!atEnd || view.scrubbing) { el.banner.classList.add('hidden'); return; }
    if (!el.banner.classList.contains('hidden') && el.banner.dataset.path === view.path) return;
    const res = R.result || {};
    const outcome = String(res.outcome || 'end').replace(/_/g, ' ');
    const team = res.team != null && res.team >= 0 && res.team <= 1 ? res.team : null;
    const big = team != null
      ? `<span style="color:${theme.teamCss[team]}">${esc(outcome.toUpperCase())}</span> &middot; ${esc(teamName(team))}`
      : esc(outcome.toUpperCase());
    const bits = [];
    if (res.duration != null) bits.push(`${(+res.duration).toFixed(1)} s`);
    if (res.throws != null) bits.push(`${res.throws} throws`);
    if (res.turnovers != null) bits.push(`${res.turnovers} turnover${res.turnovers === 1 ? '' : 's'}`);
    for (const [k, v] of Object.entries(res)) {
      if (['outcome', 'team', 'duration', 'throws', 'turnovers'].includes(k)) continue;
      bits.push(`${esc(k.replace(/_/g, ' '))}: ${fmt(v)}`);
    }
    el.banner.innerHTML = `<div class="big">${big}</div><div class="sub">${bits.join(' &middot; ') || 'end of replay'}</div>` +
      `<button type="button" id="bannerReplay">Replay ↻</button>`;
    el.banner.dataset.path = view.path;
    el.banner.classList.remove('hidden');
    $('bannerReplay').onclick = () => { seek(0); setPlaying(true); };
  }

  function buildEventMarks() {
    const R = view.R;
    el.marks.innerHTML = '';
    if (!R || R.duration <= 0) return;
    const frag = document.createDocumentFragment();
    for (const e of R.events) {
      if (e.type === 'throw' || e.type === 'fake') continue;
      const m = document.createElement('div');
      m.className = 'event-mark';
      m.style.left = `${clamp(e.t / R.duration, 0, 1) * 100}%`;
      m.style.background = e.type === 'score' ? 'var(--good)' : TURNOVER_TYPES.has(e.type) ? 'var(--bad)'
        : e.type === 'layout' ? 'var(--special)' : 'var(--muted)';
      m.title = `${(+e.t).toFixed(1)}s  ${e.text || e.type}`;
      m.addEventListener('click', () => { seek(e.t); });
      frag.appendChild(m);
    }
    el.marks.appendChild(frag);
  }

  function buildLegend() {
    const items = [
      { draw: (c, s) => legendPlayer(c, s, 0, 0, true), text: 'Team colours (white rim = offense)' },
      { draw: (c, s) => legendPlayer(c, s, 1, 0, false), text: 'Defense (dark rim)' },
      { draw: (c, s) => legendPlayer(c, s, 0, 6, true), text: 'Disc holder (gold halo)' },
      { draw: (c, s) => legendPlayer(c, s, 0, 1, true), text: 'Cutting / chasing (bright + trail)' },
      { draw: (c, s) => legendPlayer(c, s, 1, 5, false), text: 'Marker (outlined) + wedge = mark' },
      { draw: (c, s) => legendPlayer(c, s, 1, 4, false), text: 'On the ground after a layout' },
      { draw: (c, s) => legendArc(c, s, 'backhand', false), text: 'Backhand flight' },
      { draw: (c, s) => legendArc(c, s, 'forehand', false), text: 'Forehand flight' },
      { draw: (c, s) => legendArc(c, s, 'hammer', false), text: 'Hammer flight' },
      { draw: (c, s) => legendArc(c, s, 'backhand', true), text: 'Huck (dashed)' },
    ];
    el.legend.innerHTML = '';
    const dpr = window.devicePixelRatio || 1;
    for (const it of items) {
      const li = document.createElement('li');
      const cv = document.createElement('canvas');
      cv.width = 34 * dpr; cv.height = 22 * dpr;
      cv.style.width = '34px'; cv.style.height = '22px';
      const c = cv.getContext('2d');
      c.setTransform(dpr, 0, 0, dpr, 0, 0);
      c.fillStyle = theme.grass;
      roundRectOn(c, 0, 0, 34, 22, 4); c.fill();
      it.draw(c, 34);
      li.appendChild(cv);
      const span = document.createElement('span');
      span.textContent = it.text;
      li.appendChild(span);
      el.legend.appendChild(li);
    }
  }
  function roundRectOn(c, x, y, w, h, r) {
    c.beginPath();
    c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r);
    c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath();
  }
  function legendPlayer(c, w, team, st, isOff) {
    const x = 17, y = 11, r = 6.5;
    const base = theme.team[team];
    c.save();
    if (st === 1) {
      c.strokeStyle = rgba(theme.teamBright[team], 0.8); c.lineWidth = 2.4; c.lineCap = 'round';
      c.beginPath(); c.moveTo(2, 17); c.lineTo(x - 4, y + 2); c.stroke();
    }
    if (st === 5) {
      c.fillStyle = rgba(base, 0.3);
      c.beginPath(); c.moveTo(x, y); c.arc(x, y, 11, -Math.PI * 0.75, -Math.PI * 0.2); c.closePath(); c.fill();
    }
    if (st === 6) {
      c.beginPath(); c.arc(x, y, r + 3.5, 0, Math.PI * 2);
      c.fillStyle = rgba(theme.holder, 0.35); c.fill(); c.lineWidth = 2; c.strokeStyle = rgba(theme.holder, 1); c.stroke();
    }
    if (st === 4) c.globalAlpha = 0.5;
    c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2);
    if (st === 5) {
      c.fillStyle = 'rgba(255,255,255,0.88)'; c.fill(); c.lineWidth = 2.5; c.strokeStyle = rgba(base, 1); c.stroke();
    } else {
      c.fillStyle = rgba(st === 1 ? theme.teamBright[team] : base, 1); c.fill();
      c.lineWidth = isOff ? 1.8 : 1.2; c.strokeStyle = isOff ? '#fff' : 'rgba(0,0,0,.55)'; c.stroke();
    }
    if (st === 4) {
      c.globalAlpha = 1; c.strokeStyle = 'rgba(20,20,20,.85)'; c.lineWidth = 1.8;
      c.beginPath(); c.moveTo(x - 6, y - 6); c.lineTo(x + 6, y + 6); c.moveTo(x - 6, y + 6); c.lineTo(x + 6, y - 6); c.stroke();
    }
    c.restore();
  }
  function legendArc(c, w, type, huck) {
    const col = theme.arc[type];
    c.save();
    c.setLineDash(huck ? [4, 3] : []);
    c.strokeStyle = rgba(col, 0.95); c.lineWidth = 2; c.lineCap = 'round';
    c.beginPath(); c.moveTo(4, 17); c.quadraticCurveTo(17, -2, 30, 15); c.stroke();
    c.restore();
  }

  // ---------------------------------------------------------------------------
  // Playback control
  // ---------------------------------------------------------------------------
  function setPlaying(on) {
    if (!view.R) on = false;
    if (on && view.t >= view.R.duration - 1e-6) view.t = 0;   // play at the end = restart
    view.playing = on;
    el.play.innerHTML = on ? '&#x275A;&#x275A;' : '&#x25B6;';
    el.play.title = on ? 'Pause (Space)' : 'Play (Space)';
    dirty = true;
  }
  function seek(t) {
    if (!view.R) return;
    view.t = clamp(t, 0, view.R.duration);
    dirty = true;
  }
  function stepFrames(n) {
    if (!view.R) return;
    setPlaying(false);
    const f = view.t / view.R.dt;
    const target = n > 0 ? Math.floor(f + 1e-6) + n : Math.ceil(f - 1e-6) + n;
    seek(clamp(target, 0, view.R.nF - 1) * view.R.dt);
  }
  function setSpeed(s) {
    view.speed = s;
    for (const b of el.speeds.querySelectorAll('button')) b.classList.toggle('active', +b.dataset.speed === s);
  }
  function bumpSpeed(dir) {
    const i = SPEEDS.indexOf(view.speed);
    setSpeed(SPEEDS[clamp((i < 0 ? 2 : i) + dir, 0, SPEEDS.length - 1)]);
  }

  let lastTs = null;
  function frameLoop(ts) {
    const real = lastTs == null ? 0 : Math.min(0.1, (ts - lastTs) / 1000);
    lastTs = ts;
    if (view.R && view.playing && !view.scrubbing) {
      view.t += real * view.speed;
      if (view.t >= view.R.duration) { view.t = view.R.duration; setPlaying(false); }
      dirty = true;
    }
    if (dirty) { dirty = false; render(); }
    requestAnimationFrame(frameLoop);
  }

  // ---------------------------------------------------------------------------
  // Loading, replay list, live mode
  // ---------------------------------------------------------------------------
  function setStatus(msg, isError) {
    el.status.textContent = msg || '';
    el.status.classList.toggle('error', !!isError);
    el.status.title = msg || '';
  }
  const bust = (path) => path + (path.includes('?') ? '&' : '?') + '_=' + Date.now();
  const sigOf = (e) => (e ? `${e.size}:${e.mtime}` : null);

  async function fetchList() {
    const res = await fetch(bust('/api/replays'), { cache: 'no-store' });
    if (!res.ok) throw new Error(`/api/replays: HTTP ${res.status}`);
    const list = await res.json();
    view.list = Array.isArray(list) ? list : [];
    return view.list;
  }

  function fmtTime(mtime) {
    const d = new Date(mtime * 1000);
    const p = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
  }
  function fmtSize(n) {
    return n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1e3))} kB`;
  }

  function fillSelect() {
    const cur = view.path;
    const opts = view.list.map((e) => {
      let name = e.path.replace(/^\/runs\//, '');
      try { name = decodeURIComponent(name); } catch (_) { /* keep encoded */ }
      return { value: e.path, label: `${name}   (${fmtTime(e.mtime)}, ${fmtSize(e.size)})` };
    });
    if (cur && !opts.some((o) => o.value === cur)) opts.unshift({ value: cur, label: `${cur}  (not under runs/)` });
    const key = JSON.stringify(opts) + '|' + cur;
    if (el.select.dataset.key === key) return;
    el.select.dataset.key = key;
    el.select.innerHTML = '';
    if (!opts.length) {
      const o = document.createElement('option');
      o.textContent = 'No replays found in runs/';
      o.value = '';
      el.select.appendChild(o);
      return;
    }
    for (const o of opts) {
      const opt = document.createElement('option');
      opt.value = o.value;
      opt.textContent = o.label;
      el.select.appendChild(opt);
    }
    if (cur) el.select.value = cur;
  }

  function showEmpty(html) {
    el.empty.innerHTML = html;
    el.empty.classList.toggle('hidden', !html);
  }

  async function loadReplay(path, opts) {
    opts = opts || {};
    setStatus(`Loading ${path} ...`);
    let res;
    try {
      res = await fetch(bust(path), { cache: 'no-store' });
    } catch (err) {
      setStatus(`Could not fetch ${path}: ${err.message}`, true);
      return false;
    }
    if (!res.ok) { setStatus(`Could not load ${path}: HTTP ${res.status}`, true); return false; }
    let R;
    try {
      const text = await res.text();
      R = normalize(JSON.parse(text));
    } catch (err) {
      // In live mode this is usually a file caught mid-write; keep the old replay and retry.
      setStatus(`${path}: ${err.message}${opts.live ? ' (will retry)' : ''}`, true);
      return false;
    }
    const prevPath = view.path;
    view.R = R;
    view.path = path;
    const entry = view.list.find((e) => e.path === path);
    view.sig = opts.sig || sigOf(entry) || `${res.headers.get('content-length')}:${res.headers.get('last-modified')}`;
    if (prevPath !== path || !(view.selected < R.nP)) view.selected = -1;
    view.hover = -1;
    view.tickerKey = '';
    view.panelKey = '';
    el.banner.classList.add('hidden');
    el.banner.dataset.path = '';
    showEmpty('');

    el.title.textContent = R.meta.title || 'Untitled replay';
    document.title = `${R.meta.title || 'Replay'} - UltiSim`;
    scrub.max = String(R.nF - 1);
    geom.cssW = 0;          // field dims may differ: force resize + new field cache
    resize();
    buildEventMarks();
    fillSelect();

    const url = new URL(window.location.href);
    url.searchParams.set('replay', path);
    window.history.replaceState(null, '', url.toString());

    const warn = R.version !== 1 ? ` (format version ${R.version}; viewer expects 1)` : '';
    const when = entry ? `, modified ${fmtTime(entry.mtime)}` : '';
    setStatus(`${R.nF} frames, ${R.duration.toFixed(1)} s, ${R.throws.length} throws${when}${warn}`, !!warn);
    seek(0);
    setPlaying(opts.autoplay !== false);
    return true;
  }

  async function refreshList(quiet) {
    try {
      await fetchList();
      fillSelect();
      return true;
    } catch (err) {
      if (!quiet) setStatus(`Replay list unavailable: ${err.message}`, true);
      return false;
    }
  }

  async function pollLive() {
    if (polling || !view.path) return;
    polling = true;
    try {
      await fetchList();
      fillSelect();
      const entry = view.list.find((e) => e.path === view.path);
      let sig = sigOf(entry);
      if (!sig) {
        const r = await fetch(bust(view.path), { method: 'HEAD', cache: 'no-store' });
        if (r.ok) sig = `${r.headers.get('content-length')}:${r.headers.get('last-modified')}`;
      }
      const now = new Date().toLocaleTimeString();
      if (sig && sig !== view.sig) {
        el.liveStatus.textContent = `reloading (${now})`;
        const ok = await loadReplay(view.path, { autoplay: true, live: true, sig });
        el.liveStatus.textContent = ok ? `reloaded ${now}` : `retrying (${now})`;
      } else {
        el.liveStatus.textContent = `checked ${now}`;
      }
    } catch (err) {
      el.liveStatus.textContent = `poll failed: ${err.message}`;
    } finally {
      polling = false;
    }
  }

  function setLive(on) {
    if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
    el.liveStatus.classList.toggle('on', on);
    if (on) {
      el.liveStatus.textContent = 'watching';
      liveTimer = setInterval(pollLive, POLL_MS);
      pollLive();
    } else {
      el.liveStatus.textContent = '';
    }
    const url = new URL(window.location.href);
    if (on) url.searchParams.set('live', '1'); else url.searchParams.delete('live');
    window.history.replaceState(null, '', url.toString());
  }

  // ---------------------------------------------------------------------------
  // Input
  // ---------------------------------------------------------------------------
  function pickPlayer(evt) {
    if (!view.R || !view.pos.length) return -1;
    const rect = canvas.getBoundingClientRect();
    const mx = evt.clientX - rect.left, my = evt.clientY - rect.top;
    const lim = Math.max(playerRadius() + 4, 12);
    let best = -1, bd = lim;
    for (let k = 0; k < view.pos.length; k++) {
      const d = Math.hypot(view.pos[k][0] - mx, view.pos[k][1] - my);
      if (d <= bd) { bd = d; best = k; }
    }
    return best;
  }

  function wireInput() {
    for (const s of SPEEDS) {
      const b = document.createElement('button');
      b.type = 'button';
      b.dataset.speed = String(s);
      b.textContent = `${s}×`;
      b.addEventListener('click', () => setSpeed(s));
      el.speeds.appendChild(b);
    }
    setSpeed(1);

    el.play.addEventListener('click', () => setPlaying(!view.playing));
    el.back.addEventListener('click', (e) => stepFrames(e.shiftKey ? -10 : -1));
    el.fwd.addEventListener('click', (e) => stepFrames(e.shiftKey ? 10 : 1));
    el.restart.addEventListener('click', () => { seek(0); setPlaying(true); });
    el.closePlayer.addEventListener('click', () => { view.selected = -1; dirty = true; });

    // keep buttons from holding focus, so Space/arrows always reach the viewer
    document.addEventListener('mouseup', (e) => {
      const b = e.target.closest && e.target.closest('button');
      if (b) b.blur();
    });

    scrub.addEventListener('pointerdown', () => { view.scrubbing = true; });
    const endScrub = () => { if (view.scrubbing) { view.scrubbing = false; dirty = true; } };
    scrub.addEventListener('pointerup', endScrub);
    scrub.addEventListener('pointercancel', endScrub);
    scrub.addEventListener('change', endScrub);
    scrub.addEventListener('input', () => {
      if (!view.R) return;
      seek(+scrub.value * view.R.dt);
    });

    canvas.addEventListener('mousemove', (e) => {
      const k = pickPlayer(e);
      if (k !== view.hover) { view.hover = k; dirty = true; }
      canvas.classList.toggle('pointer', k >= 0);
    });
    canvas.addEventListener('mouseleave', () => { if (view.hover !== -1) { view.hover = -1; dirty = true; } });
    canvas.addEventListener('click', (e) => {
      const k = pickPlayer(e);
      view.selected = k === view.selected ? -1 : k;
      dirty = true;
    });

    el.ticker.addEventListener('click', (e) => {
      const li = e.target.closest('li[data-t]');
      if (li) { setPlaying(false); seek(+li.dataset.t); }
    });

    el.select.addEventListener('change', () => {
      const p = el.select.value;
      el.select.blur();
      if (p && p !== view.path) loadReplay(p, { autoplay: true });
    });
    el.refresh.addEventListener('click', () => refreshList(false));
    el.live.addEventListener('change', () => setLive(el.live.checked));

    document.addEventListener('keydown', (e) => {
      const tag = e.target && e.target.tagName;
      if (tag === 'SELECT' || (tag === 'INPUT' && e.target.type !== 'range' && e.target.type !== 'checkbox')) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      let handled = true;
      switch (e.key) {
        case ' ': case 'k': setPlaying(!view.playing); break;
        case 'ArrowLeft': stepFrames(e.shiftKey ? -10 : -1); break;
        case 'ArrowRight': stepFrames(e.shiftKey ? 10 : 1); break;
        case 'Home': case 'r': case 'R': seek(0); break;
        case 'End': if (view.R) { setPlaying(false); seek(view.R.duration); } break;
        case '+': case '=': bumpSpeed(1); break;
        case '-': case '_': bumpSpeed(-1); break;
        case 'Escape': view.selected = -1; dirty = true; break;
        default: handled = false;
      }
      if (handled) e.preventDefault();
    });
    // Space on a focused button would also "click" it on keyup; swallow that.
    document.addEventListener('keyup', (e) => {
      if (e.key === ' ' && e.target && e.target.tagName === 'BUTTON') e.preventDefault();
    });

    if (window.ResizeObserver) new ResizeObserver(() => resize()).observe(wrap);
    window.addEventListener('resize', resize);
    const mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
    const onTheme = () => { readTheme(); buildFieldCache(); buildLegend(); hudKey = ''; view.panelKey = ''; el.teams.dataset.key = ''; dirty = true; };
    if (mq) {
      if (mq.addEventListener) mq.addEventListener('change', onTheme);
      else if (mq.addListener) mq.addListener(onTheme);
    }
  }

  // ---------------------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------------------
  async function boot() {
    readTheme();
    wireInput();
    buildLegend();
    resize();
    requestAnimationFrame(frameLoop);

    const params = new URLSearchParams(window.location.search);
    let want = params.get('replay');
    if (want && !want.startsWith('/')) want = '/' + want;   // same-origin paths only
    const listed = await refreshList(!!want);

    if (!want && view.list.length) want = view.list[0].path;
    if (want) {
      await loadReplay(want, { autoplay: true });
    } else {
      fillSelect();
      showEmpty(listed
        ? 'No replays found under <code>runs/</code>.<br>Create one with <code>python viz/make_sample.py</code> and press &#x21bb;.'
        : 'Could not reach <code>/api/replays</code>. Start the viewer with <code>python viz/server.py</code>.');
      setStatus(listed ? 'No replays yet' : 'Server API unavailable', !listed);
    }
    if (params.get('live') === '1') { el.live.checked = true; setLive(true); }
  }

  boot();
})();
