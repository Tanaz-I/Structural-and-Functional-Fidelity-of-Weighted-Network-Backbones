// Backbone Lab UI: one dependency-free ES module. Sections: core (store, data), components, charts,
// screens, app shell. Data comes from ui/data/*.json exported by `python run.py export`.
// ========================================================================
// core.js
// ========================================================================
// Core: DOM helpers, shared store (URL-synced), data loading, constants.
const METHODS = ["W", "DF", "NC", "SMDS", "HSS", "wEffRes"];
const METHOD_NAMES = {
  W: "Global threshold", DF: "Disparity filter", NC: "Noise-corrected", SMDS: "Semi-metric distortion",
  HSS: "High-salience skeleton", wEffRes: "Effective resistance",
};
const METHOD_BLURB = {
  W: "keeps the heaviest edges", DF: "keeps locally significant edges", NC: "keeps edges above noise",
  SMDS: "keeps shortest-path edges first", HSS: "keeps edges on many shortest-path trees", wEffRes: "keeps spectrally critical edges",
};
const FUNCS = ["struct", "spread", "comm", "path", "rank"];
const FUNC_NAMES = { struct: "Structure", spread: "Spreading", comm: "Communities", path: "Shortest paths", rank: "Node rankings" };
const GOALS = ["spread", "comm", "path", "rank"];
const BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70];
const qKey = (q) => q.toFixed(2);
const pct = (x, d = 0) => (x * 100).toFixed(d) + "%";
const f2 = (x) => (Number.isFinite(x) ? x.toFixed(2) : "–");
const f3 = (x) => (Number.isFinite(x) ? x.toFixed(3) : "–");

const SVGNS = "http://www.w3.org/2000/svg";
function h(tag, attrs = {}, ...kids) {
  const el = attrs.__svg ? document.createElementNS(SVGNS, tag) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "__svg" || v === undefined || v === null || v === false) continue;
    if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "class") el.setAttribute("class", v);
    else if (k === "style" && typeof v === "object") { for (const [sk, sv] of Object.entries(v)) { if (sk.startsWith("--")) el.style.setProperty(sk, sv); else el.style[sk] = sv; } }
    else if (k === "html") el.innerHTML = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
const S = (tag, attrs = {}, ...kids) => h(tag, { ...attrs, __svg: true }, ...kids);
const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); return el; };
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

// ---------------------------------------------------------------- store
const DEFAULTS = { net: "", q: 0.2, a: "SMDS", b: "W", fn: "spread", goal: "spread", w: "5,5,5,5", conn: 0, theme: "", tier: "high", edge: -1 };
const listeners = new Set();
const state = { ...DEFAULTS };
function loadStateFromHash() {
  const [, query = ""] = location.hash.split("?");
  const p = new URLSearchParams(query);
  for (const k of Object.keys(DEFAULTS)) if (p.has(k)) state[k] = ["q"].includes(k) ? parseFloat(p.get(k)) : ["conn", "edge"].includes(k) ? parseInt(p.get(k)) : p.get(k);
}
function route() { return (location.hash.split("?")[0] || "#/arena").slice(2) || "arena"; }
function writeHash() {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(state)) if (v !== DEFAULTS[k] && v !== "") p.set(k, v);
  history.replaceState(null, "", `#/${route()}${p.toString() ? "?" + p : ""}`);
}
function set(patch) {
  Object.assign(state, patch);
  writeHash();
  listeners.forEach((fn) => fn(patch));
}
const subscribe = (fn) => { listeners.add(fn); return () => listeners.delete(fn); };
function go(screen, patch = {}) {
  Object.assign(state, patch);
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(state)) if (v !== DEFAULTS[k] && v !== "") p.set(k, v);
  location.hash = `#/${screen}${p.toString() ? "?" + p : ""}`;
}
function weights() {
  const a = state.w.split(",").map(Number);
  return Object.fromEntries(GOALS.map((g, i) => [g, Number.isFinite(a[i]) ? a[i] : 0]));
}

// ---------------------------------------------------------------- data
const cache = new Map();
async function getJSON(path) {
  if (!cache.has(path)) cache.set(path, fetch(`data/${path}`).then((r) => { if (!r.ok) throw new Error(`${path}: ${r.status}`); return r.json(); }));
  return cache.get(path);
}
function unpackBits(b64, n) {
  const bin = atob(b64), out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = (bin.charCodeAt(i >> 3) >> (7 - (i & 7))) & 1;
  return out;
}
async function loadNetwork(name) {
  const [graph, edges, bb, metrics, boot] = await Promise.all([
    getJSON(`${name}/graph.json`), getJSON(`${name}/edges.json`), getJSON(`${name}/backbones.json`),
    getJSON(`${name}/metrics.json`), getJSON(`${name}/bootstrap.json`).catch(() => null)]);
  const masks = {};
  for (const m of METHODS) { masks[m] = {}; for (const k of Object.keys(bb.bits[m])) masks[m][k] = unpackBits(bb.bits[m][k], graph.m); }
  return { name, graph, edges, masks, metrics, boot };
}
const kOf = (q, m) => Math.round(q * m);
/** Mask of the top-k edges for any continuous budget, from stored ranks (backbones are nested prefixes). */
function maskAt(net, method, q) {
  const r = net.edges.rank[method], k = kOf(q, net.graph.m), out = new Uint8Array(net.graph.m);
  for (let i = 0; i < out.length; i++) out[i] = r[i] <= k ? 1 : 0;
  return out;
}
function failures(metrics, method, q) {
  const s = metrics.scores[method][qKey(q)];
  return GOALS.filter((g) => s.struct >= 0.8 && s[g] < 0.6);
}

// ========================================================================
// components.js
// ========================================================================
// Shared components: budget dial, network canvas, tooltip, score bars.
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const methodColor = (m) => cssVar(`--m-${m}`);

// ---------------------------------------------------------------- tooltip
let tipEl;
function showTip(evt, html) {
  if (!tipEl) { tipEl = h("div", { class: "tip", role: "tooltip" }); document.body.append(tipEl); }
  tipEl.innerHTML = html; tipEl.style.display = "block";
  const pad = 14, w = tipEl.offsetWidth, hh = tipEl.offsetHeight;
  tipEl.style.left = Math.min(evt.clientX + pad, innerWidth - w - 8) + "px";
  tipEl.style.top = Math.min(evt.clientY + pad, innerHeight - hh - 8) + "px";
}
const hideTip = () => { if (tipEl) tipEl.style.display = "none"; };

// ---------------------------------------------------------------- budget dial
/** Circular dial with a tick per budget. Drag, click a tick, or use arrow keys. */
function BudgetDial(net) {
  const size = 214, cx = size / 2, cy = size / 2, r = 82, a0 = -225, a1 = 45; // sweep 270 degrees
  const ang = (i) => a0 + (a1 - a0) * (i / (BUDGETS.length - 1));
  const pt = (deg, rad) => [cx + rad * Math.cos((deg * Math.PI) / 180), cy + rad * Math.sin((deg * Math.PI) / 180)];
  const arc = (d0, d1, rad) => { const [x0, y0] = pt(d0, rad), [x1, y1] = pt(d1, rad); return `M${x0} ${y0} A${rad} ${rad} 0 ${d1 - d0 > 180 ? 1 : 0} 1 ${x1} ${y1}`; };
  const idx = () => Math.max(0, BUDGETS.findIndex((b) => Math.abs(b - state.q) < 1e-9));
  const knob = S("circle", { r: 9, class: "knob", fill: "var(--ink)", stroke: "var(--panel)", "stroke-width": 3 });
  const prog = S("path", { fill: "none", stroke: "var(--ink)", "stroke-width": 5, "stroke-linecap": "round" });
  const qText = h("div", { class: "q mono" }), subText = h("div", { class: "sub faint" });
  const svg = S("svg", { viewBox: `0 0 ${size} ${size}`, role: "slider", tabindex: 0, "aria-label": "Edge budget", "aria-valuemin": 5, "aria-valuemax": 70 },
    S("path", { d: arc(a0, a1, r), fill: "none", stroke: "var(--rule)", "stroke-width": 5, "stroke-linecap": "round" }), prog,
    ...BUDGETS.map((b, i) => { const [x, y] = pt(ang(i), r + 16), [tx, ty] = pt(ang(i), r);
      return S("g", { style: { cursor: "pointer" }, onclick: () => set({ q: b }) },
        S("circle", { cx: tx, cy: ty, r: 3, fill: "var(--ink-3)" }),
        S("text", { x, y: y + 3, "text-anchor": "middle", "font-size": 9, class: "num" }, Math.round(b * 100)),
        S("circle", { cx: x, cy: y, r: 11, fill: "transparent" })); }), knob);
  let animFrom = idx();
  const render = () => {
    const i = idx(), [x, y] = pt(ang(i), r);
    knob.setAttribute("cx", x); knob.setAttribute("cy", y);
    prog.setAttribute("d", i === 0 ? "" : arc(a0, ang(i), r));
    svg.setAttribute("aria-valuenow", Math.round(state.q * 100));
    qText.textContent = pct(state.q);
    subText.textContent = net ? `${kOf(state.q, net.graph.m).toLocaleString()} of ${net.graph.m.toLocaleString()} edges` : "";
  };
  const fromEvent = (e) => {
    const rect = svg.getBoundingClientRect(), sx = size / rect.width;
    let deg = (Math.atan2((e.clientY - rect.top) * sx - cy, (e.clientX - rect.left) * sx - cx) * 180) / Math.PI;
    if (deg > 45 && deg < 135) deg = deg < 90 ? 45 : -225; // dead zone at the bottom
    if (deg >= 135) deg -= 360;
    const t = clamp((deg - a0) / (a1 - a0), 0, 1), i = Math.round(t * (BUDGETS.length - 1));
    if (BUDGETS[i] !== state.q) set({ q: BUDGETS[i] });
  };
  let drag = false;
  svg.addEventListener("pointerdown", (e) => { drag = true; svg.setPointerCapture(e.pointerId); fromEvent(e); });
  svg.addEventListener("pointermove", (e) => drag && fromEvent(e));
  svg.addEventListener("pointerup", () => (drag = false));
  svg.addEventListener("keydown", (e) => {
    const i = idx();
    if (["ArrowRight", "ArrowUp"].includes(e.key) && i < BUDGETS.length - 1) set({ q: BUDGETS[i + 1] });
    if (["ArrowLeft", "ArrowDown"].includes(e.key) && i > 0) set({ q: BUDGETS[i - 1] });
    if (e.key.startsWith("Arrow")) e.preventDefault();
  });
  const el = h("div", { class: "dial" }, h("div", { class: "smallcaps" }, "edge budget"), svg, qText, subText);
  render();
  return { el, render };
}

// ---------------------------------------------------------------- network canvas
/**
 * Draw a network. opts: mask (Uint8Array|null kept edges), ghost (bool), color ("community"|"mono"),
 * overlay (Uint8Array of 0 none,1 A only,2 B only,3 both), highlight (edge idx), bridges (bool),
 * nodeColor (fn i -> css colour), edgeColor (css), rank-free. Edge width = weight, node size = strength.
 */
function drawNetwork(canvas, g, opts = {}) {
  const dpr = devicePixelRatio || 1, W = canvas.clientWidth || 300, H = canvas.clientHeight || W;
  if (canvas.width !== Math.round(W * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  const pad = 14, X = (i) => pad + g.nodes.x[i] * (W - 2 * pad), Y = (i) => pad + g.nodes.y[i] * (H - 2 * pad);
  const { u, v, w } = g.edges, m = u.length;
  let wmax = 0; for (let e = 0; e < m; e++) if (w[e] > wmax) wmax = w[e];
  const ink = cssVar("--ink"), ghost = cssVar("--ghost");
  const width = (e) => 0.35 + 2.4 * Math.sqrt(w[e] / wmax);
  const keepColor = opts.edgeColor || ink;
  const draw = (e, col, a, wd) => { ctx.strokeStyle = col; ctx.globalAlpha = a; ctx.lineWidth = wd ?? width(e);
    ctx.beginPath(); ctx.moveTo(X(u[e]), Y(u[e])); ctx.lineTo(X(v[e]), Y(v[e])); ctx.stroke(); };
  if (opts.overlay) {
    const cols = [null, cssVar("--m-" + opts.A), cssVar("--m-" + opts.B), ink];
    for (const kind of [0, 1, 2, 3]) for (let e = 0; e < m; e++) if (opts.overlay[e] === kind) {
      if (kind === 0) draw(e, ghost, 1, 0.5); else draw(e, cols[kind], kind === 3 ? 0.85 : 0.9, width(e) * (kind === 3 ? 1 : 1.15));
    }
  } else {
    if (opts.ghost !== false) for (let e = 0; e < m; e++) if (!opts.mask || !opts.mask[e]) draw(e, ghost, 1, 0.5);
    const bridge = opts.bridges ? opts.bridgeFlags : null, br = cssVar("--stamp");
    for (let e = 0; e < m; e++) if (!opts.mask || opts.mask[e]) draw(e, bridge && bridge[e] ? br : keepColor, bridge && bridge[e] ? 0.95 : 0.78);
  }
  if (opts.highlight >= 0) { draw(opts.highlight, cssVar("--stamp"), 1, 3.4); }
  ctx.globalAlpha = 1;
  const s = g.nodes.strength; let smax = 0; for (let i = 0; i < s.length; i++) if (s[i] > smax) smax = s[i];
  const pal = ["#4e79a7", "#e15759", "#59a14f", "#b07aa1", "#edc948", "#76b7b2", "#ff9da7", "#9c755f", "#bab0ac", "#f28e2b"];
  const dark = document.documentElement.dataset.theme === "dark" || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  const ring = cssVar("--node-ring");
  for (let i = 0; i < s.length; i++) {
    const r = 1.6 + 3.6 * Math.sqrt(s[i] / smax);
    ctx.beginPath(); ctx.arc(X(i), Y(i), r, 0, 7);
    ctx.fillStyle = opts.nodeColor ? opts.nodeColor(i) : opts.color === "mono" ? ink : pal[g.nodes.community[i] % pal.length];
    ctx.globalAlpha = dark ? 0.95 : 0.92; ctx.fill(); ctx.lineWidth = 0.8; ctx.strokeStyle = ring; ctx.stroke();
  }
  ctx.globalAlpha = 1;
  return { X, Y, W, H };
}

/** Index of the edge closest to (px, py) within tol pixels, or -1. */
function hitEdge(g, geom, px, py, tol = 5) {
  const { u, v } = g.edges; let best = -1, bd = tol * tol;
  for (let e = 0; e < u.length; e++) {
    const x1 = geom.X(u[e]), y1 = geom.Y(u[e]), x2 = geom.X(v[e]), y2 = geom.Y(v[e]);
    const dx = x2 - x1, dy = y2 - y1, L = dx * dx + dy * dy || 1, t = clamp(((px - x1) * dx + (py - y1) * dy) / L, 0, 1);
    const ex = x1 + t * dx - px, ey = y1 + t * dy - py, d = ex * ex + ey * ey;
    if (d < bd) { bd = d; best = e; }
  }
  return best;
}

// ---------------------------------------------------------------- score bars
function ScoreBars(net, method, q) {
  const k = qKey(q), s = net.metrics.scores[method][k], ci = net.metrics.ci?.[method]?.[k];
  const rows = FUNCS.map((f) => {
    const c = ci?.[f];
    return [h("span", { class: "muted" }, FUNC_NAMES[f].replace("Shortest paths", "Paths").replace("Node rankings", "Rankings").replace("Communities", "Community")),
      h("span", { class: "track" }, h("span", { class: `fill ${f === "struct" ? "struct" : ""}`, style: { width: pct(clamp(s[f], 0, 1)) } }),
        c ? h("span", { class: "ci", style: { left: pct(clamp(c[0], 0, 1)), width: pct(clamp(c[1] - c[0], 0, 1)) } }) : null),
      h("span", { class: "mono", style: { textAlign: "right" } }, f2(s[f]))];
  });
  return h("div", { class: "bars", style: { "--mc": `var(--m-${method})` } }, rows.flat());
}

function failureStamp(net, method, q) {
  const fl = failures(net.metrics, method, q);
  if (!fl.length) return null;
  return h("div", { class: "stamp", title: "Structure looks fine (≥ 0.80) but function is poor (< 0.60)" }, "MISLEADING",
    h("small", {}, fl.map((f) => FUNC_NAMES[f].toLowerCase()).join(" · ")));
}

function MethodLegend(active, onToggle) {
  return h("div", { class: "legend", role: "group", "aria-label": "Methods" }, METHODS.map((m) =>
    h("button", { "aria-pressed": active.has(m), style: { "--mc": `var(--m-${m})` }, title: METHOD_NAMES[m],
      onclick: () => { active.has(m) ? active.delete(m) : active.add(m); onToggle(); } }, h("span", { class: "sw" }), m)));
}

// ========================================================================
// charts.js
// ========================================================================
// Small SVG chart helpers (no dependencies).
const linScale = (d0, d1, r0, r1) => (x) => r0 + ((x - d0) / (d1 - d0 || 1)) * (r1 - r0);
const mix = (a, b, t) => { // mix two #rrggbb or rgb() css colours
  const p = (c) => { const m = c.match(/#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})/i); return m ? [1, 2, 3].map((i) => parseInt(m[i], 16)) : (c.match(/\d+/g) || [0, 0, 0]).slice(0, 3).map(Number); };
  const A = p(a), B = p(b); return `rgb(${A.map((x, i) => Math.round(x + (B[i] - x) * t)).join(",")})`;
};

/**
 * Line chart on a categorical-numeric x axis.
 * series: [{key, color, dash, width, marker, pts:[{x, y, lo?, hi?}]}]; xs: tick values; cursor: x value for the budget cursor.
 * tipRows(x) -> html for the crosshair tooltip.
 */
function lineChart({ width = 360, height = 230, series, xs, yDomain = [0, 1], title, cursor, onClickX, tipHtml, xFmt = (x) => Math.round(x * 100) + "%" }) {
  const m = { l: 34, r: 10, t: 22, b: 26 }, iw = width - m.l - m.r, ih = height - m.t - m.b;
  const X = linScale(xs[0], xs[xs.length - 1], m.l, m.l + iw), Y = linScale(yDomain[0], yDomain[1], m.t + ih, m.t);
  const svg = S("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", role: "img", "aria-label": title });
  svg.append(S("text", { x: m.l, y: 13, "font-size": 12, "font-weight": 600, style: { fill: "var(--ink)" } }, title));
  for (const t of [0, 0.25, 0.5, 0.75, 1].map((p) => yDomain[0] + p * (yDomain[1] - yDomain[0]))) {
    svg.append(S("line", { x1: m.l, x2: m.l + iw, y1: Y(t), y2: Y(t), stroke: "var(--rule-2)", "stroke-width": 1 }),
      S("text", { x: m.l - 5, y: Y(t) + 3, "text-anchor": "end", "font-size": 9, class: "num" }, t.toFixed(t % 1 ? 2 : 0)));
  }
  xs.forEach((x, i) => svg.append(S("text", { x: X(x), y: height - 8, "text-anchor": "middle", "font-size": 9, class: "num" }, i % 2 === 0 || xs.length < 6 ? xFmt(x) : "")));
  if (cursor !== undefined) svg.append(S("line", { x1: X(cursor), x2: X(cursor), y1: m.t, y2: m.t + ih, stroke: "var(--ink)", "stroke-width": 1, "stroke-dasharray": "2 3" }));
  for (const s of series) {
    const pts = s.pts.filter((p) => Number.isFinite(p.y));
    if (!pts.length) continue;
    if (pts[0].lo !== undefined) {
      const top = pts.map((p) => `${X(p.x)},${Y(p.hi)}`), bot = pts.slice().reverse().map((p) => `${X(p.x)},${Y(p.lo)}`);
      svg.append(S("polygon", { points: [...top, ...bot].join(" "), fill: s.color, opacity: 0.14 }));
    }
    svg.append(S("polyline", { points: pts.map((p) => `${X(p.x)},${Y(p.y)}`).join(" "), fill: "none", stroke: s.color, "stroke-width": s.width || 1.8,
      "stroke-dasharray": s.dash || "none", "stroke-linejoin": "round", opacity: s.faint ? 0.75 : 1 }));
    if (s.marker) for (const p of pts) svg.append(markerShape(s.marker, X(p.x), Y(p.y), s.color));
    const last = pts[pts.length - 1];
  }
  const cross = S("line", { y1: m.t, y2: m.t + ih, stroke: "var(--ink-3)", "stroke-width": 1, style: { display: "none" } });
  svg.append(cross);
  const hit = S("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent", style: { cursor: onClickX ? "pointer" : "default" } });
  const nearest = (e) => { const r = svg.getBoundingClientRect(), px = ((e.clientX - r.left) / r.width) * width;
    return xs.reduce((a, b) => (Math.abs(X(b) - px) < Math.abs(X(a) - px) ? b : a)); };
  hit.addEventListener("mousemove", (e) => { const x = nearest(e); cross.setAttribute("x1", X(x)); cross.setAttribute("x2", X(x)); cross.style.display = ""; if (tipHtml) showTip(e, tipHtml(x)); });
  hit.addEventListener("mouseleave", () => { cross.style.display = "none"; hideTip(); });
  if (onClickX) hit.addEventListener("click", (e) => onClickX(nearest(e)));
  svg.append(hit);
  return svg;
}

function markerShape(kind, x, y, c) {
  const a = { fill: c, stroke: "var(--panel)", "stroke-width": 1.2 };
  if (kind === "square") return S("rect", { x: x - 3, y: y - 3, width: 6, height: 6, ...a });
  if (kind === "triangle") return S("polygon", { points: `${x},${y - 4} ${x + 4},${y + 3} ${x - 4},${y + 3}`, ...a });
  if (kind === "diamond") return S("polygon", { points: `${x},${y - 4} ${x + 4},${y} ${x},${y + 4} ${x - 4},${y}`, ...a });
  if (kind === "ring") return S("circle", { cx: x, cy: y, r: 3.4, fill: "var(--panel)", stroke: c, "stroke-width": 1.6 });
  if (kind === "cross") return S("path", { d: `M${x - 3} ${y - 3}L${x + 3} ${y + 3}M${x + 3} ${y - 3}L${x - 3} ${y + 3}`, stroke: c, "stroke-width": 1.8, fill: "none" });
  return S("circle", { cx: x, cy: y, r: 3, ...a });
}
const MARKERS = { W: "circle", DF: "square", NC: "triangle", SMDS: "diamond", HSS: "ring", wEffRes: "cross" };
const DASHES = { W: "none", DF: "6 3", NC: "2 3", SMDS: "none", HSS: "8 3 2 3", wEffRes: "1 0" };

// ========================================================================
// screens/arena.js
// ========================================================================
// Arena: original + six backbones at one shared budget, with score bars and head-to-head overlay.
function Arena(net) {
  const root = h("div");
  const arena = h("div", { class: "arena" });
  const h2h = h("div", { class: "card", style: { marginTop: "18px", padding: "14px" } });
  const canvases = new Map();

  const panel = (key, title, blurb, mask, isOrig) => {
    const canvas = h("canvas", { "aria-label": `${title} drawn at the current budget` });
    canvases.set(key, { canvas, mask });
    const sel = state.a === key ? "sel-a" : state.b === key ? "sel-b" : "";
    return h("article", { class: `card specimen ${sel}`, style: { "--mc": isOrig ? "var(--orig)" : `var(--m-${key})` }, tabindex: 0,
      title: isOrig ? "" : "Click to set method A (shift-click for B) for the head-to-head view",
      onclick: (e) => { if (!isOrig) set(e.shiftKey ? { b: key } : { a: key }); update(); },
      onkeydown: (e) => { if (e.key === "Enter" && !isOrig) { set({ a: key }); update(); } } },
      h("header", {}, h("b", {}, title), h("span", { class: "faint", style: { fontSize: ".75rem" } }, blurb)),
      canvas, isOrig ? h("div", { class: "hint" }, `${net.graph.n} nodes · ${net.graph.m.toLocaleString()} edges`) : [ScoreBars(net, key, state.q), failureStamp(net, key, state.q)]);
  };

  function update() {
    arena.replaceChildren(panel("orig", "Original", "the specimen", null, true),
      ...METHODS.map((m) => panel(m, m, METHOD_BLURB[m], maskAt(net, m, state.q), false)));
    requestAnimationFrame(() => {
      for (const [key, { canvas, mask }] of canvases) drawNetwork(canvas, net.graph, { mask, ghost: true, edgeColor: key === "orig" ? undefined : methodColor(key) });
      renderH2H();
    });
  }

  function renderH2H() {
    const A = state.a, B = state.b, mA = maskAt(net, A, state.q), mB = maskAt(net, B, state.q);
    const ov = new Uint8Array(net.graph.m); let both = 0, onlyA = 0, onlyB = 0, none = 0;
    for (let e = 0; e < ov.length; e++) { ov[e] = (mA[e] ? 1 : 0) | (mB[e] ? 2 : 0); if (ov[e] === 3) both++; else if (ov[e] === 1) onlyA++; else if (ov[e] === 2) onlyB++; else none++; }
    const jac = both / Math.max(1, both + onlyA + onlyB);
    const canvas = h("canvas", { style: { width: "100%", aspectRatio: "1/1", maxWidth: "520px" } });
    const sa = net.metrics.scores[A][qKey(state.q)], sb = net.metrics.scores[B][qKey(state.q)];
    const sel = (cur, key) => h("select", { "aria-label": `Method ${key}`, style: { width: "auto", borderLeft: `4px solid var(--m-${cur})` }, onchange: (e) => { set({ [key.toLowerCase()]: e.target.value }); update(); } },
      METHODS.map((m) => h("option", { value: m, selected: m === cur }, `${m} · ${METHOD_NAMES[m]}`)));
    h2h.replaceChildren(
      h("div", { class: "row", style: { justifyContent: "space-between" } }, h("h3", {}, "Head to head"),
        h("div", { class: "row" }, sel(A, "A"), h("span", { class: "muted" }, "vs"), sel(B, "B"))),
      h("div", { class: "split", style: { marginTop: "10px" } }, canvas,
        h("div", {}, h("p", { class: "muted", style: { marginTop: 0 } }, `At ${pct(state.q)} of edges, ${A} and ${B} share ${both} edges (overlap ${pct(jac)}). Edges are coloured by who keeps them.`),
          h("div", { class: "row", style: { gap: "16px", margin: "8px 0" } },
            [[`var(--m-${A})`, `only ${A}`, onlyA], [`var(--m-${B})`, `only ${B}`, onlyB], ["var(--ink)", "both", both], ["var(--ghost)", "neither", none]].map(([c, l, n]) =>
              h("span", { class: "chip" }, h("i", { style: { background: c, border: "1px solid var(--rule)" } }), `${l} `, h("b", { class: "mono" }, n)))),
          h("table", { class: "t" }, h("thead", {}, h("tr", {}, h("th", {}, "Fidelity"), h("th", {}, A), h("th", {}, B), h("th", {}, "Δ"))),
            h("tbody", {}, ["struct", "spread", "comm", "path", "rank"].map((f) => h("tr", {}, h("td", {}, FUNC_NAMES[f]), h("td", { class: "num" }, f2(sa[f])), h("td", { class: "num" }, f2(sb[f])),
              h("td", { class: "num", style: { color: sa[f] - sb[f] > 0.005 ? "var(--ok)" : sa[f] - sb[f] < -0.005 ? "var(--stamp)" : "var(--ink-3)" } }, (sa[f] - sb[f] >= 0 ? "+" : "") + (sa[f] - sb[f]).toFixed(2)))))))));
    requestAnimationFrame(() => drawNetwork(canvas, net.graph, { overlay: ov, A, B }));
  }

  root.append(arena, h2h);
  update();
  return { el: root, update };
}

// ========================================================================
// screens/curves.js
// ========================================================================
// Preservation curves: fidelity versus budget for every function, six methods and two random baselines.
const NOTE = { struct: "Structural fidelity: connectivity, non-isolated nodes, strength distribution.",
  spread: "Epidemic spreading (SI): infection order and half-infection time versus the original.",
  comm: "Louvain communities, NMI normalised by Louvain's self-agreement.", path: "Share of node pairs whose shortest distance grows by ≤ 10%.",
  rank: "Spearman correlation of strength, eigenvector and betweenness rankings." };

function Curves(net) {
  const root = h("div"), active = new Set(METHODS);
  let tableMode = false;
  const legend = () => MethodLegend(active, draw);
  const body = h("div", { class: "grid2" });

  function seriesFor(f) {
    const s = [];
    for (const nl of ["R-U", "R-W"]) s.push({ key: nl, color: cssVar("--null"), dash: nl === "R-U" ? "2 3" : "6 3", width: 1.3, faint: true, label: nl,
      pts: BUDGETS.map((q) => ({ x: q, y: net.metrics.nulls[nl][qKey(q)][f] })) });
    for (const m of METHODS) if (active.has(m)) s.push({ key: m, color: methodColor(m), dash: DASHES[m], marker: MARKERS[m], label: m,
      pts: BUDGETS.map((q) => { const y = net.metrics.scores[m][qKey(q)][f], c = net.metrics.ci?.[m]?.[qKey(q)]?.[f]; return c ? { x: q, y, lo: c[0], hi: c[1] } : { x: q, y }; }) });
    return s;
  }
  function tip(f) {
    return (x) => `<div class="smallcaps" style="color:var(--bg)">${FUNC_NAMES[f]} · budget ${Math.round(x * 100)}%</div>` +
      [...METHODS.filter((m) => active.has(m)).map((m) => [m, net.metrics.scores[m][qKey(x)][f]]), ["R-U", net.metrics.nulls["R-U"][qKey(x)][f]], ["R-W", net.metrics.nulls["R-W"][qKey(x)][f]]]
        .sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} <b>${f2(v)}</b>`).join("<br>");
  }
  function table() {
    return h("div", { class: "card", style: { padding: "12px", gridColumn: "1 / -1", overflowX: "auto" } },
      h("table", { class: "t" }, h("thead", {}, h("tr", {}, h("th", {}, "Method"), h("th", {}, "Function"), BUDGETS.map((q) => h("th", {}, Math.round(q * 100) + "%")))),
        h("tbody", {}, [...METHODS, "R-U", "R-W"].flatMap((m) => FUNCS.map((f) => h("tr", {}, h("td", {}, m), h("td", {}, FUNC_NAMES[f]),
          BUDGETS.map((q) => h("td", { class: "num" }, f2(m.startsWith("R") ? net.metrics.nulls[m][qKey(q)][f] : net.metrics.scores[m][qKey(q)][f])))))))));
  }
  function draw() {
    legendHost.replaceChildren(legend());
    body.replaceChildren(...(tableMode ? [table()] : FUNCS.map((f) => h("figure", { class: "card", style: { margin: 0, padding: "10px 12px" } },
      lineChart({ title: FUNC_NAMES[f], series: seriesFor(f), xs: BUDGETS, cursor: state.q, tipHtml: tip(f), onClickX: (q) => { set({ q }); draw(); } }),
      h("figcaption", { class: "hint" }, NOTE[f])))));
  }
  const legendHost = h("div");
  root.append(h("div", { class: "row", style: { justifyContent: "space-between" } }, legendHost,
    h("button", { class: "btn", "aria-pressed": tableMode, onclick: (e) => { tableMode = !tableMode; e.target.setAttribute("aria-pressed", tableMode); draw(); } }, "Table view")),
    h("p", { class: "hint" }, "Shaded bands are 95% bootstrap intervals over simulation seeds, Louvain runs and source nodes. Grey lines are random removal (dotted: uniform, dashed: weight-proportional). Click a chart to move the budget."),
    body);
  draw();
  return { el: root, update: draw };
}

// ========================================================================
// screens/atlas.js
// ========================================================================
// Failure atlas: where does a passing structural check mislead? Heatmap of the structure-to-function gap + scatter.
async function Atlas(net) {
  const atlas = await getJSON("atlas.json").catch(() => null);
  if (!atlas) return { el: h("div", { class: "empty" }, "Run experiments/e1_e2.py and the exporter to create atlas.json."), update() {} };
  const col = Object.fromEntries(atlas.columns.map((c, i) => [c, i]));
  const rows = atlas.rows;
  let scope = "net"; // "net" | "pooled"
  const root = h("div"), host = h("div");

  const gapColor = (g) => { // diverging around 0: function better than structure (blue) / worse (vermillion)
    const t = Math.max(-1, Math.min(1, g / 0.7));
    return t >= 0 ? mix(cssVar("--div-mid"), cssVar("--div-pos"), t) : mix(cssVar("--div-mid"), cssVar("--div-neg"), -t);
  };
  function heat(fn) {
    const sel = rows.filter((r) => r[col.function] === fn && (scope === "pooled" || r[col.network] === net.name));
    const cw = 62, ch = 38, ml = 64, mt = 34, W = ml + cw * BUDGETS.length + 10, H = mt + ch * METHODS.length + 8;
    const svg = S("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": `Gap heatmap for ${FUNC_NAMES[fn]}` });
    BUDGETS.forEach((q, j) => svg.append(S("text", { x: ml + cw * j + cw / 2, y: 22, "text-anchor": "middle", "font-size": 11, class: "num" }, Math.round(q * 100) + "%")));
    METHODS.forEach((m, i) => {
      svg.append(S("text", { x: ml - 8, y: mt + ch * i + ch / 2 + 4, "text-anchor": "end", "font-size": 12, style: { fill: methodColor(m), fontWeight: 600 } }, m));
      BUDGETS.forEach((q, j) => {
        const cell = sel.filter((r) => r[col.method] === m && Math.abs(r[col.q] - q) < 1e-9);
        if (!cell.length) return;
        const gap = cell.reduce((a, r) => a + r[col.gap], 0) / cell.length, fail = cell.reduce((a, r) => a + (r[col.failure] ? 1 : 0), 0) / cell.length;
        const x = ml + cw * j, y = mt + ch * i;
        const g = S("g", { class: "cellx", style: { cursor: "pointer" }, tabindex: 0, role: "button", "aria-label": `${m} at ${Math.round(q * 100)} percent, gap ${gap.toFixed(2)}`,
          onclick: () => go("arena", { q, a: m }), onkeydown: (e) => e.key === "Enter" && go("arena", { q, a: m }),
          onmousemove: (e) => showTip(e, `<b>${m}</b> at ${Math.round(q * 100)}%<br>mean gap <b>${gap.toFixed(2)}</b> (structure − ${FUNC_NAMES[fn].toLowerCase()})<br>failures in <b>${pct(fail)}</b> of ${cell.length} ${cell.length > 1 ? "networks" : "network"}<br><span style="opacity:.7">click to open in the arena</span>`),
          onmouseleave: hideTip },
          S("rect", { x: x + 1, y: y + 1, width: cw - 2, height: ch - 2, fill: gapColor(gap), rx: 2 }),
          S("text", { x: x + cw / 2, y: y + ch / 2 + 4, "text-anchor": "middle", "font-size": 11, class: "num", style: { fill: Math.abs(gap) > 0.35 ? "#fff" : "var(--ink)" } }, gap.toFixed(2)));
        if (fail > 0) g.append(S("rect", { x: x + 3, y: y + 3, width: cw - 6, height: ch - 6, fill: "none", stroke: cssVar("--stamp"), "stroke-width": 1 + 2.5 * fail, rx: 2, "stroke-dasharray": fail < 1 ? "4 2" : "none" }));
        svg.append(g);
      });
    });
    return svg;
  }
  function scatter(fn) {
    const sel = rows.filter((r) => r[col.function] === fn && (scope === "pooled" || r[col.network] === net.name));
    const W = 340, H = 300, m = { l: 36, r: 10, t: 14, b: 32 }, X = linScale(0, 1, m.l, W - m.r), Y = linScale(0, 1, H - m.b, m.t);
    const svg = S("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "Structure versus function fidelity" });
    svg.append(S("rect", { x: X(0.8), y: Y(0.6), width: X(1) - X(0.8), height: Y(0) - Y(0.6), fill: cssVar("--stamp"), opacity: 0.1 }),
      S("text", { x: X(0.9), y: Y(0.05), "text-anchor": "middle", "font-size": 8.5, style: { fill: cssVar("--stamp") } }, "misleading pass"),
      S("line", { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), stroke: "var(--ink-3)", "stroke-dasharray": "3 3" }));
    for (const t of [0, 0.25, 0.5, 0.75, 1]) svg.append(S("text", { x: m.l - 5, y: Y(t) + 3, "text-anchor": "end", "font-size": 9, class: "num" }, t), S("text", { x: X(t), y: H - 16, "text-anchor": "middle", "font-size": 9, class: "num" }, t));
    svg.append(S("text", { x: W / 2, y: H - 3, "text-anchor": "middle", "font-size": 10 }, "structural fidelity"), S("text", { x: 10, y: H / 2, "font-size": 10, transform: `rotate(-90 10 ${H / 2})`, "text-anchor": "middle" }, `${FUNC_NAMES[fn].toLowerCase()} fidelity`));
    for (const r of sel) svg.append(S("circle", { cx: X(r[col.struct]), cy: Y(r[col.F_g]), r: r[col.failure] ? 4.2 : 3, fill: methodColor(r[col.method]), opacity: 0.8, stroke: r[col.failure] ? cssVar("--stamp") : "none", "stroke-width": 1.5,
      onmousemove: (e) => showTip(e, `<b>${r[col.method]}</b> · ${r[col.network]} · ${Math.round(r[col.q] * 100)}%<br>structure ${f2(r[col.struct])} · function ${f2(r[col.F_g])}`), onmouseleave: hideTip }));
    return svg;
  }
  function render() {
    const fn = state.fn === "struct" || !GOALS.includes(state.fn) ? "spread" : state.fn;
    const sel = rows.filter((r) => r[col.function] === fn && (scope === "pooled" || r[col.network] === net.name));
    const nNet = new Set(sel.map((r) => r[col.network])).size, rate = sel.filter((r) => r[col.failure]).length / Math.max(1, sel.length);
    host.replaceChildren(
      h("div", { class: "row", style: { justifyContent: "space-between", marginBottom: "10px" } },
        h("div", { class: "row" }, GOALS.map((g) => h("button", { class: "btn", "aria-pressed": g === fn, onclick: () => { set({ fn: g }); render(); } }, FUNC_NAMES[g]))),
        h("div", { class: "row" }, h("button", { class: "btn", "aria-pressed": scope === "net", onclick: () => { scope = "net"; render(); } }, `This network`),
          h("button", { class: "btn", "aria-pressed": scope === "pooled", onclick: () => { scope = "pooled"; render(); } }, "All networks"))),
      h("div", { class: "split" },
        h("div", { class: "card", style: { padding: "12px" } }, h("h3", {}, "Gap: structure minus function"), h("p", { class: "hint" }, "Red: structure overstates function. Blue: function is better preserved than structure. Outlined cells are failures (structure ≥ 0.80 while function < 0.60); thicker outline = more networks failing. Click a cell to open it in the arena."), heat(fn)),
        h("div", { class: "card", style: { padding: "12px" } }, h("h3", {}, "Structure vs function"), scatter(fn),
          h("p", { class: "hint" }, `${sel.length} cases from ${nNet} network${nNet === 1 ? "" : "s"}; ${pct(rate)} are misleading passes. Points on the dashed line would mean structure predicts function.`))));
  }
  root.append(host);
  render();
  return { el: root, update: render };
}

// ========================================================================
// screens/lens.js
// ========================================================================
// Network lens: one large drawing with layers and an edge inspector.
function Lens(net) {
  const layers = { ghost: true, bridges: false };
  const canvas = h("canvas", { style: { width: "100%", aspectRatio: "1/1", maxHeight: "78vh", cursor: "crosshair" }, "aria-label": "Network drawing; click an edge to inspect it" });
  const inspector = h("div", { class: "card", style: { padding: "14px" } });
  const controls = h("div", { class: "row", style: { marginBottom: "10px" } });
  let geom = null;

  const method = () => (METHODS.includes(state.a) ? state.a : "SMDS");
  function draw() {
    const mask = state.a === "orig" ? null : maskAt(net, method(), state.q);
    geom = drawNetwork(canvas, net.graph, { mask, ghost: layers.ghost, bridges: layers.bridges, bridgeFlags: net.edges.role.bridge, highlight: state.edge });
    renderInspector();
  }
  function renderControls() {
    controls.replaceChildren(
      h("select", { "aria-label": "Method", style: { width: "auto" }, onchange: (e) => { set({ a: e.target.value }); renderControls(); draw(); } },
        h("option", { value: "orig", selected: state.a === "orig" }, "Original"), METHODS.map((m) => h("option", { value: m, selected: m === state.a }, `${m} · ${METHOD_NAMES[m]}`))),
      h("button", { class: "btn", "aria-pressed": layers.ghost, onclick: (e) => { layers.ghost = !layers.ghost; e.target.setAttribute("aria-pressed", layers.ghost); draw(); } }, "Ghost edges"),
      h("button", { class: "btn", "aria-pressed": layers.bridges, onclick: (e) => { layers.bridges = !layers.bridges; e.target.setAttribute("aria-pressed", layers.bridges); draw(); } }, "Highlight bridges"));
  }
  function renderInspector() {
    const e = state.edge;
    if (e < 0) { inspector.replaceChildren(h("h3", {}, "Edge inspector"), h("p", { class: "muted" }, "Click an edge in the drawing. You will see its score and rank under every method, whether each keeps it at the current budget, and its role in the network.")); return; }
    const g = net.graph, k = kOf(state.q, g.m), role = net.edges.role;
    inspector.replaceChildren(h("h3", {}, `Edge ${g.edges.u[e]} – ${g.edges.v[e]}`),
      h("p", { class: "muted", style: { marginTop: "4px" } }, `weight ${g.edges.w[e]} · community ${g.nodes.community[g.edges.u[e]]}${role.bridge[e] ? " → " + g.nodes.community[g.edges.v[e]] : ""}`),
      h("div", { class: "row", style: { margin: "6px 0" } },
        role.bridge[e] ? h("span", { class: "badge b-rejected", title: "connects two communities" }, "bridge") : h("span", { class: "badge b-not_run" }, "within community"),
        role.metric[e] ? h("span", { class: "badge b-supported", title: "lies on a shortest path (metric backbone)" }, "on shortest path") : h("span", { class: "badge b-not_run" }, "redundant for paths"),
        role.hub[e] ? h("span", { class: "badge b-inconclusive" }, "touches hub") : null),
      h("table", { class: "t" }, h("thead", {}, h("tr", {}, h("th", {}, "Method"), h("th", {}, "Rank"), h("th", {}, "Score"), h("th", {}, `Kept at ${pct(state.q)}`))),
        h("tbody", {}, METHODS.map((m) => { const r = net.edges.rank[m][e], kept = r <= k;
          return h("tr", {}, h("td", {}, h("span", { class: "chip", style: { "--mc": `var(--m-${m})` } }, h("i"), m)), h("td", { class: "num" }, `${r} / ${g.m}`), h("td", { class: "num" }, f3(net.edges.score[m][e])),
            h("td", {}, h("span", { class: `badge ${kept ? "b-supported" : "b-rejected"}` }, kept ? "kept" : "dropped"))); }))),
      h("p", { class: "hint" }, `${METHODS.filter((m) => net.edges.rank[m][e] <= k).length} of 6 methods keep this edge.`));
  }
  canvas.addEventListener("click", (ev) => {
    const r = canvas.getBoundingClientRect();
    const e = hitEdge(net.graph, geom, ev.clientX - r.left, ev.clientY - r.top, 6);
    set({ edge: e }); draw();
  });
  renderControls();
  const el = h("div", {}, controls, h("div", { class: "split" }, h("div", { class: "card", style: { padding: "10px" } }, canvas), inspector));
  requestAnimationFrame(draw);
  return { el, update: draw };
}

// ========================================================================
// screens/theatre.js
// ========================================================================
// Spread theatre: replay of mean infection times on the original and a backbone, plus infected-fraction curves.
async function Theatre(net) {
  const sp = await getJSON(`${net.name}/spread.json`).catch(() => null);
  if (!sp) return { el: h("div", { class: "empty" }, "spread.json missing: export without --no-spread."), update() {} };
  const T = sp.steps, ts = sp.t;
  let t = 0, timer = null;
  const cA = h("canvas", { style: { width: "100%", aspectRatio: "1/1" } }), cB = h("canvas", { style: { width: "100%", aspectRatio: "1/1" } });
  const clock = h("span", { class: "mono" }), slider = h("input", { type: "range", min: 0, max: T, value: 0, "aria-label": "Time step", oninput: (e) => { t = +e.target.value; frame(); } });
  const play = h("button", { class: "btn", onclick: toggle }, "▶ Play");
  const chartHost = h("div", { class: "card", style: { padding: "10px 12px" } });
  const title = h("div", { class: "row", style: { justifyContent: "space-between" } });
  const seedNode = () => sp.seeds[state.tier];
  const method = () => (METHODS.includes(state.a) ? state.a : "SMDS");

  const infColor = (mean, tt, seed, i) => (i === seed ? cssVar("--stamp") : mean[i] <= tt ? mix(cssVar("--seq-3"), cssVar("--seq-4"), Math.min(1, tt - mean[i] > 15 ? 1 : (tt - mean[i]) / 15)) : cssVar("--rule"));
  function frame() {
    const mG = sp.data.G[state.tier].mean_time, mB = sp.data[method()][qKey(state.q)][state.tier].mean_time, seed = seedNode();
    drawNetwork(cA, net.graph, { mask: null, ghost: false, edgeColor: cssVar("--ink-3"), nodeColor: (i) => infColor(mG, t, seed, i) });
    drawNetwork(cB, net.graph, { mask: net.masks[method()][qKey(state.q)], ghost: true, nodeColor: (i) => infColor(mB, t, seed, i) });
    clock.textContent = `t = ${t}`; slider.value = t;
  }
  function toggle() {
    if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ Play"; return; }
    if (t >= T) t = 0;
    play.textContent = "❚❚ Pause";
    timer = setInterval(() => { t = Math.min(T, t + 1); frame(); if (t >= T) toggle(); }, 55);
  }
  function chart() {
    const series = [{ key: "G", color: cssVar("--orig"), width: 2.6, label: "orig", pts: ts.map((x, i) => ({ x, y: sp.data.G[state.tier].curve[i] })) },
      ...METHODS.map((m) => ({ key: m, color: methodColor(m), dash: DASHES[m], marker: m === method() ? MARKERS[m] : null, width: m === method() ? 2.4 : 1.3, faint: m !== method(), label: m,
        pts: ts.map((x, i) => ({ x, y: sp.data[m][qKey(state.q)][state.tier].curve[i] })) }))];
    chartHost.replaceChildren(h("h3", {}, "Share of nodes infected over time"),
      lineChart({ width: 1000, height: 300, title: `seed tier: ${state.tier} centrality · budget ${Math.round(state.q * 100)}%`, series, xs: ts, cursor: t, xFmt: (x) => x,
        onClickX: (x) => { t = x; frame(); chart(); },
        tipHtml: (x) => { const i = ts.indexOf(x); return `t = <b>${x}</b><br>orig <b>${f2(sp.data.G[state.tier].curve[i])}</b><br>` + METHODS.map((m) => `${m} <b>${f2(sp.data[m][qKey(state.q)][state.tier].curve[i])}</b>`).join("<br>"); } }));
  }
  function render() {
    title.replaceChildren(h("div", { class: "row" },
      h("select", { "aria-label": "Method", style: { width: "auto" }, onchange: (e) => { set({ a: e.target.value }); render(); } }, METHODS.map((m) => h("option", { value: m, selected: m === method() }, `${m} · ${METHOD_NAMES[m]}`))),
      h("select", { "aria-label": "Seed centrality tier", style: { width: "auto" }, onchange: (e) => { set({ tier: e.target.value }); render(); } },
        ["high", "median", "low"].map((x) => h("option", { value: x, selected: x === state.tier }, `${x}-centrality seed`)))),
      h("div", { class: "row" }, play, slider, clock));
    frame(); chart();
  }
  const el = h("div", {}, title, h("div", { class: "grid2", style: { margin: "12px 0" } },
    h("figure", { class: "card", style: { margin: 0, padding: "10px" } }, h("div", { class: "smallcaps" }, "original"), cA),
    h("figure", { class: "card", style: { margin: 0, padding: "10px" } }, h("div", { class: "smallcaps" }, "backbone"), cB)),
    h("p", { class: "hint" }, `A node lights up once its mean infection time (over ${sp.n_runs} runs from the red seed) has passed — a replay of the average outbreak, not a single run. Delays on the right panel are the cost of the removed edges.`), chartHost);
  requestAnimationFrame(render);
  return { el, update: render, destroy: () => timer && clearInterval(timer) };
}

// ========================================================================
// screens/mechanisms.js
// ========================================================================
// Mechanism bench: gap versus bridge retention (real networks, associational) and the synthetic factorial (causal).
function ols(xs, ys) {
  const n = xs.length, mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sxx += (xs[i] - mx) ** 2; sxy += (xs[i] - mx) * (ys[i] - my); }
  const b = sxy / (sxx || 1);
  return { b, a: my - b * mx };
}

async function Mechanisms() {
  const [atlas, conc] = await Promise.all([getJSON("atlas.json").catch(() => null), getJSON("conclusions.json").catch(() => null)]);
  if (!atlas) return { el: h("div", { class: "empty" }, "atlas.json missing."), update() {} };
  const col = Object.fromEntries(atlas.columns.map((c, i) => [c, i]));
  if (!("bridge_ratio" in col)) return { el: h("div", { class: "empty" }, "Re-export: atlas.json lacks bridge_ratio."), update() {} };
  const root = h("div");
  const currentFn = () => (["spread", "comm"].includes(state.fn) ? state.fn : "spread");

  function scatter(fn) {
    const rows = atlas.rows.filter((r) => r[col.function] === fn && Number.isFinite(r[col.bridge_ratio]) && r[col.bridge_ratio] <= 3);
    const W = 560, H = 360, m = { l: 44, r: 12, t: 12, b: 38 };
    const xmax = Math.max(1.5, ...rows.map((r) => r[col.bridge_ratio])), X = linScale(0, xmax, m.l, W - m.r);
    const gmin = Math.min(-0.1, ...rows.map((r) => r[col.gap])), gmax = Math.max(0.8, ...rows.map((r) => r[col.gap])), Y = linScale(gmin, gmax, H - m.b, m.t);
    const svg = S("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "Gap versus bridge-retention ratio" });
    for (const t of [0, 0.25, 0.5, 0.75]) {
      svg.append(S("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: "var(--rule-2)" }), S("text", { x: m.l - 6, y: Y(t) + 3, "text-anchor": "end", "font-size": 9, class: "num" }, t));
    }
    svg.append(S("line", { x1: X(1), x2: X(1), y1: m.t, y2: H - m.b, stroke: "var(--ink-3)", "stroke-dasharray": "3 3" }),
      S("text", { x: X(1) + 4, y: m.t + 10, "font-size": 9 }, "ratio 1: bridges kept in proportion"));
    for (let t = 0; t <= xmax; t += 0.5) svg.append(S("text", { x: X(t), y: H - 22, "text-anchor": "middle", "font-size": 9, class: "num" }, t));
    svg.append(S("text", { x: W / 2, y: H - 6, "text-anchor": "middle", "font-size": 10 }, "bridge-retention ratio (inter-community edges kept / intra-community edges kept)"),
      S("text", { x: 12, y: H / 2, "font-size": 10, transform: `rotate(-90 12 ${H / 2})`, "text-anchor": "middle" }, `gap: structure - ${FUNC_NAMES[fn].toLowerCase()}`));
    for (const r of rows) {
      svg.append(S("circle", { cx: X(r[col.bridge_ratio]), cy: Y(r[col.gap]), r: 3.2, fill: methodColor(r[col.method]), opacity: 0.7,
        onmousemove: (e) => showTip(e, `<b>${r[col.method]}</b> · ${r[col.network]} · ${Math.round(r[col.q] * 100)}%<br>bridge ratio <b>${f2(r[col.bridge_ratio])}</b> · gap <b>${f2(r[col.gap])}</b>`),
        onmouseleave: hideTip }));
    }
    if (rows.length > 3) {
      const { a, b } = ols(rows.map((r) => r[col.bridge_ratio]), rows.map((r) => r[col.gap]));
      svg.append(S("line", { x1: X(0), y1: Y(a), x2: X(xmax), y2: Y(a + b * xmax), stroke: "var(--ink)", "stroke-width": 1.6 }));
    }
    return { svg, n: rows.length, nets: new Set(rows.map((r) => r[col.network])).size };
  }

  function forest(het) {
    const rows = het.filter((r) => r.function === "spread"), W = 520, H = 40 + rows.length * 38, ml = 130;
    const lo = Math.min(-0.05, ...rows.map((r) => r.ci_lo)), hi = Math.max(0.05, ...rows.map((r) => r.ci_hi)), X = linScale(lo, hi, ml, W - 16);
    const svg = S("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "Interaction effects" });
    svg.append(S("line", { x1: X(0), x2: X(0), y1: 10, y2: H - 26, stroke: "var(--ink)", "stroke-dasharray": "3 3" }));
    const lab = { "wb:sigma": "weight spread (sigma)", "wb:coupling": "weight-degree coupling", "wb:mixing": "community mixing" };
    rows.forEach((r, i) => {
      const y = 24 + i * 38, good = r.ci_lo > 0;
      svg.append(S("text", { x: ml - 8, y: y + 4, "text-anchor": "end", "font-size": 11 }, lab[r.term] || r.term),
        S("line", { x1: X(r.ci_lo), x2: X(r.ci_hi), y1: y, y2: y, stroke: good ? cssVar("--stamp") : cssVar("--ink-3"), "stroke-width": 3 }),
        S("circle", { cx: X(r.coef), cy: y, r: 5, fill: good ? cssVar("--stamp") : cssVar("--ink") }));
    });
    svg.append(S("text", { x: (ml + W) / 2, y: H - 6, "text-anchor": "middle", "font-size": 10 }, "extra gap for W and DF relative to SMDS, per unit of the factor (95% CI)"));
    return svg;
  }

  function bridgeTable(br) {
    if (!br.length) return h("p", { class: "muted" }, "Not computed yet.");
    const rows = br.map((r) => h("tr", {}, h("td", {}, FUNC_NAMES[r.function]), h("td", { class: "num" }, f3(r.coef_bridge_ratio)),
      h("td", { class: "num" }, `[${f3(r.ci_lo)}, ${f3(r.ci_hi)}]`), h("td", { class: "num" }, r.p_lr_holm < 0.001 ? "<0.001" : f3(r.p_lr_holm)),
      h("td", {}, h("span", { class: `badge ${r.supported ? "b-supported" : "b-inconclusive"}` }, r.supported ? "supported" : "not supported"))));
    return h("table", { class: "t" }, h("thead", {}, h("tr", {}, ["Function", "Bridge coef.", "95% CI", "Holm p", "Verdict"].map((x) => h("th", {}, x)))), h("tbody", {}, rows));
  }

  function render() {
    const fn = currentFn(), sc = scatter(fn), br = conc?.tables?.bridge || [], het = conc?.tables?.heterogeneity;
    const left = h("section", { class: "card", style: { padding: "14px" } },
      h("div", { class: "row", style: { justifyContent: "space-between" } }, h("h3", {}, "Bridges and the gap"), h("span", { class: "badge b-inconclusive" }, "associational")),
      h("p", { class: "hint" }, "Each point is one method at one budget on one network. If losing bridges drives failures, points with a low ratio should sit higher."), sc.svg,
      h("p", { class: "hint" }, `${sc.n} cases from ${sc.nets} networks. Line: simple least-squares fit (the report uses a mixed model with network as a random effect).`));
    const right = h("section", { class: "card", style: { padding: "14px" } }, h("h3", {}, "Mixed-model test (H3)"), bridgeTable(br),
      h("p", { class: "hint" }, "A negative coefficient means more bridges kept, smaller gap. Real-network regressions show association, not cause."),
      h("h3", { style: { marginTop: "18px" } }, "Weight heterogeneity (RQ3)"), h("span", { class: "badge b-supported", style: { margin: "4px 0 8px" } }, "causal, synthetic"),
      het ? forest(het) : h("p", { class: "muted" }, "The synthetic factorial sweep (270 networks) has not been analysed yet. When it is, this panel shows whether heterogeneous or degree-coupled weights hurt W and DF more than SMDS."));
    root.replaceChildren(h("div", { class: "row", style: { marginBottom: "12px" } },
      ["spread", "comm"].map((g) => h("button", { class: "btn", "aria-pressed": g === fn, onclick: () => { set({ fn: g }); render(); } }, FUNC_NAMES[g]))),
      h("div", { class: "split" }, left, right));
  }
  render();
  return { el: root, update: render };
}

// ========================================================================
// screens/conclusions.js
// ========================================================================
// Conclusions: auto-generated from conclusions.json (verdicts, findings, leaderboard, regimes, replication).
const ICON = { supported: "✓", rejected: "✕", inconclusive: "≈", not_run: "…" };
const FIG = { atlas: "atlas", curves: "curves", leaderboard: "conclusions" };

const chip = (m) => h("span", { class: "chip", style: { "--mc": `var(--m-${m})` } }, h("i"), m);

function leaderboardTable(lb, heat) {
  const cell = (m, g) => {
    const r = lb.find((x) => x.method === m && x.function === g);
    if (!r) return h("td", {}, "–");
    return h("td", { class: "num", style: { background: heat(r.mean_skill) }, title: `n=${r.n_networks}` }, `${f2(r.mean_skill)} [${f2(r.ci_lo)}, ${f2(r.ci_hi)}]`);
  };
  const rows = METHODS.map((m) => h("tr", {}, h("td", {}, chip(m)), GOALS.map((g) => cell(m, g))));
  return h("div", { class: "card", style: { padding: "12px", overflowX: "auto" } }, h("table", { class: "t" },
    h("thead", {}, h("tr", {}, h("th", {}, "Method"), GOALS.map((g) => h("th", {}, FUNC_NAMES[g])))), h("tbody", {}, rows)));
}

function regimeTable(regimes) {
  const names = [...new Set(regimes.map((r) => r.regime))];
  const cell = (g, reg) => {
    const r = regimes.find((x) => x.function === g && x.regime === reg);
    return h("td", {}, chip(r.best), h("span", { class: "hint" }, ` +${f2(r.margin)} over ${r.runner_up}`));
  };
  const rows = GOALS.map((g) => h("tr", {}, h("td", {}, FUNC_NAMES[g]), names.map((reg) => cell(g, reg))));
  return h("div", { class: "card", style: { padding: "12px", overflowX: "auto" } }, h("table", { class: "t" },
    h("thead", {}, h("tr", {}, h("th", {}, "Function"), names.map((r) => h("th", {}, r)))), h("tbody", {}, rows)));
}

function robustness(c) {
  const rob = c.tables.robustness;
  if (!rob) return null;
  const flag = (x) => h("td", {}, h("span", { class: `badge ${x ? "b-supported" : "b-rejected"}` }, x ? "holds" : "fails"));
  const best = (r, f) => h("td", {}, h("span", { style: { fontWeight: r[`best_${f}_changed`] ? 700 : 400, color: r[`best_${f}_changed`] ? "var(--stamp)" : "inherit" } }, r[`best_${f}`]));
  const rows = rob.map((r) => h("tr", {}, h("td", {}, r.variant), flag(r["RQ1 (>=2 functions with mean tau < 0.6)"]), flag(r["H1 direction (SMDS, HSS > W, DF)"]), flag(r["RQ4 (> 50% cells differ)"]),
    best(r, "spread"), best(r, "comm"), best(r, "path"), best(r, "rank")));
  return h("div", {}, h("h2", { style: { margin: "26px 0 8px" } }, "Robustness to modelling choices"),
    h("p", { class: "hint" }, "Each row re-runs the study with one change (SIR instead of SI, another proximity map, infection speed, path tolerance, Louvain resolution, uniform seeds). Red = the best method changed relative to the default."),
    h("div", { class: "card", style: { padding: "12px", overflowX: "auto" } }, h("table", { class: "t" },
      h("thead", {}, h("tr", {}, ["Variant", "RQ1", "H1", "RQ4", "Best: spreading", "communities", "paths", "rankings"].map((x) => h("th", {}, x)))), h("tbody", {}, rows))));
}

function replication(c) {
  if (!c.tables.replication.length) return null;
  const rows = c.tables.replication.map((r) => h("tr", {}, h("td", {}, r.source), h("td", {}, r.claim), h("td", {}, r.ours),
    h("td", {}, h("span", { class: `badge ${r.agrees ? "b-supported" : "b-rejected"}` }, r.agrees ? "yes" : "no"))));
  return h("div", {}, h("h2", { style: { margin: "26px 0 8px" } }, "Replication of earlier claims"),
    h("div", { class: "card", style: { padding: "12px", overflowX: "auto" } }, h("table", { class: "t" },
      h("thead", {}, h("tr", {}, ["Source", "Claim", "These data", "Agrees"].map((x) => h("th", {}, x)))), h("tbody", {}, rows))));
}

async function Conclusions() {
  const c = await getJSON("conclusions.json").catch(() => null);
  if (!c) return { el: h("div", { class: "empty" }, "No conclusions.json yet. Run experiments/conclusions.py, then export."), update() {} };
  const lb = c.tables.leaderboard;
  const heat = (v) => mix(cssVar("--seq-0"), cssVar("--seq-3"), Math.max(0, Math.min(1, v)));
  const el = h("div", {},
    c.confirmatory ? null : h("div", { class: "notice" }, `Exploratory results: the protocol is "${c.protocol_status}", not frozen. Treat verdicts as development findings until the protocol is tagged.`),
    h("section", { class: "card headline" }, h("div", { class: "smallcaps" }, `answer to the central question · ${c.n_networks} networks${c.n_synthetic ? ` + ${c.n_synthetic} synthetic` : ""}`), h("p", {}, c.summary.answer)),
    h("h2", { style: { margin: "22px 0 10px" } }, "Hypotheses (H1–H3) and research questions"),
    h("div", { class: "verdicts" }, c.verdicts.map((v) => h("article", { class: "card verdict" },
      h("div", { class: "row", style: { justifyContent: "space-between" } }, h("span", { class: "id" }, v.id), h("span", { class: `badge b-${v.verdict}` }, `${ICON[v.verdict]} ${v.verdict.replace("_", " ")}`)),
      h("div", {}, v.statement), h("div", { class: "hint" }, "Rule: " + v.rule),
      h("div", { class: "mono hint" }, `${v.n_networks} networks`), v.caveats.map((x) => h("div", { class: "hint" }, "· " + x))))),
    h("div", { class: "split", style: { marginTop: "22px" } },
      h("section", {}, h("h2", { style: { marginBottom: "8px" } }, "Findings"),
        h("ul", { style: { paddingLeft: "18px", margin: 0 } }, c.findings.map((f) => h("li", { style: { margin: "6px 0" } }, f.text, " ",
          h("a", { href: `#/${FIG[f.figure] || "atlas"}`, class: "hint", title: "show the supporting figure" }, "evidence →"), " ", h("span", { class: "hint" }, `(${f.strength})`))))),
      h("section", { class: "card", style: { padding: "14px" } }, h("h3", {}, "Why, and which method"),
        c.summary.why.concat(c.summary.methods).map((t) => h("p", { style: { margin: "8px 0" } }, t)))),
    h("h2", { style: { margin: "26px 0 8px" } }, "Leaderboard"), h("p", { class: "hint" }, "Mean skill versus weight-proportional random removal (0 = no better than chance, 1 = perfect). Intervals are cluster bootstrap over networks."),
    leaderboardTable(lb, heat),
    h("h2", { style: { margin: "26px 0 8px" } }, "Best method by budget regime"),
    regimeTable(c.tables.regimes),
    replication(c),
    robustness(c),
    h("h2", { style: { margin: "26px 0 8px" } }, "Limits"), h("ul", { style: { paddingLeft: "18px" } }, c.summary.limits.map((t) => h("li", {}, t))),
    h("p", { class: "hint" }, `Generated ${c.generated}.`));
  return { el, update() {} };
}

// ========================================================================
// screens/advisor.js
// ========================================================================
// Backbone Advisor: objective weights + budget -> recommendation with probability of being best, trade-offs, evidence.
const quant = (arr, p) => { const a = Float64Array.from(arr).sort(); return a[Math.min(a.length - 1, Math.max(0, Math.floor(p * (a.length - 1))))]; };

/** Pure computation from exported metrics + bootstrap draws (same logic as backbone/advisor/evidence.py). */
function advise(net, q, wts, requireConnected) {
  const k = qKey(q), u = GOALS.map((g) => wts[g] || 0), su = u.reduce((a, b) => a + b, 0);
  if (su <= 0 || !net.boot) return null;
  const un = u.map((x) => x / su);
  const methods = METHODS.filter((m) => net.metrics.flags[m][k].connected || !requireConnected);
  const excluded = METHODS.filter((m) => !methods.includes(m));
  if (!methods.length) return { error: "No method gives a connected backbone at this budget." };
  const B = net.boot.B;
  const draws = methods.map((m) => { const d = net.boot.data[m][k], out = new Float64Array(B);
    GOALS.forEach((g, gi) => { const v = d[g]; for (let b = 0; b < B; b++) out[b] += un[gi] * (Array.isArray(v) ? v[b] : v); });
    return out; });
  const point = methods.map((m) => GOALS.reduce((a, g, gi) => a + un[gi] * net.metrics.scores[m][k][g], 0));
  const pBest = new Array(methods.length).fill(0);
  for (let b = 0; b < B; b++) { let bi = 0; for (let i = 1; i < methods.length; i++) if (draws[i][b] > draws[bi][b]) bi = i; pBest[bi] += 1 / B; }
  const ci = draws.map((d) => [quant(d, 0.025), quant(d, 0.975)]);
  const order = methods.map((_, i) => i).sort((a, b) => point[b] - point[a]);
  const top = order[0], second = order[1] ?? order[0];
  const overlap = order.length > 1 && ci[second][1] >= ci[top][0];
  const tied = order.length > 1 && point[top] - point[second] < 0.02; // practically tied even if noise is small
  const label = tied ? "close call" : pBest[top] >= 0.8 ? "clear" : overlap ? "close call" : "likely";
  const F = methods.map((m) => GOALS.map((g) => net.metrics.scores[m][k][g]));
  const pareto = methods.filter((_, i) => !methods.some((__, j) => j !== i && F[j].every((x, c) => x >= F[i][c]) && F[j].some((x, c) => x > F[i][c])));
  const rec = methods[top];
  const give = GOALS.map((g, gi) => { const best = Math.max(...F.map((r) => r[gi])), bm = methods[F.findIndex((r) => r[gi] === best)]; return { g, best, bm, mine: F[top][gi], give: best - F[top][gi] }; });
  const warn = [];
  const sc = net.metrics.scores[rec][k];
  GOALS.forEach((g, gi) => { if (sc.struct >= 0.8 && sc[g] < 0.6 && un[gi] > 0) warn.push(`${rec} looks structurally fine (structure ${f2(sc.struct)}) but ${FUNC_NAMES[g].toLowerCase()} fidelity is only ${f2(sc[g])}.`);
    if (net.metrics.skill["R-W"][rec][k][g] <= 0 && un[gi] > 0) warn.push(`${rec} is no better than weight-proportional random removal on ${FUNC_NAMES[g].toLowerCase()}.`); });
  if (excluded.length) warn.push(`Excluded because their backbone is disconnected at ${pct(q)}: ${excluded.join(", ")}.`);
  return { q, un, methods, excluded, point, ci, pBest, order, rec, label, pareto, give, warn, F, top };
}

function adviseNeighbours(net, q, wts, conn) {
  const i = BUDGETS.findIndex((b) => Math.abs(b - q) < 1e-9), out = {};
  for (const j of [i - 1, i + 1]) if (j >= 0 && j < BUDGETS.length) out[BUDGETS[j]] = advise(net, BUDGETS[j], wts, conn)?.rec;
  return out;
}

function Advisor(net) {
  const host = h("div");
  if (!net.boot) return { el: h("div", { class: "empty" }, "bootstrap.json missing for this network: run `runner samples` then the exporter."), update() {} };

  function render() {
    const wts = weights(), conn = !!state.conn;
    const r = advise(net, state.q, wts, conn);
    if (!r) { host.replaceChildren(h("div", { class: "empty" }, "Give at least one goal a positive weight.")); return; }
    if (r.error) { host.replaceChildren(h("div", { class: "notice" }, r.error)); return; }
    const neigh = adviseNeighbours(net, state.q, wts, conn), stable = Object.values(neigh).every((x) => x === r.rec);
    const inputs = h("div", { class: "card", style: { padding: "16px" } }, h("h3", {}, "Your goal"), h("p", { class: "hint" }, "How much does each function matter for what you will do with the backbone?"),
      h("div", { class: "sliders" }, GOALS.map((g) => h("label", {}, h("span", {}, FUNC_NAMES[g]),
        h("input", { type: "range", min: 0, max: 10, value: wts[g], "aria-label": FUNC_NAMES[g] + " weight", oninput: (e) => { const w = weights(); w[g] = +e.target.value; set({ w: GOALS.map((x) => w[x]).join(",") }); render(); } }),
        h("span", { class: "mono" }, Math.round((wts[g] / (Object.values(wts).reduce((a, b) => a + b, 0) || 1)) * 100) + "%")))),
      h("div", { class: "row", style: { margin: "14px 0 6px" } }, h("button", { class: "btn", "aria-pressed": conn, onclick: () => { set({ conn: conn ? 0 : 1 }); render(); } }, "Backbone must be connected")),
      h("div", { class: "row", style: { gap: "6px" } }, [["Spreading", [10, 0, 0, 0]], ["Communities", [0, 10, 0, 0]], ["Paths", [0, 0, 10, 0]], ["Balanced", [5, 5, 5, 5]]].map(([l, v]) =>
        h("button", { class: "btn", onclick: () => { set({ w: v.join(",") }); render(); } }, l))),
      h("p", { class: "hint", style: { marginTop: "12px" } }, `Budget ${pct(state.q)} — move the dial on the left. Recomputed live from stored bootstrap draws (${net.boot.B} resamples). Uncertainty covers simulation and algorithm noise on this network only, not variation across networks.`));
    const rec = h("div", { class: "card rec" }, h("div", { class: "smallcaps" }, "recommended method"),
      h("div", { class: "big", style: { color: `var(--m-${r.rec})` } }, r.rec), h("div", { class: "muted" }, METHOD_NAMES[r.rec]),
      h("div", { style: { margin: "10px 0" } }, h("span", { class: `badge ${r.label === "clear" ? "b-supported" : "b-inconclusive"}` }, r.label)),
      h("p", { class: "muted", style: { margin: "0 0 12px" } }, r.label === "clear" ? `${pct(r.pBest[r.top])} probability of being the best method for this goal.` :
        r.label === "close call" ? `Close call: ${r.methods[r.order[1]]} is statistically close (${pct(r.pBest[r.order[1]])} chance of being best).` : `${pct(r.pBest[r.top])} probability of being best.`),
      r.order.map((i) => h("div", { class: "pbar", style: { "--mc": `var(--m-${r.methods[i]})` } }, h("b", {}, r.methods[i]),
        h("div", { class: "track" }, h("div", { class: "fill", style: { width: pct(Math.max(0, Math.min(1, r.point[i]))), background: `var(--m-${r.methods[i]})`, opacity: i === r.top ? 1 : 0.55 } }),
          h("div", { class: "w", style: { left: pct(Math.max(0, r.ci[i][0])), width: pct(Math.max(0.004, r.ci[i][1] - r.ci[i][0])) } })),
        h("span", { class: "mono" }, f2(r.point[i])))),
      h("div", { class: "hint" }, "bar = utility, whisker = 95% interval"));
    const evid = h("div", { class: "card", style: { padding: "16px" } }, h("h3", {}, "Trade-offs and evidence"),
      h("p", { class: "hint" }, `Non-dominated methods: ${r.pareto.join(", ")}.`),
      h("table", { class: "t" }, h("thead", {}, h("tr", {}, h("th", {}, "Function"), h("th", {}, r.rec), h("th", {}, "Best"), h("th", {}, "Gives up"))),
        h("tbody", {}, r.give.map((x) => h("tr", {}, h("td", {}, FUNC_NAMES[x.g]), h("td", { class: "num" }, f2(x.mine)), h("td", { class: "num" }, `${x.bm} ${f2(x.best)}`),
          h("td", { class: "num", style: { color: x.give > 0.05 ? "var(--stamp)" : "var(--ink-3)" } }, x.give > 0.0005 ? "−" + f2(x.give) : "–"))))),
      r.warn.length ? h("div", { class: "notice", style: { marginTop: "12px", color: "var(--stamp)", borderColor: "var(--stamp)" } }, r.warn.map((w) => h("div", {}, "⚠ " + w))) : h("p", { class: "hint" }, "No warnings for this recommendation."),
      h("p", { class: "hint", style: { marginTop: "10px" } }, Object.keys(neigh).length ? `Sensitivity: at ${Object.entries(neigh).map(([q, m]) => `${pct(+q)} → ${m}`).join(" and ")}. ${stable ? "The recommendation is stable." : "The recommendation changes with the budget."}` : ""),
      h("div", { class: "row", style: { marginTop: "10px" } }, h("button", { class: "btn", onclick: () => download(r, neigh, stable) }, "Export report (.md)")));
    host.replaceChildren(h("div", { style: { display: "grid", gridTemplateColumns: "minmax(250px,1fr) minmax(280px,1.1fr) minmax(280px,1.2fr)", gap: "16px", alignItems: "start" } }, inputs, rec, evid));
  }
  function download(r, neigh, stable) {
    const L = [`# Backbone Advisor report: ${net.name}`, "", `- Budget: ${pct(state.q)}`, `- Goal weights: ${GOALS.map((g, i) => `${g} ${pct(r.un[i])}`).join(", ")}`,
      `- **Recommendation: ${r.rec}** (${r.label}; probability of being best ${pct(r.pBest[r.top])})`, "", "| Method | Utility | 95% CI | P(best) |", "|---|---|---|---|",
      ...r.order.map((i) => `| ${r.methods[i]} | ${f3(r.point[i])} | [${f3(r.ci[i][0])}, ${f3(r.ci[i][1])}] | ${pct(r.pBest[i])} |`), "", `Non-dominated: ${r.pareto.join(", ")}`, "",
      ...r.give.map((x) => `- ${x.g}: ${f2(x.mine)} vs best ${x.bm} ${f2(x.best)}`), "", ...r.warn.map((w) => `- WARNING: ${w}`), "", `Sensitivity: ${JSON.stringify(neigh)} (${stable ? "stable" : "changes"})`];
    const a = h("a", { href: URL.createObjectURL(new Blob([L.join("\n")], { type: "text/markdown" })), download: `advisor_${net.name}.md` }); a.click();
  }
  render();
  return { el: host, update: render };
}

// ========================================================================
// main.js
// ========================================================================
// App shell: rail (network, budget dial, navigation, theme) + routed screens.
const SCREENS = {
  arena: { label: "Arena", title: "The arena", sub: "One network, one budget, six ways to cut it. Every backbone keeps the same number of edges, so differences come from the method.", make: Arena },
  atlas: { label: "Failure atlas", title: "Failure atlas", sub: "Where does a backbone pass the structural check but fail at its job?", make: Atlas },
  curves: { label: "Curves", title: "Preservation curves", sub: "How fidelity changes as edges are removed, against random removal.", make: Curves },
  mechanisms: { label: "Mechanism bench", title: "Mechanism bench", sub: "Why do structural checks mislead? Bridge edges and weight heterogeneity.", make: Mechanisms, global: true },
  lens: { label: "Network lens", title: "Network lens", sub: "Inspect one backbone edge by edge: what was kept, what was lost, and why.", make: Lens },
  theatre: { label: "Spread theatre", title: "Spread theatre", sub: "Replay an epidemic on the original and on a backbone.", make: Theatre },
  conclusions: { label: "Conclusions", title: "Conclusions", sub: "What the evidence says, written from the stored results.", make: Conclusions, global: true },
  advisor: { label: "Advisor", title: "Backbone advisor", sub: "Tell it what you need the backbone for; it recommends a method and shows how sure it is.", make: Advisor },
};
const rail = document.getElementById("rail"), main = document.getElementById("main");
let nets = null, net = null, dial = null, screenObj = null, token = 0;

function applyTheme() {
  const t = state.theme; if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
}
function renderRail() {
  const cur = route();
  dial = BudgetDial(net);
  rail.replaceChildren(
    h("div", { class: "brand" }, h("b", {}, "Backbone Lab"), h("span", { class: "smallcaps" }, "specimen bench")),
    h("div", { class: "field" }, h("label", { class: "smallcaps", for: "netsel" }, "network"),
      h("select", { id: "netsel", onchange: (e) => { set({ net: e.target.value, edge: -1 }); mount(); } },
        nets.networks.map((n) => h("option", { value: n.name, selected: n.name === state.net }, `${n.name} (${n.n})`)))),
    dial.el,
    h("nav", { class: "tabs", "aria-label": "Screens" }, Object.entries(SCREENS).map(([k, s]) =>
      h("a", { href: `#/${k}${location.hash.includes("?") ? "?" + location.hash.split("?")[1] : ""}`, "aria-current": k === cur ? "page" : null }, s.label))),
    h("div", { class: "rail-foot" }, h("button", { class: "btn", onclick: () => { const dark = document.documentElement.dataset.theme === "dark" || (!state.theme && matchMedia("(prefers-color-scheme: dark)").matches); set({ theme: dark ? "light" : "dark" }); applyTheme(); mount(); } }, "◐ Theme"),
      h("span", { class: "faint" }, "←→ budget")));
}

async function mount() {
  const my = ++token, cur = SCREENS[route()] ? route() : "arena", sc = SCREENS[cur];
  screenObj?.destroy?.();
  main.replaceChildren(h("div", { class: "empty" }, "Loading…"));
  try {
    net = await loadNetwork(state.net);
    if (my !== token) return;
    renderRail();
    const obj = await sc.make(net);
    if (my !== token) return;
    screenObj = obj;
    main.replaceChildren(h("header", { class: "page-head" }, h("div", {}, h("h1", {}, sc.title), h("p", {}, sc.sub)),
      h("div", { class: "hint mono" }, sc.global ? "all networks" : `${net.graph.name} · ${net.graph.n} nodes · ${net.graph.m.toLocaleString()} edges`)), obj.el);
  } catch (err) {
    console.error(err);
    main.replaceChildren(h("div", { class: "notice" }, `Could not load data for "${state.net}": ${err.message}. Run the exporter: python -m runner export --network ${state.net}`));
  }
}

async function boot() {
  loadStateFromHash();
  if (!location.hash) location.hash = "#/arena";
  nets = await getJSON("networks.json");
  if (!nets.networks.some((n) => n.name === state.net)) state.net = nets.networks[0].name;
  applyTheme();
  window.addEventListener("hashchange", () => { loadStateFromHash(); applyTheme(); mount(); });
  subscribe((patch) => { if ("q" in patch) { dial?.render(); screenObj?.update?.(); } else if ("a" in patch || "b" in patch || "tier" in patch || "fn" in patch) { /* screens re-render themselves */ } });
  mount();
}
boot().catch((e) => { main.replaceChildren(h("div", { class: "notice" }, `Could not load data/networks.json (${e.message}). Serve the ui folder with: python -m http.server -d ui 8000`)); });
