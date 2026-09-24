"""Интерактивный проигрыватель прогона: один автономный HTML-файл (без сервера и интернета).

    swarmcore replay runs/baseline_static_roundrobin_s1      → runs/.../replay.html

Внутрь встраиваются trajectories.csv, timeseries.csv, events.jsonl, summary.json и world.json
(цели, маршруты секторов, дома; в старых прогонах его нет — тогда без целей и секторов).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .eventlog import load_events

MODES = ["idle", "mission", "rtl", "landed", "failed"]


def _num(v: str):
    if v == "":
        return None
    f = float(v)
    return int(f) if f.is_integer() else f


def build_data(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    cfg = manifest["config"]
    world_path = run_dir / "world.json"
    world = json.loads(world_path.read_text(encoding="utf-8")) if world_path.exists() else None

    times: list[float] = []
    agents: dict[int, dict[str, list]] = {}
    traj = run_dir / "trajectories.csv"
    if not traj.exists():
        raise SystemExit(f"{traj} не найден: включите output.trajectories в сценарии")
    with traj.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = float(row["t"])
            if not times or times[-1] != t:
                times.append(t)
            a = agents.setdefault(int(row["agent"]),
                                  {"x": [], "y": [], "alt": [], "m": [], "task": [], "e": [],
                                   "vx": [], "vy": [], "wp": []})
            a["x"].append(round(float(row["x"]), 2))
            a["y"].append(round(float(row["y"]), 2))
            a["alt"].append(_num(row["alt"]))
            a["m"].append(MODES.index(row["mode"]) if row["mode"] in MODES else 0)
            a["task"].append(_num(row["task"]))
            a["e"].append(round(float(row["energy_frac"]), 4))
            a["vx"].append(round(float(row["vx"]), 3))
            a["vy"].append(round(float(row["vy"]), 3))
            a["wp"].append(_num(row.get("wp", "")))

    ts: dict[str, list] = {}
    with (run_dir / "timeseries.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                ts.setdefault(k, []).append(_num(v))

    return {
        "meta": {"scenario": manifest["scenario"], "allocator": manifest["allocator"],
                 "seed": manifest["seed"], "size": cfg["world"]["size"],
                 "nofly": cfg["world"]["nofly"], "sensor": cfg.get("sensor"),
                 "altitude": cfg["agents"]["altitude"], "motion": cfg["agents"]["motion"],
                 "v_cruise": cfg["agents"]["v_cruise"], "v_max": cfg["agents"]["v_max"],
                 "a_max": cfg["agents"]["a_max"], "yaw_rate_max_deg": cfg["agents"]["yaw_rate_max_deg"],
                 "arrive_radius": cfg["agents"].get("arrive_radius"), "dt": cfg["dt"],
                 "record_every": cfg["output"]["record_every"]},
        "summary": summary,
        "world": world,
        "modes": MODES,
        "times": times,
        "agents": {str(k): v for k, v in sorted(agents.items())},
        "ts": ts,
        "events": load_events(run_dir),
    }


def make_replay(run_dir: str | Path, out: str | Path | None = None) -> Path:
    run_dir = Path(run_dir)
    data = build_data(run_dir)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    title = f"{data['meta']['scenario']} · {data['meta']['allocator']} · seed {data['meta']['seed']}"
    html = _TEMPLATE.replace("__TITLE__", title).replace("__DATA__", payload)
    out = Path(out) if out else run_dir / "replay.html"
    out.write_text(html, encoding="utf-8")
    return out


def replay_command(args) -> int:
    out = make_replay(args.run_dir, args.out)
    print(f"проигрыватель: {out}")
    if args.open:
        import webbrowser
        webbrowser.open(out.resolve().as_uri())
    return 0


def add_replay_args(p) -> None:
    p.add_argument("run_dir")
    p.add_argument("--out", default=None)
    p.add_argument("--open", action="store_true", help="сразу открыть в браузере")


_TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Replay · __TITLE__</title>
<style>
:root {
  --bg: #f6f6f3; --panel: #ffffff; --ink: #1d1d1b; --muted: #6b6b66; --line: #e2e1dc;
  --map: #fbfbf9; --grid: #ecebe6; --accent: #2f6fdf; --warn: #d9822b; --bad: #d64545; --ok: #2e9e5b;
  --route: rgba(0,0,0,.16); --sel: #fff3c4;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #141413; --panel: #1d1d1b; --ink: #ecebe6; --muted: #9a9a93; --line: #33332f;
    --map: #191918; --grid: #26261f; --accent: #6b9cff; --warn: #f0a050; --bad: #ff6b6b; --ok: #4cc47f;
    --route: rgba(255,255,255,.16); --sel: #3a3420;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { display: flex; flex-wrap: wrap; gap: 8px 20px; align-items: baseline; padding: 12px 16px 4px; }
header h1 { font-size: 16px; margin: 0; font-weight: 600; }
header .sub { color: var(--muted); }
main { display: grid; grid-template-columns: minmax(0, 1fr) 400px; gap: 12px; padding: 8px 16px 16px; }
@media (max-width: 980px) { main { grid-template-columns: minmax(0, 1fr); } }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 10px; min-width: 0; }
.card h2 { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em;
  color: var(--muted); margin: 0 0 8px; }
#mapWrap { position: relative; width: 100%; aspect-ratio: 1 / 1; max-height: calc(100vh - 190px); }
#map { width: 100%; height: 100%; display: block; border-radius: 6px; background: var(--map); cursor: crosshair; }
#tip { position: absolute; pointer-events: none; background: var(--panel); border: 1px solid var(--line);
  border-radius: 6px; padding: 4px 8px; font-size: 12px; display: none; white-space: nowrap; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 10px; }
.controls button, .controls select { font: inherit; color: var(--ink); background: var(--bg);
  border: 1px solid var(--line); border-radius: 6px; padding: 4px 10px; cursor: pointer; }
.controls button.primary { background: var(--accent); color: #fff; border-color: var(--accent); min-width: 44px; }
#clock { font-variant-numeric: tabular-nums; font-weight: 600; min-width: 90px; }
#scrub { position: relative; margin-top: 8px; }
#timeline { width: 100%; height: 26px; display: block; cursor: pointer; }
.toggles { display: flex; flex-wrap: wrap; gap: 4px 14px; margin-top: 8px; color: var(--muted); }
.toggles label { cursor: pointer; user-select: none; }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
.stat { background: var(--bg); border-radius: 6px; padding: 6px 8px; }
.stat b { display: block; font-size: 16px; font-variant-numeric: tabular-nums; }
.stat span { color: var(--muted); font-size: 11px; }
side { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
#agents { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
#agents td, #agents th { padding: 3px 4px; text-align: left; border-bottom: 1px solid var(--line); }
#agents th { color: var(--muted); font-weight: 500; font-size: 11px; }
#agents tr { cursor: pointer; }
#agents tr.sel { background: var(--sel); }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; vertical-align: -1px; }
.bar { width: 70px; height: 7px; background: var(--line); border-radius: 4px; overflow: hidden; display: inline-block; }
.bar i { display: block; height: 100%; }
.mode-rtl { color: var(--warn); } .mode-failed { color: var(--bad); font-weight: 600; }
.mode-landed, .mode-idle { color: var(--muted); }
#chart { width: 100%; height: 150px; display: block; cursor: pointer; }
.legend { display: flex; gap: 12px; flex-wrap: wrap; color: var(--muted); font-size: 11px; margin-top: 4px; }
.legend i { display: inline-block; width: 12px; height: 3px; vertical-align: 3px; margin-right: 4px; }
#log { height: 300px; overflow-y: auto; font-variant-numeric: tabular-nums; font-size: 12px; }
#log div { padding: 2px 4px; border-radius: 4px; cursor: pointer; display: flex; gap: 8px; }
#log div:hover { background: var(--bg); }
#log div.future { opacity: .35; }
#log div.now { background: var(--sel); }
#log .t { color: var(--muted); min-width: 52px; }
#logFilters { display: flex; flex-wrap: wrap; gap: 4px 10px; margin-bottom: 6px; font-size: 11px; color: var(--muted); }
.kbd { color: var(--muted); font-size: 11px; }
#algo { font-size: 12px; font-variant-numeric: tabular-nums; }
#algo .f { font-family: ui-monospace, Consolas, monospace; background: var(--bg); border-radius: 6px;
  padding: 6px 8px; margin: 6px 0; white-space: pre-wrap; }
#algo table { border-collapse: collapse; width: 100%; }
#algo td { padding: 1px 4px; border-bottom: 1px solid var(--line); }
#algo .hint { color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1 id="title"></h1><span class="sub" id="subtitle"></span>
</header>
<main>
  <section class="card">
    <div id="mapWrap"><canvas id="map"></canvas><div id="tip"></div></div>
    <div id="scrub"><canvas id="timeline"></canvas></div>
    <div class="controls">
      <button id="play" class="primary" title="Пробел">▶</button>
      <button id="back" title="← (Shift: 10 с)">−1 с</button>
      <button id="fwd" title="→ (Shift: 10 с)">+1 с</button>
      <button id="prevEv" title="P">◀ событие</button>
      <button id="nextEv" title="N">событие ▶</button>
      <select id="speed">
        <option value="1">×1</option><option value="5">×5</option><option value="10">×10</option>
        <option value="20" selected>×20</option><option value="50">×50</option><option value="100">×100</option>
      </select>
      <span id="clock"></span>
      <span class="kbd">пробел — пуск/пауза · ←/→ — шаг · N/P — события · клик по дрону — выбрать</span>
    </div>
    <div class="toggles">
      <label><input type="checkbox" id="optSectors" checked> секторы</label>
      <label><input type="checkbox" id="optRoutes"> маршруты «змейки»</label>
      <label><input type="checkbox" id="optTargets" checked> все цели (истина)</label>
      <label><input type="checkbox" id="optFootprint"> пятно камеры</label>
      <label><input type="checkbox" id="optGoal" checked> цель контроллера</label>
      <label><input type="checkbox" id="optVel" checked> вектор скорости</label>
      <label><input type="checkbox" id="optPlan" checked> план выбранного</label>
      <label>след <select id="optTrail">
        <option value="30">30 с</option><option value="120" selected>2 мин</option>
        <option value="1e9">весь</option><option value="0">нет</option></select></label>
    </div>
  </section>
  <side>
    <section class="card">
      <div class="stats">
        <div class="stat"><b id="sDet">–</b><span>цели найдены</span></div>
        <div class="stat"><b id="sTasks">–</b><span>секторы</span></div>
        <div class="stat"><b id="sAir">–</b><span>в воздухе</span></div>
        <div class="stat"><b id="sWh">–</b><span>энергия, Вт·ч</span></div>
      </div>
    </section>
    <section class="card">
      <h2>Динамика</h2>
      <canvas id="chart"></canvas>
      <div class="legend">
        <span><i style="background:var(--ok)"></i>найдено целей</span>
        <span><i style="background:var(--accent)"></i>ожидаемая доля (Pd)</span>
        <span><i style="background:var(--warn)"></i>секторы выполнены</span>
        <span><i style="background:var(--muted)"></i>в воздухе</span>
      </div>
    </section>
    <section class="card">
      <h2>Алгоритм · шаг за шагом</h2>
      <div id="algo"></div>
    </section>
    <section class="card">
      <h2>БПЛА</h2>
      <table id="agents"><thead><tr><th></th><th>режим</th><th>сектор</th><th>очередь</th><th>заряд</th></tr></thead><tbody></tbody></table>
    </section>
    <section class="card">
      <h2>Журнал событий <span id="logCount" style="text-transform:none;font-weight:400"></span></h2>
      <div id="logFilters"></div>
      <div id="log"></div>
    </section>
  </side>
</main>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const T = D.times, N = T.length, T0 = T[0], T1 = T[N - 1];
const ids = Object.keys(D.agents).map(Number);
const W = D.meta.size[0], H = D.meta.size[1];
const world = D.world || {tasks: [], targets: [], homes: {}, nofly: D.meta.nofly};
const color = i => `hsl(${(i * 137.508) % 360} 70% 50%)`;
const colorA = (i, a) => `hsl(${(i * 137.508) % 360} 70% 50% / ${a})`;
const MODE_RU = {idle: "на земле", mission: "миссия", rtl: "домой", landed: "сел", failed: "ОТКАЗ"};

let t = T0, playing = false, speed = 20, selected = null;

$("title").textContent = `${D.meta.scenario} · ${D.meta.allocator} · seed ${D.meta.seed}`;
const S = D.summary;
$("subtitle").textContent = `${ids.length} БПЛА · ${world.targets.length || S.targets} целей · ${world.tasks.length || S.tasks_total} секторов · ` +
  `T50=${S.T_detect_50 ?? "—"} с · T90=${S.T_detect_90 ?? "—"} с · CR=${S.CR} · ${S.valid ? "valid" : "INVALID"}` +
  (D.world ? "" : " · нет world.json — перезапустите прогон, чтобы видеть цели и секторы");

// ---------- индексы по времени
function idxAt(tt) {            // последний индекс с T[i] <= tt
  let lo = 0, hi = N - 1;
  if (tt <= T[0]) return 0;
  if (tt >= T[hi]) return hi;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (T[m] <= tt) lo = m; else hi = m; }
  return lo;
}
function posAt(a, tt) {
  const A = D.agents[a], i = idxAt(tt), j = Math.min(i + 1, N - 1);
  const f = j > i ? Math.max(0, Math.min(1, (tt - T[i]) / (T[j] - T[i]))) : 0;
  return {x: A.x[i] + (A.x[j] - A.x[i]) * f, y: A.y[i] + (A.y[j] - A.y[i]) * f,
          hx: A.x[j] - A.x[i], hy: A.y[j] - A.y[i], i};
}

// ---------- задачи: bbox сектора, статус во времени
const tasks = world.tasks.map(tk => {
  const xs = tk.waypoints.map(p => p[0]), ys = tk.waypoints.map(p => p[1]);
  const uy = [...new Set(ys)].sort((a, b) => a - b);
  const pad = uy.length > 1 ? (uy[1] - uy[0]) / 2 : 10;
  return {...tk, x0: Math.min(...xs) - pad, x1: Math.max(...xs) + pad, y0: Math.min(...ys) - pad, y1: Math.max(...ys) + pad};
});
const taskById = Object.fromEntries(tasks.map(tk => [tk.id, tk]));
const EV = D.events;
function taskState(tt) {        // id -> {owner, status: assigned|active|done|free}
  const st = {};
  for (const e of EV) {
    if (e.t > tt) break;
    if (e.type === "allocation") {
      for (const [a, tids] of Object.entries(e.assignment))
        for (const id of tids) if (!st[id] || st[id].status !== "done") st[id] = {owner: +a, status: "assigned"};
    } else if (e.type === "plan") {
      for (const id of e.tasks) if (!st[id] || st[id].status !== "done") st[id] = {owner: e.agent, status: "assigned"};
    } else if (e.type === "task_started") st[e.task] = {owner: e.agent, status: "active"};
    else if (e.type === "task_done") st[e.task] = {owner: e.agent, status: "done"};
    else if (e.type === "task_released") st[e.task] = {owner: null, status: "free"};
  }
  return st;
}
function queueAt(a, tt) {       // последняя очередь задач агента
  let q = [];
  for (const e of EV) {
    if (e.t > tt) break;
    if (e.type === "plan" && e.agent === a) q = e.tasks.slice();
    if (e.type === "task_done" && e.agent === a) q = q.filter(x => x !== e.task);
    if ((e.type === "task_released") && e.agent === a) q = q.filter(x => x !== e.task);
  }
  return q;
}

// ---------- карта
const cv = $("map"), ctx = cv.getContext("2d");
let scale = 1, ox = 0, oy = 0, cw = 0, ch = 0;
function resize() {
  const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  cv.width = r.width * dpr; cv.height = r.height * dpr; cw = r.width; ch = r.height;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const m = 14;
  scale = Math.min((cw - 2 * m) / W, (ch - 2 * m) / H);
  ox = (cw - W * scale) / 2; oy = (ch - H * scale) / 2;
  const tl = $("timeline"), tr = tl.getBoundingClientRect();
  tl.width = tr.width * dpr; tl.height = tr.height * dpr; tl.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
  const c2 = $("chart"), r2 = c2.getBoundingClientRect();
  c2.width = r2.width * dpr; c2.height = r2.height * dpr; c2.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
  drawAll();
}
const sx = x => ox + x * scale, sy = y => oy + (H - y) * scale;

function drawMap() {
  const st = taskState(t);
  ctx.clearRect(0, 0, cw, ch);
  // сетка
  ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
  for (let g = 0; g <= W; g += 100) { ctx.beginPath(); ctx.moveTo(sx(g), sy(0)); ctx.lineTo(sx(g), sy(H)); ctx.stroke(); }
  for (let g = 0; g <= H; g += 100) { ctx.beginPath(); ctx.moveTo(sx(0), sy(g)); ctx.lineTo(sx(W), sy(g)); ctx.stroke(); }
  ctx.strokeStyle = css("--line"); ctx.strokeRect(sx(0), sy(H), W * scale, H * scale);
  // бесполётные зоны
  for (const poly of world.nofly || []) {
    ctx.beginPath(); poly.forEach((p, k) => k ? ctx.lineTo(sx(p[0]), sy(p[1])) : ctx.moveTo(sx(p[0]), sy(p[1])));
    ctx.closePath(); ctx.fillStyle = "rgba(214,69,69,.15)"; ctx.fill(); ctx.strokeStyle = css("--bad"); ctx.stroke();
  }
  // секторы
  if ($("optSectors").checked) for (const tk of tasks) {
    const s = st[tk.id], x = sx(tk.x0), y = sy(tk.y1), w = (tk.x1 - tk.x0) * scale, h = (tk.y1 - tk.y0) * scale;
    if (s && s.owner != null) {
      const dim = selected != null && s.owner !== selected;
      ctx.fillStyle = colorA(s.owner, (s.status === "done" ? .22 : s.status === "active" ? .12 : .04) * (dim ? .35 : 1));
      ctx.fillRect(x, y, w, h);
      ctx.setLineDash(s.status === "active" ? [5, 3] : []);
      ctx.strokeStyle = colorA(s.owner, s.status === "assigned" ? .25 : .7);
      ctx.lineWidth = s.status === "active" ? 1.5 : 1; ctx.strokeRect(x + .5, y + .5, w - 1, h - 1); ctx.setLineDash([]);
    } else { ctx.strokeStyle = css("--line"); ctx.lineWidth = 1; ctx.strokeRect(x + .5, y + .5, w - 1, h - 1); }
    ctx.fillStyle = css("--muted"); ctx.font = "10px system-ui";
    ctx.fillText(`#${tk.id}` + (s && s.status === "done" ? " ✓" : ""), x + 4, y + 12);
  }
  if ($("optRoutes").checked) {
    ctx.strokeStyle = css("--route"); ctx.lineWidth = 1;
    for (const tk of tasks) { ctx.beginPath(); tk.waypoints.forEach((p, k) => k ? ctx.lineTo(sx(p[0]), sy(p[1])) : ctx.moveTo(sx(p[0]), sy(p[1]))); ctx.stroke(); }
  }
  // дома
  for (const [a, h] of Object.entries(world.homes || {})) { ctx.fillStyle = colorA(+a, .5); ctx.fillRect(sx(h[0]) - 3, sy(h[1]) - 3, 6, 6); }
  // следы
  const trail = +$("optTrail").value, ti = idxAt(t);
  if (trail > 0) for (const a of ids) {
    const A = D.agents[a], k0 = idxAt(t - (a === selected ? 1e9 : trail));
    ctx.strokeStyle = colorA(a, selected == null || a === selected ? .85 : .2);
    ctx.lineWidth = a === selected ? 2.2 : 1.4; ctx.beginPath();
    for (let k = k0; k <= ti; k++) k === k0 ? ctx.moveTo(sx(A.x[k]), sy(A.y[k])) : ctx.lineTo(sx(A.x[k]), sy(A.y[k]));
    const p = posAt(a, t); ctx.lineTo(sx(p.x), sy(p.y)); ctx.stroke();
  }
  // план выбранного: остаток текущей «змейки» → следующие секторы очереди → дом
  if ($("optPlan").checked && selected != null) {
    const A = D.agents[selected], i = idxAt(t), p = posAt(selected, t), mode = D.modes[A.m[i]];
    const pts = [[p.x, p.y]];
    if (mode === "mission") {
      const cur = taskById[A.task[i]];
      if (cur && A.wp[i] != null) pts.push(...cur.waypoints.slice(A.wp[i]));
      for (const q of queueAt(selected, t)) if (q !== A.task[i] && taskById[q]) pts.push(...taskById[q].waypoints);
    }
    if (mode === "mission" || mode === "rtl") { const h = world.homes[selected]; if (h) pts.push(h); }
    ctx.strokeStyle = colorA(selected, .55); ctx.lineWidth = 1.2; ctx.setLineDash([2, 3]); ctx.beginPath();
    pts.forEach((q, k) => k ? ctx.lineTo(sx(q[0]), sy(q[1])) : ctx.moveTo(sx(q[0]), sy(q[1]))); ctx.stroke(); ctx.setLineDash([]);
  }
  // цель контроллера (goto target) и вектор скорости
  for (const a of ids) {
    const A = D.agents[a], i = idxAt(t), p = posAt(a, t), g = goalOf(a, i);
    if (selected != null && a !== selected) continue;
    if (g && $("optGoal").checked) {
      ctx.strokeStyle = colorA(a, .9); ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(sx(p.x), sy(p.y)); ctx.lineTo(sx(g[0]), sy(g[1])); ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(sx(g[0]), sy(g[1]), Math.max(3, (D.meta.arrive_radius || 2) * scale), 0, 7); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(sx(g[0]) - 4, sy(g[1])); ctx.lineTo(sx(g[0]) + 4, sy(g[1]));
      ctx.moveTo(sx(g[0]), sy(g[1]) - 4); ctx.lineTo(sx(g[0]), sy(g[1]) + 4); ctx.stroke();
    }
    if ($("optVel").checked && (A.vx[i] || A.vy[i])) {        // стрелка = путь за 5 с при текущей v
      const ex = sx(p.x + A.vx[i] * 5), ey = sy(p.y + A.vy[i] * 5), ang = Math.atan2(ey - sy(p.y), ex - sx(p.x));
      ctx.strokeStyle = css("--ink"); ctx.fillStyle = css("--ink"); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(sx(p.x), sy(p.y)); ctx.lineTo(ex, ey); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(ex - 7 * Math.cos(ang - .4), ey - 7 * Math.sin(ang - .4));
      ctx.lineTo(ex - 7 * Math.cos(ang + .4), ey - 7 * Math.sin(ang + .4)); ctx.closePath(); ctx.fill();
    }
  }
  // цели
  for (const tg of world.targets) {
    if (tg.t_appear > t) continue;
    const found = tg.t_detected != null && tg.t_detected <= t, x = sx(tg.x), y = sy(tg.y);
    if (!found && !$("optTargets").checked) continue;
    const det = found ? detector[tg.id] : null;
    ctx.beginPath(); ctx.moveTo(x, y - 5); ctx.lineTo(x + 5, y); ctx.lineTo(x, y + 5); ctx.lineTo(x - 5, y); ctx.closePath();
    if (found) { ctx.fillStyle = det != null ? color(det) : css("--ok"); ctx.fill(); ctx.strokeStyle = css("--ink"); ctx.lineWidth = .8; ctx.stroke(); }
    else { ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1; ctx.stroke(); }
    if (found && t - tg.t_detected < 5) {
      const k = (t - tg.t_detected) / 5;
      ctx.beginPath(); ctx.arc(x, y, 6 + 24 * k, 0, 7); ctx.strokeStyle = `rgba(46,158,91,${1 - k})`; ctx.lineWidth = 2; ctx.stroke();
    }
  }
  // дроны
  const foot = $("optFootprint").checked && D.meta.sensor;
  for (const a of ids) {
    const A = D.agents[a], p = posAt(a, t), mode = D.modes[A.m[p.i]], x = sx(p.x), y = sy(p.y);
    if (foot && (mode === "mission" || mode === "rtl")) {
      const half = (A.alt[p.i] || D.meta.altitude) * Math.tan(D.meta.sensor.fov_deg * Math.PI / 360);
      ctx.fillStyle = colorA(a, .1); ctx.fillRect(sx(p.x - half), sy(p.y + half), 2 * half * scale, 2 * half * scale);
    }
    const r = a === selected ? 7 : 5.5;
    ctx.globalAlpha = selected == null || a === selected ? 1 : .45;
    if (mode === "failed") {
      ctx.strokeStyle = css("--bad"); ctx.lineWidth = 2.5; ctx.beginPath();
      ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r); ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r); ctx.stroke();
    } else {
      const ang = Math.atan2(-p.hy, p.hx), moving = Math.hypot(p.hx, p.hy) > .1;
      ctx.fillStyle = mode === "landed" || mode === "idle" ? css("--muted") : color(a);
      ctx.beginPath();
      if (moving) { ctx.moveTo(x + Math.cos(ang) * r * 1.6, y + Math.sin(ang) * r * 1.6);
        ctx.lineTo(x + Math.cos(ang + 2.5) * r, y + Math.sin(ang + 2.5) * r);
        ctx.lineTo(x + Math.cos(ang - 2.5) * r, y + Math.sin(ang - 2.5) * r); ctx.closePath(); }
      else ctx.arc(x, y, r, 0, 7);
      ctx.fill();
      ctx.strokeStyle = mode === "rtl" ? css("--warn") : css("--panel"); ctx.lineWidth = mode === "rtl" ? 2.5 : 1.2; ctx.stroke();
    }
    ctx.fillStyle = css("--ink"); ctx.font = "600 10px system-ui"; ctx.fillText(a, x + 8, y - 6);
    ctx.globalAlpha = 1;
  }
}
const detector = {};
for (const e of EV) if (e.type === "target_detected" && !(e.target in detector)) detector[e.target] = e.agent;

// ---------- шкала времени с отметками событий
const EV_COLOR = {target_detected: "--ok", task_done: "--warn", agent_failed: "--bad", collision: "--bad",
  low_battery_rtl: "--bad", allocation: "--accent", task_released: "--bad", near_miss: "--bad"};
function drawTimeline() {
  const c = $("timeline"), g = c.getContext("2d"), w = c.getBoundingClientRect().width, h = 26;
  g.clearRect(0, 0, w, h);
  g.fillStyle = css("--line"); g.fillRect(0, 11, w, 4);
  const X = tt => (tt - T0) / (T1 - T0 || 1) * w;
  for (const e of EV) {
    const c0 = EV_COLOR[e.type]; if (!c0) continue;
    g.fillStyle = css(c0); const big = c0 === "--bad" || e.type === "allocation";
    g.fillRect(X(e.t) - .5, big ? 2 : 6, big ? 2 : 1, big ? 22 : 14);
  }
  g.fillStyle = css("--accent"); g.fillRect(0, 11, X(t), 4);
  g.beginPath(); g.arc(X(t), 13, 6, 0, 7); g.fill();
}
function seekFromEvent(ev, el) {
  const r = el.getBoundingClientRect(); setT(T0 + (ev.clientX - r.left) / r.width * (T1 - T0));
}
let dragging = false;
$("timeline").addEventListener("pointerdown", e => { dragging = true; $("timeline").setPointerCapture(e.pointerId); seekFromEvent(e, $("timeline")); });
$("timeline").addEventListener("pointermove", e => dragging && seekFromEvent(e, $("timeline")));
$("timeline").addEventListener("pointerup", () => dragging = false);

// ---------- график
function drawChart() {
  const c = $("chart"), g = c.getContext("2d"), r = c.getBoundingClientRect(), w = r.width, h = r.height;
  const L = 30, R = 6, Tp = 6, B = 18, pw = w - L - R, ph = h - Tp - B;
  g.clearRect(0, 0, w, h);
  const X = tt => L + (tt - T0) / (T1 - T0 || 1) * pw, Y = v => Tp + (1 - v) * ph;
  g.strokeStyle = css("--grid"); g.fillStyle = css("--muted"); g.font = "10px system-ui"; g.lineWidth = 1;
  for (const v of [0, .5, 1]) { g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(w - R, Y(v)); g.stroke(); g.fillText(`${v * 100}%`, 2, Y(v) + 3); }
  for (let k = 0; k <= 4; k++) { const tt = T0 + (T1 - T0) * k / 4; g.fillText(`${Math.round(tt)}с`, X(tt) - 10, h - 4); }
  const ts = D.ts, nt = D.summary.tasks_total || tasks.length || 1, na = ids.length;
  const line = (vals, col, f = v => v, dash = []) => {
    g.strokeStyle = css(col); g.lineWidth = 1.6; g.setLineDash(dash); g.beginPath();
    ts.t.forEach((tt, k) => k ? g.lineTo(X(tt), Y(f(vals[k]))) : g.moveTo(X(tt), Y(f(vals[k]))));
    g.stroke(); g.setLineDash([]);
  };
  line(ts.airborne, "--muted", v => v / na, [3, 3]);
  line(ts.tasks_done, "--warn", v => v / nt);
  line(ts.expected_detect_frac, "--accent");
  line(ts.detected_frac, "--ok");
  for (const lv of [.5, .9]) { g.strokeStyle = css("--line"); g.setLineDash([2, 4]); g.beginPath(); g.moveTo(L, Y(lv)); g.lineTo(w - R, Y(lv)); g.stroke(); g.setLineDash([]); }
  g.strokeStyle = css("--ink"); g.lineWidth = 1; g.beginPath(); g.moveTo(X(t), Tp); g.lineTo(X(t), Tp + ph); g.stroke();
}
$("chart").addEventListener("click", e => {
  const r = $("chart").getBoundingClientRect(), L = 30, pw = r.width - 36;
  setT(T0 + (e.clientX - r.left - L) / pw * (T1 - T0));
});

// ---------- панель агентов и статистика
const tbody = $("agents").querySelector("tbody");
for (const a of ids) {
  const tr = document.createElement("tr"); tr.dataset.a = a;
  tr.innerHTML = `<td><span class="dot" style="background:${color(a)}"></span> ${a}</td><td></td><td></td><td></td><td><span class="bar"><i></i></span> <span></span></td>`;
  tr.onclick = () => select(selected === a ? null : a);
  tbody.appendChild(tr);
}
function drawPanel() {
  const i = idxAt(t), st = taskState(t);
  for (const tr of tbody.children) {
    const a = +tr.dataset.a, A = D.agents[a], mode = D.modes[A.m[i]], td = tr.children;
    td[1].textContent = MODE_RU[mode] || mode; td[1].className = "mode-" + mode;
    td[2].textContent = A.task[i] ?? "—";
    td[3].textContent = queueAt(a, t).filter(x => x !== A.task[i]).join(", ") || "—";
    const e = A.e[i], bar = td[4].querySelector("i");
    bar.style.width = `${e * 100}%`; bar.style.background = e < .3 ? css("--bad") : e < .5 ? css("--warn") : css("--ok");
    td[4].querySelector("span:last-child").textContent = `${Math.round(e * 100)}%`;
    tr.classList.toggle("sel", a === selected);
  }
  const k = Math.min(idxAt(t), D.ts.t.length - 1), nt = world.tasks.length || D.summary.tasks_total;
  const nTg = world.targets.filter(x => x.t_appear <= t).length || D.summary.targets;
  const nFound = world.targets.length ? world.targets.filter(x => x.t_detected != null && x.t_detected <= t).length
                                      : Math.round(D.ts.detected_frac[k] * nTg);
  $("sDet").textContent = `${nFound}/${nTg}`;
  $("sTasks").textContent = `${Object.values(st).filter(s => s.status === "done").length}/${nt}`;
  $("sAir").textContent = `${D.ts.airborne[k]}/${ids.length}`;
  $("sWh").textContent = (+D.ts.energy_used_wh[k]).toFixed(1);
  const m = Math.floor(t / 60), s = t - m * 60;
  $("clock").textContent = `${m}:${s.toFixed(1).padStart(4, "0")} / ${Math.floor(T1 / 60)}:${String(Math.round(T1 % 60)).padStart(2, "0")}`;
}

// ---------- разбор алгоритма
function goalOf(a, i) {          // точка, которую агент передаёт в MotionCommand.goto
  const A = D.agents[a], mode = D.modes[A.m[i]];
  if (mode === "mission" && A.task[i] != null && A.wp[i] != null && taskById[A.task[i]])
    return taskById[A.task[i]].waypoints[A.wp[i]];
  if (mode === "rtl" || (mode === "mission" && A.task[i] == null)) return world.homes[a] || null;
  return null;
}
const f1 = v => (+v).toFixed(1), deg = r => (r * 180 / Math.PI).toFixed(0);
function drawAlgo() {
  const box = $("algo");
  if (selected == null) {
    let al = null; for (const e of EV) { if (e.t > t) break; if (e.type === "allocation") al = e; }
    if (!al) { box.innerHTML = `<span class="hint">распределение ещё не было</span>`; return; }
    const agents = Object.keys(al.assignment).map(Number).sort((x, y) => x - y), n = agents.length;
    const order = []; for (const [a, v] of Object.entries(al.assignment)) for (const id of v) order.push([id, +a]);
    order.sort((x, y) => x[0] - y[0]);
    const rr = al.allocator === "static_roundrobin";
    box.innerHTML = `<div>Распределение <b>${al.allocator}</b> в t=${f1(al.t)} с (${al.reason}), исправных БПЛА N=${n}, задач ${order.length}.</div>` +
      (rr ? `<div class="f">for k, task in enumerate(невыполненные):\n    plan[agents[k mod N]].append(task)</div>` : "") +
      `<table>` + order.map(([id, a], k) => `<tr><td>k=${k}</td><td>сектор #${id}</td><td>` +
        (rr ? `${k} mod ${n} = ${k % n} → ` : "→ ") + `<span class="dot" style="background:${color(a)}"></span> БПЛА ${a}</td></tr>`).join("") +
      `</table><div class="hint" style="margin-top:6px">Кликните дрон на карте или в таблице, чтобы разобрать его закон управления.</div>`;
    return;
  }
  const a = selected, A = D.agents[a], i = idxAt(t), mode = D.modes[A.m[i]], g = goalOf(a, i);
  const px = A.x[i], py = A.y[i], v = Math.hypot(A.vx[i], A.vy[i]), m = D.meta, dtr = m.record_every;
  let html = `<div><span class="dot" style="background:${color(a)}"></span> <b>БПЛА ${a}</b> · запись t=${f1(T[i])} с (шаг лога ${dtr} с, шаг сим. ${m.dt} с)</div>`;
  html += `<div>1) Режим: <b>${mode.toUpperCase()}</b>` + (mode === "mission" ? `, сектор #${A.task[i] ?? "—"}` : "") + `</div>`;
  if (g) {
    const dx = g[0] - px, dy = g[1] - py, d = Math.hypot(dx, dy), tk = taskById[A.task[i]];
    html += `<div>2) Цель goto: ` + (mode === "mission" && tk ? `точка ${A.wp[i] + 1}/${tk.waypoints.length} «змейки»` : "дом") +
      ` = (${f1(g[0])}, ${f1(g[1])})</div>`;
    html += `<div>3) Ошибка: wp − p = (${f1(dx)}, ${f1(dy)}), |d| = ${f1(d)} м, курс на цель ${deg(Math.atan2(dy, dx))}°</div>`;
    if (m.motion === "kinematic") {
      const sp = Math.min(m.v_cruise, m.v_max, d / m.dt);
      html += `<div class="f">v = min(v_cruise, v_max, |d|/dt) · d/|d|\n  = min(${m.v_cruise}, ${m.v_max}, ${f1(d / m.dt)}) · (${(dx / (d || 1)).toFixed(2)}, ${(dy / (d || 1)).toFixed(2)})\n  = (${f1(sp * dx / (d || 1))}, ${f1(sp * dy / (d || 1))}) м/с\np ← p + v·dt</div>`;
    } else {
      html += `<div class="f">v_des = v_cruise · d/|d|  (${m.v_cruise} м/с)\nΔv ограничено a_max = ${m.a_max} м/с²,\nповорот ≤ ψ̇_max = ${m.yaw_rate_max_deg}°/с</div>`;
    }
    html += `<div>4) Факт в логе: v = (${f1(A.vx[i])}, ${f1(A.vy[i])}), |v| = ${f1(v)} м/с, курс ${deg(Math.atan2(A.vy[i], A.vx[i]))}°</div>`;
    if (i > 0) {
      const h0 = Math.atan2(A.vy[i - 1], A.vx[i - 1]), h1 = Math.atan2(A.vy[i], A.vx[i]);
      let dpsi = h1 - h0; dpsi = (dpsi + 3 * Math.PI) % (2 * Math.PI) - Math.PI;
      const acc = Math.hypot(A.vx[i] - A.vx[i - 1], A.vy[i] - A.vy[i - 1]) / (T[i] - T[i - 1] || 1);
      html += `<div>5) За шаг лога: Δψ = ${deg(dpsi)}° (${f1(Math.abs(dpsi) * 180 / Math.PI / (T[i] - T[i - 1] || 1))}°/с), |Δv|/Δt = ${f1(acc)} м/с²` +
        (m.motion === "kinematic" ? ` <span class="hint">— кинематика не ограничивает ускорение и поворот, отсюда резкие развороты</span>` : "") + `</div>`;
    }
    html += `<div>6) Переключение точки, когда |d| ≤ arrive_radius = ${m.arrive_radius} м; ETA ≈ ${v > .1 ? f1(d / v) : "—"} с</div>`;
  } else html += `<div class="hint">агент не движется (${MODE_RU[mode]})</div>`;
  html += `<div>Очередь: [${queueAt(a, t).join(", ") || "—"}] · заряд ${Math.round(A.e[i] * 100)}%</div>`;
  box.innerHTML = html;
}

// ---------- журнал событий
const L = (a) => `БПЛА ${a}`, lst = xs => xs && xs.length ? xs.join(", ") : "—";
function describe(e) {
  switch (e.type) {
    case "allocation": return `распределение (${e.reason}, ${e.allocator}): ` + Object.entries(e.assignment).map(([a, v]) => `${a}→[${v}]`).join(" ");
    case "plan": return `${L(e.agent)}: план [${lst(e.tasks)}]`;
    case "takeoff": return `${L(e.agent)}: взлёт`;
    case "task_started": return `${L(e.agent)}: начал сектор #${e.task}`;
    case "task_done": return `${L(e.agent)}: закончил сектор #${e.task}`;
    case "task_released": return `${L(e.agent)}: освободил сектор #${e.task} (${e.reason})`;
    case "target_detected": return `${L(e.agent)}: обнаружил цель ${e.target}`;
    case "new_targets": return `появились цели [${lst(e.targets)}]`;
    case "plan_complete_rtl": return `${L(e.agent)}: план выполнен, домой`;
    case "low_battery_rtl": return `${L(e.agent)}: низкий заряд → домой, освобождены [${lst(e.released)}]`;
    case "battery_depleted": return `${L(e.agent)}: батарея разряжена`;
    case "agent_failed": return `${L(e.agent)}: ОТКАЗ (${e.reason}), освобождены [${lst(e.released)}]`;
    case "landed": return `${L(e.agent)}: посадка, остаток ${(e.energy_j / 3600).toFixed(1)} Вт·ч`;
    case "collision": return `СТОЛКНОВЕНИЕ ${e.agents}`;
    case "near_miss": return `опасное сближение ${e.agents}`;
    case "geofence_violation": return `${L(e.agent)}: нарушение геозоны`;
  }
  const {t: _t, type: _k, ...rest} = e; return `${e.type} ${JSON.stringify(rest)}`;
}
const types = [...new Set(EV.map(e => e.type))];
const hidden = new Set(["plan", "takeoff"]);
for (const k of types) {
  const lb = document.createElement("label");
  lb.innerHTML = `<input type="checkbox" ${hidden.has(k) ? "" : "checked"}> ${k} (${EV.filter(e => e.type === k).length})`;
  lb.querySelector("input").onchange = ev => { ev.target.checked ? hidden.delete(k) : hidden.add(k); buildLog(); };
  $("logFilters").appendChild(lb);
}
const involves = (e, a) => e.agent === a || (e.agents || []).includes(a) || (e.assignment && String(a) in e.assignment);
let logRows = [];
function buildLog() {
  const box = $("log"); box.innerHTML = ""; logRows = [];
  EV.forEach(e => {
    if (hidden.has(e.type) || (selected != null && !involves(e, selected))) return;
    const d = document.createElement("div");
    const m = Math.floor(e.t / 60), s = e.t - m * 60;
    d.innerHTML = `<span class="t">${m}:${s.toFixed(1).padStart(4, "0")}</span><span></span>`;
    d.lastChild.textContent = describe(e);
    if (e.agent != null) d.lastChild.style.borderLeft = `3px solid ${color(e.agent)}`, d.lastChild.style.paddingLeft = "6px";
    d.onclick = () => { setT(e.t + 1e-6); if (e.agent != null) select(e.agent); };
    box.appendChild(d); logRows.push([e.t, d]);
  });
  $("logCount").textContent = `· ${logRows.length}` + (selected != null ? ` (БПЛА ${selected})` : "");
  lastNow = null; drawLog();
}
let lastNow = null;
function drawLog() {
  let now = null;
  for (const [et, d] of logRows) { const fut = et > t + 1e-9; d.classList.toggle("future", fut); d.classList.remove("now"); if (!fut) now = d; }
  if (now) { now.classList.add("now"); if (now !== lastNow) { const box = $("log"); box.scrollTop = now.offsetTop - box.offsetTop - box.clientHeight / 2; } }
  lastNow = now;
}

// ---------- управление
function drawAll() { drawMap(); drawTimeline(); drawChart(); drawPanel(); drawLog(); drawAlgo(); }
function setT(v) { t = Math.max(T0, Math.min(T1, v)); drawAll(); }
function select(a) { selected = a; buildLog(); drawAll(); }
function setPlaying(p) { playing = p; $("play").textContent = p ? "❚❚" : "▶"; if (p && t >= T1) t = T0; last = performance.now(); }
let last = performance.now();
function frame(now) {
  if (playing) { t += (now - last) / 1000 * speed; if (t >= T1) { t = T1; setPlaying(false); } drawAll(); }
  last = now; requestAnimationFrame(frame);
}
const jumpEv = dir => {
  const vis = logRows.map(r => r[0]);
  const nxt = dir > 0 ? vis.find(x => x > t + 1e-6) : [...vis].reverse().find(x => x < t - 1e-6);
  if (nxt != null) setT(nxt + 1e-6);
};
$("play").onclick = () => setPlaying(!playing);
$("back").onclick = () => setT(t - 1); $("fwd").onclick = () => setT(t + 1);
$("prevEv").onclick = () => jumpEv(-1); $("nextEv").onclick = () => jumpEv(1);
$("speed").onchange = e => speed = +e.target.value;
for (const id of ["optSectors", "optRoutes", "optTargets", "optFootprint", "optTrail", "optGoal", "optVel", "optPlan"]) $(id).onchange = drawAll;
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT") return;
  if (e.code === "Space") { e.preventDefault(); setPlaying(!playing); }
  else if (e.key === "ArrowRight") setT(t + (e.shiftKey ? 10 : 1));
  else if (e.key === "ArrowLeft") setT(t - (e.shiftKey ? 10 : 1));
  else if (e.key === "n" || e.key === "т") jumpEv(1);
  else if (e.key === "p" || e.key === "з") jumpEv(-1);
  else if (e.key === "Escape") select(null);
});
// клик и подсказка на карте
const toWorld = e => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left - ox) / scale, H - (e.clientY - r.top - oy) / scale, e.clientX - r.left, e.clientY - r.top]; };
function nearestAgent(wx, wy) {
  let best = null, bd = 12 / scale;
  for (const a of ids) { const p = posAt(a, t), d = Math.hypot(p.x - wx, p.y - wy); if (d < bd) { bd = d; best = a; } }
  return best;
}
cv.addEventListener("click", e => { const [wx, wy] = toWorld(e); select(nearestAgent(wx, wy)); });
cv.addEventListener("mousemove", e => {
  const [wx, wy, px, py] = toWorld(e), tip = $("tip"), a = nearestAgent(wx, wy);
  let html = null;
  if (a != null) {
    const A = D.agents[a], i = idxAt(t), p = posAt(a, t);
    html = `<b>БПЛА ${a}</b> · ${MODE_RU[D.modes[A.m[i]]]} · сектор ${A.task[i] ?? "—"}<br>(${p.x.toFixed(0)}, ${p.y.toFixed(0)}) м · h ${A.alt[i]} м · заряд ${Math.round(A.e[i] * 100)}%`;
  } else {
    const tg = world.targets.find(g => g.t_appear <= t && Math.hypot(g.x - wx, g.y - wy) < 8 / scale);
    if (tg) html = `<b>цель ${tg.id}</b> · ` + (tg.t_detected == null ? "не найдена за прогон" :
      tg.t_detected <= t ? `найдена БПЛА ${detector[tg.id]} в ${tg.t_detected.toFixed(1)} с` : `будет найдена в ${tg.t_detected.toFixed(1)} с`);
    else if (wx >= 0 && wx <= W && wy >= 0 && wy <= H) {
      const tk = tasks.find(k => wx >= k.x0 && wx <= k.x1 && wy >= k.y0 && wy <= k.y1);
      html = `(${wx.toFixed(0)}, ${wy.toFixed(0)}) м` + (tk ? ` · сектор #${tk.id}` + (tk.t_done != null ? `, готов в ${tk.t_done.toFixed(0)} с` : "") : "");
    }
  }
  if (html) { tip.innerHTML = html; tip.style.display = "block"; tip.style.left = Math.min(px + 14, cw - tip.offsetWidth - 4) + "px"; tip.style.top = (py + 14) + "px"; }
  else tip.style.display = "none";
});
cv.addEventListener("mouseleave", () => $("tip").style.display = "none");
window.addEventListener("resize", resize);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawAll);
buildLog(); resize(); requestAnimationFrame(frame);
</script>
</body>
</html>
"""
