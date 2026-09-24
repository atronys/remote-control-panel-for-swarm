"""Локальная страница «пульт + проигрыватель»: новые прогоны прямо из браузера.

    swarmcore serve              # откроет http://127.0.0.1:8765 в браузере
    swarmcore serve --port 9000 --no-browser

Слева — настройки (сценарий, распределитель, число дронов, отказы, связь, ветер, ORCA, зерно)
и кнопка «Новый прогон»; справа — интерактивный проигрыватель (apps/replay.py) результата.
Прогон идёт в фоне на этом же компьютере, результаты — в runs/web/<время>_<распределитель>_s<зерно>/.
Сервер слушает только 127.0.0.1 — снаружи недоступен. Только стандартная библиотека Python.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
WEB = RUNS / "web"
SCENARIOS = ["demo", "baseline", "failures", "nofly"]
ALLOCATORS = {
    "hybrid": "B5 гибрид (станция + CBBA без станции)",
    "cbba": "B4 CBBA (без центра)",
    "central_greedy": "B2 центр, жадный аукцион",
    "central_optimal": "B3 центр, оптимизатор",
    "static_oracle": "B0 оракул (знает цели, без перераспределения)",
    "static_roundrobin": "B1 по кругу (без перераспределения)",
}
SUMMARY_KEYS = ["CR", "U", "makespan", "T_detect_90", "E_total_Wh", "agents_failed", "T_recovery_max",
                "N_dup", "N_lost", "N_col", "N_nm", "N_conf", "B_per_agent", "A_gcs", "valid"]

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


# ------------------------------------------------------------------ прогон
def build_overrides(p: dict) -> tuple[str, dict, int]:
    """Параметры формы → (сценарий, переопределения, зерно)."""
    scen = p.get("scenario") if p.get("scenario") in SCENARIOS else "demo"
    o: dict = {"output.trajectories": True}
    if p.get("allocator") in ALLOCATORS:
        o["allocator"] = {"kind": p["allocator"]}
    n = int(p.get("agents") or 10)
    o["agents.count"] = max(1, min(n, 50))
    rate = float(p.get("fail_rate") or 0)
    if rate > 0:
        o.update({"faults.kind": "random", "faults.rate": min(rate, 0.5), "faults.t_range": [60, 450],
                  "faults.detect": "silent" if p.get("silent", True) else "reported"})
    else:
        o.update({"faults.kind": "scheduled", "faults.scheduled": []})
    loss = float(p.get("loss") or 0)
    rng = float(p.get("range") or 0)
    outage = bool(p.get("outage"))
    if loss > 0 or rng > 0 or outage or p.get("allocator") in ("cbba", "hybrid"):
        comms = {"kind": "network", "range": rng if rng > 0 else 2000, "range_gcs": 1500,
                 "latency": 0.05, "jitter": 0.03, "loss": min(loss, 0.6)}
        if outage:
            comms["gcs_outages"] = [[float(p.get("outage_from") or 200), float(p.get("outage_to") or 320)]]
        o["comms"] = comms
    else:
        o["comms"] = {"kind": "ideal", "range": 2000}
    o["realism.enabled"] = bool(p.get("wind"))
    o["safety.filter"] = "orca" if p.get("orca", True) else "none"
    if p.get("one_layer"):
        o["agents.altitude_layers"] = 1
    if p.get("hotspots"):
        o["world.prior.kind"] = "hotspots"
    seed = p.get("seed")
    seed = int(seed) if str(seed or "").strip().lstrip("-").isdigit() else secrets.randbelow(1_000_000)
    return scen, o, seed


def _run_job(jid: str, params: dict) -> None:
    from apps.replay import make_replay
    from swarmsim import apply_overrides, load_scenario, run
    from swarmsim.io import write_run
    job = _jobs[jid]
    try:
        scen, over, seed = build_overrides(params)
        job.update(seed=seed, scenario=scen)
        scn = apply_overrides(load_scenario(ROOT / "scenarios" / f"{scen}.yaml"), over)
        t0 = time.perf_counter()
        res = run(scn, seed)
        name = f"{time.strftime('%H%M%S')}_{scen}_{res.summary['allocator']}_s{seed}"
        out = WEB / name
        write_run(res, out)
        (out / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")
        make_replay(out)
        job.update(state="done", url=f"/runs/web/{name}/replay.html", name=name,
                   wall=round(time.perf_counter() - t0, 1),
                   summary={k: res.summary.get(k) for k in SUMMARY_KEYS})
    except Exception as exc:                       # показать ошибку на странице, сервер не падает
        job.update(state="error", error=f"{exc!r}", trace=traceback.format_exc()[-2000:])


def history() -> list[dict]:
    out = []
    if WEB.exists():
        for d in sorted(WEB.iterdir(), reverse=True)[:40]:
            if (d / "replay.html").exists():
                s = {}
                try:
                    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
                except Exception:
                    pass
                out.append({"name": d.name, "url": f"/runs/web/{d.name}/replay.html",
                            "summary": {k: s.get(k) for k in SUMMARY_KEYS}})
    return out


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "swarmcore"

    def log_message(self, fmt, *args):             # тихо: без строки на каждый запрос
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/history":
            self._json(history())
        elif path == "/api/options":
            self._json({"scenarios": SCENARIOS, "allocators": ALLOCATORS})
        elif path.startswith("/api/status/"):
            job = _jobs.get(path.rsplit("/", 1)[-1])
            self._json(job if job else {"state": "unknown"}, 200 if job else 404)
        elif path.startswith("/runs/"):
            f = (ROOT / path.lstrip("/")).resolve()
            if RUNS.resolve() not in f.parents or not f.is_file():   # только файлы внутри runs/
                self._send(404, b"not found", "text/plain")
                return
            ctype = {".html": "text/html; charset=utf-8", ".json": "application/json",
                     ".csv": "text/csv; charset=utf-8"}.get(f.suffix, "application/octet-stream")
            self._send(200, f.read_bytes(), ctype)
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if urlparse(self.path).path != "/api/run":
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            params = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "bad json"}, 400)
            return
        with _lock:
            if any(j["state"] == "running" for j in _jobs.values()):
                self._json({"error": "прогон уже идёт — дождитесь окончания"}, 409)
                return
            jid = secrets.token_hex(4)
            _jobs[jid] = {"id": jid, "state": "running", "started": time.time()}
        threading.Thread(target=_run_job, args=(jid, params), daemon=True).start()
        self._json({"id": jid})


def serve(port: int = 8765, open_browser: bool = True) -> None:
    WEB.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"пульт симуляции: {url}   (остановить — Ctrl+C)", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("остановлен")
    finally:
        httpd.server_close()


def serve_command(args) -> int:
    serve(args.port, not args.no_browser)
    return 0


def add_serve_args(p) -> None:
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="не открывать браузер")


# ------------------------------------------------------------------ страница
PAGE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Пульт роя</title>
<style>
:root { --bg:#f4f5f7; --panel:#fff; --text:#1d2330; --muted:#6b7385; --line:#dfe3ea; --accent:#2563eb;
        --ok:#15803d; --bad:#b91c1c; --warn:#b45309; }
@media (prefers-color-scheme: dark) { :root { --bg:#12151b; --panel:#1b2029; --text:#e6e9ef; --muted:#8f98a8;
        --line:#2c3340; --accent:#60a5fa; --ok:#4ade80; --bad:#f87171; --warn:#fbbf24; } }
* { box-sizing: border-box; }
body { margin:0; font:14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; background:var(--bg); color:var(--text);
       display:grid; grid-template-columns: 330px 1fr; height:100vh; }
aside { background:var(--panel); border-right:1px solid var(--line); padding:14px 16px; overflow:auto; }
main { display:flex; flex-direction:column; min-width:0; }
h1 { font-size:17px; margin:0 0 2px; } .sub { color:var(--muted); font-size:12px; margin-bottom:12px; }
fieldset { border:1px solid var(--line); border-radius:8px; margin:0 0 10px; padding:8px 10px 10px; }
legend { color:var(--muted); font-size:12px; padding:0 4px; }
label { display:flex; align-items:center; justify-content:space-between; gap:8px; margin:5px 0; }
label.chk { justify-content:flex-start; } label small { color:var(--muted); }
select, input[type=number], input[type=text] { background:var(--bg); color:var(--text); border:1px solid var(--line);
       border-radius:6px; padding:4px 6px; font:inherit; width:150px; }
input[type=range] { width:130px; } .val { min-width:38px; text-align:right; font-variant-numeric:tabular-nums; }
button { font:inherit; border-radius:8px; border:1px solid var(--line); background:var(--bg); color:var(--text);
       padding:6px 10px; cursor:pointer; }
#go { width:100%; background:var(--accent); color:#fff; border:0; font-weight:600; padding:10px; font-size:15px; }
#go:disabled { opacity:.6; cursor:wait; }
#status { margin:8px 0; min-height:20px; font-size:13px; } .err { color:var(--bad); white-space:pre-wrap; }
.sum { display:grid; grid-template-columns:auto auto; gap:2px 10px; font-size:12px; margin-top:6px; }
.sum b { font-variant-numeric:tabular-nums; } .good { color:var(--ok); } .badv { color:var(--bad); }
#hist { list-style:none; padding:0; margin:6px 0 0; font-size:12px; }
#hist li { padding:5px 6px; border-radius:6px; cursor:pointer; display:flex; justify-content:space-between; gap:6px; }
#hist li:hover, #hist li.on { background:var(--bg); } #hist .m { color:var(--muted); }
#bar { padding:6px 12px; border-bottom:1px solid var(--line); background:var(--panel); font-size:13px; color:var(--muted);
       display:flex; justify-content:space-between; gap:8px; }
iframe { flex:1; border:0; width:100%; background:var(--bg); }
#empty { flex:1; display:flex; align-items:center; justify-content:center; color:var(--muted); text-align:center; padding:20px; }
@media (max-width: 800px) { body { grid-template-columns:1fr; height:auto; } iframe { height:80vh; } }
</style></head><body>
<aside>
  <h1>Пульт роя БПЛА</h1>
  <div class="sub">Настройте прогон и нажмите «Новый прогон». Пустое зерно — каждый раз новый мир.</div>
  <fieldset><legend>Миссия</legend>
    <label>Сценарий <select id="scenario"></select></label>
    <label>Алгоритм <select id="allocator"></select></label>
    <label>Дронов <input type="range" id="agents" min="2" max="30" value="10"><span class="val" id="agents_v">10</span></label>
    <label class="chk"><input type="checkbox" id="hotspots" checked> цели скоплениями (карта меняется)</label>
  </fieldset>
  <fieldset><legend>Отказы</legend>
    <label>Доля отказов <input type="range" id="fail_rate" min="0" max="0.4" step="0.1" value="0.2"><span class="val" id="fail_rate_v">20 %</span></label>
    <label class="chk"><input type="checkbox" id="silent" checked> тихие (заметят по пропаже связи)</label>
  </fieldset>
  <fieldset><legend>Связь</legend>
    <label>Дальность, м <input type="range" id="range" min="300" max="2000" step="100" value="700"><span class="val" id="range_v">700</span></label>
    <label>Потери пакетов <input type="range" id="loss" min="0" max="0.5" step="0.05" value="0.1"><span class="val" id="loss_v">10 %</span></label>
    <label class="chk"><input type="checkbox" id="outage" checked> разрыв со станцией</label>
    <label><small>с … по, с</small><span><input type="number" id="outage_from" value="200" style="width:64px"> <input type="number" id="outage_to" value="320" style="width:64px"></span></label>
  </fieldset>
  <fieldset><legend>Мир и безопасность</legend>
    <label class="chk"><input type="checkbox" id="wind" checked> ветер и порывы</label>
    <label class="chk"><input type="checkbox" id="orca" checked> ORCA (уклонение от столкновений)</label>
    <label class="chk"><input type="checkbox" id="one_layer"> все на одной высоте</label>
    <label>Зерно <input type="text" id="seed" placeholder="случайное"></label>
  </fieldset>
  <button id="go">▶ Новый прогон</button>
  <div id="status"></div>
  <div id="summary"></div>
  <fieldset style="margin-top:10px"><legend>История (щёлкните, чтобы открыть)</legend><ul id="hist"></ul></fieldset>
</aside>
<main>
  <div id="bar"><span id="title">нет открытого прогона</span><span><a id="newtab" href="#" target="_blank" style="display:none">открыть в новой вкладке ↗</a></span></div>
  <div id="empty">Нажмите «▶ Новый прогон» слева —<br>через 10–30 секунд здесь появится анимация.</div>
  <iframe id="view" style="display:none"></iframe>
</main>
<script>
const $ = id => document.getElementById(id);
const pct = v => Math.round(v * 100) + " %";
const fmt = { fail_rate: pct, loss: pct, agents: v => v, range: v => v };
for (const k in fmt) { const el = $(k); const upd = () => $(k + "_v").textContent = fmt[k](+el.value); el.oninput = upd; upd(); }

fetch("/api/options").then(r => r.json()).then(o => {
  for (const s of o.scenarios) $("scenario").add(new Option(s, s));
  for (const [k, v] of Object.entries(o.allocators)) $("allocator").add(new Option(v, k));
  try { const saved = JSON.parse(localStorage.getItem("swarm-form") || "{}"); applyForm(saved); } catch (e) {}
});

const FIELDS = ["scenario", "allocator", "agents", "hotspots", "fail_rate", "silent", "range", "loss", "outage",
                "outage_from", "outage_to", "wind", "orca", "one_layer", "seed"];
function readForm() {
  const p = {};
  for (const k of FIELDS) { const el = $(k); p[k] = el.type === "checkbox" ? el.checked : el.value; }
  return p;
}
function applyForm(p) {
  for (const k of FIELDS) { if (!(k in p)) continue; const el = $(k);
    if (el.type === "checkbox") el.checked = p[k]; else el.value = p[k]; if (el.oninput) el.oninput(); }
}

function summaryHtml(s, seed, wall) {
  if (!s) return "";
  const row = (name, v, good) => `<span>${name}</span><b class="${good === undefined ? "" : good ? "good" : "badv"}">${v ?? "—"}</b>`;
  const n = (v, d = 0) => v == null ? "—" : (+v).toFixed(d);
  return `<div class="sum">
    ${row("зерно", seed ?? "—")}
    ${row("выполнено задач", s.CR == null ? "—" : pct(s.CR), s.CR >= 0.95)}
    ${row("время миссии, с", n(s.makespan))}
    ${row("90 % целей найдено к, с", n(s.T_detect_90))}
    ${row("отказало дронов", s.agents_failed ?? 0)}
    ${row("восстановление, с", n(s.T_recovery_max, 1))}
    ${row("дублей задач", s.N_dup ?? 0, !s.N_dup)}
    ${row("потерянных задач", s.N_lost ?? 0, !s.N_lost)}
    ${row("столкновений", s.N_col ?? 0, !s.N_col)}
    ${row("уклонений ORCA (шагов)", s.N_conf ?? 0)}
    ${row("трафик, байт/с на дрон", n(s.B_per_agent))}
    ${wall ? row("счёт занял, с", wall) : ""}
  </div>`;
}

function openRun(url, name, s, seed) {
  $("empty").style.display = "none"; $("view").style.display = "block";
  $("view").src = url; $("title").textContent = name; $("newtab").href = url; $("newtab").style.display = "";
  $("summary").innerHTML = summaryHtml(s, seed);
  for (const li of $("hist").children) li.classList.toggle("on", li.dataset.name === name);
}

function loadHistory() {
  fetch("/api/history").then(r => r.json()).then(h => {
    $("hist").innerHTML = "";
    for (const it of h) {
      const li = document.createElement("li"); li.dataset.name = it.name;
      const s = it.summary || {}; const seed = (it.name.match(/_s(\d+)$/) || [])[1];
      li.innerHTML = `<span>${it.name.replace(/_s\d+$/, "")}</span><span class="m">CR ${s.CR == null ? "—" : pct(s.CR)} · s${seed}</span>`;
      li.onclick = () => openRun(it.url, it.name, s, seed);
      $("hist").appendChild(li);
    }
  });
}
loadHistory();

$("go").onclick = async () => {
  const p = readForm();
  try { localStorage.setItem("swarm-form", JSON.stringify(p)); } catch (e) {}
  $("go").disabled = true; $("status").className = ""; $("summary").innerHTML = "";
  const r = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(p) });
  const j = await r.json();
  if (!r.ok) { $("status").className = "err"; $("status").textContent = j.error || "ошибка"; $("go").disabled = false; return; }
  const t0 = Date.now();
  const tick = async () => {
    const st = await (await fetch("/api/status/" + j.id)).json();
    if (st.state === "running") {
      $("status").textContent = `⏳ считаю прогон… ${Math.round((Date.now() - t0) / 1000)} с`;
      setTimeout(tick, 700); return;
    }
    $("go").disabled = false;
    if (st.state === "done") {
      $("status").textContent = `✅ готово за ${st.wall} с`;
      loadHistory(); openRun(st.url, st.name, st.summary, st.seed);
      $("summary").innerHTML = summaryHtml(st.summary, st.seed, st.wall);
    } else { $("status").className = "err"; $("status").textContent = "ошибка: " + (st.error || st.state); }
  };
  tick();
};
</script></body></html>
"""
