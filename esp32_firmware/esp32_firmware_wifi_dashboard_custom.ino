/*
 * SNN-SHM ESP32 Firmware — WiFi Dashboard Edition
 *
 * Endpoints:
 *   GET /      → full HTML dashboard (chunked transfer, never truncates)
 *   GET /data  → JSON with latest inference result
 *
 * Setup:
 *  1. Set WIFI_SSID / WIFI_PASSWORD below.
 *  2. Flash. Open Serial Monitor → note the IP.
 *  3. Open http://<IP> in any browser on the same WiFi.
 *
 * Hardware: ESP32 + MPU6050
 */

#include <Wire.h>
#include <MPU6050.h>
#include <WiFi.h>
#include <WebServer.h>

#include "weights/snn_config.h"
#include "src/lif_neuron.h"
#include "src/snn_inference.h"
#include "src/feature_engineering.h"

// ── WiFi credentials ──────────────────────────────────────────────────────────
#define WIFI_SSID     "LAPTOP-Q5IS18KS 9743"
#define WIFI_PASSWORD "Q0#o0108"

// ── Pin definitions ───────────────────────────────────────────────────────────
#define LED_PIN            2
#define BUZZER_PIN         4
#define MPU_SDA            21
#define MPU_SCL            22
#define SAMPLE_INTERVAL_MS 100

// ── Global objects ────────────────────────────────────────────────────────────
MPU6050      mpu;
SNNInference snn;
WebServer    server(80);
portMUX_TYPE resultMux = portMUX_INITIALIZER_UNLOCKED;

// ── Shared result struct ──────────────────────────────────────────────────────
struct InferenceResult {
    int      prediction;
    float    accel_x, accel_y, accel_z;
    int32_t  spike_normal, spike_anomaly;
    uint32_t inference_us;
    uint32_t timestamp_ms;
    uint32_t total_samples;
    uint32_t anomaly_count;
} result;

// ── Dashboard HTML ────────────────────────────────────────────────────────────
static const char HTML_1[] PROGMEM = R"rawhtml(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SNN-SHM EDGE</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;700;900&display=swap" rel="stylesheet"/>
<style>
:root{
  --bg:#050a0f;--surface:#080f18;--border:#0d2035;
  --accent:#00e5ff;--accent2:#00ff88;--warn:#ffaa00;
  --danger:#ff2244;--text:#c8dde8;--dim:#3a5a70;
  --glow:0 0 12px #00e5ff55;--glow-g:0 0 12px #00ff8855;
  --glow-r:0 0 20px #ff224488;
}
*{box-sizing:border-box;margin:0;padding:0}
body{
  background:var(--bg);color:var(--text);
  font-family:'Share Tech Mono',monospace;
  min-height:100vh;overflow-x:hidden;
}
body::before{
  content:'';position:fixed;inset:0;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.12) 2px,rgba(0,0,0,0.12) 4px);
  pointer-events:none;z-index:999;
}

/* ── HEADER ── */
header{
  display:flex;align-items:center;justify-content:space-between;
  padding:18px 28px;border-bottom:1px solid var(--border);
  background:linear-gradient(90deg,#050a0f 60%,#061520 100%);
  position:sticky;top:0;z-index:100;
}
.logo{display:flex;align-items:center;gap:14px}
.logo-icon{
  width:38px;height:38px;border:2px solid var(--accent);
  border-radius:6px;display:grid;place-items:center;
  box-shadow:var(--glow);animation:pulse-border 2.5s infinite;
}
@keyframes pulse-border{
  0%,100%{box-shadow:0 0 8px #00e5ff44}
  50%{box-shadow:0 0 22px #00e5ffaa}
}
.logo-icon svg{width:22px;height:22px}
.logo-text h1{
  font-family:'Exo 2',sans-serif;font-weight:900;
  font-size:1.1rem;color:var(--accent);
  letter-spacing:3px;text-transform:uppercase;
}
.logo-text p{font-size:.62rem;color:var(--dim);letter-spacing:2px}
.header-right{display:flex;align-items:center;gap:24px}
.controls{display:flex;gap:10px;align-items:center}
.status-pill{
  display:flex;align-items:center;gap:8px;
  padding:6px 14px;border:1px solid var(--border);
  border-radius:2px;font-size:.72rem;letter-spacing:1.5px;
}
.dot{
  width:8px;height:8px;border-radius:50%;
  background:var(--accent2);box-shadow:var(--glow-g);
  animation:blink 1.2s infinite;
}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
#nClock{font-size:.78rem;color:var(--dim)}

/* ── BUTTONS ── */
.btn{
  padding:8px 18px;border:1px solid var(--accent);
  background:transparent;color:var(--accent);
  font-family:'Share Tech Mono',monospace;
  font-size:.72rem;letter-spacing:2px;
  cursor:pointer;border-radius:2px;
  transition:background .2s,box-shadow .2s;
  text-transform:uppercase;
}
.btn:hover,.btn.on{background:#00e5ff18;box-shadow:var(--glow)}
.btn.danger{border-color:var(--danger);color:var(--danger)}
.btn.danger:hover{background:#ff224418}
.speed-label{font-size:.65rem;color:var(--dim)}
select{
  background:var(--surface);border:1px solid var(--border);
  color:var(--text);font-family:'Share Tech Mono',monospace;
  font-size:.72rem;padding:6px 10px;border-radius:2px;cursor:pointer;
}

/* ── MAIN GRID ── */
main{
  display:grid;
  grid-template-columns:280px 1fr;
  grid-template-rows:auto auto 1fr;
  gap:16px;
  padding:20px 24px;
  max-width:1440px;margin:0 auto;
}

/* ── CARDS ── */
.card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:4px;padding:18px;
}
.card-title{
  font-size:.62rem;letter-spacing:3px;color:var(--dim);
  text-transform:uppercase;margin-bottom:14px;
  display:flex;align-items:center;gap:8px;
}
.card-title::before{
  content:'';display:block;width:3px;height:12px;
  background:var(--accent);box-shadow:var(--glow);
}

/* ── SIDEBAR ── */
.sidebar{
  grid-column:1;grid-row:1/5;
  display:flex;flex-direction:column;gap:14px;
  overflow-y:auto;max-height:calc(100vh - 80px);
  scrollbar-width:none;
}

/* big prediction stat */
.big-stat{text-align:center;padding:22px 10px}
.big-stat .value{
  font-family:'Exo 2',sans-serif;font-weight:900;
  font-size:3.2rem;line-height:1;color:var(--accent2);
  text-shadow:var(--glow-g);transition:color .3s,text-shadow .3s;
}
.big-stat.anomaly .value{color:var(--danger);text-shadow:var(--glow-r)}
.big-stat .label{font-size:.65rem;letter-spacing:2px;color:var(--dim);margin-top:6px}

/* gauge */
.gauge-wrap{position:relative;width:160px;height:80px;margin:10px auto}
.gauge-wrap canvas{width:100%;height:100%}
.gauge-pct{
  position:absolute;bottom:0;left:50%;transform:translateX(-50%);
  font-family:'Exo 2',sans-serif;font-weight:700;
  font-size:1.4rem;color:var(--accent);text-shadow:var(--glow);
  transition:color .25s;
}

/* metric rows */
.metric-row{
  display:flex;justify-content:space-between;align-items:center;
  padding:9px 0;border-bottom:1px solid var(--border);font-size:.75rem;
}
.metric-row:last-child{border-bottom:none}
.mkey{color:var(--dim);letter-spacing:1px}
.mval{color:var(--accent);font-weight:bold;font-variant-numeric:tabular-nums}

/* accel xyz grid */
.accel-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:0}
.accel-box{
  text-align:center;padding:10px 4px;
  background:var(--bg);border:1px solid var(--border);border-radius:2px;
}
.accel-axis{font-size:.6rem;color:var(--dim);letter-spacing:2px}
.accel-val{
  font-family:'Exo 2',sans-serif;font-weight:700;font-size:1rem;
  color:var(--accent);text-shadow:var(--glow);
  font-variant-numeric:tabular-nums;margin-top:2px;
}

/* magnitude */
.mag-val{
  font-family:'Exo 2',sans-serif;font-weight:900;font-size:2.5rem;
  color:var(--accent2);text-shadow:var(--glow-g);
  line-height:1;text-align:center;transition:color .25s,text-shadow .25s;
}
.mag-val.danger{color:var(--danger);text-shadow:var(--glow-r)}
.mag-unit{font-size:.62rem;color:var(--dim);letter-spacing:2px;text-align:center;margin-top:4px}

/* spike output */
.spike-out-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.spike-out-box{
  text-align:center;padding:10px 6px;
  background:var(--bg);border:1px solid var(--border);border-radius:2px;
}
.spike-out-lbl{font-size:.6rem;color:var(--dim);letter-spacing:2px}
.spike-out-val{
  font-family:'Exo 2',sans-serif;font-weight:900;font-size:1.4rem;margin-top:3px;
}

/* confidence bars */
.conf-row{margin-bottom:10px}
.conf-row:last-child{margin-bottom:0}
.conf-lbl{display:flex;justify-content:space-between;font-size:.62rem;margin-bottom:4px}
.prob-bar-wrap{height:10px;background:#0d2035;border-radius:2px;overflow:hidden}
.prob-bar{height:100%;border-radius:2px;transition:width .3s}

/* ── VERDICT (top-right) ── */
.verdict{
  grid-column:2;grid-row:1;
  display:flex;align-items:center;justify-content:space-between;
  padding:20px 28px;
}
.verdict-label{
  font-family:'Exo 2',sans-serif;font-weight:900;
  font-size:2rem;letter-spacing:6px;text-transform:uppercase;
  transition:color .3s,text-shadow .3s;
}
.verdict-label.normal{color:var(--accent2);text-shadow:var(--glow-g)}
.verdict-label.anomaly{color:var(--danger);text-shadow:var(--glow-r);animation:shake .4s}
@keyframes shake{
  0%,100%{transform:translateX(0)}
  25%{transform:translateX(-5px)}
  75%{transform:translateX(5px)}
}
.verdict-sub{font-size:.65rem;color:var(--dim);margin-top:6px;letter-spacing:2px}
.verdict-meta{font-size:.7rem;color:var(--dim);line-height:1.8;text-align:right}
.verdict-meta span{color:var(--text)}

/* ── WAVES (middle-right) ── */
.waves{
  grid-column:2;grid-row:2;
  display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;
}
.wave-card{overflow:hidden}
.wave-card canvas{width:100%;height:80px;display:block}

/* ── LOG AREA (bottom-right) ── */
.log-area{
  grid-column:2;grid-row:3;
  display:grid;grid-template-columns:1fr 320px;gap:14px;min-height:0;
}
.log-scroll{
  overflow-y:auto;max-height:340px;
  scrollbar-width:thin;scrollbar-color:var(--border) transparent;
}
.log-header{
  display:grid;
  grid-template-columns:60px 110px 80px 80px 80px 80px 1fr;
  gap:6px;padding:5px 10px;font-size:.6rem;
  letter-spacing:1.5px;color:var(--dim);
  border-bottom:1px solid var(--border);
  background:var(--bg);position:sticky;top:0;z-index:1;
}
.log-row{
  display:grid;
  grid-template-columns:60px 110px 80px 80px 80px 80px 1fr;
  gap:6px;align-items:center;padding:6px 10px;
  border-bottom:1px solid #0a1c2a;font-size:.7rem;
  transition:background .2s;animation:fadeIn .3s;
}
@keyframes fadeIn{from{opacity:0;transform:translateY(-4px)}to{opacity:1}}
.log-row:hover{background:#0a1828}
.log-row.an{background:#1a060a;border-left:2px solid var(--danger)}
.log-row.nm{border-left:2px solid transparent}
.tag{
  padding:2px 7px;border-radius:2px;font-size:.62rem;
  letter-spacing:1px;font-weight:bold;text-align:center;
}
.tag.n{background:#002d20;color:var(--accent2)}
.tag.a{background:#2d0010;color:var(--danger)}
.pb{height:10px;background:#0d2035;border-radius:2px;overflow:hidden;margin-top:3px}
.pbi{height:100%;border-radius:2px;transition:width .3s}

/* spike raster card */
.spike-card{overflow:hidden}
.spike-card canvas{width:100%;height:100%;display:block}

/* ── FLOATING STATUS BADGES ── */
.connection-status{
  position:fixed;bottom:20px;right:20px;
  padding:8px 16px;border-radius:4px;font-size:.7rem;
  background:var(--surface);border:1px solid var(--border);
  z-index:1000;display:flex;align-items:center;gap:8px;
}
.connection-status.connected{border-color:var(--accent2);color:var(--accent2)}
.connection-status.disconnected{border-color:var(--danger);color:var(--danger)}
.connection-status.connecting{border-color:var(--warn);color:var(--warn)}
.esp-info{
  position:fixed;bottom:20px;left:20px;
  padding:8px 16px;border-radius:4px;font-size:.7rem;
  background:var(--surface);border:1px solid var(--border);
  z-index:1000;color:var(--accent);
}
.dot2{width:6px;height:6px;border-radius:50%;display:inline-block}
</style>
</head>
<body>

<!-- ── HEADER ── -->
<header>
  <div class="logo">
    <div class="logo-icon">
      <svg viewBox="0 0 24 24" fill="none" stroke="#00e5ff" stroke-width="1.8"
           xmlns="http://www.w3.org/2000/svg">
        <path d="M2 12h3l3-7 4 14 3-9 2 2h5" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="18" cy="6" r="3" fill="#00e5ff"/>
      </svg>
    </div>
    <div class="logo-text">
      <h1>SNN · SHM</h1>
      <p>ESP32 · LIF · EDGE INFERENCE</p>
    </div>
  </div>
  <div class="header-right">
    <div class="controls">
      <span class="speed-label">SPEED</span>
      <select id="spd">
        <option value="1000">0.5×</option>
        <option value="500" selected>1×</option>
        <option value="250">2×</option>
        <option value="100">5×</option>
      </select>
      <button class="btn" id="bP">PAUSE</button>
      <button class="btn danger" id="bC">CLEAR LOG</button>
    </div>
    <div class="status-pill">
      <div class="dot" id="nDot"></div>
      <span id="nSt">CONNECTING...</span>
    </div>
    <div id="nClock">--:--:--</div>
  </div>
</header>

<!-- ── MAIN GRID ── -->
<main>

  <!-- LEFT SIDEBAR -->
  <aside class="sidebar">

    <!-- Prediction big stat -->
    <div class="card big-stat" id="vp">
      <div class="value" id="vw">–</div>
      <div class="label">PREDICTION</div>
    </div>

    <!-- Gauge -->
    <div class="card">
      <div class="card-title">ANOMALY PROBABILITY</div>
      <div class="gauge-wrap">
        <canvas id="gC" width="320" height="160"></canvas>
        <div class="gauge-pct" id="gP">0%</div>
      </div>
    </div>

    <!-- Session stats -->
    <div class="card">
      <div class="card-title">SESSION STATS</div>
      <div class="metric-row"><span class="mkey">SAMPLES</span><span class="mval" id="mT">0</span></div>
      <div class="metric-row"><span class="mkey">ANOMALIES</span><span class="mval" id="mA">0</span></div>
      <div class="metric-row"><span class="mkey">ANOM RATE</span><span class="mval" id="mR">—</span></div>
      <div class="metric-row"><span class="mkey">POLL RATE</span><span class="mval" id="mH">—</span></div>
      <div class="metric-row"><span class="mkey">INFER TIME</span><span class="mval" id="mI">—</span></div>
      <div class="metric-row"><span class="mkey">UPTIME</span><span class="mval" id="mU">—</span></div>
      <div class="metric-row"><span class="mkey">SPIKES N/A</span><span class="mval" id="mS">—</span></div>
    </div>

    <!-- Accelerometer XYZ -->
    <div class="card">
      <div class="card-title">ACCELEROMETER m/s²</div>
      <div class="accel-grid">
        <div class="accel-box"><div class="accel-axis">X</div><div class="accel-val" id="rX">—</div></div>
        <div class="accel-box"><div class="accel-axis">Y</div><div class="accel-val" id="rY">—</div></div>
        <div class="accel-box"><div class="accel-axis">Z</div><div class="accel-val" id="rZ">—</div></div>
      </div>
    </div>

    <!-- Vector magnitude -->
    <div class="card">
      <div class="card-title">VECTOR MAGNITUDE</div>
      <div class="mag-val" id="bM">—</div>
      <div class="mag-unit">m/s²</div>
    </div>

    <!-- Output spikes -->
    <div class="card">
      <div class="card-title">OUTPUT SPIKES</div>
      <div class="spike-out-grid">
        <div class="spike-out-box">
          <div class="spike-out-lbl">NORMAL</div>
          <div class="spike-out-val" id="sN" style="color:var(--accent2);text-shadow:var(--glow-g)"></div>
        </div>
        <div class="spike-out-box">
          <div class="spike-out-lbl">ANOMALY</div>
          <div class="spike-out-val" id="sA" style="color:var(--danger);text-shadow:var(--glow-r)"></div>
        </div>
      </div>
    </div>

    <!-- Confidence bars -->
    <div class="card">
      <div class="card-title">CONFIDENCE</div>
      <div class="conf-row">
        <div class="conf-lbl">
          <span style="color:var(--accent2)">NORMAL</span>
          <span id="pNt" style="color:var(--accent2)"></span>
        </div>
        <div class="prob-bar-wrap">
          <div class="prob-bar" id="pNb" style="background:var(--accent2)"></div>
        </div>
      </div>
      <div class="conf-row">
        <div class="conf-lbl">
          <span style="color:var(--danger)">ANOMALY</span>
          <span id="pAt" style="color:var(--danger)"></span>
        </div>
        <div class="prob-bar-wrap">
          <div class="prob-bar" id="pAb" style="background:var(--danger)"></div>
        </div>
      </div>
    </div>

    <!-- Node info -->
    <div class="card">
      <div class="card-title">NODE INFO</div>
      <div class="metric-row"><span class="mkey">IP</span><span class="mval" id="eIP">—</span></div>
      <div class="metric-row"><span class="mkey">ARCH</span><span class="mval">LIF·SNN</span></div>
      <div class="metric-row"><span class="mkey">TIME STEPS</span><span class="mval">120</span></div>
    </div>

  </aside>

  <!-- VERDICT BANNER -->
  <div class="card verdict">
    <div>
      <div class="verdict-label normal" id="verdictLabel">AWAITING DATA</div>
      <div class="verdict-sub">LIF-SNN · SPIKE THRESHOLD 0.800</div>
    </div>
    <div class="verdict-meta">
      SAMPLE <span id="sbSamp">–</span><br>
      P(NORMAL) <span id="vNorm">–</span><br>
      P(ANOMALY) <span id="vAnom">–</span><br>
      UPTIME <span id="sSamp">–</span>
    </div>
  </div>

  <!-- WAVEFORMS -->
  <div class="waves">
    <div class="card wave-card">
      <div class="card-title">ACCEL X</div>
      <canvas id="wAX" width="600" height="80"></canvas>
    </div>
    <div class="card wave-card">
      <div class="card-title">ACCEL Y</div>
      <canvas id="wAY" width="600" height="80"></canvas>
    </div>
    <div class="card wave-card">
      <div class="card-title">ACCEL Z · MAGNITUDE</div>
      <canvas id="wAZ" width="600" height="80"></canvas>
    </div>
  </div>

  <!-- LOG + SPIKE RASTER -->
  <div class="log-area">
    <div class="card">
      <div class="card-title">PREDICTION LOG</div>
      <div class="log-header">
        <span>#</span><span>TIME</span><span>STATUS</span>
        <span>AX</span><span>AY</span><span>AZ</span><span>P(ANOM)</span>
      </div>
      <div class="log-scroll" id="lS"></div>
    </div>
    <div class="card spike-card">
      <div class="card-title">SPIKE RASTER · LAYER 1</div>
      <canvas id="sC" width="300" height="280"></canvas>
    </div>
  </div>

</main>

<!-- FLOATING BADGES -->
<div class="connection-status disconnected" id="connStatus">
  <span class="dot2" id="sDot" style="background:var(--warn)"></span>
  <span id="sTxt">INITIALISING</span>
</div>

<div class="esp-info">
  &#128225; ESP32: <span id="eIP2">—</span>
</div>
)rawhtml";

static const char HTML_2[] PROGMEM = R"rawhtml(
<script>
const WLEN=120,NEU=32,HIST=50,MLOG=300;
let paused=false,cnt=0,acnt=0,t0=Date.now(),lastTs=-1,lastPt=Date.now(),phz=0;
let ptimer=null,PMS=500;
const wv={ax:new Array(WLEN).fill(0),ay:new Array(WLEN).fill(0),
          az:new Array(WLEN).fill(0),mg:new Array(WLEN).fill(0)};
const spk=Array.from({length:NEU},()=>new Array(HIST).fill(0));

function tick(){const n=new Date(),z=x=>String(x).padStart(2,'0');
  document.getElementById('nClock').textContent=`${z(n.getHours())}:${z(n.getMinutes())}:${z(n.getSeconds())}`;}
setInterval(tick,1000);tick();

function conn(s){
  const d=document.getElementById('sDot'),t=document.getElementById('sTxt'),
        nd=document.getElementById('nDot'),ns=document.getElementById('nSt');
  const cs=document.getElementById('connStatus');
  const C={ok:['#00ff88','LIVE','connected'],warn:['#ffaa00','WAIT','connecting'],err:['#ff2244','OFFLINE','disconnected']};
  const [c,l,cls]=C[s]||C.err;
  d.style.background=c;d.style.boxShadow=`0 0 8px ${c}80`;
  nd.style.background=c;nd.style.boxShadow=`0 0 8px ${c}80`;
  t.textContent=s==='ok'?'LIVE · CONNECTED':s==='warn'?'CONNECTING...':'DISCONNECTED';
  ns.textContent=l;
  cs.className='connection-status '+cls;
}

function drawG(p){
  const c=document.getElementById('gC');if(!c)return;
  const x=c.getContext('2d'),W=c.width,H=c.height,cx=W/2,cy=H-10,r=110;
  x.clearRect(0,0,W,H);
  x.beginPath();x.arc(cx,cy,r,Math.PI,0);x.strokeStyle='#0d2035';x.lineWidth=18;x.stroke();
  const col=p<.5?`hsl(${150-p*100},100%,55%)`:`hsl(${50-(p-.5)*100},100%,55%)`;
  x.beginPath();x.arc(cx,cy,r,Math.PI,Math.PI+p*Math.PI);
  x.strokeStyle=col;x.lineWidth=18;x.shadowColor=col;x.shadowBlur=16;x.stroke();x.shadowBlur=0;
  for(let i=0;i<=10;i++){
    const a=Math.PI+i/10*Math.PI;
    x.beginPath();x.moveTo(cx+(r-24)*Math.cos(a),cy+(r-24)*Math.sin(a));
    x.lineTo(cx+(r-12)*Math.cos(a),cy+(r-12)*Math.sin(a));
    x.strokeStyle='#1a3050';x.lineWidth=1.5;x.stroke();
  }
  const g=document.getElementById('gP');
  g.textContent=(p*100).toFixed(0)+'%';
  g.style.color=p>.7?'var(--danger)':p>.4?'var(--warn)':'var(--accent2)';
}

function drawW(id,data,col,lo,hi,ex,ec){
  const c=document.getElementById(id);if(!c)return;
  const x=c.getContext('2d'),W=c.width,H=c.height;
  x.clearRect(0,0,W,H);
  x.strokeStyle='#0d2035';x.lineWidth=1;
  [.25,.5,.75].forEach(f=>{x.beginPath();x.moveTo(0,f*H);x.lineTo(W,f*H);x.stroke();});
  const zy=H*(1-(0-lo)/(hi-lo));
  x.strokeStyle='#1a3050';x.lineWidth=1;x.setLineDash([3,3]);
  x.beginPath();x.moveTo(0,zy);x.lineTo(W,zy);x.stroke();x.setLineDash([]);
  const ln=(arr,lc,lw)=>{
    x.strokeStyle=lc;x.lineWidth=lw||1.5;x.shadowColor=lc;x.shadowBlur=6;x.beginPath();
    arr.forEach((v,i)=>{const px=i/(arr.length-1)*W,py=H*(1-(v-lo)/(hi-lo));
      i?x.lineTo(px,py):x.moveTo(px,py);});
    x.stroke();x.shadowBlur=0;
  };
  ln(data,col);if(ex)ln(ex,ec,.7);
}

function pushSpk(p){spk.forEach((r,n)=>{r.shift();
  const prob=p*(0.3+0.7*n/NEU)+Math.random()*.1;r.push(Math.random()<prob?1:0);});}
function drawSpk(){
  const c=document.getElementById('sC');if(!c)return;
  const x=c.getContext('2d'),W=c.width,H=c.height;
  x.fillStyle='#050a0f';x.fillRect(0,0,W,H);
  const rh=H/NEU,cw=W/HIST;
  const pal=['#00e5ff','#00ff88','#ffaa00'];
  spk.forEach((row,n)=>row.forEach((s,t)=>{if(s){
    const col=pal[Math.floor(n/(NEU/3))];
    x.fillStyle=col;x.shadowColor=col;x.shadowBlur=3;
    x.fillRect(t*cw+1,n*rh+1,cw-1,rh-1);
  }}));x.shadowBlur=0;
}

function addLog(ax,ay,az,pA,pred){
  const sc=document.getElementById('lS');if(!sc)return;
  const isA=pred===1,n=new Date(),z=x=>String(x).padStart(2,'0');
  const ts=`${z(n.getHours())}:${z(n.getMinutes())}:${z(n.getSeconds())}`;
  const vc=v=>v>=0?'var(--accent)':'#ff8888';
  const r=document.createElement('div');
  r.className='log-row '+(isA?'an':'nm');
  r.innerHTML=`<span style="color:var(--dim)">${cnt}</span>
<span style="color:var(--dim);font-size:.62rem">${ts}</span>
<span><span class="tag ${isA?'a':'n'}">${isA?'ANOMALY':'NORMAL'}</span></span>
<span style="color:${vc(ax)}">${ax.toFixed(2)}</span>
<span style="color:${vc(ay)}">${ay.toFixed(2)}</span>
<span style="color:${vc(az)}">${az.toFixed(2)}</span>
<div>
  <div style="font-size:.65rem;margin-bottom:3px;color:${isA?'var(--danger)':'var(--accent2)'}">${(pA*100).toFixed(1)}%</div>
  <div class="pb"><div class="pbi" style="width:${pA*100}%;background:${isA?'var(--danger)':'var(--accent2)'}"></div></div>
</div>`;
  sc.prepend(r);
  while(sc.children.length>MLOG)sc.removeChild(sc.lastChild);
}

function applyData(d){
  const isA=d.prediction===1;
  const pA=Math.min(1,Math.max(0,d.spike_anomaly/120));
  const pN=Math.min(1,Math.max(0,d.spike_normal/120));
  const mag=Math.sqrt(d.accel_x**2+d.accel_y**2+d.accel_z**2);
  const now=Date.now();phz=1000/(now-lastPt);lastPt=now;
  cnt++;if(isA)acnt++;

  // verdict banner
  const vl=document.getElementById('verdictLabel'),vp=document.getElementById('vp');
  if(vl){
    vl.textContent=isA?'ANOMALY DETECTED':'NORMAL';
    vl.className='verdict-label '+(isA?'anomaly':'normal');
    if(isA){vl.style.animation='none';void vl.offsetWidth;vl.style.animation='shake .4s';}
  }
  // big sidebar stat
  const vw=document.getElementById('vw');
  if(vw){vw.textContent=isA?'ANOM':'OK';}
  if(vp){vp.className='card big-stat'+(isA?' anomaly':'');}

  drawG(pA);

  const s=(id,v)=>{const e=document.getElementById(id);if(e)e.textContent=v;};
  const up=((Date.now()-t0)/1000).toFixed(0);
  s('mT',cnt);s('mA',acnt);
  s('mR',cnt?(acnt/cnt*100).toFixed(1)+'%':'—');
  s('mH',phz.toFixed(2)+' Hz');
  s('mI',d.inference_us+' µs');
  s('mU',up+'s');
  s('mS',d.spike_normal+' / '+d.spike_anomaly);
  s('sbSamp',cnt);
  s('sSamp',up+'s');
  s('vNorm',(pN*100).toFixed(1)+'%');
  s('vAnom',(pA*100).toFixed(1)+'%');

  const ip=window.location.hostname;
  s('eIP',ip);s('eIP2',ip);

  s('rX',d.accel_x.toFixed(3));s('rY',d.accel_y.toFixed(3));s('rZ',d.accel_z.toFixed(3));

  const bm=document.getElementById('bM');
  if(bm){bm.textContent=mag.toFixed(3);bm.className='mag-val'+(isA?' danger':'');}

  s('sN',d.spike_normal);s('sA',d.spike_anomaly);

  s('pNt',(pN*100).toFixed(1)+'%');s('pAt',(pA*100).toFixed(1)+'%');
  const pnb=document.getElementById('pNb'),pab=document.getElementById('pAb');
  if(pnb)pnb.style.width=(pN*100)+'%';if(pab)pab.style.width=(pA*100)+'%';

  wv.ax.shift();wv.ax.push(d.accel_x);
  wv.ay.shift();wv.ay.push(d.accel_y);
  wv.az.shift();wv.az.push(d.accel_z);
  wv.mg.shift();wv.mg.push(mag);
  drawW('wAX',wv.ax,'#00e5ff',-15,15);
  drawW('wAY',wv.ay,'#00ff88',-15,15);
  drawW('wAZ',wv.az,'#7755ff',0,25,wv.mg,'#ffaa00');

  pushSpk(pA);drawSpk();
  addLog(d.accel_x,d.accel_y,d.accel_z,pA,d.prediction);
}

async function poll(){
  if(paused)return;
  try{
    const resp=await fetch('/data',{cache:'no-store'});
    if(!resp.ok){conn('err');return;}
    const txt=await resp.text();
    let d;
    try{d=JSON.parse(txt);}catch(e){console.error('JSON err:',txt);conn('err');return;}
    if(lastTs!==-1 && d.total_samples>0 && d.timestamp_ms===lastTs)return;
    lastTs=d.timestamp_ms;
    conn('ok');
    applyData(d);
  }catch(e){console.error('poll:',e);conn('err');}
}

document.getElementById('bP').addEventListener('click',function(){
  paused=!paused;this.textContent=paused?'RESUME':'PAUSE';this.classList.toggle('on',paused);
});
document.getElementById('bC').addEventListener('click',()=>{
  document.getElementById('lS').innerHTML='';
  cnt=0;acnt=0;t0=Date.now();
  Object.values(wv).forEach(a=>a.fill(0));spk.forEach(r=>r.fill(0));
});
document.getElementById('spd').addEventListener('change',function(){
  PMS=parseInt(this.value);clearInterval(ptimer);ptimer=setInterval(poll,PMS);
});

conn('warn');drawG(0);drawSpk();
ptimer=setInterval(poll,PMS);
poll();
</script>
</body>
</html>
)rawhtml";

// ── Web handlers ──────────────────────────────────────────────────────────────

void handleRoot() {
    server.setContentLength(CONTENT_LENGTH_UNKNOWN);
    server.sendHeader("Content-Type", "text/html; charset=utf-8");
    server.sendHeader("Cache-Control", "no-cache");
    server.send(200);
    server.sendContent_P(HTML_1);
    server.sendContent_P(HTML_2);
    server.sendContent("");
}

void handleData() {
    InferenceResult snap;
    portENTER_CRITICAL(&resultMux);
    snap = result;
    portEXIT_CRITICAL(&resultMux);

    char json[320];
    snprintf(json, sizeof(json),
        "{"
          "\"prediction\":%d,"
          "\"accel_x\":%.4f,"
          "\"accel_y\":%.4f,"
          "\"accel_z\":%.4f,"
          "\"spike_normal\":%d,"
          "\"spike_anomaly\":%d,"
          "\"inference_us\":%u,"
          "\"timestamp_ms\":%u,"
          "\"total_samples\":%u,"
          "\"anomaly_count\":%u"
        "}",
        snap.prediction,
        snap.accel_x, snap.accel_y, snap.accel_z,
        (int)snap.spike_normal, (int)snap.spike_anomaly,
        (unsigned)snap.inference_us, (unsigned)snap.timestamp_ms,
        (unsigned)snap.total_samples, (unsigned)snap.anomaly_count
    );

    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.sendHeader("Cache-Control", "no-cache, no-store, must-revalidate");
    server.send(200, "application/json", json);
}

void handleNotFound() {
    server.send(404, "text/plain", "Not found");
}

// ── WiFi ──────────────────────────────────────────────────────────────────────

void connectWiFi() {
    Serial.print("Connecting to: "); Serial.println(WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED) {
        delay(500); Serial.print(".");
        digitalWrite(LED_PIN, !digitalRead(LED_PIN));
        if (++attempts > 40) {
            Serial.println("\nWiFi failed.");
            digitalWrite(LED_PIN, LOW); return;
        }
    }
    digitalWrite(LED_PIN, LOW);
    Serial.println();
    Serial.print("Dashboard: http://");
    Serial.println(WiFi.localIP());
}

// ── Web server task (Core 0) ──────────────────────────────────────────────────

void webServerTask(void* pv) {
    for (;;) { server.handleClient(); vTaskDelay(1); }
}

// ── Setup ─────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    pinMode(LED_PIN,    OUTPUT);
    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(LED_PIN,    LOW);
    digitalWrite(BUZZER_PIN, LOW);

    Wire.begin(MPU_SDA, MPU_SCL);
    Serial.println("Initializing MPU6050...");
    mpu.initialize();
    if (mpu.testConnection()) {
        Serial.println("MPU6050 OK");
    } else {
        Serial.println("MPU6050 FAILED");
        while (1) { digitalWrite(LED_PIN, HIGH); delay(400); digitalWrite(LED_PIN, LOW); delay(400); }
    }
    mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);
    mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);
    memset(&result, 0, sizeof(result));

    connectWiFi();
    server.on("/",     handleRoot);
    server.on("/data", handleData);
    server.onNotFound(handleNotFound);
    server.begin();
    Serial.println("Web server started");

    xTaskCreatePinnedToCore(webServerTask, "WebSrv", 8192, NULL, 1, NULL, 0);

    Serial.println("==============================================");
    Serial.println("SNN-SHM Edge Node Ready");
    Serial.printf("Architecture: %d->%d->%d->%d->%d\n",
        INPUT_SIZE, HIDDEN_1_SIZE, HIDDEN_2_SIZE, HIDDEN_3_SIZE, OUTPUT_SIZE);
    Serial.printf("Time steps: %d  |  Threshold: %.3f\n",
        TIME_STEPS, ANOMALY_THRESHOLD / 32768.0f);
    Serial.println("==============================================");
    delay(500);
}

// ── Loop — Core 1, inference ──────────────────────────────────────────────────

void loop() {
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getAcceleration(&ax, &ay, &az);
    mpu.getRotation(&gx, &gy, &gz);

    RawSensorData raw;
    raw.accel_x = ax / 16384.0f * 9.81f;
    raw.accel_y = ay / 16384.0f * 9.81f;
    raw.accel_z = az / 16384.0f * 9.81f;
    raw.strain  = 85.0f;
    raw.temp    = 25.0f;

    int16_t features[INPUT_SIZE];
    compute_features(raw, features);

    unsigned long t0     = micros();
    int prediction       = snn.predict(features);
    unsigned long inf_us = micros() - t0;

    int32_t spikes[OUTPUT_SIZE];
    snn.get_output_spikes(spikes);

    portENTER_CRITICAL(&resultMux);
    result.prediction    = prediction;
    result.accel_x       = raw.accel_x;
    result.accel_y       = raw.accel_y;
    result.accel_z       = raw.accel_z;
    result.spike_normal  = spikes[0];
    result.spike_anomaly = spikes[1];
    result.inference_us  = inf_us;
    result.timestamp_ms  = millis();
    result.total_samples++;
    if (prediction == 1) result.anomaly_count++;
    portEXIT_CRITICAL(&resultMux);

    Serial.printf("[%lums] %s  AX=%.3f AY=%.3f AZ=%.3f  spk=[%d,%d]  %luus\n",
        (unsigned long)millis(),
        prediction ? "ANOMALY!" : "normal  ",
        raw.accel_x, raw.accel_y, raw.accel_z,
        (int)spikes[0], (int)spikes[1],
        (unsigned long)inf_us);

    if (prediction == 1) {
        digitalWrite(LED_PIN,    HIGH);
        digitalWrite(BUZZER_PIN, HIGH);
        delay(200);
        digitalWrite(BUZZER_PIN, LOW);
    } else {
        digitalWrite(LED_PIN, LOW);
    }

    delay(SAMPLE_INTERVAL_MS);
}
