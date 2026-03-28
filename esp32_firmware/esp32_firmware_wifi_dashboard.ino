/*
 * SNN-SHM ESP32 Firmware — WiFi Edition (Rich Dashboard)
 *
 * The ESP32 hosts a full visual dashboard at http://<ESP32-IP>/
 * Two endpoints:
 *   GET /        -> full HTML dashboard
 *   GET /data    -> JSON with latest inference result
 *
 * ── Setup ────────────────────────────────────────────────────────────────
 *  1. Fill in WIFI_SSID and WIFI_PASSWORD below.
 *  2. Flash as normal.
 *  3. Open Serial Monitor once to see the assigned IP address.
 *  4. Open http://<that-IP> in any browser on the same WiFi.
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Hardware: ESP32 + MPU6050
 */

#include <Wire.h>
#include <MPU6050.h>
#include <WiFi.h>
#include <WebServer.h>

// Custom SNN headers
#include "weights/snn_config.h"
#include "src/lif_neuron.h"
#include "src/snn_inference.h"
#include "src/feature_engineering.h"

// ==================== WiFi credentials ====================
#define WIFI_SSID     "LAPTOP-Q5IS18KS 9743"
#define WIFI_PASSWORD "Q0#o0108"

// ==================== Global objects ======================
MPU6050      mpu;
SNNInference snn;
WebServer    server(80);

// ==================== Shared inference result =============
struct InferenceResult {
    int       prediction;
    float     accel_x, accel_y, accel_z;
    int32_t   spike_normal, spike_anomaly;
    uint32_t  inference_us;
    uint32_t  timestamp_ms;
    uint32_t  total_samples;
    uint32_t  anomaly_count;
} result;

// ==================== Dashboard HTML (PROGMEM) ============
static const char DASHBOARD_HTML[] PROGMEM = R"rawhtml(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>SNN-SHM Monitor</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;700;900&display=swap" rel="stylesheet"/>
<style>
  :root{--bg:#050a0f;--surface:#080f18;--border:#0d2035;--accent:#00e5ff;--accent2:#00ff88;--warn:#ffaa00;--danger:#ff2244;--text:#c8dde8;--dim:#3a5a70;--glow:0 0 12px #00e5ff55;--glow-g:0 0 12px #00ff8855;--glow-r:0 0 20px #ff224488}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Share Tech Mono',monospace;min-height:100vh;overflow-x:hidden}
  body::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.12) 2px,rgba(0,0,0,0.12) 4px);pointer-events:none;z-index:999}
  header{display:flex;align-items:center;justify-content:space-between;padding:18px 28px;border-bottom:1px solid var(--border);background:linear-gradient(90deg,#050a0f 60%,#061520 100%);position:sticky;top:0;z-index:100}
  .logo{display:flex;align-items:center;gap:14px}
  .logo-icon{width:38px;height:38px;border:2px solid var(--accent);border-radius:6px;display:grid;place-items:center;box-shadow:var(--glow);animation:pulse-border 2.5s infinite}
  @keyframes pulse-border{0%,100%{box-shadow:0 0 8px #00e5ff44}50%{box-shadow:0 0 22px #00e5ffaa}}
  .logo-icon svg{width:22px;height:22px}
  .logo-text h1{font-family:'Exo 2',sans-serif;font-weight:900;font-size:1.1rem;color:var(--accent);letter-spacing:3px;text-transform:uppercase}
  .logo-text p{font-size:0.62rem;color:var(--dim);letter-spacing:2px}
  .header-right{display:flex;align-items:center;gap:24px}
  .status-pill{display:flex;align-items:center;gap:8px;padding:6px 14px;border:1px solid var(--border);border-radius:2px;font-size:0.72rem;letter-spacing:1.5px}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--accent2);box-shadow:var(--glow-g);animation:blink 1.2s infinite}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}
  #clock{font-size:0.78rem;color:var(--dim)}
  main{display:grid;grid-template-columns:280px 1fr;grid-template-rows:auto auto 1fr;gap:16px;padding:20px 24px;max-width:1440px;margin:0 auto}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:18px}
  .card-title{font-size:0.62rem;letter-spacing:3px;color:var(--dim);text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;gap:8px}
  .card-title::before{content:'';display:block;width:3px;height:12px;background:var(--accent);box-shadow:var(--glow)}
  .sidebar{grid-column:1;grid-row:1/4;display:flex;flex-direction:column;gap:14px}
  .big-stat{text-align:center;padding:22px 10px}
  .big-stat .value{font-family:'Exo 2',sans-serif;font-weight:900;font-size:3.2rem;line-height:1;color:var(--accent2);text-shadow:var(--glow-g);transition:color 0.3s}
  .big-stat.anomaly .value{color:var(--danger);text-shadow:var(--glow-r)}
  .big-stat .label{font-size:0.65rem;letter-spacing:2px;color:var(--dim);margin-top:6px}
  .metric-row{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid var(--border);font-size:0.75rem}
  .metric-row:last-child{border-bottom:none}
  .metric-row .mkey{color:var(--dim);letter-spacing:1px}
  .metric-row .mval{color:var(--accent);font-weight:bold}
  .gauge-wrap{position:relative;width:160px;height:80px;margin:10px auto}
  .gauge-wrap canvas{width:100%;height:100%}
  .gauge-pct{position:absolute;bottom:0;left:50%;transform:translateX(-50%);font-family:'Exo 2',sans-serif;font-weight:700;font-size:1.4rem;color:var(--accent);text-shadow:var(--glow)}
  .verdict{grid-column:2;grid-row:1;display:flex;align-items:center;justify-content:space-between;padding:20px 28px}
  .verdict-label{font-family:'Exo 2',sans-serif;font-weight:900;font-size:2rem;letter-spacing:6px;text-transform:uppercase;transition:color 0.3s,text-shadow 0.3s}
  .verdict-label.normal{color:var(--accent2);text-shadow:var(--glow-g)}
  .verdict-label.anomaly{color:var(--danger);text-shadow:var(--glow-r);animation:shake 0.4s}
  @keyframes shake{0%,100%{transform:translateX(0)}25%{transform:translateX(-5px)}75%{transform:translateX(5px)}}
  .verdict-meta{font-size:0.7rem;color:var(--dim);line-height:1.8;text-align:right}
  .verdict-meta span{color:var(--text)}
  .waves{grid-column:2;grid-row:2;display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
  .wave-card{overflow:hidden}
  .wave-card canvas{width:100%;height:80px;display:block}
  .log-area{grid-column:2;grid-row:3;display:grid;grid-template-columns:1fr 320px;gap:14px;min-height:0}
  .log-scroll{overflow-y:auto;max-height:340px;scrollbar-width:thin;scrollbar-color:var(--border) transparent}
  .log-row{display:grid;grid-template-columns:60px 110px 80px 80px 80px 80px 1fr;gap:6px;align-items:center;padding:6px 10px;border-bottom:1px solid #0a1c2a;font-size:0.7rem;transition:background 0.2s;animation:fadeIn 0.3s}
  @keyframes fadeIn{from{opacity:0;transform:translateY(-4px)}to{opacity:1}}
  .log-row:hover{background:#0a1828}
  .log-row.anomaly-row{background:#1a060a;border-left:2px solid var(--danger)}
  .log-row.normal-row{border-left:2px solid transparent}
  .log-header{display:grid;grid-template-columns:60px 110px 80px 80px 80px 80px 1fr;gap:6px;padding:5px 10px;font-size:0.6rem;letter-spacing:1.5px;color:var(--dim);border-bottom:1px solid var(--border);background:#050a0f;position:sticky;top:0;z-index:1}
  .tag{padding:2px 7px;border-radius:2px;font-size:0.62rem;letter-spacing:1px;font-weight:bold;text-align:center}
  .tag.n{background:#002d20;color:var(--accent2)}
  .tag.a{background:#2d0010;color:var(--danger)}
  .prob-bar-wrap{position:relative;height:10px;background:#0d2035;border-radius:2px;overflow:hidden}
  .prob-bar{height:100%;border-radius:2px;transition:width 0.3s}
  .spike-card{overflow:hidden}
  .spike-card canvas{width:100%;height:100%;display:block}
  .controls{display:flex;gap:10px;align-items:center}
  .btn{padding:8px 18px;border:1px solid var(--accent);background:transparent;color:var(--accent);font-family:'Share Tech Mono',monospace;font-size:0.72rem;letter-spacing:2px;cursor:pointer;border-radius:2px;transition:background 0.2s,box-shadow 0.2s;text-transform:uppercase}
  .btn:hover{background:#00e5ff18;box-shadow:var(--glow)}
  .btn.active{background:#00e5ff25}
  .btn.danger{border-color:var(--danger);color:var(--danger)}
  .btn.danger:hover{background:#ff224418}
  .speed-label{font-size:0.65rem;color:var(--dim)}
  select{background:var(--surface);border:1px solid var(--border);color:var(--text);font-family:'Share Tech Mono',monospace;font-size:0.72rem;padding:6px 10px;border-radius:2px;cursor:pointer}
  .connection-status{position:fixed;bottom:20px;right:20px;padding:8px 16px;border-radius:4px;font-size:0.7rem;background:var(--surface);border:1px solid var(--border);z-index:1000;display:flex;align-items:center;gap:8px}
  .connection-status.connected{border-color:var(--accent2);color:var(--accent2)}
  .connection-status.disconnected{border-color:var(--danger);color:var(--danger)}
  .connection-status.connecting{border-color:var(--warn);color:var(--warn)}
  .esp-info{position:fixed;bottom:20px;left:20px;padding:8px 16px;border-radius:4px;font-size:0.7rem;background:var(--surface);border:1px solid var(--border);z-index:1000;color:var(--accent)}
  @media(max-width:900px){
    main{grid-template-columns:1fr}
    .sidebar{grid-column:1;grid-row:auto}
    .verdict,.waves,.log-area{grid-column:1;grid-row:auto}
    .waves{grid-template-columns:1fr}
    .log-area{grid-template-columns:1fr}
  }
</style>
</head>
<body>
<header>
  <div class="logo">
    <div class="logo-icon">
      <svg viewBox="0 0 24 24" fill="none" stroke="#00e5ff" stroke-width="1.8">
        <path d="M2 12h3l3-7 4 14 3-9 2 2h5"/>
        <circle cx="18" cy="6" r="3" fill="#00e5ff"/>
      </svg>
    </div>
    <div class="logo-text">
      <h1>SNN-SHM Monitor</h1>
      <p>ESP32 · WiFi · LIF-SNN · EDGE AI</p>
    </div>
  </div>
  <div class="header-right">
    <div class="controls">
      <span class="speed-label">SPEED</span>
      <select id="speedSel">
        <option value="200">0.5x</option>
        <option value="100" selected>1x</option>
        <option value="50">2x</option>
        <option value="20">5x</option>
      </select>
      <button class="btn" id="pauseBtn">PAUSE</button>
      <button class="btn danger" id="clearBtn">CLEAR LOG</button>
    </div>
    <div class="status-pill"><div class="dot" id="statusDot"></div><span id="statusTxt">CONNECTING...</span></div>
    <div id="clock"></div>
  </div>
</header>

<main>
  <aside class="sidebar">
    <div class="card big-stat" id="verdictCard">
      <div class="value" id="bigVerdict">-</div>
      <div class="label">PREDICTION</div>
    </div>
    <div class="card">
      <div class="card-title">ANOMALY PROBABILITY</div>
      <div class="gauge-wrap">
        <canvas id="gaugeCanvas" width="320" height="160"></canvas>
        <div class="gauge-pct" id="gaugePct">0%</div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">SESSION STATS</div>
      <div class="metric-row"><span class="mkey">SAMPLES</span><span class="mval" id="mTotal">0</span></div>
      <div class="metric-row"><span class="mkey">ANOMALIES</span><span class="mval" id="mAnom">0</span></div>
      <div class="metric-row"><span class="mkey">ANOM RATE</span><span class="mval" id="mRate">0.0%</span></div>
      <div class="metric-row"><span class="mkey">SAMPLE RATE</span><span class="mval" id="mHz">-</span></div>
      <div class="metric-row"><span class="mkey">INFER TIME</span><span class="mval" id="mInfer">-</span></div>
      <div class="metric-row"><span class="mkey">SPIKES [N,A]</span><span class="mval" id="mSpikes">-</span></div>
      <div class="metric-row"><span class="mkey">TIME STEPS</span><span class="mval">120</span></div>
    </div>
    <div class="card">
      <div class="card-title">RAW FEATURES</div>
      <div class="metric-row"><span class="mkey">AX</span><span class="mval" id="fAX">-</span></div>
      <div class="metric-row"><span class="mkey">AY</span><span class="mval" id="fAY">-</span></div>
      <div class="metric-row"><span class="mkey">AZ</span><span class="mval" id="fAZ">-</span></div>
      <div class="metric-row"><span class="mkey">MAG</span><span class="mval" id="fMAG">-</span></div>
      <div class="metric-row"><span class="mkey">SUM_ABS</span><span class="mval" id="fSUM">-</span></div>
    </div>
  </aside>

  <div class="card verdict">
    <div>
      <div class="verdict-label normal" id="verdictLabel">AWAITING DATA</div>
      <div style="font-size:0.65rem;color:var(--dim);margin-top:6px;letter-spacing:2px;">LIF-SNN · SPIKE RATE THRESHOLD >= 0.800</div>
    </div>
    <div class="verdict-meta">
      SAMPLE <span id="vSample">-</span><br>
      P(NORMAL) <span id="vNorm">-</span><br>
      P(ANOMALY) <span id="vAnom">-</span><br>
      ELAPSED <span id="vElapsed">-</span>
    </div>
  </div>

  <div class="waves">
    <div class="card wave-card"><div class="card-title">ACCEL X</div><canvas id="waveAX" width="600" height="80"></canvas></div>
    <div class="card wave-card"><div class="card-title">ACCEL Y</div><canvas id="waveAY" width="600" height="80"></canvas></div>
    <div class="card wave-card"><div class="card-title">ACCEL Z / MAGNITUDE</div><canvas id="waveAZ" width="600" height="80"></canvas></div>
  </div>

  <div class="log-area">
    <div class="card">
      <div class="card-title">PREDICTION LOG</div>
      <div class="log-header">
        <span>#</span><span>TIME</span><span>STATUS</span>
        <span>AX</span><span>AY</span><span>AZ</span><span>P(ANOM)</span>
      </div>
      <div class="log-scroll" id="logScroll"></div>
    </div>
    <div class="card spike-card">
      <div class="card-title">SPIKE RASTER</div>
      <canvas id="spikeCanvas" width="300" height="280"></canvas>
    </div>
  </div>
</main>

<div class="connection-status disconnected" id="connStatus">
  <span class="dot" id="connDot"></span><span id="connText">Disconnected</span>
</div>
<div class="esp-info">ESP32: <span id="espIp">-</span></div>

<script>
const POLL_MS=500,WAVE_LEN=120,NEURONS=32,SPIKE_HISTORY=40,MAX_LOG=200;
let count=0,anomalyCount=0,startTime=Date.now(),paused=false,spikeInterval=null;
let lastTimestamp=0,lastHzTime=Date.now(),lastHz=0;
const waveData={ax:new Array(WAVE_LEN).fill(0),ay:new Array(WAVE_LEN).fill(0),az:new Array(WAVE_LEN).fill(0),mag:new Array(WAVE_LEN).fill(0)};
const spikes=Array.from({length:NEURONS},()=>new Array(SPIKE_HISTORY).fill(0));

function updateClock(){const n=new Date(),p=x=>String(x).padStart(2,'0'),el=document.getElementById('clock');if(el)el.textContent=`${p(n.getHours())}:${p(n.getMinutes())}:${p(n.getSeconds())}`;}
setInterval(updateClock,1000);updateClock();

function setText(id,val){const el=document.getElementById(id);if(el)el.textContent=val;}

function setConnStatus(s){
  const el=document.getElementById('connStatus'),dot=document.getElementById('connDot'),txt=document.getElementById('connText');
  el.className='connection-status '+s;
  if(s==='connected'){dot.style.background='var(--accent2)';dot.style.boxShadow='var(--glow-g)';txt.textContent='Connected to ESP32';setText('statusTxt','LIVE · CONNECTED');}
  else if(s==='connecting'){dot.style.background='var(--warn)';dot.style.boxShadow='none';txt.textContent='Connecting...';}
  else{dot.style.background='var(--danger)';dot.style.boxShadow='var(--glow-r)';txt.textContent='Disconnected';setText('statusTxt','OFFLINE');}
}

async function poll(){
  if(paused)return;
  try{
    const r=await fetch('/data');
    if(!r.ok)throw new Error('HTTP '+r.status);
    const d=await r.json();
    if(d.timestamp_ms===lastTimestamp)return;
    lastTimestamp=d.timestamp_ms;
    setConnStatus('connected');
    document.getElementById('espIp').textContent=window.location.hostname;
    count++;
    const isAnom=d.prediction===1;
    if(isAnom)anomalyCount++;
    const pAnom=d.spike_anomaly/120.0,pNorm=d.spike_normal/120.0;
    const mag=Math.sqrt(d.accel_x**2+d.accel_y**2+d.accel_z**2);
    const sumAbs=Math.abs(d.accel_x)+Math.abs(d.accel_y)+Math.abs(d.accel_z);
    const now=Date.now();lastHz=1000/(now-lastHzTime);lastHzTime=now;

    const vl=document.getElementById('verdictLabel'),vc=document.getElementById('verdictCard'),bv=document.getElementById('bigVerdict');
    vl.textContent=isAnom?'ANOMALY DETECTED':'NORMAL';
    vl.className='verdict-label '+(isAnom?'anomaly':'normal');
    if(isAnom){vl.style.animation='none';void vl.offsetWidth;vl.style.animation='shake 0.4s';}
    bv.textContent=isAnom?'ANOM':'OK';
    vc.className='card big-stat'+(isAnom?' anomaly':'');

    setText('vNorm',(pNorm*100).toFixed(1)+'%');
    setText('vAnom',(pAnom*100).toFixed(1)+'%');
    setText('vSample',count);
    setText('vElapsed',((Date.now()-startTime)/1000).toFixed(1)+'s');
    drawGauge(pAnom);
    const gpEl=document.getElementById('gaugePct');
    gpEl.textContent=(pAnom*100).toFixed(0)+'%';
    gpEl.style.color=isAnom?'var(--danger)':(pAnom>0.3?'var(--warn)':'var(--accent2)');

    setText('mTotal',count);
    setText('mAnom',anomalyCount);
    setText('mRate',(anomalyCount/count*100).toFixed(1)+'%');
    setText('mHz',lastHz.toFixed(2)+' Hz');
    setText('mInfer',d.inference_us+' us');
    setText('mSpikes','['+d.spike_normal+', '+d.spike_anomaly+']');
    setText('fAX',d.accel_x.toFixed(3));
    setText('fAY',d.accel_y.toFixed(3));
    setText('fAZ',d.accel_z.toFixed(3));
    setText('fMAG',mag.toFixed(3));
    setText('fSUM',sumAbs.toFixed(3));

    waveData.ax.shift();waveData.ax.push(d.accel_x);
    waveData.ay.shift();waveData.ay.push(d.accel_y);
    waveData.az.shift();waveData.az.push(d.accel_z);
    waveData.mag.shift();waveData.mag.push(mag);
    drawWave('waveAX',waveData.ax,'#00e5ff',-15,15);
    drawWave('waveAY',waveData.ay,'#00ff88',-15,15);
    drawWave('waveAZ',waveData.az,'#7755ff',0,20,waveData.mag,'#ffaa00');

    const nf=[(d.accel_x+15)/30,(d.accel_y+15)/30,d.accel_z/20,mag/25,sumAbs/30];
    updateSpikes(nf);drawSpikes();
    addLogRow(count,d.accel_x,d.accel_y,d.accel_z,pAnom,d.prediction);
  }catch(e){setConnStatus('disconnected');}
}

function drawWave(id,data,color,minV,maxV,extraData,extraColor){
  const c=document.getElementById(id);if(!c)return;
  const ctx=c.getContext('2d'),W=c.width,H=c.height;
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle='#0d2035';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){const y=(i/4)*H;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
  const zeroY=H-((0-minV)/(maxV-minV))*H;
  ctx.strokeStyle='#1a3050';ctx.beginPath();ctx.moveTo(0,zeroY);ctx.lineTo(W,zeroY);ctx.stroke();
  const drawLine=(arr,col)=>{
    ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.shadowColor=col;ctx.shadowBlur=6;ctx.beginPath();
    arr.forEach((v,i)=>{const x=(i/(arr.length-1))*W,y=H-((v-minV)/(maxV-minV))*H;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});
    ctx.stroke();ctx.shadowBlur=0;
  };
  drawLine(data,color);if(extraData)drawLine(extraData,extraColor);
}

function drawGauge(prob){
  const c=document.getElementById('gaugeCanvas');if(!c)return;
  const ctx=c.getContext('2d'),W=c.width,H=c.height;
  ctx.clearRect(0,0,W,H);
  const cx=W/2,cy=H-10,r=110;
  ctx.beginPath();ctx.arc(cx,cy,r,Math.PI,2*Math.PI);ctx.strokeStyle='#0d2035';ctx.lineWidth=18;ctx.stroke();
  const col=prob<0.5?`hsl(${150-prob*100},100%,55%)`:`hsl(${50-(prob-0.5)*100},100%,55%)`;
  ctx.beginPath();ctx.arc(cx,cy,r,Math.PI,Math.PI+prob*Math.PI);ctx.strokeStyle=col;ctx.lineWidth=18;ctx.shadowColor=col;ctx.shadowBlur=16;ctx.stroke();ctx.shadowBlur=0;
  for(let i=0;i<=10;i++){const a=Math.PI+(i/10)*Math.PI;ctx.beginPath();ctx.moveTo(cx+(r-24)*Math.cos(a),cy+(r-24)*Math.sin(a));ctx.lineTo(cx+(r-12)*Math.cos(a),cy+(r-12)*Math.sin(a));ctx.strokeStyle='#1a3050';ctx.lineWidth=1.5;ctx.stroke();}
}

function updateSpikes(nf){spikes.forEach((row,n)=>{row.shift();const p=nf[n%nf.length]+Math.random()*0.15;row.push(Math.random()<p?1:0);});}
function drawSpikes(){
  const c=document.getElementById('spikeCanvas');if(!c)return;
  const ctx=c.getContext('2d'),W=c.width,H=c.height;
  ctx.clearRect(0,0,W,H);ctx.fillStyle='#050a0f';ctx.fillRect(0,0,W,H);
  const rowH=H/NEURONS,colW=W/SPIKE_HISTORY;
  spikes.forEach((row,n)=>{row.forEach((s,t)=>{if(s){ctx.fillStyle=n<10?'#00e5ff':n<21?'#00ff88':'#ffaa00';ctx.shadowColor=ctx.fillStyle;ctx.shadowBlur=3;ctx.fillRect(t*colW+1,n*rowH+1,colW-1,rowH-1);}});});
  ctx.shadowBlur=0;
}

function addLogRow(n,ax,ay,az,pAnom,pred){
  const scroll=document.getElementById('logScroll');if(!scroll)return;
  const now=new Date(),p=x=>String(x).padStart(2,'0');
  const ts=`${p(now.getHours())}:${p(now.getMinutes())}:${p(now.getSeconds())}`;
  const isAnom=pred===1;
  const row=document.createElement('div');
  row.className='log-row '+(isAnom?'anomaly-row':'normal-row');
  row.innerHTML=`<span style="color:var(--dim)">${n}</span><span style="color:var(--dim);font-size:0.62rem">${ts}</span><span><span class="tag ${isAnom?'a':'n'}">${isAnom?'ANOMALY':'NORMAL'}</span></span><span style="color:${ax>=0?'#00e5ff':'#ff8888'}">${ax.toFixed(2)}</span><span style="color:${ay>=0?'#00e5ff':'#ff8888'}">${ay.toFixed(2)}</span><span style="color:${az>=0?'#00e5ff':'#ff8888'}">${az.toFixed(2)}</span><div><div style="font-size:0.65rem;margin-bottom:3px;color:${isAnom?'var(--danger)':'var(--accent2)'}">${(pAnom*100).toFixed(1)}%</div><div class="prob-bar-wrap"><div class="prob-bar" style="width:${pAnom*100}%;background:${isAnom?'var(--danger)':'var(--accent2)'}"></div></div></div>`;
  scroll.prepend(row);
  while(scroll.children.length>MAX_LOG)scroll.removeChild(scroll.lastChild);
}

document.getElementById('pauseBtn').addEventListener('click',function(){
  paused=!paused;this.textContent=paused?'RESUME':'PAUSE';this.classList.toggle('active',paused);
  setText('statusTxt',paused?'PAUSED':'LIVE · CONNECTED');
});
document.getElementById('clearBtn').addEventListener('click',()=>{
  const s=document.getElementById('logScroll');if(s)s.innerHTML='';
  count=0;anomalyCount=0;startTime=Date.now();
  Object.values(waveData).forEach(a=>a.fill(0));
  spikes.forEach(r=>r.fill(0));
});
document.getElementById('speedSel').addEventListener('change',function(){
  clearInterval(spikeInterval);
  spikeInterval=setInterval(()=>{if(!paused)drawSpikes();},parseInt(this.value));
});

setConnStatus('connecting');
setInterval(poll,POLL_MS);
spikeInterval=setInterval(()=>{if(!paused)drawSpikes();},100);
poll();
</script>
</body>
</html>
)rawhtml";

// ==================== Web server handlers =====================

void handleRoot() {
    server.send_P(200, "text/html", DASHBOARD_HTML);
}

void handleData() {
    char json[320];
    snprintf(json, sizeof(json),
        "{"
          "\"prediction\":%d,"
          "\"accel_x\":%.4f,"
          "\"accel_y\":%.4f,"
          "\"accel_z\":%.4f,"
          "\"spike_normal\":%ld,"
          "\"spike_anomaly\":%ld,"
          "\"inference_us\":%lu,"
          "\"timestamp_ms\":%lu,"
          "\"total_samples\":%lu,"
          "\"anomaly_count\":%lu"
        "}",
        result.prediction,
        result.accel_x, result.accel_y, result.accel_z,
        result.spike_normal, result.spike_anomaly,
        result.inference_us, result.timestamp_ms,
        (unsigned long)result.total_samples,
        (unsigned long)result.anomaly_count
    );
    server.send(200, "application/json", json);
}

void handleNotFound() {
    server.send(404, "text/plain", "Not found");
}

// ==================== WiFi connection helper ==================

void connectWiFi() {
    Serial.print("Connecting to WiFi: ");
    Serial.println(WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED) {
        delay(500); Serial.print(".");
        digitalWrite(LED_PIN, !digitalRead(LED_PIN));
        if (++attempts > 40) {
            Serial.println("\nWiFi failed - continuing without network.");
            digitalWrite(LED_PIN, LOW); return;
        }
    }
    digitalWrite(LED_PIN, LOW);
    Serial.println();
    Serial.print("WiFi connected!  Open in browser:  http://");
    Serial.println(WiFi.localIP());
}

// ==================== Setup ===================================

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
        Serial.println("MPU6050 connected");
    } else {
        Serial.println("MPU6050 connection failed!");
        while (1) { digitalWrite(LED_PIN, HIGH); delay(500); digitalWrite(LED_PIN, LOW); delay(500); }
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
    Serial.println("==================================================");
    Serial.println("SNN-SHM Edge Node Ready");
    Serial.print("Architecture: ");
    Serial.print(INPUT_SIZE);    Serial.print(" -> ");
    Serial.print(HIDDEN_1_SIZE); Serial.print(" -> ");
    Serial.print(HIDDEN_2_SIZE); Serial.print(" -> ");
    Serial.print(HIDDEN_3_SIZE); Serial.print(" -> ");
    Serial.println(OUTPUT_SIZE);
    Serial.print("Time Steps: "); Serial.println(TIME_STEPS);
    Serial.print("Threshold:  "); Serial.println(ANOMALY_THRESHOLD / 32768.0f);
    Serial.println("==================================================");
    delay(1000);
}

// ==================== Main loop ===============================

void loop() {
    server.handleClient();

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

    Serial.print("["); Serial.print(millis()); Serial.print("ms] ");
    if (prediction == 1) {
        Serial.println("ANOMALY DETECTED!");
        digitalWrite(LED_PIN,    HIGH);
        digitalWrite(BUZZER_PIN, HIGH);
        delay(200);
        digitalWrite(BUZZER_PIN, LOW);
    } else {
        Serial.println("Normal");
        digitalWrite(LED_PIN, LOW);
    }
    Serial.print("  Accel: [");
    Serial.print(raw.accel_x,3); Serial.print(", ");
    Serial.print(raw.accel_y,3); Serial.print(", ");
    Serial.print(raw.accel_z,3); Serial.println("] m/s2");
    Serial.print("  Inference Time: "); Serial.print(inf_us); Serial.println(" us");
    Serial.print("  Output Spikes: [");
    Serial.print(spikes[0]); Serial.print(", "); Serial.print(spikes[1]); Serial.println("]");
    Serial.println();

    unsigned long wait_until = millis() + SAMPLE_INTERVAL_MS;
    while (millis() < wait_until) { server.handleClient(); delay(5); }
}
