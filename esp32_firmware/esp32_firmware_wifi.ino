/*
 * SNN-SHM ESP32 Firmware — WiFi Edition
 *
 * Replaces the USB serial monitor with a live browser dashboard.
 * The ESP32 hosts a tiny web server on your local WiFi network.
 * Open  http://<ESP32-IP>  in any browser to watch live readings.
 *
 * Two endpoints:
 *   GET /        → the HTML dashboard page (auto-refreshes every 500 ms)
 *   GET /data    → JSON snapshot of the latest inference result
 *
 * No external broker, cloud service, or extra software needed.
 *
 * ── Setup ────────────────────────────────────────────────────────────────
 *  1. Fill in WIFI_SSID and WIFI_PASSWORD below.
 *  2. Flash as normal.
 *  3. Open the Serial Monitor once to see the IP address printed on boot.
 *  4. Type that IP into any browser on the same WiFi — done.
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
// Change these to match your network before flashing.
#define WIFI_SSID     ""
#define WIFI_PASSWORD ""

// ==================== Global objects ======================
MPU6050      mpu;
SNNInference snn;
WebServer    server(80);

// ==================== Latest result (shared with web handlers) ===========
struct InferenceResult {
    int       prediction;       // 0 = Normal, 1 = Anomaly
    float     accel_x;
    float     accel_y;
    float     accel_z;
    int32_t   spike_normal;     // output_spikes[0]
    int32_t   spike_anomaly;    // output_spikes[1]
    uint32_t  inference_us;     // inference time in microseconds
    uint32_t  timestamp_ms;     // millis() at inference
} result;

// ==================== HTML dashboard ==========================
// Served once at GET /  — JavaScript polls GET /data every 500 ms
// and updates the page without a full reload.
static const char DASHBOARD_HTML[] PROGMEM = R"rawhtml(
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SNN-SHM Monitor</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Courier New', monospace;
      background: #0d1117;
      color: #c9d1d9;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 24px 16px;
    }
    h1 { color: #58a6ff; font-size: 1.4rem; margin-bottom: 4px; }
    .subtitle { color: #8b949e; font-size: 0.8rem; margin-bottom: 24px; }
    .card {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 20px 28px;
      width: 100%;
      max-width: 480px;
      margin-bottom: 16px;
    }
    .status {
      font-size: 1.6rem;
      font-weight: bold;
      text-align: center;
      padding: 12px;
      border-radius: 8px;
      margin-bottom: 8px;
    }
    .status.normal  { background:#0d2e1a; color:#3fb950; }
    .status.anomaly { background:#2e0d0d; color:#f85149; animation: pulse 0.6s infinite alternate; }
    @keyframes pulse { from { opacity:1; } to { opacity:0.5; } }
    .row {
      display: flex;
      justify-content: space-between;
      padding: 6px 0;
      border-bottom: 1px solid #21262d;
      font-size: 0.9rem;
    }
    .row:last-child { border-bottom: none; }
    .label { color: #8b949e; }
    .value { color: #e6edf3; }
    .log-box {
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 8px;
      width: 100%;
      max-width: 480px;
      height: 260px;
      overflow-y: auto;
      padding: 10px 14px;
      font-size: 0.78rem;
      color: #8b949e;
    }
    .log-box .entry { margin-bottom: 4px; line-height: 1.5; }
    .log-box .entry.anomaly { color: #f85149; }
    .log-box .entry.normal  { color: #3fb950; }
    #conn { font-size:0.75rem; color:#8b949e; margin-top:8px; }
  </style>
</head>
<body>
  <h1>SNN-SHM Edge Monitor</h1>
  <p class="subtitle">ESP32 · LIF Spiking Neural Network · Live</p>

  <div class="card">
    <div id="status" class="status normal">NORMAL</div>
    <div class="row"><span class="label">Accel X</span><span class="value" id="ax">--</span></div>
    <div class="row"><span class="label">Accel Y</span><span class="value" id="ay">--</span></div>
    <div class="row"><span class="label">Accel Z</span><span class="value" id="az">--</span></div>
    <div class="row"><span class="label">Output Spikes</span><span class="value" id="spikes">--</span></div>
    <div class="row"><span class="label">Inference Time</span><span class="value" id="itime">--</span></div>
    <div class="row"><span class="label">Uptime</span><span class="value" id="uptime">--</span></div>
  </div>

  <div class="log-box" id="log"></div>
  <div id="conn">Connecting...</div>

  <script>
    const MAX_LOG = 80;
    let entries = [];

    function fmtUptime(ms) {
      let s = Math.floor(ms / 1000);
      let m = Math.floor(s / 60); s %= 60;
      let h = Math.floor(m / 60); m %= 60;
      return (h ? h + 'h ' : '') + (m ? m + 'm ' : '') + s + 's';
    }

    async function poll() {
      try {
        const r = await fetch('/data');
        if (!r.ok) throw new Error('HTTP ' + r.status);
        const d = await r.json();

        const isAnomaly = d.prediction === 1;
        const statusEl = document.getElementById('status');
        statusEl.textContent = isAnomaly ? 'ANOMALY DETECTED' : 'NORMAL';
        statusEl.className   = 'status ' + (isAnomaly ? 'anomaly' : 'normal');

        document.getElementById('ax').textContent     = d.accel_x.toFixed(3) + ' m/s2';
        document.getElementById('ay').textContent     = d.accel_y.toFixed(3) + ' m/s2';
        document.getElementById('az').textContent     = d.accel_z.toFixed(3) + ' m/s2';
        document.getElementById('spikes').textContent = '[' + d.spike_normal + ', ' + d.spike_anomaly + ']';
        document.getElementById('itime').textContent  = d.inference_us + ' us';
        document.getElementById('uptime').textContent = fmtUptime(d.timestamp_ms);

        const line = '[' + fmtUptime(d.timestamp_ms) + '] '
          + (isAnomaly ? 'ANOMALY' : 'Normal')
          + '  ax=' + d.accel_x.toFixed(2)
          + ' ay=' + d.accel_y.toFixed(2)
          + ' az=' + d.accel_z.toFixed(2)
          + '  spikes=[' + d.spike_normal + ',' + d.spike_anomaly + ']'
          + '  ' + d.inference_us + 'us';

        entries.unshift({ text: line, anomaly: isAnomaly });
        if (entries.length > MAX_LOG) entries.pop();

        const box = document.getElementById('log');
        box.innerHTML = entries.map(e =>
          '<div class="entry ' + (e.anomaly ? 'anomaly' : 'normal') + '">'
          + e.text + '</div>'
        ).join('');

        document.getElementById('conn').textContent =
          'Connected - last update ' + new Date().toLocaleTimeString();
      } catch(e) {
        document.getElementById('conn').textContent =
          'Connection lost - retrying... (' + e.message + ')';
      }
    }

    poll();
    setInterval(poll, 500);
  </script>
</body>
</html>
)rawhtml";

// ==================== Web server handlers =====================

void handleRoot() {
    server.send_P(200, "text/html", DASHBOARD_HTML);
}

void handleData() {
    char json[256];
    snprintf(json, sizeof(json),
        "{"
          "\"prediction\":%d,"
          "\"accel_x\":%.4f,"
          "\"accel_y\":%.4f,"
          "\"accel_z\":%.4f,"
          "\"spike_normal\":%ld,"
          "\"spike_anomaly\":%ld,"
          "\"inference_us\":%lu,"
          "\"timestamp_ms\":%lu"
        "}",
        result.prediction,
        result.accel_x, result.accel_y, result.accel_z,
        result.spike_normal, result.spike_anomaly,
        result.inference_us, result.timestamp_ms
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
        delay(500);
        Serial.print(".");
        digitalWrite(LED_PIN, !digitalRead(LED_PIN));   // blink while connecting
        if (++attempts > 40) {                           // 20 s timeout
            Serial.println("\nWiFi failed - continuing without network.");
            digitalWrite(LED_PIN, LOW);
            return;
        }
    }
    digitalWrite(LED_PIN, LOW);
    Serial.println();
    Serial.print("WiFi connected! Open this in your browser:  http://");
    Serial.println(WiFi.localIP());
}

// ==================== Setup ===================================

void setup() {
    Serial.begin(115200);
    pinMode(LED_PIN,    OUTPUT);
    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(LED_PIN,    LOW);
    digitalWrite(BUZZER_PIN, LOW);

    // I2C + MPU6050
    Wire.begin(MPU_SDA, MPU_SCL);
    Serial.println("Initializing MPU6050...");
    mpu.initialize();

    if (mpu.testConnection()) {
        Serial.println("MPU6050 connected");
    } else {
        Serial.println("MPU6050 connection failed!");
        while (1) {
            digitalWrite(LED_PIN, HIGH); delay(500);
            digitalWrite(LED_PIN, LOW);  delay(500);
        }
    }
    mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);
    mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);

    // WiFi + web server
    connectWiFi();
    server.on("/",     handleRoot);
    server.on("/data", handleData);
    server.onNotFound(handleNotFound);
    server.begin();
    Serial.println("Web server started");

    // Boot info
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
    // Handle any pending web requests immediately
    server.handleClient();

    // ── Read accelerometer ────────────────────────────────────
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getAcceleration(&ax, &ay, &az);
    mpu.getRotation(&gx, &gy, &gz);

    RawSensorData raw;
    raw.accel_x = ax / 16384.0f * 9.81f;
    raw.accel_y = ay / 16384.0f * 9.81f;
    raw.accel_z = az / 16384.0f * 9.81f;
    raw.strain  = 85.0f;
    raw.temp    = 25.0f;

    // ── Feature engineering ───────────────────────────────────
    int16_t features[INPUT_SIZE];
    compute_features(raw, features);

    // ── SNN inference ─────────────────────────────────────────
    unsigned long t0       = micros();
    int prediction         = snn.predict(features);
    unsigned long inf_us   = micros() - t0;

    int32_t spikes[OUTPUT_SIZE];
    snn.get_output_spikes(spikes);

    // ── Update result struct (read by /data handler) ──────────
    result.prediction    = prediction;
    result.accel_x       = raw.accel_x;
    result.accel_y       = raw.accel_y;
    result.accel_z       = raw.accel_z;
    result.spike_normal  = spikes[0];
    result.spike_anomaly = spikes[1];
    result.inference_us  = inf_us;
    result.timestamp_ms  = millis();

    // ── Serial output (identical to original) ─────────────────
    Serial.print("[");
    Serial.print(millis());
    Serial.print("ms] ");

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
    Serial.print(raw.accel_x, 3); Serial.print(", ");
    Serial.print(raw.accel_y, 3); Serial.print(", ");
    Serial.print(raw.accel_z, 3); Serial.println("] m/s2");

    Serial.print("  Inference Time: ");
    Serial.print(inf_us);
    Serial.println(" us");

    Serial.print("  Output Spikes: [");
    Serial.print(spikes[0]); Serial.print(", ");
    Serial.print(spikes[1]); Serial.println("]");
    Serial.println();

    // ── Delay — keep calling handleClient() so the server stays
    //    responsive during the 500 ms wait between samples ─────
    unsigned long wait_until = millis() + SAMPLE_INTERVAL_MS;
    while (millis() < wait_until) {
        server.handleClient();
        delay(5);
    }
}
