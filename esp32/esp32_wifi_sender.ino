// esp32_wifi_sender_debug.ino
#include <WiFi.h>
#include <Wire.h>
#include <MPU6050.h>

// ============================================================
// YOUR NETWORK DETAILS - UPDATE THESE!
// ============================================================
const char* ssid = "LAPTOP-Q5IS18KS 9743";        // Your phone's hotspot name
const char* password = "Q0#o0108";          // Your hotspot password

// Your PC's IP address - from the Python server output
const char* server_ip = "192.168.137.1";      // ← Your PC's IP from Python output
const int server_port = 8080;

// ============================================================

MPU6050 mpu;
WiFiClient client;
unsigned long lastReconnectAttempt = 0;
unsigned long lastDataSend = 0;
unsigned long lastStatusPrint = 0;

void setup() {
  Serial.begin(115200);
  Serial.println("\n\n=================================");
  Serial.println("ESP32 WiFi SNN Monitor (Debug)");
  Serial.println("=================================");
  
  // Initialize MPU6050
  Wire.begin();
  mpu.initialize();
  if (mpu.testConnection()) {
    Serial.println("✅ MPU6050 connected");
  } else {
    Serial.println("❌ MPU6050 not found - check wiring!");
  }
  
  // Connect to WiFi
  connectToWiFi();
}

void connectToWiFi() {
  Serial.print("\n📡 Connecting to: ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ WiFi connected!");
    Serial.print("   ESP32 IP: ");
    Serial.println(WiFi.localIP());
    Serial.print("   Gateway: ");
    Serial.println(WiFi.gatewayIP());
    Serial.print("   Signal strength: ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
    Serial.print("   Server IP: ");
    Serial.println(server_ip);
    
    // Test connectivity to server IP
    Serial.println("\n🔍 Testing network connectivity...");
    if (pingServer()) {
      Serial.println("✅ Server IP is reachable!");
    } else {
      Serial.println("❌ Cannot reach server IP - check network");
    }
  } else {
    Serial.println("\n❌ WiFi connection failed!");
    Serial.println("   Check password and hotspot is on");
  }
}

bool pingServer() {
  // Try to establish a connection to test reachability
  WiFiClient testClient;
  if (testClient.connect(server_ip, server_port)) {
    testClient.stop();
    return true;
  }
  return false;
}

void connectToServer() {
  int attempts = 0;
  while (!client.connected() && attempts < 20) {  // try for 60 seconds
    Serial.printf("🔌 Attempt %d: Connecting to %s:%d... ", attempts+1, server_ip, server_port);
    if (client.connect(server_ip, server_port)) {
      Serial.println("Connected!");
      return;
    }
    Serial.println("Failed, retrying in 3s...");
    attempts++;
    delay(3000);
  }
  if (!client.connected()) {
    Serial.println("❌ Could not connect after 20 attempts.");
  }
}
void loop() {
  // Reconnect WiFi if disconnected
  if (WiFi.status() != WL_CONNECTED) {
    if (millis() - lastReconnectAttempt > 5000) {
      lastReconnectAttempt = millis();
      connectToWiFi();
    }
    delay(100);
    return;
  }
  
  // Reconnect to server if needed
  if (!client.connected()) {
    if (millis() - lastReconnectAttempt > 3000) {
      lastReconnectAttempt = millis();
      connectToServer();
    }
    delay(100);
    return;
  }
  
  // Send data at ~20 Hz (every 50ms)
  if (millis() - lastDataSend >= 50) {
    lastDataSend = millis();
    
    // Read MPU6050
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    
    // Convert to m/s² (for ±2g range)
    float ax_ms2 = ax / 16384.0 * 9.81;
    float ay_ms2 = ay / 16384.0 * 9.81;
    float az_ms2 = az / 16384.0 * 9.81;
    
    // Calculate magnitude
    float mag = sqrt(ax_ms2*ax_ms2 + ay_ms2*ay_ms2 + az_ms2*az_ms2);
    
    // Format: timestamp,ax,ay,az,gx,gy,gz,mag
    String data = String(millis()) + "," +
                  String(ax_ms2, 2) + "," +
                  String(ay_ms2, 2) + "," +
                  String(az_ms2, 2) + "," +
                  "0,0,0," + String(mag, 2);
    
    // Send to server
    if (client.connected()) {
      client.println(data);
      
      // Print to serial every 20 samples for debugging
      static int count = 0;
      if (count++ % 20 == 0) {
        Serial.printf("📊 X=%6.2f Y=%6.2f Z=%6.2f Mag=%6.2f\n", 
                      ax_ms2, ay_ms2, az_ms2, mag);
      }
    } else {
      Serial.println("⚠️  Client disconnected, will reconnect");
      client.stop();
    }
  }
  
  // Print status every 10 seconds
  if (millis() - lastStatusPrint > 10000) {
    lastStatusPrint = millis();
    Serial.printf("📡 Status: Connected=%s, RSSI=%d dBm\n", 
                  client.connected() ? "Yes" : "No",
                  WiFi.RSSI());
  }
}

