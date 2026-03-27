/*
 * ESP32 + MPU6050 → USB Serial → Laptop
 * No WiFi, No Internet Required!
 */

#include <Wire.h>
#include <MPU6050.h>
#include <math.h>

// ==================== CONFIGURATION ====================
#define MPU_SDA 21
#define MPU_SCL 22
#define BAUD_RATE 115200
#define SAMPLE_INTERVAL_MS 500  // Send data every 500ms

// ==================== GLOBAL OBJECTS ====================
MPU6050 mpu;

// ==================== SETUP ====================
void setup() {
  Serial.begin(BAUD_RATE);
  
  // Initialize I2C
  Wire.begin(MPU_SDA, MPU_SCL);
  
  // Initialize MPU6050
  mpu.initialize();
  
  // Test connection
  if (mpu.testConnection()) {
    Serial.println("✅ MPU6050 connected");
  } else {
    Serial.println("❌ MPU6050 connection failed");
    while (1);  // Halt if sensor not found
  }
  
  // Configure MPU6050
  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);  // ±2g range
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);  // ±250°/s
  
  Serial.println("🚀 ESP32-MPU6050 Ready");
  Serial.println("Sending data via USB Serial...");
  // Fixed: Print 50 equal signs using a loop or String multiplication alternative
  for(int i = 0; i < 50; i++) {
    Serial.print("=");
  }
  Serial.println();  // New line after the separator
  delay(1000);
}

// ==================== MAIN LOOP ====================
void loop() {
  // Read accelerometer (convert to m/s²)
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getAcceleration(&ax, &ay, &az);
  mpu.getRotation(&gx, &gy, &gz);
  
  // Convert to physical units
  float accel_x = ax / 16384.0 * 9.81;  // m/s²
  float accel_y = ay / 16384.0 * 9.81;
  float accel_z = az / 16384.0 * 9.81;
  
  float gyro_x = gx / 131.0;  // °/s
  float gyro_y = gy / 131.0;
  float gyro_z = gz / 131.0;
  
  // Calculate acceleration magnitude (useful feature)
  float accel_mag = sqrt(accel_x*accel_x + accel_y*accel_y + accel_z*accel_z);
  
  // Send data as CSV format (easy to parse in Python)
  // Format: timestamp,ax,ay,az,gx,gy,gz,mag
  Serial.print(millis());
  Serial.print(",");
  Serial.print(accel_x, 4);
  Serial.print(",");
  Serial.print(accel_y, 4);
  Serial.print(",");
  Serial.print(accel_z, 4);
  Serial.print(",");
  Serial.print(gyro_x, 2);
  Serial.print(",");
  Serial.print(gyro_y, 2);
  Serial.print(",");
  Serial.print(gyro_z, 2);
  Serial.print(",");
  Serial.println(accel_mag, 4);
  
  // Wait for next sample
  delay(SAMPLE_INTERVAL_MS);
}