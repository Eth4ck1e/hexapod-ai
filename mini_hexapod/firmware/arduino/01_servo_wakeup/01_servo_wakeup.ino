// mini_hexapod stage 1: front-right leg servo wakeup.
//
// Hardware: Seeed XIAO ESP32-C3 + PCA9685 board @ 0x40 (right side, no addr jumpers).
// Wiring:   XIAO 3V3 -> PCA9685 VCC, GND -> GND, D4(GPIO6) -> SDA, D5(GPIO7) -> SCL.
//           External battery -> PCA9685 V+ (servo rail, isolated from logic 3V3).
//           Battery GND must tie to XIAO GND for common reference.
// Servos:   ch0=FR coxa, ch1=FR femur, ch2=FR knee.
// Behavior: centers all three, then sweeps each one +/-20 deg from center and returns.
//           One-shot in setup(); press reset to re-run.
//
// Library: "Adafruit PWM Servo Driver Library" via Arduino IDE Library Manager.

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm(0x40);

constexpr uint16_t SERVO_MIN_US = 500;    // approx 0 deg
constexpr uint16_t SERVO_MAX_US = 2500;   // approx 180 deg
constexpr uint16_t SERVO_CTR_US = 1500;   // approx 90 deg, treated as joint zero
constexpr uint8_t  PWM_FREQ_HZ  = 50;
constexpr float    SWEEP_DEG    = 20.0f;  // conservative for assembled leg

constexpr uint8_t CH_COXA  = 0;
constexpr uint8_t CH_FEMUR = 1;
constexpr uint8_t CH_KNEE  = 2;

uint16_t usToTicks(uint16_t us) {
  return (uint16_t)((uint32_t)us * PWM_FREQ_HZ * 4096UL / 1000000UL);
}

uint16_t degToUs(float deg) {
  deg = constrain(deg, -90.0f, 90.0f);
  return (uint16_t)(SERVO_CTR_US + deg * ((float)(SERVO_MAX_US - SERVO_MIN_US) / 180.0f));
}

void setServoDeg(uint8_t ch, float deg) {
  pwm.setPWM(ch, 0, usToTicks(degToUs(deg)));
}

void sweepChannel(uint8_t ch, const char* name) {
  Serial.print("Sweeping "); Serial.print(name);
  Serial.print(" (ch "); Serial.print(ch); Serial.println(") +/-20 deg");
  setServoDeg(ch, 0.0f);
  delay(500);
  for (float d = 0;          d >= -SWEEP_DEG; d -= 1.0f) { setServoDeg(ch, d); delay(20); }
  delay(300);
  for (float d = -SWEEP_DEG; d <=  SWEEP_DEG; d += 1.0f) { setServoDeg(ch, d); delay(20); }
  delay(300);
  for (float d =  SWEEP_DEG; d >=  0.0f;      d -= 1.0f) { setServoDeg(ch, d); delay(20); }
  delay(500);
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("=== mini_hexapod stage 1: FR servo wakeup ===");
  Serial.println("Support the FR leg by hand -- it will snap to centered pose on first command.");
  delay(2000);

  Wire.begin();  // XIAO ESP32-C3 default I2C: SDA=D4(GPIO6), SCL=D5(GPIO7)
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(PWM_FREQ_HZ);
  delay(100);

  setServoDeg(CH_COXA,  0.0f);
  setServoDeg(CH_FEMUR, 0.0f);
  setServoDeg(CH_KNEE,  0.0f);
  Serial.println("Centered. Sweep starts in 3s.");
  delay(3000);

  sweepChannel(CH_COXA,  "COXA");
  sweepChannel(CH_FEMUR, "FEMUR");
  sweepChannel(CH_KNEE,  "KNEE");

  Serial.println("Done. Servos centered. Press reset to re-run.");
}

void loop() {}
