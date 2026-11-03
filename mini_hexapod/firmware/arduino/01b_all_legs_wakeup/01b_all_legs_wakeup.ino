// mini_hexapod stage 1b: all-legs servo wakeup.
//
// Hardware: XIAO ESP32-C3 + 2x PCA9685 (right=0x40, left=0x41).
// Behavior: For each of 6 legs in sequence (FR -> MR -> RR -> FL -> ML -> RL):
//             1. Announce on Serial + pause 3s so you can support the leg.
//             2. Center coxa/femur/knee.
//             3. Sweep each joint +/-20 deg from center, return to center.
//           One-shot in setup(); ~65s total. Reset to re-run.
//
// Confirms all 18 servos respond on both boards before stage 2 calibration.

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

constexpr uint16_t SERVO_MIN_US = 500;
constexpr uint16_t SERVO_MAX_US = 2500;
constexpr uint16_t SERVO_CTR_US = 1500;
constexpr uint8_t  PWM_FREQ_HZ  = 50;
constexpr float    SWEEP_DEG    = 20.0f;

constexpr uint8_t ADDR_RIGHT = 0x40;
constexpr uint8_t ADDR_LEFT  = 0x41;

Adafruit_PWMServoDriver pwm_right(ADDR_RIGHT);
Adafruit_PWMServoDriver pwm_left (ADDR_LEFT);

struct LegSpec {
  const char* name;
  uint8_t board_addr;
  uint8_t coxa_ch;
  uint8_t femur_ch;
  uint8_t knee_ch;
};

const LegSpec LEGS[6] = {
  {"FR", ADDR_RIGHT, 0, 1, 2},
  {"MR", ADDR_RIGHT, 3, 4, 5},
  {"RR", ADDR_RIGHT, 6, 7, 8},
  {"FL", ADDR_LEFT,  0, 1, 2},
  {"ML", ADDR_LEFT,  3, 4, 5},
  {"RL", ADDR_LEFT,  6, 7, 8},
};

Adafruit_PWMServoDriver& driverFor(const LegSpec& leg) {
  return leg.board_addr == ADDR_RIGHT ? pwm_right : pwm_left;
}

uint16_t usToTicks(uint16_t us) {
  return (uint16_t)((uint32_t)us * PWM_FREQ_HZ * 4096UL / 1000000UL);
}

uint16_t degToUs(float deg) {
  deg = constrain(deg, -90.0f, 90.0f);
  return (uint16_t)(SERVO_CTR_US + deg * ((float)(SERVO_MAX_US - SERVO_MIN_US) / 180.0f));
}

void setServoDeg(Adafruit_PWMServoDriver& drv, uint8_t ch, float deg) {
  drv.setPWM(ch, 0, usToTicks(degToUs(deg)));
}

void sweepChannel(Adafruit_PWMServoDriver& drv, uint8_t ch, const char* name) {
  Serial.print("  Sweeping "); Serial.print(name);
  Serial.print(" (ch "); Serial.print(ch); Serial.println(") +/-20 deg");
  setServoDeg(drv, ch, 0.0f);
  delay(300);
  for (float d = 0;          d >= -SWEEP_DEG; d -= 1.0f) { setServoDeg(drv, ch, d); delay(15); }
  delay(200);
  for (float d = -SWEEP_DEG; d <=  SWEEP_DEG; d += 1.0f) { setServoDeg(drv, ch, d); delay(15); }
  delay(200);
  for (float d =  SWEEP_DEG; d >=  0.0f;      d -= 1.0f) { setServoDeg(drv, ch, d); delay(15); }
  delay(300);
}

void testLeg(const LegSpec& leg) {
  Serial.println();
  Serial.print("=== Leg "); Serial.print(leg.name);
  Serial.print(" (board 0x"); Serial.print(leg.board_addr, HEX);
  Serial.print(", ch "); Serial.print(leg.coxa_ch); Serial.print("/");
  Serial.print(leg.femur_ch); Serial.print("/"); Serial.print(leg.knee_ch);
  Serial.println(") ===");
  Serial.println("  Support this leg. Centering + sweep in 3s.");
  delay(3000);

  Adafruit_PWMServoDriver& drv = driverFor(leg);
  setServoDeg(drv, leg.coxa_ch,  0.0f);
  setServoDeg(drv, leg.femur_ch, 0.0f);
  setServoDeg(drv, leg.knee_ch,  0.0f);
  delay(500);

  sweepChannel(drv, leg.coxa_ch,  "COXA");
  sweepChannel(drv, leg.femur_ch, "FEMUR");
  sweepChannel(drv, leg.knee_ch,  "KNEE");
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("=== mini_hexapod stage 1b: all-legs wakeup ===");
  Serial.println("Order: FR -> MR -> RR -> FL -> ML -> RL");
  Serial.println("3s pause before each leg so you can support it.");

  Wire.begin();  // XIAO ESP32-C3 default: SDA=D4(GPIO6), SCL=D5(GPIO7)
  pwm_right.begin();
  pwm_right.setOscillatorFrequency(27000000);
  pwm_right.setPWMFreq(PWM_FREQ_HZ);
  pwm_left.begin();
  pwm_left.setOscillatorFrequency(27000000);
  pwm_left.setPWMFreq(PWM_FREQ_HZ);
  delay(200);

  for (uint8_t i = 0; i < 6; ++i) {
    testLeg(LEGS[i]);
  }

  Serial.println();
  Serial.println("Done. All 18 servos exercised. Press reset to re-run.");
}

void loop() {}
