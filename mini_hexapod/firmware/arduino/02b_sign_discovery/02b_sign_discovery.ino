// mini_hexapod stage 2b: per-joint sign discovery.
//
// Hardware: XIAO ESP32-C3 + 2x PCA9685 (right=0x40, left=0x41).
// Purpose:  For each of 18 joints, sweep +25 deg kinematic and ask the user
//           which direction the joint actually moved. Save the per-joint sign
//           to NVS so the IK demo (stage 3) and walking gait (stage 4) can
//           command joints in the correct physical direction.
//
// Prereq:   trims from stage 2 (read from NVS).
//
// Workflow:
//   1. Home all legs to trim (fully extended outward).
//   2. For each (leg, joint), one at a time:
//        a. Sweep joint smoothly to +25 deg kinematic over ~750ms.
//        b. Hold, then sweep back to 0.
//        c. Prompt: which direction did it move?
//        d. Save sign based on answer. Move to next joint.
//   3. After all 18 done, dump the SIGN[] table as a C++ literal.
//
// Sign convention saved to NVS:
//   sign = +1 means the servo's "increase us" direction matches the
//          kinematic positive direction for that joint.
//   sign = -1 means they're opposite (flip servo us delta when commanding).
//
// Kinematic positive directions (what the answer maps to):
//   COXA  - leg rotates CCW about body +Z. For RIGHT legs this swings the
//           foot toward the FRONT of the bot; for LEFT legs it swings the
//           foot toward the REAR.
//   FEMUR - femur tilts UP (knee rises above coxa pivot). Same all 6 legs.
//   KNEE  - tibia bends downward (foot drops below femur extension). Same all 6.
//
// Serial Monitor: 115200, "No line ending".

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <Preferences.h>
#include <math.h>

constexpr uint16_t SERVO_MIN_US = 500;
constexpr uint16_t SERVO_MAX_US = 2500;
constexpr uint16_t SERVO_CTR_US = 1500;
constexpr uint8_t  PWM_FREQ_HZ  = 50;
constexpr uint8_t  ADDR_RIGHT   = 0x40;
constexpr uint8_t  ADDR_LEFT    = 0x41;

Adafruit_PWMServoDriver pwm_right(ADDR_RIGHT);
Adafruit_PWMServoDriver pwm_left (ADDR_LEFT);

struct LegSpec {
  const char* name;
  uint8_t board_addr;
  uint8_t channels[3];
  bool    is_right;
};

const LegSpec LEGS[6] = {
  {"FR", ADDR_RIGHT, {0, 1, 2}, true },
  {"MR", ADDR_RIGHT, {3, 4, 5}, true },
  {"RR", ADDR_RIGHT, {6, 7, 8}, true },
  {"FL", ADDR_LEFT,  {0, 1, 2}, false},
  {"ML", ADDR_LEFT,  {3, 4, 5}, false},
  {"RL", ADDR_LEFT,  {6, 7, 8}, false},
};

const char* JOINT_NAMES[3] = {"COXA", "FEMUR", "KNEE"};

Adafruit_PWMServoDriver& driverFor(const LegSpec& leg) {
  return leg.board_addr == ADDR_RIGHT ? pwm_right : pwm_left;
}

uint16_t usToTicks(uint16_t us) {
  return (uint16_t)((uint32_t)us * PWM_FREQ_HZ * 4096UL / 1000000UL);
}

constexpr float US_PER_DEG = (float)(SERVO_MAX_US - SERVO_MIN_US) / 180.0f;

Preferences prefs;
uint16_t trim_us[6][3];
int8_t   sign_table[6][3];

// --- NVS ---
void loadTrims() {
  prefs.begin("hexa_trim", true);
  for (uint8_t l = 0; l < 6; ++l) {
    for (uint8_t j = 0; j < 3; ++j) {
      char k[16]; snprintf(k, 16, "l%d_j%d", l, j);
      trim_us[l][j] = prefs.getUShort(k, SERVO_CTR_US);
    }
  }
  prefs.end();
}

void loadSigns() {
  prefs.begin("hexa_sign", true);
  for (uint8_t l = 0; l < 6; ++l) {
    for (uint8_t j = 0; j < 3; ++j) {
      char k[16]; snprintf(k, 16, "l%d_j%d", l, j);
      sign_table[l][j] = (int8_t)prefs.getChar(k, (int8_t)+1);
    }
  }
  prefs.end();
}

void saveSign(uint8_t leg, uint8_t joint, int8_t s) {
  prefs.begin("hexa_sign", false);
  char k[16]; snprintf(k, 16, "l%d_j%d", leg, joint);
  prefs.putChar(k, (signed char)s);
  prefs.end();
  sign_table[leg][joint] = s;
}

// --- Servo control ---
// Command joint to a kinematic angle (degrees from trim), using sign=+1
// regardless of saved sign. The user observes which physical direction it
// moves, and we infer the correct sign from their answer.
void cmdJointTentative(uint8_t leg, uint8_t joint, float deg) {
  float us = (float)trim_us[leg][joint] + deg * US_PER_DEG;
  if (us < (float)SERVO_MIN_US) us = SERVO_MIN_US;
  if (us > (float)SERVO_MAX_US) us = SERVO_MAX_US;
  const LegSpec& L = LEGS[leg];
  driverFor(L).setPWM(L.channels[joint], 0, usToTicks((uint16_t)(us + 0.5f)));
}

void cmdJointTrim(uint8_t leg, uint8_t joint) {
  const LegSpec& L = LEGS[leg];
  driverFor(L).setPWM(L.channels[joint], 0, usToTicks(trim_us[leg][joint]));
}

void homeAllToTrim() {
  for (uint8_t l = 0; l < 6; ++l) {
    for (uint8_t j = 0; j < 3; ++j) cmdJointTrim(l, j);
  }
}

// Smooth sweep from from_deg to to_deg over ~750ms.
void sweepFromTo(uint8_t leg, uint8_t joint, float from_deg, float to_deg) {
  const int STEPS = 25;
  for (int i = 1; i <= STEPS; ++i) {
    float t = (float)i / STEPS;
    float d = from_deg + (to_deg - from_deg) * t;
    cmdJointTentative(leg, joint, d);
    delay(30);
  }
}

// --- UI ---
char waitForKey(const char* valid) {
  while (true) {
    if (Serial.available()) {
      char c = Serial.read();
      if (c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
      for (const char* p = valid; *p; ++p) {
        if (c == *p) return c;
      }
    }
    delay(10);
  }
}

void waitForEnter() {
  while (!Serial.available()) delay(10);
  while (Serial.available()) Serial.read();
}

void doSignTest(uint8_t leg, uint8_t joint) {
  const LegSpec& L = LEGS[leg];

  for (uint8_t j = 0; j < 3; ++j) cmdJointTrim(leg, j);
  delay(500);

  Serial.println();
  Serial.print(">>> "); Serial.print(L.name);
  Serial.print(" "); Serial.print(JOINT_NAMES[joint]);
  Serial.print("   (board=0x"); Serial.print(L.board_addr, HEX);
  Serial.print(" ch="); Serial.print(L.channels[joint]); Serial.println(")");

  bool decided = false;
  int8_t result = +1;

  while (!decided) {
    Serial.println("    Watch the leg. Sweeping in 1s...");
    delay(1000);
    sweepFromTo(leg, joint, 0.0f, +25.0f);
    delay(500);
    sweepFromTo(leg, joint, +25.0f, 0.0f);
    delay(300);

    if (joint == 0) {
      Serial.println("    During the sweep, did the foot swing:");
      Serial.println("      f = FORWARD (toward the front of the bot)");
      Serial.println("      b = BACKWARD (toward the rear of the bot)");
      Serial.println("      r = re-sweep");
      char c = waitForKey("fbr");
      if (c == 'r') continue;
      // CCW about body +Z is "forward" for right-side legs, "backward" for left.
      if (L.is_right) result = (c == 'f') ? +1 : -1;
      else            result = (c == 'f') ? -1 : +1;
    } else if (joint == 1) {
      Serial.println("    During the sweep, did the knee:");
      Serial.println("      u = RISE up (knee lifted above coxa)");
      Serial.println("      d = DROP down (knee fell below coxa)");
      Serial.println("      r = re-sweep");
      char c = waitForKey("udr");
      if (c == 'r') continue;
      result = (c == 'u') ? +1 : -1;
    } else {
      Serial.println("    During the sweep, did the tibia:");
      Serial.println("      d = bend DOWN (foot dropped below the femur line)");
      Serial.println("      u = bend UP backward (foot rose above the femur line)");
      Serial.println("      r = re-sweep");
      char c = waitForKey("dur");
      if (c == 'r') continue;
      result = (c == 'd') ? +1 : -1;
    }
    decided = true;
  }

  saveSign(leg, joint, result);
  Serial.print("    Saved sign = "); Serial.println((int)result);
}

void dumpSigns() {
  Serial.println();
  Serial.println("// Discovered signs (also persisted to NVS namespace 'hexa_sign'):");
  Serial.println("constexpr int8_t SIGN[6][3] = {");
  for (uint8_t l = 0; l < 6; ++l) {
    Serial.print("  /* ");
    Serial.print(LEGS[l].name);
    Serial.print(" */ {");
    for (uint8_t j = 0; j < 3; ++j) {
      Serial.print((sign_table[l][j] > 0) ? "+1" : "-1");
      if (j < 2) Serial.print(", ");
    }
    Serial.println("},");
  }
  Serial.println("};");
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("=== mini_hexapod stage 2b: per-joint sign discovery ===");

  Wire.begin();
  pwm_right.begin();
  pwm_right.setOscillatorFrequency(27000000);
  pwm_right.setPWMFreq(PWM_FREQ_HZ);
  pwm_left.begin();
  pwm_left.setOscillatorFrequency(27000000);
  pwm_left.setPWMFreq(PWM_FREQ_HZ);
  delay(200);

  loadTrims();
  loadSigns();

  Serial.println("Loaded trims + any existing signs from NVS.");
  Serial.println("WARNING: about to home all 18 joints to trim. Body on a stand.");
  Serial.println("Holding 3s...");
  delay(3000);
  homeAllToTrim();
  delay(500);

  Serial.println();
  Serial.println("All legs at kinematic zero (fully extended outward).");
  Serial.println("Will test each of 18 joints by sweeping +25 deg, then asking");
  Serial.println("you which direction it moved. Press ENTER (or any key) to begin.");
  waitForEnter();

  for (uint8_t l = 0; l < 6; ++l) {
    for (uint8_t j = 0; j < 3; ++j) {
      doSignTest(l, j);
    }
  }

  Serial.println();
  Serial.println("=== Sign discovery complete! ===");
  dumpSigns();
  Serial.println();
  Serial.println("Signs saved to NVS. Re-upload stage 3 (now reads signs from NVS)");
  Serial.println("and type 'h' to home FR via IK -- the leg should now fold correctly.");
}

void loop() {}
