// mini_hexapod stage 2: interactive pose calibration.
//
// Hardware: XIAO ESP32-C3 + 2x PCA9685 (right=0x40, left=0x41).
// Purpose:  Find the PWM-us value corresponding to each joint's kinematic zero,
//           for all 18 joints. Saves to NVS so values survive reflash.
//
// Kinematic zero conventions:
//   COXA  - leg points along its coxa axis (outward from body)
//   FEMUR - femur horizontal (parallel to ground)
//   KNEE  - tibia in straight extension of femur (knee unbent)
//
// All 18 at zero  =>  legs fully extended outward, feet at horizontal distance
// L2+L3 = 144.5mm from coxa pivots, at body height.
//
// SETUP: body on a stand (or held up) so legs hang free without bearing weight.
//        On boot the sketch homes all joints to their saved trims (1500us default
//        for any not yet calibrated). 3-second warning printed before homing.
//
// Serial Monitor: 115200 baud, "No line ending" (single-keystroke commands).
//
// Commands:
//   Move:  a/d  -1/+1 deg    q/e  -5/+5 deg    z/c  -20/+20 deg
//   Reset: r    snap to 1500us (servo center)
//   Save:  s    record current us as trim, advance to next joint
//   Skip:  n    advance without saving
//   Back:  b    back to previous joint
//   Home:  h    drive current LEG to its saved trims
//          H    drive ALL legs to their saved trims
//   View:  p    print current joint state
//          l    list all 18 trims
//          D    dump trims as C++ literal (for paste-into-header)
//   Wipe:  X    clear all NVS trims (back to 1500us)
//   Help:  ?

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <Preferences.h>

constexpr uint16_t SERVO_MIN_US = 500;
constexpr uint16_t SERVO_MAX_US = 2500;
constexpr uint16_t SERVO_CTR_US = 1500;
constexpr uint8_t  PWM_FREQ_HZ  = 50;

constexpr uint8_t ADDR_RIGHT = 0x40;
constexpr uint8_t ADDR_LEFT  = 0x41;

Adafruit_PWMServoDriver pwm_right(ADDR_RIGHT);
Adafruit_PWMServoDriver pwm_left (ADDR_LEFT);

struct LegSpec {
  const char* name;
  uint8_t board_addr;
  uint8_t channels[3];  // [coxa, femur, knee]
};

const LegSpec LEGS[6] = {
  {"FR", ADDR_RIGHT, {0, 1, 2}},
  {"MR", ADDR_RIGHT, {3, 4, 5}},
  {"RR", ADDR_RIGHT, {6, 7, 8}},
  {"FL", ADDR_LEFT,  {0, 1, 2}},
  {"ML", ADDR_LEFT,  {3, 4, 5}},
  {"RL", ADDR_LEFT,  {6, 7, 8}},
};

const char* JOINT_NAMES[3] = {"COXA", "FEMUR", "KNEE"};
const char* ZERO_HINTS[3] = {
  "leg points along its coxa axis (outward from body)",
  "femur horizontal (parallel to ground)",
  "tibia in straight extension of femur (knee unbent)"
};

Adafruit_PWMServoDriver& driverFor(const LegSpec& leg) {
  return leg.board_addr == ADDR_RIGHT ? pwm_right : pwm_left;
}

uint16_t usToTicks(uint16_t us) {
  return (uint16_t)((uint32_t)us * PWM_FREQ_HZ * 4096UL / 1000000UL);
}

float degToUsDelta(float deg) {
  return deg * ((float)(SERVO_MAX_US - SERVO_MIN_US) / 180.0f);
}

// --- State ---
Preferences prefs;
uint16_t trim_us[6][3];      // saved zeros (mirrored from NVS)
uint16_t current_us[6][3];   // currently commanded values
uint8_t  cur_leg = 0;
uint8_t  cur_joint = 0;

// --- NVS persistence ---

void trimKey(char* buf, uint8_t leg, uint8_t joint) {
  snprintf(buf, 16, "l%d_j%d", leg, joint);
}

void loadTrims() {
  prefs.begin("hexa_trim", true);
  for (uint8_t l = 0; l < 6; ++l) {
    for (uint8_t j = 0; j < 3; ++j) {
      char k[16];
      trimKey(k, l, j);
      trim_us[l][j] = prefs.getUShort(k, SERVO_CTR_US);
      current_us[l][j] = trim_us[l][j];
    }
  }
  prefs.end();
}

void saveTrim(uint8_t leg, uint8_t joint, uint16_t us) {
  prefs.begin("hexa_trim", false);
  char k[16];
  trimKey(k, leg, joint);
  prefs.putUShort(k, us);
  prefs.end();
  trim_us[leg][joint] = us;
}

void wipeTrims() {
  prefs.begin("hexa_trim", false);
  prefs.clear();
  prefs.end();
  for (uint8_t l = 0; l < 6; ++l) {
    for (uint8_t j = 0; j < 3; ++j) {
      trim_us[l][j] = SERVO_CTR_US;
      current_us[l][j] = SERVO_CTR_US;
    }
  }
  Serial.println("All trims wiped (reset to 1500us).");
}

// --- Servo control ---

void cmdJoint(uint8_t leg, uint8_t joint, uint16_t us) {
  const LegSpec& L = LEGS[leg];
  driverFor(L).setPWM(L.channels[joint], 0, usToTicks(us));
  current_us[leg][joint] = us;
}

void applyCurrent() {
  cmdJoint(cur_leg, cur_joint, current_us[cur_leg][cur_joint]);
}

void homeLegToTrims(uint8_t leg) {
  for (uint8_t j = 0; j < 3; ++j) {
    cmdJoint(leg, j, trim_us[leg][j]);
  }
}

void homeAllToTrims() {
  for (uint8_t l = 0; l < 6; ++l) homeLegToTrims(l);
}

// --- UI ---

void printPrompt() {
  const LegSpec& L = LEGS[cur_leg];
  Serial.println();
  Serial.print("[");
  Serial.print(L.name);
  Serial.print(" ");
  Serial.print(JOINT_NAMES[cur_joint]);
  Serial.print("  board=0x");
  Serial.print(L.board_addr, HEX);
  Serial.print(" ch=");
  Serial.print(L.channels[cur_joint]);
  Serial.print("]  us=");
  Serial.print(current_us[cur_leg][cur_joint]);
  Serial.print("  saved=");
  Serial.println(trim_us[cur_leg][cur_joint]);
  Serial.print("  Goal: ");
  Serial.println(ZERO_HINTS[cur_joint]);
}

void printHelp() {
  Serial.println();
  Serial.println("Commands:");
  Serial.println("  a/d  -1/+1 deg    q/e  -5/+5 deg    z/c  -20/+20 deg");
  Serial.println("  r    reset to 1500us");
  Serial.println("  s    save & next   n    skip & next   b    back");
  Serial.println("  h    home current leg     H    home all legs");
  Serial.println("  p    print state    l    list trims    D    dump as C++");
  Serial.println("  X    wipe all trims        ?    help");
}

void listTrims() {
  Serial.println();
  Serial.println("Trims (us):  COXA   FEMUR  KNEE");
  for (uint8_t l = 0; l < 6; ++l) {
    Serial.print("  ");
    Serial.print(LEGS[l].name);
    Serial.print("         ");
    for (uint8_t j = 0; j < 3; ++j) {
      Serial.print(trim_us[l][j]);
      Serial.print("   ");
    }
    Serial.println();
  }
}

void dumpTrims() {
  Serial.println();
  Serial.println("// Paste into your IK sketch (FR, MR, RR, FL, ML, RL):");
  Serial.println("constexpr uint16_t TRIM_US[6][3] = {");
  for (uint8_t l = 0; l < 6; ++l) {
    Serial.print("  /* ");
    Serial.print(LEGS[l].name);
    Serial.print(" */ {");
    for (uint8_t j = 0; j < 3; ++j) {
      Serial.print(trim_us[l][j]);
      if (j < 2) Serial.print(", ");
    }
    Serial.println("},");
  }
  Serial.println("};");
}

void nudge(float deg) {
  float new_us = (float)current_us[cur_leg][cur_joint] + degToUsDelta(deg);
  new_us = constrain(new_us, (float)SERVO_MIN_US, (float)SERVO_MAX_US);
  current_us[cur_leg][cur_joint] = (uint16_t)(new_us + 0.5f);
  applyCurrent();
  Serial.print("  us=");
  Serial.println(current_us[cur_leg][cur_joint]);
}

void advance() {
  cur_joint++;
  if (cur_joint > 2) {
    cur_joint = 0;
    cur_leg = (cur_leg + 1) % 6;
  }
  current_us[cur_leg][cur_joint] = trim_us[cur_leg][cur_joint];
  applyCurrent();
  printPrompt();
}

void retreat() {
  if (cur_joint == 0) {
    cur_joint = 2;
    cur_leg = (cur_leg + 5) % 6;
  } else {
    cur_joint--;
  }
  current_us[cur_leg][cur_joint] = trim_us[cur_leg][cur_joint];
  applyCurrent();
  printPrompt();
}

void handleCommand(char c) {
  switch (c) {
    case 'a': nudge(-1.0f); break;
    case 'd': nudge(+1.0f); break;
    case 'q': nudge(-5.0f); break;
    case 'e': nudge(+5.0f); break;
    case 'z': nudge(-20.0f); break;
    case 'c': nudge(+20.0f); break;
    case 'r':
      current_us[cur_leg][cur_joint] = SERVO_CTR_US;
      applyCurrent();
      Serial.print("  us=");
      Serial.println(SERVO_CTR_US);
      break;
    case 's':
      saveTrim(cur_leg, cur_joint, current_us[cur_leg][cur_joint]);
      Serial.print("  SAVED ");
      Serial.print(LEGS[cur_leg].name);
      Serial.print(" ");
      Serial.print(JOINT_NAMES[cur_joint]);
      Serial.print(" = ");
      Serial.print(current_us[cur_leg][cur_joint]);
      Serial.println(" us");
      advance();
      break;
    case 'n':
      Serial.println("  (skipped)");
      advance();
      break;
    case 'b':
      retreat();
      break;
    case 'h':
      homeLegToTrims(cur_leg);
      Serial.print("  Homed leg ");
      Serial.println(LEGS[cur_leg].name);
      break;
    case 'H':
      homeAllToTrims();
      Serial.println("  Homed all legs to saved trims.");
      break;
    case 'p':
      printPrompt();
      break;
    case 'l':
      listTrims();
      break;
    case 'D':
      dumpTrims();
      break;
    case 'X':
      wipeTrims();
      break;
    case '?':
      printHelp();
      break;
    default:
      break;
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("=== mini_hexapod stage 2: pose calibration ===");

  Wire.begin();
  pwm_right.begin();
  pwm_right.setOscillatorFrequency(27000000);
  pwm_right.setPWMFreq(PWM_FREQ_HZ);
  pwm_left.begin();
  pwm_left.setOscillatorFrequency(27000000);
  pwm_left.setPWMFreq(PWM_FREQ_HZ);
  delay(200);

  loadTrims();

  Serial.println("Loaded trims from NVS (1500us default for joints not yet calibrated).");
  Serial.println("WARNING: about to home all 18 joints to their saved trims.");
  Serial.println("Body should be on a stand or held up. Homing in 3s...");
  delay(3000);

  homeAllToTrims();
  delay(500);

  Serial.println("Ready. Type ? for help.");
  printPrompt();
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r' || c == ' ' || c == '\t') return;
    handleCommand(c);
  }
}
