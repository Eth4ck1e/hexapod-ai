// mini_hexapod unified firmware
//
// Single sketch combining geometry config + trim calibration + sign discovery
// + IK demo + walking controller + BT pairing (stub). Menu-driven via Serial.
//
// Workflow for a fresh bot (or new chassis):
//   1. G  geometry  — set L2, L3, coxa positions, axis angles to match chassis
//   2. T  trim      — per-joint trim calibration at L-pose rest
//   3. S  sign      — per-joint sign discovery via sweep test
//   4. I  IK demo   — verify foot tracking + signs
//   5. W  walking   — verify gait via serial commands
//   6. B  BT pairing — pair 8BitDo Pro 3 (stub for now)
//   7. L  limits    — per-joint travel-range calibration (jog to each stop)
//   8. P  policy stream — host PC streams 18 joint targets at ~50 Hz over
//                     serial (scripts/drive_serial.py runs scaffold + policy)
//   X  export    — dump full config as C++
//   J  export    — dump full config as JSON (feeds the simulator profile)
//   R  reset NVS — factory defaults (confirm)
//
// All persists to ESP32 NVS:
//   hexa_geom   body geometry + leg lengths
//   hexa_trim   per-joint trim us
//   hexa_sign   per-joint sign +1/-1
//   hexa_gait   step length / height / period
//   hexa_range  per-joint travel limits (us at each mechanical stop; 0 = unset)
//
// Serial Monitor: 115200, "Newline" line ending (works for both single-char
// menu commands and multi-token geometry commands).

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <Preferences.h>
#include <math.h>

// ============================== HARDWARE ==============================
constexpr uint16_t SERVO_MIN_US = 500;
constexpr uint16_t SERVO_MAX_US = 2500;
constexpr uint16_t SERVO_CTR_US = 1500;
constexpr uint8_t  PWM_FREQ_HZ  = 50;
constexpr uint8_t  ADDR_RIGHT   = 0x40;
constexpr uint8_t  ADDR_LEFT    = 0x41;

constexpr float US_PER_DEG = (float)(SERVO_MAX_US - SERVO_MIN_US) / 180.0f;

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

struct IkResult {
  float coxa_rad, femur_rad, knee_rad;
  bool reachable;
};

// =========================== DEFAULT GEOMETRY ===========================
// Edit these to change the compiled defaults, OR override at runtime in
// geometry mode (saves to NVS).
constexpr float DEFAULT_L2 = 55.0f;
constexpr float DEFAULT_L3 = 89.5f;
const float DEFAULT_COXA_X[6] = {+56.75f, 0.0f, -56.75f, +56.75f, 0.0f, -56.75f};
const float DEFAULT_COXA_Y[6] = {-32.0f, -42.4f, -32.0f, +32.0f, +42.4f, +32.0f};
const float DEFAULT_COXA_Z[6] = {0, 0, 0, 0, 0, 0};
const float DEFAULT_AXIS_RAD[6] = {
  -PI/4.0f, -PI/2.0f, -3.0f*PI/4.0f,
  +PI/4.0f, +PI/2.0f, +3.0f*PI/4.0f
};

// =========================== RUNTIME GEOMETRY ===========================
float L2 = DEFAULT_L2;
float L3 = DEFAULT_L3;
float COXA_X[6], COXA_Y[6], COXA_Z[6], AXIS_RAD[6];
float R_HOME, D_HOME;

// =========================== CALIBRATION ===========================
uint16_t trim_us[6][3];
int8_t   SIGN[6][3];
float    HOME_KIN_ANGLE[3] = {0, 0, 0};
// Travel limits measured in range mode: pulse width where the joint binds in
// each direction. 0 = never calibrated. Kinematic angle at a pulse width:
//   angle_rad = HOME_KIN_ANGLE[j] + SIGN[l][j] * (us - trim_us[l][j]) / US_PER_DEG * PI/180
// (inverse of angleToUs), which is also the simulator's joint-angle convention
// at the L-pose, so the JSON export maps straight into gait/robot_profile.py.
uint16_t range_lo_us[6][3];
uint16_t range_hi_us[6][3];

// =========================== GAIT ===========================
float         step_length    = 30.0f;
float         step_height    = 25.0f;
unsigned long step_period_ms = 1500;
constexpr unsigned long ACTION_DURATION_MS = 5000;

const uint8_t TRIPOD_LEGS[6] = {0, 4, 2, 3, 1, 5};
const float   TRIPOD_OFFS[6] = {0, 0, 0, 0.5f, 0.5f, 0.5f};

// =========================== STATE MACHINE ===========================
enum Mode {
  MODE_MAIN, MODE_GEOMETRY, MODE_TRIM, MODE_SIGN,
  MODE_IK, MODE_WALK, MODE_BT, MODE_EXPORT, MODE_RANGE, MODE_JSON, MODE_STREAM
};
Mode current_mode = MODE_MAIN;

Preferences prefs;
char pending_cmd = 0;

// =========================== HELPERS ===========================
Adafruit_PWMServoDriver& driverFor(uint8_t leg) {
  return LEGS[leg].board_addr == ADDR_RIGHT ? pwm_right : pwm_left;
}

uint16_t usToTicks(uint16_t us) {
  return (uint16_t)((uint32_t)us * PWM_FREQ_HZ * 4096UL / 1000000UL);
}

bool isWS(char c) { return c == '\n' || c == '\r' || c == ' ' || c == '\t'; }

String readLine(unsigned long timeout_ms) {
  String s;
  unsigned long start = millis();
  while (millis() - start < timeout_ms) {
    if (Serial.available()) {
      char c = Serial.read();
      if (c == '\n') return s;
      if (c == '\r') continue;
      s += c;
    }
    delay(2);
  }
  return s;
}

char readKey() {
  while (true) {
    if (Serial.available()) {
      char c = Serial.read();
      if (isWS(c)) continue;
      return c;
    }
    delay(5);
  }
}

int findLeg(const String& name) {
  for (uint8_t l = 0; l < 6; ++l) if (name == LEGS[l].name) return l;
  return -1;
}

// =========================== KINEMATICS ===========================
IkResult ikFootBody(uint8_t leg, float fx, float fy, float fz) {
  float dx = fx - COXA_X[leg];
  float dy = fy - COXA_Y[leg];
  float dz = fz - COXA_Z[leg];

  IkResult r;
  r.reachable = false;
  r.coxa_rad = r.femur_rad = r.knee_rad = 0.0f;

  float target_angle = atan2f(dy, dx);
  float coxa = target_angle - AXIS_RAD[leg];
  while (coxa >  PI) coxa -= 2.0f * PI;
  while (coxa < -PI) coxa += 2.0f * PI;

  float horiz = sqrtf(dx*dx + dy*dy);
  float d2 = horiz*horiz + dz*dz;
  float d = sqrtf(d2);

  if (d > (L2 + L3 + 0.01f) || d < fabsf(L2 - L3) - 0.01f) return r;

  float femur = atan2f(dz, horiz) + acosf((L2*L2 + d2 - L3*L3) / (2.0f * L2 * d));
  float knee_interior = acosf((L2*L2 + L3*L3 - d2) / (2.0f * L2 * L3));
  float knee = PI - knee_interior;

  r.coxa_rad = coxa;
  r.femur_rad = femur;
  r.knee_rad = knee;
  r.reachable = true;
  return r;
}

uint16_t angleToUs(uint8_t leg, uint8_t joint, float angle_rad) {
  float delta_rad = angle_rad - HOME_KIN_ANGLE[joint];
  float deg = delta_rad * 180.0f / PI;
  deg *= (float)SIGN[leg][joint];
  float us = (float)trim_us[leg][joint] + deg * US_PER_DEG;
  if (us < (float)SERVO_MIN_US) us = SERVO_MIN_US;
  if (us > (float)SERVO_MAX_US) us = SERVO_MAX_US;
  return (uint16_t)(us + 0.5f);
}

void cmdJointAngle(uint8_t leg, uint8_t joint, float angle_rad) {
  driverFor(leg).setPWM(LEGS[leg].channels[joint], 0, usToTicks(angleToUs(leg, joint, angle_rad)));
}

void cmdJointTrim(uint8_t leg, uint8_t joint) {
  driverFor(leg).setPWM(LEGS[leg].channels[joint], 0, usToTicks(trim_us[leg][joint]));
}

void cmdJointUs(uint8_t leg, uint8_t joint, uint16_t us) {
  if (us < SERVO_MIN_US) us = SERVO_MIN_US;
  if (us > SERVO_MAX_US) us = SERVO_MAX_US;
  driverFor(leg).setPWM(LEGS[leg].channels[joint], 0, usToTicks(us));
}

bool commandFoot(uint8_t leg, float fx, float fy, float fz) {
  IkResult ik = ikFootBody(leg, fx, fy, fz);
  if (!ik.reachable) return false;
  cmdJointAngle(leg, 0, ik.coxa_rad);
  cmdJointAngle(leg, 1, ik.femur_rad);
  cmdJointAngle(leg, 2, ik.knee_rad);
  return true;
}

void homePoseForLeg(uint8_t leg, float& x, float& y, float& z) {
  x = COXA_X[leg] + R_HOME * cosf(AXIS_RAD[leg]);
  y = COXA_Y[leg] + R_HOME * sinf(AXIS_RAD[leg]);
  z = COXA_Z[leg] - D_HOME;
}

void computeHomeKinAngles() {
  float x, y, z;
  homePoseForLeg(0, x, y, z);
  IkResult ik = ikFootBody(0, x, y, z);
  if (ik.reachable) {
    HOME_KIN_ANGLE[0] = ik.coxa_rad;
    HOME_KIN_ANGLE[1] = ik.femur_rad;
    HOME_KIN_ANGLE[2] = ik.knee_rad;
  }
}

void homeLeg(uint8_t leg) { for (uint8_t j = 0; j < 3; ++j) cmdJointTrim(leg, j); }
void homeAll() { for (uint8_t l = 0; l < 6; ++l) homeLeg(l); }

// =========================== NVS ===========================
void resetGeometryToDefaults() {
  L2 = DEFAULT_L2;
  L3 = DEFAULT_L3;
  for (uint8_t l = 0; l < 6; ++l) {
    COXA_X[l] = DEFAULT_COXA_X[l];
    COXA_Y[l] = DEFAULT_COXA_Y[l];
    COXA_Z[l] = DEFAULT_COXA_Z[l];
    AXIS_RAD[l] = DEFAULT_AXIS_RAD[l];
  }
  R_HOME = L2;
  D_HOME = L3;
}

void loadGeometry() {
  resetGeometryToDefaults();
  prefs.begin("hexa_geom", true);
  L2 = prefs.getFloat("l2", L2);
  L3 = prefs.getFloat("l3", L3);
  R_HOME = prefs.getFloat("r_home", R_HOME);
  D_HOME = prefs.getFloat("d_home", D_HOME);
  for (uint8_t l = 0; l < 6; ++l) {
    char k[16];
    snprintf(k, 16, "cx_%d", l); COXA_X[l]   = prefs.getFloat(k, COXA_X[l]);
    snprintf(k, 16, "cy_%d", l); COXA_Y[l]   = prefs.getFloat(k, COXA_Y[l]);
    snprintf(k, 16, "cz_%d", l); COXA_Z[l]   = prefs.getFloat(k, COXA_Z[l]);
    snprintf(k, 16, "aa_%d", l); AXIS_RAD[l] = prefs.getFloat(k, AXIS_RAD[l]);
  }
  prefs.end();
}

void saveGeometry() {
  prefs.begin("hexa_geom", false);
  prefs.putFloat("l2", L2);
  prefs.putFloat("l3", L3);
  prefs.putFloat("r_home", R_HOME);
  prefs.putFloat("d_home", D_HOME);
  for (uint8_t l = 0; l < 6; ++l) {
    char k[16];
    snprintf(k, 16, "cx_%d", l); prefs.putFloat(k, COXA_X[l]);
    snprintf(k, 16, "cy_%d", l); prefs.putFloat(k, COXA_Y[l]);
    snprintf(k, 16, "cz_%d", l); prefs.putFloat(k, COXA_Z[l]);
    snprintf(k, 16, "aa_%d", l); prefs.putFloat(k, AXIS_RAD[l]);
  }
  prefs.end();
}

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

void saveTrim(uint8_t leg, uint8_t joint, uint16_t us) {
  prefs.begin("hexa_trim", false);
  char k[16]; snprintf(k, 16, "l%d_j%d", leg, joint);
  prefs.putUShort(k, us);
  prefs.end();
  trim_us[leg][joint] = us;
}

void loadSigns() {
  prefs.begin("hexa_sign", true);
  for (uint8_t l = 0; l < 6; ++l) {
    for (uint8_t j = 0; j < 3; ++j) {
      char k[16]; snprintf(k, 16, "l%d_j%d", l, j);
      SIGN[l][j] = (int8_t)prefs.getChar(k, (int8_t)+1);
    }
  }
  prefs.end();
}

void saveSign(uint8_t leg, uint8_t joint, int8_t s) {
  prefs.begin("hexa_sign", false);
  char k[16]; snprintf(k, 16, "l%d_j%d", leg, joint);
  prefs.putChar(k, (signed char)s);
  prefs.end();
  SIGN[leg][joint] = s;
}

void loadRanges() {
  prefs.begin("hexa_range", true);
  char k[16];
  for (uint8_t l = 0; l < 6; ++l) {
    for (uint8_t j = 0; j < 3; ++j) {
      snprintf(k, 16, "lo_%d_%d", l, j); range_lo_us[l][j] = prefs.getUShort(k, 0);
      snprintf(k, 16, "hi_%d_%d", l, j); range_hi_us[l][j] = prefs.getUShort(k, 0);
    }
  }
  prefs.end();
}

void saveRange(uint8_t leg, uint8_t joint, uint16_t lo, uint16_t hi) {
  prefs.begin("hexa_range", false);
  char k[16];
  snprintf(k, 16, "lo_%d_%d", leg, joint); prefs.putUShort(k, lo);
  snprintf(k, 16, "hi_%d_%d", leg, joint); prefs.putUShort(k, hi);
  prefs.end();
  range_lo_us[leg][joint] = lo;
  range_hi_us[leg][joint] = hi;
}

void loadGait() {
  prefs.begin("hexa_gait", true);
  step_length    = prefs.getFloat("step_len", step_length);
  step_height    = prefs.getFloat("step_ht",  step_height);
  step_period_ms = (unsigned long)prefs.getUInt("step_per", (uint32_t)step_period_ms);
  prefs.end();
}

void saveGait() {
  prefs.begin("hexa_gait", false);
  prefs.putFloat("step_len", step_length);
  prefs.putFloat("step_ht",  step_height);
  prefs.putUInt("step_per", (uint32_t)step_period_ms);
  prefs.end();
}

void wipeAllNvs() {
  prefs.begin("hexa_geom", false); prefs.clear(); prefs.end();
  prefs.begin("hexa_trim", false); prefs.clear(); prefs.end();
  prefs.begin("hexa_sign", false); prefs.clear(); prefs.end();
  prefs.begin("hexa_gait", false); prefs.clear(); prefs.end();
  prefs.begin("hexa_range", false); prefs.clear(); prefs.end();
  resetGeometryToDefaults();
  for (uint8_t l = 0; l < 6; ++l) {
    for (uint8_t j = 0; j < 3; ++j) {
      trim_us[l][j] = SERVO_CTR_US;
      SIGN[l][j] = +1;
      range_lo_us[l][j] = 0;
      range_hi_us[l][j] = 0;
    }
  }
  step_length = 30.0f;
  step_height = 25.0f;
  step_period_ms = 1500;
  computeHomeKinAngles();
}

// ====================================================================
// =                          MAIN MENU                                =
// ====================================================================
void printMainMenu() {
  Serial.println();
  Serial.println("============= HEXAPOD FIRMWARE — MAIN MENU =============");
  Serial.println("  G  geometry config (chassis dimensions)");
  Serial.println("  T  trim calibration");
  Serial.println("  S  sign discovery");
  Serial.println("  I  IK demo (foot control)");
  Serial.println("  W  walking controller");
  Serial.println("  B  bluetooth controller pairing (stub)");
  Serial.println("  L  joint travel-limit calibration (jog to each stop)");
  Serial.println("  P  policy stream (host PC sends joint targets over serial)");
  Serial.println("  X  export full config as C++");
  Serial.println("  J  export full config as JSON (for the simulator)");
  Serial.println("  R  reset all NVS to defaults (confirm)");
  Serial.println("  ?  reprint this menu");
  Serial.println("(within any mode: 'm' returns here)");
}

void handleMainMenu() {
  printMainMenu();
  while (current_mode == MODE_MAIN) {
    char c = readKey();
    switch (c) {
      case 'G': case 'g': current_mode = MODE_GEOMETRY; return;
      case 'T': case 't': current_mode = MODE_TRIM;     return;
      case 'S': case 's': current_mode = MODE_SIGN;     return;
      case 'I': case 'i': current_mode = MODE_IK;       return;
      case 'W': case 'w': current_mode = MODE_WALK;     return;
      case 'B': case 'b': current_mode = MODE_BT;       return;
      case 'L': case 'l': current_mode = MODE_RANGE;    return;
      case 'P': case 'p': current_mode = MODE_STREAM;   return;
      case 'X': case 'x': current_mode = MODE_EXPORT;   return;
      case 'J': case 'j': current_mode = MODE_JSON;     return;
      case 'R': case 'r': {
        Serial.println("Reset ALL NVS to defaults? Type 'Y' to confirm.");
        char y = readKey();
        if (y == 'Y') {
          wipeAllNvs();
          Serial.println("All NVS wiped, defaults restored.");
        } else {
          Serial.println("Cancelled.");
        }
        printMainMenu();
        break;
      }
      case '?': printMainMenu(); break;
      default:  break;
    }
  }
}

// ====================================================================
// =                       GEOMETRY MODE                               =
// ====================================================================
void printGeometry() {
  Serial.println();
  Serial.println("Current geometry:");
  Serial.print("  L2="); Serial.print(L2); Serial.print(" L3="); Serial.print(L3);
  Serial.print("  R_HOME="); Serial.print(R_HOME); Serial.print(" D_HOME="); Serial.println(D_HOME);
  for (uint8_t l = 0; l < 6; ++l) {
    Serial.print("  "); Serial.print(LEGS[l].name);
    Serial.print("  coxa=("); Serial.print(COXA_X[l]);
    Serial.print(","); Serial.print(COXA_Y[l]);
    Serial.print(","); Serial.print(COXA_Z[l]);
    Serial.print(")  axis="); Serial.print(AXIS_RAD[l] * 180.0f / PI); Serial.println(" deg");
  }
}

void printGeomHelp() {
  Serial.println();
  Serial.println("Geometry commands (type then Enter):");
  Serial.println("  v                       view current geometry");
  Serial.println("  l2 <mm>                 set femur length");
  Serial.println("  l3 <mm>                 set tibia length");
  Serial.println("  rhome <mm>              set R_HOME (foot horiz offset at rest)");
  Serial.println("  dhome <mm>              set D_HOME (foot depth at rest)");
  Serial.println("  coxa <leg> <x> <y> <z>  set leg's coxa pivot (leg=FR..RL)");
  Serial.println("  axis <leg> <deg>        set leg's coxa axis angle");
  Serial.println("  s                       save to NVS");
  Serial.println("  d                       reset to compiled defaults (in-memory)");
  Serial.println("  m                       back to main menu");
  Serial.println("  ?                       help");
}

bool parseGeomCmd(const String& line) {
  if (line.length() == 0) return true;
  String t[6];
  int n = 0, i = 0, L = line.length();
  while (i < L && n < 6) {
    while (i < L && (line[i] == ' ' || line[i] == '\t')) i++;
    int start = i;
    while (i < L && line[i] != ' ' && line[i] != '\t') i++;
    if (i > start) t[n++] = line.substring(start, i);
  }
  if (n == 0) return true;
  String cmd = t[0]; cmd.toLowerCase();

  if (cmd == "v") { printGeometry(); return true; }
  if (cmd == "?") { printGeomHelp(); return true; }
  if (cmd == "m") { current_mode = MODE_MAIN; return true; }
  if (cmd == "s") {
    saveGeometry(); computeHomeKinAngles();
    Serial.println("Geometry saved.");
    return true;
  }
  if (cmd == "d") {
    resetGeometryToDefaults(); computeHomeKinAngles();
    Serial.println("Reset to compiled defaults (not saved).");
    return true;
  }
  if (cmd == "l2" && n >= 2) { L2 = t[1].toFloat(); computeHomeKinAngles(); Serial.print("L2="); Serial.println(L2); return true; }
  if (cmd == "l3" && n >= 2) { L3 = t[1].toFloat(); computeHomeKinAngles(); Serial.print("L3="); Serial.println(L3); return true; }
  if (cmd == "rhome" && n >= 2) { R_HOME = t[1].toFloat(); computeHomeKinAngles(); Serial.print("R_HOME="); Serial.println(R_HOME); return true; }
  if (cmd == "dhome" && n >= 2) { D_HOME = t[1].toFloat(); computeHomeKinAngles(); Serial.print("D_HOME="); Serial.println(D_HOME); return true; }
  if (cmd == "coxa" && n >= 5) {
    int leg = findLeg(t[1]);
    if (leg < 0) { Serial.println("Bad leg name (FR/MR/RR/FL/ML/RL)."); return true; }
    COXA_X[leg] = t[2].toFloat();
    COXA_Y[leg] = t[3].toFloat();
    COXA_Z[leg] = t[4].toFloat();
    Serial.print(LEGS[leg].name); Serial.print(" coxa = (");
    Serial.print(COXA_X[leg]); Serial.print(", "); Serial.print(COXA_Y[leg]);
    Serial.print(", "); Serial.print(COXA_Z[leg]); Serial.println(")");
    return true;
  }
  if (cmd == "axis" && n >= 3) {
    int leg = findLeg(t[1]);
    if (leg < 0) { Serial.println("Bad leg name."); return true; }
    AXIS_RAD[leg] = t[2].toFloat() * PI / 180.0f;
    Serial.print(LEGS[leg].name); Serial.print(" axis = ");
    Serial.print(AXIS_RAD[leg] * 180.0f / PI); Serial.println(" deg");
    return true;
  }
  Serial.println("Unknown command. '?' for help.");
  return true;
}

void runGeometryMode() {
  Serial.println();
  Serial.println("=== GEOMETRY MODE ===");
  printGeometry();
  Serial.println("'?' for commands, 'm' for main menu.");
  while (current_mode == MODE_GEOMETRY) {
    Serial.print("geom> ");
    String line = readLine(120000);
    Serial.println(line);
    parseGeomCmd(line);
  }
}

// ====================================================================
// =                          TRIM MODE                                =
// ====================================================================
uint8_t  trim_cur_leg = 0, trim_cur_joint = 0;
uint16_t trim_cur_us;

void trimPrompt() {
  Serial.println();
  Serial.print("[");
  Serial.print(LEGS[trim_cur_leg].name);
  Serial.print(" ");
  Serial.print(JOINT_NAMES[trim_cur_joint]);
  Serial.print("  board=0x"); Serial.print(LEGS[trim_cur_leg].board_addr, HEX);
  Serial.print(" ch="); Serial.print(LEGS[trim_cur_leg].channels[trim_cur_joint]);
  Serial.print("]  us="); Serial.print(trim_cur_us);
  Serial.print("  saved="); Serial.println(trim_us[trim_cur_leg][trim_cur_joint]);
}

void trimHelp() {
  Serial.println();
  Serial.println("Trim commands:");
  Serial.println("  a/d  -1/+1 deg     q/e  -5/+5 deg     z/c  -20/+20 deg");
  Serial.println("  r    reset to 1500us");
  Serial.println("  s    save & next     n    skip & next     b    back");
  Serial.println("  h    home current leg     H    home all");
  Serial.println("  p    print state    l    list all     D    dump as C++");
  Serial.println("  m    main menu      ?    help");
}

void trimApply() { cmdJointUs(trim_cur_leg, trim_cur_joint, trim_cur_us); }

void trimNudge(float deg) {
  float us = (float)trim_cur_us + deg * US_PER_DEG;
  if (us < SERVO_MIN_US) us = SERVO_MIN_US;
  if (us > SERVO_MAX_US) us = SERVO_MAX_US;
  trim_cur_us = (uint16_t)(us + 0.5f);
  trimApply();
  Serial.print("  us="); Serial.println(trim_cur_us);
}

void trimAdvance() {
  if (++trim_cur_joint > 2) { trim_cur_joint = 0; trim_cur_leg = (trim_cur_leg + 1) % 6; }
  trim_cur_us = trim_us[trim_cur_leg][trim_cur_joint];
  trimApply();
  trimPrompt();
}

void trimRetreat() {
  if (trim_cur_joint == 0) { trim_cur_joint = 2; trim_cur_leg = (trim_cur_leg + 5) % 6; }
  else                     { trim_cur_joint--; }
  trim_cur_us = trim_us[trim_cur_leg][trim_cur_joint];
  trimApply();
  trimPrompt();
}

void trimList() {
  Serial.println();
  Serial.println("Trims (us):  COXA   FEMUR  KNEE");
  for (uint8_t l = 0; l < 6; ++l) {
    Serial.print("  "); Serial.print(LEGS[l].name); Serial.print("       ");
    for (uint8_t j = 0; j < 3; ++j) { Serial.print(trim_us[l][j]); Serial.print("   "); }
    Serial.println();
  }
}

void trimDump() {
  Serial.println();
  Serial.println("constexpr uint16_t TRIM_US[6][3] = {");
  for (uint8_t l = 0; l < 6; ++l) {
    Serial.print("  /* "); Serial.print(LEGS[l].name); Serial.print(" */ {");
    for (uint8_t j = 0; j < 3; ++j) { Serial.print(trim_us[l][j]); if (j < 2) Serial.print(", "); }
    Serial.println("},");
  }
  Serial.println("};");
}

void runTrimMode() {
  Serial.println();
  Serial.println("=== TRIM CALIBRATION ===");
  Serial.println("Body on stand. Homing in 3s...");
  delay(3000);
  for (uint8_t l = 0; l < 6; ++l) for (uint8_t j = 0; j < 3; ++j) cmdJointTrim(l, j);
  delay(500);
  trim_cur_us = trim_us[trim_cur_leg][trim_cur_joint];
  trimApply();
  trimPrompt();

  while (current_mode == MODE_TRIM) {
    char c = readKey();
    switch (c) {
      case 'a': trimNudge(-1);  break;
      case 'd': trimNudge(+1);  break;
      case 'q': trimNudge(-5);  break;
      case 'e': trimNudge(+5);  break;
      case 'z': trimNudge(-20); break;
      case 'c': trimNudge(+20); break;
      case 'r': trim_cur_us = SERVO_CTR_US; trimApply();
                Serial.print("  us="); Serial.println(trim_cur_us); break;
      case 's': saveTrim(trim_cur_leg, trim_cur_joint, trim_cur_us);
                Serial.print("  SAVED "); Serial.print(LEGS[trim_cur_leg].name);
                Serial.print(" "); Serial.print(JOINT_NAMES[trim_cur_joint]);
                Serial.print(" = "); Serial.print(trim_cur_us); Serial.println(" us");
                trimAdvance(); break;
      case 'n': Serial.println("  (skipped)"); trimAdvance(); break;
      case 'b': trimRetreat(); break;
      case 'h': for (uint8_t j = 0; j < 3; ++j) cmdJointTrim(trim_cur_leg, j);
                Serial.print("  Homed "); Serial.println(LEGS[trim_cur_leg].name); break;
      case 'H': homeAll(); Serial.println("  Homed all."); break;
      case 'p': trimPrompt(); break;
      case 'l': trimList(); break;
      case 'D': trimDump(); break;
      case 'm': computeHomeKinAngles(); current_mode = MODE_MAIN; return;
      case '?': trimHelp(); break;
      default:  break;
    }
  }
}

// ====================================================================
// =                          SIGN MODE                                =
// ====================================================================
void signSweep(uint8_t leg, uint8_t joint, float from_deg, float to_deg) {
  const int STEPS = 25;
  for (int i = 1; i <= STEPS; ++i) {
    float t = (float)i / STEPS;
    float d = from_deg + (to_deg - from_deg) * t;
    float us = (float)trim_us[leg][joint] + d * US_PER_DEG;
    if (us < SERVO_MIN_US) us = SERVO_MIN_US;
    if (us > SERVO_MAX_US) us = SERVO_MAX_US;
    cmdJointUs(leg, joint, (uint16_t)(us + 0.5f));
    delay(30);
  }
}

char signWait(const char* valid) {
  while (true) {
    if (Serial.available()) {
      char c = Serial.read();
      if (isWS(c)) continue;
      for (const char* p = valid; *p; ++p) if (c == *p) return c;
    }
    delay(10);
  }
}

void runSignMode() {
  Serial.println();
  Serial.println("=== SIGN DISCOVERY ===");
  Serial.println("Will home all to trim in 3s, then sweep each joint and ask direction.");
  delay(3000);
  for (uint8_t l = 0; l < 6; ++l) for (uint8_t j = 0; j < 3; ++j) cmdJointTrim(l, j);
  delay(500);

  Serial.println("Press any key to begin (or 'm' to abort)...");
  char start = readKey();
  if (start == 'm') { current_mode = MODE_MAIN; return; }

  for (uint8_t l = 0; l < 6; ++l) {
    for (uint8_t j = 0; j < 3; ++j) {
      const LegSpec& L = LEGS[l];
      for (uint8_t k = 0; k < 3; ++k) cmdJointTrim(l, k);
      delay(400);

      Serial.println();
      Serial.print(">>> "); Serial.print(L.name); Serial.print(" "); Serial.print(JOINT_NAMES[j]);
      Serial.print("  (board=0x"); Serial.print(L.board_addr, HEX);
      Serial.print(" ch="); Serial.print(L.channels[j]); Serial.println(")");

      bool decided = false;
      int8_t result = +1;
      while (!decided) {
        Serial.println("    Watch the leg. Sweeping in 1s...");
        delay(1000);
        signSweep(l, j, 0.0f, +25.0f);
        delay(500);
        signSweep(l, j, +25.0f, 0.0f);
        delay(300);

        if (j == 0) {
          Serial.println("    Foot swung:  f=FORWARD  b=BACKWARD  r=re-sweep");
          char ans = signWait("fbr");
          if (ans == 'r') continue;
          if (L.is_right) result = (ans == 'f') ? +1 : -1;
          else            result = (ans == 'f') ? -1 : +1;
        } else if (j == 1) {
          Serial.println("    Knee:  u=ROSE  d=DROPPED  r=re-sweep");
          char ans = signWait("udr");
          if (ans == 'r') continue;
          result = (ans == 'u') ? +1 : -1;
        } else {
          Serial.println("    Tibia:  d=BENT DOWN (foot dropped)  u=HYPEREXTENDED  r=re-sweep");
          char ans = signWait("dur");
          if (ans == 'r') continue;
          result = (ans == 'd') ? +1 : -1;
        }
        decided = true;
      }
      saveSign(l, j, result);
      Serial.print("    Saved sign = "); Serial.println((int)result);
    }
  }

  Serial.println();
  Serial.println("=== Sign discovery complete ===");
  current_mode = MODE_MAIN;
}

// ====================================================================
// =                          IK DEMO MODE                             =
// ====================================================================
uint8_t ik_cur_leg = 0;
float ik_target_x[6], ik_target_y[6], ik_target_z[6];

void ikState() {
  Serial.println();
  Serial.print("[Leg "); Serial.print(LEGS[ik_cur_leg].name);
  Serial.print("] target_body_mm = (");
  Serial.print(ik_target_x[ik_cur_leg]); Serial.print(", ");
  Serial.print(ik_target_y[ik_cur_leg]); Serial.print(", ");
  Serial.print(ik_target_z[ik_cur_leg]); Serial.println(")");
  IkResult ik = ikFootBody(ik_cur_leg, ik_target_x[ik_cur_leg],
                                       ik_target_y[ik_cur_leg],
                                       ik_target_z[ik_cur_leg]);
  if (!ik.reachable) { Serial.println("  IK: UNREACHABLE"); return; }
  Serial.print("  deg: c="); Serial.print(ik.coxa_rad * 180 / PI);
  Serial.print(" f=");        Serial.print(ik.femur_rad * 180 / PI);
  Serial.print(" k=");        Serial.println(ik.knee_rad * 180 / PI);
  Serial.print("  us:  c=");  Serial.print(angleToUs(ik_cur_leg, 0, ik.coxa_rad));
  Serial.print(" f=");        Serial.print(angleToUs(ik_cur_leg, 1, ik.femur_rad));
  Serial.print(" k=");        Serial.println(angleToUs(ik_cur_leg, 2, ik.knee_rad));
}

void ikHelp() {
  Serial.println();
  Serial.println("IK demo commands:");
  Serial.println("  1-6  select leg                  p    print state");
  Serial.println("  w/s  +/-1mm X    W/S  +/-10mm    a/d  +/-1mm Y    A/D  +/-10mm");
  Serial.println("  r/f  +/-1mm Z    R/F  +/-10mm");
  Serial.println("  h    home leg    H    home all   z/Z  snap leg/all to trim");
  Serial.println("  m    main menu   ?    help");
}

void ikNudge(float dx, float dy, float dz) {
  float nx = ik_target_x[ik_cur_leg] + dx;
  float ny = ik_target_y[ik_cur_leg] + dy;
  float nz = ik_target_z[ik_cur_leg] + dz;
  if (commandFoot(ik_cur_leg, nx, ny, nz)) {
    ik_target_x[ik_cur_leg] = nx;
    ik_target_y[ik_cur_leg] = ny;
    ik_target_z[ik_cur_leg] = nz;
    Serial.print("  target=("); Serial.print(nx); Serial.print(", ");
    Serial.print(ny); Serial.print(", "); Serial.print(nz); Serial.println(")");
  } else {
    Serial.println("  UNREACHABLE");
  }
}

void ikResyncTargetsToHome() {
  for (uint8_t l = 0; l < 6; ++l)
    homePoseForLeg(l, ik_target_x[l], ik_target_y[l], ik_target_z[l]);
}

void runIkMode() {
  Serial.println();
  Serial.println("=== IK DEMO ===");
  Serial.println("Driving all legs to home (trim) in 3s. Body on stand.");
  delay(3000);
  homeAll();
  ikResyncTargetsToHome();
  delay(500);
  ikState();

  while (current_mode == MODE_IK) {
    char c = readKey();
    switch (c) {
      case '1': case '2': case '3': case '4': case '5': case '6':
        ik_cur_leg = c - '1';
        Serial.print("Selected: "); Serial.println(LEGS[ik_cur_leg].name);
        ikState();
        break;
      case 'w': ikNudge(+1, 0, 0); break;
      case 's': ikNudge(-1, 0, 0); break;
      case 'a': ikNudge(0, +1, 0); break;
      case 'd': ikNudge(0, -1, 0); break;
      case 'r': ikNudge(0, 0, +1); break;
      case 'f': ikNudge(0, 0, -1); break;
      case 'W': ikNudge(+10, 0, 0); break;
      case 'S': ikNudge(-10, 0, 0); break;
      case 'A': ikNudge(0, +10, 0); break;
      case 'D': ikNudge(0, -10, 0); break;
      case 'R': ikNudge(0, 0, +10); break;
      case 'F': ikNudge(0, 0, -10); break;
      case 'h':
        homeLeg(ik_cur_leg);
        homePoseForLeg(ik_cur_leg, ik_target_x[ik_cur_leg], ik_target_y[ik_cur_leg], ik_target_z[ik_cur_leg]);
        Serial.print("Homed "); Serial.println(LEGS[ik_cur_leg].name); break;
      case 'H': homeAll(); ikResyncTargetsToHome(); Serial.println("Homed all."); break;
      case 'z':
        homeLeg(ik_cur_leg);
        homePoseForLeg(ik_cur_leg, ik_target_x[ik_cur_leg], ik_target_y[ik_cur_leg], ik_target_z[ik_cur_leg]);
        break;
      case 'Z': homeAll(); ikResyncTargetsToHome(); break;
      case 'p': ikState(); break;
      case 'm': current_mode = MODE_MAIN; return;
      case '?': ikHelp(); break;
      default:  break;
    }
  }
}

// ====================================================================
// =                       WALKING MODE                                =
// ====================================================================
bool walkInterrupt() {
  while (Serial.available()) {
    char c = Serial.read();
    if (isWS(c)) continue;
    pending_cmd = c;
    return true;
  }
  return false;
}

void runWalk(const char* label, float vx, float vy, float omega_z) {
  Serial.print(">>> "); Serial.println(label);

  float dir_x[6], dir_y[6];
  for (uint8_t l = 0; l < 6; ++l) {
    float vxl = vx + (-COXA_Y[l]) * omega_z;
    float vyl = vy + ( COXA_X[l]) * omega_z;
    float mag = sqrtf(vxl * vxl + vyl * vyl);
    if (mag < 1e-6f) { dir_x[l] = 1.0f; dir_y[l] = 0.0f; }
    else             { dir_x[l] = vxl / mag; dir_y[l] = vyl / mag; }
  }

  unsigned long start_ms = millis();
  unsigned long last_update = 0;

  while (true) {
    unsigned long now = millis();
    unsigned long elapsed = now - start_ms;
    if (elapsed >= ACTION_DURATION_MS) break;
    if (walkInterrupt()) { Serial.println("  interrupted"); break; }

    if (now - last_update >= 25) {
      last_update = now;
      for (uint8_t i = 0; i < 6; ++i) {
        uint8_t leg = TRIPOD_LEGS[i];
        float phase = fmodf((float)elapsed / (float)step_period_ms + TRIPOD_OFFS[i], 1.0f);
        float disp, lift;
        if (phase < 0.5f) {
          float s = phase * 2.0f;
          disp = step_length * (s - 0.5f);
          lift = step_height * sinf(PI * s);
        } else {
          float s = (phase - 0.5f) * 2.0f;
          disp = step_length * (0.5f - s);
          lift = 0.0f;
        }
        float hx, hy, hz; homePoseForLeg(leg, hx, hy, hz);
        commandFoot(leg, hx + disp * dir_x[leg], hy + disp * dir_y[leg], hz + lift);
      }
    }
    delay(2);
  }
  homeAll();
  delay(300);
}

void walkHelp() {
  Serial.println();
  Serial.println("Walking commands (each 5s, any key interrupts):");
  Serial.println("  w/s  forward/backward       a/d  strafe left/right");
  Serial.println("  q/e  turn left/right CCW/CW");
  Serial.println("  r/t  fwd-left/fwd-right     f/g  back-left/back-right");
  Serial.println("  1/2  arc fwd-left/right     3/4  arc back-left/right");
  Serial.println("  h    home all");
  Serial.println("  [/]  step length -/+5    ,/.  step height -/+2");
  Serial.println("  </>  step period -/+200    p  print params    S  save params");
  Serial.println("  m    main menu     ?    help");
}

void walkParams() {
  Serial.print("step_length="); Serial.print(step_length);
  Serial.print(" mm  step_height="); Serial.print(step_height);
  Serial.print(" mm  step_period="); Serial.print(step_period_ms); Serial.println(" ms");
}

void walkHandle(char c) {
  switch (c) {
    case 'w': runWalk("forward",          +1, 0, 0); break;
    case 's': runWalk("backward",         -1, 0, 0); break;
    case 'a': runWalk("strafe left",       0, +1, 0); break;
    case 'd': runWalk("strafe right",      0, -1, 0); break;
    case 'q': runWalk("turn left (CCW)",   0, 0, +1); break;
    case 'e': runWalk("turn right (CW)",   0, 0, -1); break;
    case 'r': runWalk("fwd-left",  +0.707f, +0.707f, 0); break;
    case 't': runWalk("fwd-right", +0.707f, -0.707f, 0); break;
    case 'f': runWalk("back-left", -0.707f, +0.707f, 0); break;
    case 'g': runWalk("back-right",-0.707f, -0.707f, 0); break;
    case '1': runWalk("arc fwd-left",   +1, 0, +0.5f); break;
    case '2': runWalk("arc fwd-right",  +1, 0, -0.5f); break;
    case '3': runWalk("arc back-left",  -1, 0, +0.5f); break;
    case '4': runWalk("arc back-right", -1, 0, -0.5f); break;
    case 'h': homeAll(); Serial.println("Homed."); break;
    case 'p': walkParams(); break;
    case 'S': saveGait(); Serial.println("Gait params saved to NVS."); break;
    case '[': step_length -= 5; if (step_length < 10) step_length = 10; walkParams(); break;
    case ']': step_length += 5; if (step_length > 50) step_length = 50; walkParams(); break;
    case ',': step_height -= 2; if (step_height < 5)  step_height = 5;  walkParams(); break;
    case '.': step_height += 2; if (step_height > 50) step_height = 50; walkParams(); break;
    case '<': { long p = (long)step_period_ms - 200; if (p < 500)  p = 500;  step_period_ms = p; walkParams(); break; }
    case '>': { long p = (long)step_period_ms + 200; if (p > 3000) p = 3000; step_period_ms = p; walkParams(); break; }
    case 'm': current_mode = MODE_MAIN; break;
    case '?': walkHelp(); break;
    default: break;
  }
}

void runWalkMode() {
  Serial.println();
  Serial.println("=== WALKING ===");
  Serial.println("Body on stand. Homing in 3s...");
  delay(3000);
  homeAll();
  delay(500);
  walkParams();
  Serial.println("? for help");

  while (current_mode == MODE_WALK) {
    char c = 0;
    if (pending_cmd) { c = pending_cmd; pending_cmd = 0; }
    else if (Serial.available()) {
      c = Serial.read();
      if (isWS(c)) continue;
    } else {
      delay(5);
      continue;
    }
    walkHandle(c);
  }
}

// ====================================================================
// =                         BT MODE (stub)                            =
// ====================================================================
void runBtMode() {
  Serial.println();
  Serial.println("=== BLUETOOTH PAIRING ===");
  Serial.println("Not yet implemented. Planned:");
  Serial.println("  - Put 8BitDo Pro 3 in macOS mode (Start + B) and press Sync.");
  Serial.println("  - ESP32-C3 hosts BLE HID via Bluepad32, listens for the controller.");
  Serial.println("  - Left stick -> vx/vy, right stick -> yaw, shoulders -> step len/height.");
  Serial.println("Press any key to return to main menu.");
  readKey();
  current_mode = MODE_MAIN;
}

// ====================================================================
// =                   RANGE (TRAVEL LIMIT) MODE                       =
// ====================================================================
// Finds where each joint physically binds so the simulator's joint ranges
// (and the policy's action limits) match this chassis. Procedure per joint:
//   jog toward one stop a few degrees at a time; the moment the link touches
//   the body / adjacent link (or the servo starts to buzz), back off ONE
//   notch and press '[' (low stop) — repeat toward the other side and press
//   ']' (high stop) — then 's' saves both and moves on.
// Never leave a servo pressing into a stop: it draws stall current.
uint8_t  rng_cur_leg = 0, rng_cur_joint = 0;
uint16_t rng_cur_us;
uint16_t rng_lo_tmp = 0, rng_hi_tmp = 0;

float usToKinRad(uint8_t leg, uint8_t joint, uint16_t us) {
  float deg = ((float)us - (float)trim_us[leg][joint]) / US_PER_DEG;
  return HOME_KIN_ANGLE[joint] + (float)SIGN[leg][joint] * deg * PI / 180.0f;
}

void rngPrompt() {
  Serial.println();
  Serial.print("[LIMITS "); Serial.print(LEGS[rng_cur_leg].name);
  Serial.print(" "); Serial.print(JOINT_NAMES[rng_cur_joint]);
  Serial.print("]  us="); Serial.print(rng_cur_us);
  Serial.print("  trim="); Serial.print(trim_us[rng_cur_leg][rng_cur_joint]);
  Serial.print("  marks lo/hi="); Serial.print(rng_lo_tmp); Serial.print("/"); Serial.print(rng_hi_tmp);
  Serial.print("  saved="); Serial.print(range_lo_us[rng_cur_leg][rng_cur_joint]);
  Serial.print("/"); Serial.println(range_hi_us[rng_cur_leg][rng_cur_joint]);
}

void rngHelp() {
  Serial.println();
  Serial.println("Limit commands (jog ONE notch past contact, then back off one):");
  Serial.println("  a/d  -1/+1 deg     q/e  -5/+5 deg     z/c  -20/+20 deg");
  Serial.println("  [    mark current us as LOW stop    ]  mark as HIGH stop");
  Serial.println("  s    save marks & next joint   n  skip & next   b  back");
  Serial.println("  t    return this joint to trim   h  home leg   H  home all");
  Serial.println("  p    print state   l  list all (us + kinematic rad)");
  Serial.println("  m    main menu     ?  help");
}

void rngApply() { cmdJointUs(rng_cur_leg, rng_cur_joint, rng_cur_us); }

void rngNudge(float deg) {
  float us = (float)rng_cur_us + deg * US_PER_DEG;
  if (us < SERVO_MIN_US) us = SERVO_MIN_US;
  if (us > SERVO_MAX_US) us = SERVO_MAX_US;
  rng_cur_us = (uint16_t)(us + 0.5f);
  rngApply();
  Serial.print("  us="); Serial.print(rng_cur_us);
  Serial.print("  ("); Serial.print(((float)rng_cur_us - (float)trim_us[rng_cur_leg][rng_cur_joint]) / US_PER_DEG, 1);
  Serial.println(" deg from trim)");
}

void rngLoadJoint() {
  rng_cur_us = trim_us[rng_cur_leg][rng_cur_joint];
  rng_lo_tmp = range_lo_us[rng_cur_leg][rng_cur_joint];
  rng_hi_tmp = range_hi_us[rng_cur_leg][rng_cur_joint];
  rngApply();
  rngPrompt();
}

void rngAdvance() {
  cmdJointTrim(rng_cur_leg, rng_cur_joint);           // never leave a joint at a stop
  if (++rng_cur_joint > 2) { rng_cur_joint = 0; rng_cur_leg = (rng_cur_leg + 1) % 6; }
  rngLoadJoint();
}

void rngRetreat() {
  cmdJointTrim(rng_cur_leg, rng_cur_joint);
  if (rng_cur_joint == 0) { rng_cur_joint = 2; rng_cur_leg = (rng_cur_leg + 5) % 6; }
  else                    { rng_cur_joint--; }
  rngLoadJoint();
}

void rngList() {
  Serial.println();
  Serial.println("Travel limits  (us lo/hi  ->  kinematic rad min/max; 0/0 = not calibrated)");
  for (uint8_t l = 0; l < 6; ++l) {
    for (uint8_t j = 0; j < 3; ++j) {
      uint16_t lo = range_lo_us[l][j], hi = range_hi_us[l][j];
      Serial.print("  "); Serial.print(LEGS[l].name); Serial.print(" ");
      Serial.print(JOINT_NAMES[j]); Serial.print("\t");
      Serial.print(lo); Serial.print("/"); Serial.print(hi);
      if (lo && hi) {
        float a = usToKinRad(l, j, lo), b = usToKinRad(l, j, hi);
        Serial.print("\t-> "); Serial.print(fminf(a, b), 3); Serial.print(" .. "); Serial.print(fmaxf(a, b), 3);
      }
      Serial.println();
    }
  }
}

void runRangeMode() {
  Serial.println();
  Serial.println("=== JOINT TRAVEL-LIMIT CALIBRATION ===");
  Serial.println("Body on a stand, legs free to move. Homing in 3s...");
  delay(3000);
  homeAll();
  delay(500);
  rngHelp();
  rngLoadJoint();

  while (current_mode == MODE_RANGE) {
    char c = readKey();
    switch (c) {
      case 'a': rngNudge(-1);  break;
      case 'd': rngNudge(+1);  break;
      case 'q': rngNudge(-5);  break;
      case 'e': rngNudge(+5);  break;
      case 'z': rngNudge(-20); break;
      case 'c': rngNudge(+20); break;
      case '[': rng_lo_tmp = rng_cur_us; Serial.print("  LOW stop marked at "); Serial.println(rng_lo_tmp); break;
      case ']': rng_hi_tmp = rng_cur_us; Serial.print("  HIGH stop marked at "); Serial.println(rng_hi_tmp); break;
      case 's':
        if (!rng_lo_tmp || !rng_hi_tmp) { Serial.println("  mark BOTH stops first ('[' and ']')"); break; }
        if (rng_lo_tmp > rng_hi_tmp) { uint16_t t = rng_lo_tmp; rng_lo_tmp = rng_hi_tmp; rng_hi_tmp = t; }
        saveRange(rng_cur_leg, rng_cur_joint, rng_lo_tmp, rng_hi_tmp);
        Serial.print("  SAVED "); Serial.print(LEGS[rng_cur_leg].name); Serial.print(" ");
        Serial.print(JOINT_NAMES[rng_cur_joint]); Serial.print(" = ");
        Serial.print(rng_lo_tmp); Serial.print(" .. "); Serial.print(rng_hi_tmp); Serial.println(" us");
        rngAdvance(); break;
      case 'n': Serial.println("  (skipped)"); rngAdvance(); break;
      case 'b': rngRetreat(); break;
      case 't': rng_cur_us = trim_us[rng_cur_leg][rng_cur_joint]; rngApply();
                Serial.print("  us="); Serial.println(rng_cur_us); break;
      case 'h': homeLeg(rng_cur_leg); rng_cur_us = trim_us[rng_cur_leg][rng_cur_joint];
                Serial.print("  Homed "); Serial.println(LEGS[rng_cur_leg].name); break;
      case 'H': homeAll(); rng_cur_us = trim_us[rng_cur_leg][rng_cur_joint]; Serial.println("  Homed all."); break;
      case 'p': rngPrompt(); break;
      case 'l': rngList(); break;
      case 'm': homeAll(); current_mode = MODE_MAIN; return;
      case '?': rngHelp(); break;
      default:  break;
    }
  }
}

// ====================================================================
// =                      POLICY STREAM MODE                           =
// ====================================================================
// Tethered deployment: the host runs gait scaffold + trained policy
// (scripts/drive_serial.py) and streams joint targets; this side only
// converts kinematic angles to pulses. Line protocol (115200 8N1, '\n'):
//   J a0 a1 ... a17   18 joint angles in RADIANS, simulator convention and
//                     simulator leg order RR RM RF LR LM LF, each leg
//                     (coxa, femur, tibia). Applied immediately; replies "ok".
//   H                 home all legs (trim pose)
//   Q                 leave stream mode
// Safety: if no frame arrives for STREAM_TIMEOUT_MS the legs are homed and
// a line "timeout" is printed; streaming resumes on the next frame.
constexpr unsigned long STREAM_TIMEOUT_MS = 500;
// simulator leg index -> firmware LEGS[] index
const uint8_t SIM_TO_FW_LEG[6] = {2, 1, 0, 5, 4, 3};   // RR RM RF LR LM LF

void runStreamMode() {
  Serial.println();
  Serial.println("=== POLICY STREAM ===");
  Serial.println("Waiting for 'J a0 .. a17' frames (rad, sim order). 'H' home, 'Q' quit.");
  homeAll();
  unsigned long last_frame = millis();
  bool timed_out = false;
  String line;
  line.reserve(200);

  while (current_mode == MODE_STREAM) {
    while (Serial.available()) {
      char c = Serial.read();
      if (c == '\r') continue;
      if (c != '\n') { if (line.length() < 199) line += c; continue; }

      // full line
      if (line.length() == 0) { continue; }
      char head = line.charAt(0);
      if (head == 'Q' || head == 'q' || head == 'm') { homeAll(); current_mode = MODE_MAIN; line = ""; return; }
      if (head == 'H' || head == 'h') { homeAll(); Serial.println("ok home"); line = ""; continue; }
      if (head == 'J' || head == 'j') {
        float a[18];
        uint8_t n = 0;
        const char* p = line.c_str() + 1;
        char* end;
        while (n < 18) {
          float v = strtof(p, &end);
          if (end == p) break;
          a[n++] = v;
          p = end;
        }
        if (n != 18) { Serial.print("err n="); Serial.println(n); line = ""; continue; }
        for (uint8_t sl = 0; sl < 6; ++sl) {
          uint8_t fl = SIM_TO_FW_LEG[sl];
          cmdJointAngle(fl, 0, a[sl * 3 + 0]);
          cmdJointAngle(fl, 1, a[sl * 3 + 1]);
          cmdJointAngle(fl, 2, a[sl * 3 + 2]);
        }
        last_frame = millis();
        timed_out = false;
        Serial.println("ok");
      }
      line = "";
    }
    if (!timed_out && millis() - last_frame > STREAM_TIMEOUT_MS) {
      homeAll();
      timed_out = true;
      Serial.println("timeout");
    }
    delay(1);
  }
}

// ====================================================================
// =                        JSON EXPORT MODE                           =
// ====================================================================
// One JSON object between BEGIN_JSON / END_JSON. Copy it into
// mini_hexapod/calibration/<date>.json and run
//   python tools/import_mini_calibration.py <file>
// which prints the matching gait/robot_profile.py + MJCF generator values.
// Units: mm, us, rad. Leg order is the firmware's (FR MR RR FL ML RL).
void jsonArr3(const char* key, const uint16_t v[3]) {
  Serial.print("\""); Serial.print(key); Serial.print("\": [");
  for (uint8_t j = 0; j < 3; ++j) { Serial.print(v[j]); if (j < 2) Serial.print(", "); }
  Serial.print("]");
}

void runJsonMode() {
  Serial.println();
  Serial.println("BEGIN_JSON");
  Serial.println("{");
  Serial.println("  \"firmware\": \"mini_hexapod unified v0.2\",");
  Serial.print("  \"servo\": {\"min_us\": "); Serial.print(SERVO_MIN_US);
  Serial.print(", \"max_us\": "); Serial.print(SERVO_MAX_US);
  Serial.print(", \"us_per_deg\": "); Serial.print(US_PER_DEG, 4);
  Serial.print(", \"pwm_hz\": "); Serial.print(PWM_FREQ_HZ); Serial.println("},");
  Serial.print("  \"geometry_mm\": {\"L2\": "); Serial.print(L2, 2);
  Serial.print(", \"L3\": "); Serial.print(L3, 2);
  Serial.print(", \"R_HOME\": "); Serial.print(R_HOME, 2);
  Serial.print(", \"D_HOME\": "); Serial.print(D_HOME, 2); Serial.println("},");
  Serial.print("  \"home_kin_angle_rad\": [");
  for (uint8_t j = 0; j < 3; ++j) { Serial.print(HOME_KIN_ANGLE[j], 5); if (j < 2) Serial.print(", "); }
  Serial.println("],");
  Serial.print("  \"gait\": {\"step_length_mm\": "); Serial.print(step_length, 1);
  Serial.print(", \"step_height_mm\": "); Serial.print(step_height, 1);
  Serial.print(", \"step_period_ms\": "); Serial.print(step_period_ms); Serial.println("},");
  Serial.println("  \"joints\": [\"COXA\", \"FEMUR\", \"KNEE\"],");
  Serial.println("  \"legs\": [");
  for (uint8_t l = 0; l < 6; ++l) {
    Serial.print("    {\"name\": \""); Serial.print(LEGS[l].name);
    Serial.print("\", \"right\": "); Serial.print(LEGS[l].is_right ? "true" : "false");
    Serial.print(", \"board\": "); Serial.print(LEGS[l].board_addr);
    Serial.print(", \"channels\": ["); Serial.print(LEGS[l].channels[0]); Serial.print(", ");
    Serial.print(LEGS[l].channels[1]); Serial.print(", "); Serial.print(LEGS[l].channels[2]); Serial.println("],");
    Serial.print("     \"coxa_mm\": ["); Serial.print(COXA_X[l], 2); Serial.print(", ");
    Serial.print(COXA_Y[l], 2); Serial.print(", "); Serial.print(COXA_Z[l], 2);
    Serial.print("], \"axis_rad\": "); Serial.print(AXIS_RAD[l], 6); Serial.println(",");
    Serial.print("     "); jsonArr3("trim_us", trim_us[l]); Serial.println(",");
    Serial.print("     \"sign\": ["); 
    for (uint8_t j = 0; j < 3; ++j) { Serial.print((int)SIGN[l][j]); if (j < 2) Serial.print(", "); }
    Serial.println("],");
    Serial.print("     "); jsonArr3("range_lo_us", range_lo_us[l]); Serial.println(",");
    Serial.print("     "); jsonArr3("range_hi_us", range_hi_us[l]); Serial.println(",");
    Serial.print("     \"range_kin_rad\": [");
    for (uint8_t j = 0; j < 3; ++j) {
      uint16_t lo = range_lo_us[l][j], hi = range_hi_us[l][j];
      if (lo && hi) {
        float a = usToKinRad(l, j, lo), b = usToKinRad(l, j, hi);
        Serial.print("["); Serial.print(fminf(a, b), 4); Serial.print(", "); Serial.print(fmaxf(a, b), 4); Serial.print("]");
      } else {
        Serial.print("null");
      }
      if (j < 2) Serial.print(", ");
    }
    Serial.println("]");
    Serial.print("    }"); Serial.println(l < 5 ? "," : "");
  }
  Serial.println("  ]");
  Serial.println("}");
  Serial.println("END_JSON");
  Serial.println();
  Serial.println("Press any key for main menu.");
  readKey();
  current_mode = MODE_MAIN;
}

// ====================================================================
// =                          EXPORT MODE                              =
// ====================================================================
void runExportMode() {
  Serial.println();
  Serial.println("=== EXPORT — full configuration ===");

  Serial.println();
  Serial.println("// ---- Geometry ----");
  Serial.print("constexpr float L2 = ");     Serial.print(L2);     Serial.println("f;");
  Serial.print("constexpr float L3 = ");     Serial.print(L3);     Serial.println("f;");
  Serial.print("constexpr float R_HOME = "); Serial.print(R_HOME); Serial.println("f;");
  Serial.print("constexpr float D_HOME = "); Serial.print(D_HOME); Serial.println("f;");

  Serial.println("const float COXA_X[6] = {");
  for (uint8_t l = 0; l < 6; ++l) { Serial.print("  "); Serial.print(COXA_X[l]); Serial.print("f,  // "); Serial.println(LEGS[l].name); }
  Serial.println("};");
  Serial.println("const float COXA_Y[6] = {");
  for (uint8_t l = 0; l < 6; ++l) { Serial.print("  "); Serial.print(COXA_Y[l]); Serial.print("f,  // "); Serial.println(LEGS[l].name); }
  Serial.println("};");
  Serial.println("const float COXA_Z[6] = {");
  for (uint8_t l = 0; l < 6; ++l) { Serial.print("  "); Serial.print(COXA_Z[l]); Serial.print("f,  // "); Serial.println(LEGS[l].name); }
  Serial.println("};");
  Serial.println("const float AXIS_RAD[6] = {");
  for (uint8_t l = 0; l < 6; ++l) {
    Serial.print("  "); Serial.print(AXIS_RAD[l], 6);
    Serial.print("f,  // "); Serial.print(LEGS[l].name);
    Serial.print("  ("); Serial.print(AXIS_RAD[l] * 180.0f / PI); Serial.println(" deg)");
  }
  Serial.println("};");

  Serial.println();
  Serial.println("// ---- Trims (us) ----");
  trimDump();

  Serial.println();
  Serial.println("// ---- Signs ----");
  Serial.println("constexpr int8_t SIGN[6][3] = {");
  for (uint8_t l = 0; l < 6; ++l) {
    Serial.print("  /* "); Serial.print(LEGS[l].name); Serial.print(" */ {");
    for (uint8_t j = 0; j < 3; ++j) {
      Serial.print((SIGN[l][j] > 0) ? "+1" : "-1");
      if (j < 2) Serial.print(", ");
    }
    Serial.println("},");
  }
  Serial.println("};");

  Serial.println();
  Serial.println("// ---- Travel limits (us at each stop; 0 = not calibrated) ----");
  Serial.println("constexpr uint16_t RANGE_US[6][3][2] = {");
  for (uint8_t l = 0; l < 6; ++l) {
    Serial.print("  /* "); Serial.print(LEGS[l].name); Serial.print(" */ {");
    for (uint8_t j = 0; j < 3; ++j) {
      Serial.print("{"); Serial.print(range_lo_us[l][j]); Serial.print(", "); Serial.print(range_hi_us[l][j]); Serial.print("}");
      if (j < 2) Serial.print(", ");
    }
    Serial.println("},");
  }
  Serial.println("};");

  Serial.println();
  Serial.println("// ---- Gait ----");
  Serial.print("constexpr float STEP_LENGTH    = "); Serial.print(step_length);    Serial.println("f;");
  Serial.print("constexpr float STEP_HEIGHT    = "); Serial.print(step_height);    Serial.println("f;");
  Serial.print("constexpr unsigned long STEP_PERIOD_MS = "); Serial.print(step_period_ms); Serial.println(";");

  Serial.println();
  Serial.println("Press any key for main menu.");
  readKey();
  current_mode = MODE_MAIN;
}

// ====================================================================
// =                          SETUP / LOOP                             =
// ====================================================================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("==========================================");
  Serial.println("  mini_hexapod unified firmware  v0.2");
  Serial.println("==========================================");

  Wire.begin();
  pwm_right.begin();
  pwm_right.setOscillatorFrequency(27000000);
  pwm_right.setPWMFreq(PWM_FREQ_HZ);
  pwm_left.begin();
  pwm_left.setOscillatorFrequency(27000000);
  pwm_left.setPWMFreq(PWM_FREQ_HZ);
  delay(200);

  loadGeometry();
  loadTrims();
  loadSigns();
  loadGait();
  loadRanges();
  computeHomeKinAngles();

  Serial.println("Loaded geometry, trims, signs, gait params, travel limits from NVS.");
  Serial.println("Tip: set Serial Monitor to 'Newline' line ending.");
}

void loop() {
  switch (current_mode) {
    case MODE_MAIN:     handleMainMenu(); break;
    case MODE_GEOMETRY: runGeometryMode(); break;
    case MODE_TRIM:     runTrimMode(); break;
    case MODE_SIGN:     runSignMode(); break;
    case MODE_IK:       runIkMode(); break;
    case MODE_WALK:     runWalkMode(); break;
    case MODE_BT:       runBtMode(); break;
    case MODE_EXPORT:   runExportMode(); break;
    case MODE_RANGE:    runRangeMode(); break;
    case MODE_JSON:     runJsonMode(); break;
    case MODE_STREAM:   runStreamMode(); break;
  }
}
