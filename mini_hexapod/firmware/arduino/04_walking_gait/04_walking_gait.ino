// mini_hexapod stage 4: walking gait demo.
//
// Hardware: XIAO ESP32-C3 + 2x PCA9685 (right=0x40, left=0x41).
// Purpose:  Animate leg swing trajectories around the L-pose rest, to verify
//           IK + signs produce a recognizable walking motion. Body should be
//           on a stand (legs swing in air, no ground contact needed).
//
// Trajectory per leg (one cycle = STEP_PERIOD_MS):
//   Phase 0.0 - 0.5  SWING:  foot lifts, moves from -X back to +X forward
//   Phase 0.5 - 1.0  STANCE: foot at ground level, moves from +X to -X
// In body-frame coordinates, all legs move toward body +X during swing and
// toward body -X during stance => forward walking direction.
//
// Tripod gait pairs:
//   Tripod A: FR, ML, RR
//   Tripod B: FL, MR, RL
//   B runs at phase + 0.5 relative to A.
//
// Serial Monitor: 115200, "No line ending".
//
// Commands:
//   1-6  walk single leg (FR/MR/RR/FL/ML/RL), 3 cycles
//   R    walk right side  (FR + MR + RR in unison), 5 cycles
//   L    walk left side   (FL + ML + RL in unison), 5 cycles
//   T    tripod gait (all 6, alternating tripods), 10 cycles
//   g    full sequence: 1..6 -> R -> L -> T  (~65 sec total)
//   s    stop the currently running gait (handled inline)
//   h    home all legs (snap to trim/L-pose)
//   ?    help

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
};

const LegSpec LEGS[6] = {
  {"FR", ADDR_RIGHT, {0, 1, 2}},
  {"MR", ADDR_RIGHT, {3, 4, 5}},
  {"RR", ADDR_RIGHT, {6, 7, 8}},
  {"FL", ADDR_LEFT,  {0, 1, 2}},
  {"ML", ADDR_LEFT,  {3, 4, 5}},
  {"RL", ADDR_LEFT,  {6, 7, 8}},
};

struct LegGeom {
  float coxa_x, coxa_y, coxa_z;
  float axis_angle_rad;
};

const LegGeom GEOM[6] = {
  { +56.75f, -32.0f, 0.0f,        -PI / 4.0f },  // FR
  {   0.00f, -42.4f, 0.0f,        -PI / 2.0f },  // MR
  { -56.75f, -32.0f, 0.0f, -3.0f * PI / 4.0f },  // RR
  { +56.75f, +32.0f, 0.0f,        +PI / 4.0f },  // FL
  {   0.00f, +42.4f, 0.0f,        +PI / 2.0f },  // ML
  { -56.75f, +32.0f, 0.0f, +3.0f * PI / 4.0f },  // RL
};

struct IkResult {
  float coxa_rad, femur_rad, knee_rad;
  bool reachable;
};

constexpr float L2 = 55.0f;
constexpr float L3 = 89.5f;
constexpr float R_HOME = 55.0f;   // = L2: foot horizontal offset at L-pose rest
constexpr float D_HOME = 89.5f;   // = L3: foot depth below coxa pivot at L-pose rest

constexpr float         STEP_LENGTH    = 30.0f;   // mm peak-to-peak in body +X
constexpr float         STEP_HEIGHT    = 25.0f;   // mm foot lift during swing
constexpr unsigned long STEP_PERIOD_MS = 1500;    // one full swing+stance cycle
constexpr int           CYCLES_SOLO    = 3;
constexpr int           CYCLES_SIDE    = 5;
constexpr int           CYCLES_TRIPOD  = 10;

int8_t SIGN[6][3];
float  HOME_KIN_ANGLE[3] = {0.0f, 0.0f, 0.0f};

Adafruit_PWMServoDriver& driverFor(const LegSpec& leg) {
  return leg.board_addr == ADDR_RIGHT ? pwm_right : pwm_left;
}

uint16_t usToTicks(uint16_t us) {
  return (uint16_t)((uint32_t)us * PWM_FREQ_HZ * 4096UL / 1000000UL);
}

// --- State ---
Preferences prefs;
uint16_t trim_us[6][3];

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
      SIGN[l][j] = (int8_t)prefs.getChar(k, (int8_t)+1);
    }
  }
  prefs.end();
}

// --- Kinematics ---
IkResult ikFootBody(uint8_t leg, float fx, float fy, float fz) {
  const LegGeom& g = GEOM[leg];
  float dx = fx - g.coxa_x;
  float dy = fy - g.coxa_y;
  float dz = fz - g.coxa_z;

  IkResult r;
  r.reachable = false;
  r.coxa_rad = r.femur_rad = r.knee_rad = 0.0f;

  float target_angle = atan2f(dy, dx);
  float coxa = target_angle - g.axis_angle_rad;
  while (coxa >  PI) coxa -= 2.0f * PI;
  while (coxa < -PI) coxa += 2.0f * PI;

  float horiz = sqrtf(dx*dx + dy*dy);
  float d2 = horiz*horiz + dz*dz;
  float d = sqrtf(d2);

  if (d > (L2 + L3 + 0.01f) || d < fabsf(L2 - L3) - 0.01f) {
    return r;
  }

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
  float us = (float)trim_us[leg][joint] + deg * ((float)(SERVO_MAX_US - SERVO_MIN_US) / 180.0f);
  if (us < (float)SERVO_MIN_US) us = SERVO_MIN_US;
  if (us > (float)SERVO_MAX_US) us = SERVO_MAX_US;
  return (uint16_t)(us + 0.5f);
}

void cmdJointAngle(uint8_t leg, uint8_t joint, float angle_rad) {
  const LegSpec& L = LEGS[leg];
  uint16_t us = angleToUs(leg, joint, angle_rad);
  driverFor(L).setPWM(L.channels[joint], 0, usToTicks(us));
}

void cmdJointTrim(uint8_t leg, uint8_t joint) {
  const LegSpec& L = LEGS[leg];
  driverFor(L).setPWM(L.channels[joint], 0, usToTicks(trim_us[leg][joint]));
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
  const LegGeom& g = GEOM[leg];
  x = g.coxa_x + R_HOME * cosf(g.axis_angle_rad);
  y = g.coxa_y + R_HOME * sinf(g.axis_angle_rad);
  z = g.coxa_z - D_HOME;
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

void homeLeg(uint8_t leg) {
  for (uint8_t j = 0; j < 3; ++j) cmdJointTrim(leg, j);
}

void homeAll() {
  for (uint8_t l = 0; l < 6; ++l) homeLeg(l);
}

// --- Trajectory ---
// Foot trajectory at given phase, returned as body-frame (x,y,z) mm.
void footTrajectoryFor(uint8_t leg, float phase, float& out_x, float& out_y, float& out_z) {
  float hx, hy, hz;
  homePoseForLeg(leg, hx, hy, hz);

  if (phase < 0.5f) {
    // Swing: lift, move from -X back to +X forward
    float s = phase * 2.0f;  // 0..1
    out_x = hx + STEP_LENGTH * (s - 0.5f);
    out_y = hy;
    out_z = hz + STEP_HEIGHT * sinf(PI * s);
  } else {
    // Stance: at ground level, move from +X back to -X
    float s = (phase - 0.5f) * 2.0f;
    out_x = hx + STEP_LENGTH * (0.5f - s);
    out_y = hy;
    out_z = hz;
  }
}

// --- Gait runner ---
bool checkStop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 's' || c == 'S') return true;
  }
  return false;
}

void runGait(const uint8_t* legs, uint8_t n_legs, const float* phase_offsets, int cycles) {
  unsigned long start_ms = millis();
  unsigned long total_ms = (unsigned long)STEP_PERIOD_MS * cycles;
  unsigned long last_update = 0;

  while (true) {
    unsigned long now = millis();
    unsigned long elapsed = now - start_ms;
    if (elapsed >= total_ms) break;
    if (checkStop()) {
      Serial.println("  stopped.");
      break;
    }
    if (now - last_update >= 25) {  // ~40 Hz updates
      last_update = now;
      for (uint8_t i = 0; i < n_legs; ++i) {
        uint8_t leg = legs[i];
        float phase = fmodf((float)elapsed / (float)STEP_PERIOD_MS + phase_offsets[i], 1.0f);
        float fx, fy, fz;
        footTrajectoryFor(leg, phase, fx, fy, fz);
        commandFoot(leg, fx, fy, fz);
      }
    }
    delay(2);
  }
  // Return all involved legs to home
  for (uint8_t i = 0; i < n_legs; ++i) homeLeg(legs[i]);
  delay(400);
}

void runSingleLeg(uint8_t leg) {
  Serial.print(">>> walking leg "); Serial.println(LEGS[leg].name);
  uint8_t legs[1]; legs[0] = leg;
  float offsets[1] = {0.0f};
  runGait(legs, 1, offsets, CYCLES_SOLO);
}

void runRightSide() {
  Serial.println(">>> walking right side (FR + MR + RR, in unison)");
  uint8_t legs[3] = {0, 1, 2};
  float offsets[3] = {0.0f, 0.0f, 0.0f};
  runGait(legs, 3, offsets, CYCLES_SIDE);
}

void runLeftSide() {
  Serial.println(">>> walking left side (FL + ML + RL, in unison)");
  uint8_t legs[3] = {3, 4, 5};
  float offsets[3] = {0.0f, 0.0f, 0.0f};
  runGait(legs, 3, offsets, CYCLES_SIDE);
}

void runTripod() {
  Serial.println(">>> tripod gait  A: FR/ML/RR   B: FL/MR/RL");
  //               A             B
  uint8_t legs[6] = {0, 4, 2,    3, 1, 5};
  float offsets[6] = {0, 0, 0,   0.5f, 0.5f, 0.5f};
  runGait(legs, 6, offsets, CYCLES_TRIPOD);
}

void runFullSequence() {
  Serial.println("=== Full walking sequence ===");
  for (uint8_t l = 0; l < 6; ++l) {
    runSingleLeg(l);
    if (checkStop()) { Serial.println("Sequence stopped."); return; }
    delay(300);
  }
  runRightSide();
  if (checkStop()) { Serial.println("Sequence stopped."); return; }
  delay(300);
  runLeftSide();
  if (checkStop()) { Serial.println("Sequence stopped."); return; }
  delay(300);
  runTripod();
  Serial.println("=== Sequence complete ===");
}

// --- UI ---
void printHelp() {
  Serial.println();
  Serial.println("Commands:");
  Serial.println("  1-6  walk single leg (FR/MR/RR/FL/ML/RL), 3 cycles");
  Serial.println("  R    walk right side (FR+MR+RR), 5 cycles");
  Serial.println("  L    walk left side  (FL+ML+RL), 5 cycles");
  Serial.println("  T    tripod gait, 10 cycles");
  Serial.println("  g    full sequence (1..6 -> R -> L -> T)");
  Serial.println("  s    stop current gait");
  Serial.println("  h    home all to L-pose");
  Serial.println("  ?    help");
}

void handleCommand(char c) {
  switch (c) {
    case '1': case '2': case '3': case '4': case '5': case '6':
      runSingleLeg(c - '1');
      break;
    case 'R': case 'r': runRightSide(); break;
    case 'L': case 'l': runLeftSide(); break;
    case 'T': case 't': runTripod(); break;
    case 'g': case 'G': runFullSequence(); break;
    case 'h': case 'H': homeAll(); Serial.println("Homed all."); break;
    case '?': printHelp(); break;
    default: break;
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("=== mini_hexapod stage 4: walking gait demo ===");

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
  computeHomeKinAngles();

  Serial.println("Loaded trims + signs. Body on a stand. Homing in 3s...");
  delay(3000);
  homeAll();
  delay(500);

  Serial.println("Ready. ? for help.  Try '1' for FR solo, or 'g' for full sequence.");
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r' || c == ' ' || c == '\t') return;
    handleCommand(c);
  }
}
