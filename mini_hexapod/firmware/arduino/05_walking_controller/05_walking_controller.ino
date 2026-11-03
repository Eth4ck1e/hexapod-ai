// mini_hexapod stage 5: walking controller firmware.
//
// Hardware: XIAO ESP32-C3 + 2x PCA9685 (right=0x40, left=0x41).
// Purpose:  Directional walking controller, similar to the watch interactive
//           but command-driven from Serial. Each motion command runs for
//           ACTION_DURATION_MS, then returns to home and waits for next.
//
// Gait: tripod (A: FR/ML/RR  +  B: FL/MR/RL, 180 deg phase offset).
// Each leg's foot trajectory in body frame:
//   Swing  (phase 0..0.5):  foot lifts and moves +motion_dir * STEP_LENGTH
//   Stance (phase 0.5..1):  foot at ground level, moves -motion_dir * STEP_LENGTH
// motion_dir is computed per-leg from (vx, vy, omega_z):
//   v_at_leg = (vx - py*omega_z, vy + px*omega_z)
//   motion_dir = v_at_leg / |v_at_leg|
// For pure translation, all legs get the same direction. For yaw, each leg
// gets a tangent direction CCW about body +Z.
//
// Serial Monitor: 115200, "No line ending". Any keystroke during motion
// interrupts the current action; if it's a movement key, the new motion
// starts immediately after a quick home.
//
// Commands (each motion runs 5 seconds unless interrupted):
//   w  forward       s  backward
//   a  strafe left   d  strafe right
//   q  turn left     e  turn right       (yaw CCW / CW)
//   r  fwd-left      t  fwd-right        (diagonals)
//   f  back-left     g  back-right
//   h  home all (snap to L-pose, stops any motion)
//   [ / ]  step length -5 / +5 mm  (limits 10..50)
//   , / .  step height -2 / +2 mm  (limits 5..50)
//   < / >  cycle period -200 / +200 ms (limits 500..3000, lower = faster)
//   p  print current step length / height / period
//   ?  help

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
constexpr float R_HOME = 55.0f;
constexpr float D_HOME = 89.5f;

// Runtime-tunable gait parameters.
float         step_length    = 30.0f;
float         step_height    = 25.0f;
unsigned long step_period_ms = 1500;

constexpr unsigned long ACTION_DURATION_MS = 5000;

int8_t SIGN[6][3];
float  HOME_KIN_ANGLE[3] = {0.0f, 0.0f, 0.0f};

// Tripod gait layout. A = front-right, mid-left, rear-right. B = the others.
const uint8_t TRIPOD_LEGS[6] = {0, 4, 2,  3, 1, 5};
const float   TRIPOD_OFFS[6] = {0, 0, 0,  0.5f, 0.5f, 0.5f};

// Interrupt mechanism: a key pressed during motion is buffered here and
// dispatched after the motion loop exits.
char pending_cmd = 0;

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

// --- Interrupt handling ---
bool checkInterrupt() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
    pending_cmd = c;
    return true;
  }
  return false;
}

// --- Motion runner ---
// vx, vy, omega_z are unit-scale: pure translation in (vx, vy); pure rotation
// via omega_z about body +Z (CCW positive). Magnitudes only matter relative
// to each other; per-leg direction is normalized.
void runMotion(const char* label, float vx, float vy, float omega_z) {
  Serial.print(">>> "); Serial.println(label);

  float dir_x[6], dir_y[6];
  for (uint8_t l = 0; l < 6; ++l) {
    const LegGeom& g = GEOM[l];
    float vxl = vx + (-g.coxa_y) * omega_z;
    float vyl = vy + ( g.coxa_x) * omega_z;
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
    if (checkInterrupt()) {
      Serial.println("  interrupted");
      break;
    }
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

        float hx, hy, hz;
        homePoseForLeg(leg, hx, hy, hz);
        float fx = hx + disp * dir_x[leg];
        float fy = hy + disp * dir_y[leg];
        float fz = hz + lift;
        commandFoot(leg, fx, fy, fz);
      }
    }
    delay(2);
  }

  homeAll();
  delay(300);
}

// --- UI ---
void printHelp() {
  Serial.println();
  Serial.println("Movement (5 sec each, interruptible):");
  Serial.println("  w  forward         s  backward");
  Serial.println("  a  strafe left     d  strafe right");
  Serial.println("  q  turn left CCW   e  turn right CW");
  Serial.println("  r  fwd-left        t  fwd-right       (translation diagonals)");
  Serial.println("  f  back-left       g  back-right");
  Serial.println("  1  arc fwd-left    2  arc fwd-right   (curved walking)");
  Serial.println("  3  arc back-left   4  arc back-right");
  Serial.println("Adjust:");
  Serial.println("  [ / ]   step length -/+ 5mm  (10..50)");
  Serial.println("  , / .   step height -/+ 2mm  (5..50)");
  Serial.println("  < / >   step period -/+ 200ms (500..3000, smaller=faster)");
  Serial.println("Housekeeping:");
  Serial.println("  h    home all     p    print current params     ?    help");
}

void printParams() {
  Serial.print("step_length="); Serial.print(step_length);
  Serial.print(" mm  step_height="); Serial.print(step_height);
  Serial.print(" mm  step_period="); Serial.print(step_period_ms);
  Serial.println(" ms");
}

void adjustLen(float dv) {
  step_length += dv;
  if (step_length < 10.0f) step_length = 10.0f;
  if (step_length > 50.0f) step_length = 50.0f;
  printParams();
}

void adjustHt(float dv) {
  step_height += dv;
  if (step_height < 5.0f)  step_height = 5.0f;
  if (step_height > 50.0f) step_height = 50.0f;
  printParams();
}

void adjustPer(long dv) {
  long p = (long)step_period_ms + dv;
  if (p < 500)  p = 500;
  if (p > 3000) p = 3000;
  step_period_ms = (unsigned long)p;
  printParams();
}

void handleCommand(char c) {
  switch (c) {
    // Cardinal
    case 'w': runMotion("forward",         +1.0f,  0.0f,  0.0f); break;
    case 's': runMotion("backward",        -1.0f,  0.0f,  0.0f); break;
    case 'a': runMotion("strafe left",      0.0f, +1.0f,  0.0f); break;
    case 'd': runMotion("strafe right",     0.0f, -1.0f,  0.0f); break;
    // Yaw
    case 'q': runMotion("turn left  (CCW)", 0.0f,  0.0f, +1.0f); break;
    case 'e': runMotion("turn right (CW)",  0.0f,  0.0f, -1.0f); break;
    // Diagonals (unit-magnitude)
    case 'r': runMotion("fwd-left",  +0.707f, +0.707f, 0.0f); break;
    case 't': runMotion("fwd-right", +0.707f, -0.707f, 0.0f); break;
    case 'f': runMotion("back-left", -0.707f, +0.707f, 0.0f); break;
    case 'g': runMotion("back-right",-0.707f, -0.707f, 0.0f); break;
    // Arcs (translation + yaw together — curved walking)
    case '1': runMotion("arc fwd-left",   +1.0f, 0.0f, +0.5f); break;
    case '2': runMotion("arc fwd-right",  +1.0f, 0.0f, -0.5f); break;
    case '3': runMotion("arc back-left",  -1.0f, 0.0f, +0.5f); break;
    case '4': runMotion("arc back-right", -1.0f, 0.0f, -0.5f); break;
    // Housekeeping
    case 'h': case 'H':
      homeAll();
      Serial.println("Homed.");
      break;
    case 'p': case 'P': printParams(); break;
    case '?': printHelp(); break;
    // Adjustments
    case '[': adjustLen(-5.0f); break;
    case ']': adjustLen(+5.0f); break;
    case ',': adjustHt(-2.0f);  break;
    case '.': adjustHt(+2.0f);  break;
    case '<': adjustPer(-200);  break;
    case '>': adjustPer(+200);  break;
    default: break;
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("=== mini_hexapod stage 5: walking controller ===");

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

  printParams();
  Serial.println("Ready. ? for help.");
}

void loop() {
  char c = 0;
  if (pending_cmd) {
    c = pending_cmd;
    pending_cmd = 0;
  } else if (Serial.available()) {
    c = Serial.read();
    if (c == '\n' || c == '\r' || c == ' ' || c == '\t') return;
  } else {
    return;
  }
  handleCommand(c);
}
