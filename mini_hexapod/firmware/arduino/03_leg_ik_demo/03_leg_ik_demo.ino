// mini_hexapod stage 3: single-leg IK demo.
//
// Hardware: XIAO ESP32-C3 + 2x PCA9685 (right=0x40, left=0x41).
// Purpose:  Drive any leg's foot tip to commanded (x, y, z) in body frame
//           via inverse kinematics. Uses trims calibrated in stage 2 (loaded
//           from NVS). Verifies IK math + trims + per-joint sign conventions.
//
// Geometry (mm, body frame +X forward, +Y left, +Z up; origin = body center):
//   Coxa pivots and axis angles:
//     FR (+56.75, -32.0)   axis -45°
//     MR (  0.00, -42.4)   axis -90°
//     RR (-56.75, -32.0)   axis -135°
//     FL (+56.75, +32.0)   axis +45°
//     ML (  0.00, +42.4)   axis +90°
//     RL (-56.75, +32.0)   axis +135°
//   L2 (femur) = 55, L3 (tibia) = 89.5
//
// Joint conventions (the angles IK produces):
//   coxa  - CCW about body +Z, measured from this leg's coxa axis
//   femur - tilt ABOVE horizontal (positive = knee above coxa pivot)
//   knee  - bend DOWN from femur extension (positive = tibia rotates downward)
//
// SIGN[leg][joint]: +1 if servo us increases in same direction as kinematic
// angle, -1 if opposite. Defaults to +1 everywhere — edit and re-upload when
// you observe a joint moving the wrong way.
//
// Serial Monitor: 115200, "No line ending".
//
// Commands:
//   Leg select:  1=FR 2=MR 3=RR 4=FL 5=ML 6=RL
//   Move foot tip in body frame (current leg only):
//     w/s  +/- 1mm X (forward/back)    W/S  +/- 10mm
//     a/d  +/- 1mm Y (left/right)      A/D  +/- 10mm
//     r/f  +/- 1mm Z (up/down)         R/F  +/- 10mm
//   h    home current leg via IK (foot at standing pose)
//   H    home all 6 legs
//   z    zero current leg (joints to trim — kinematic zero, leg extended)
//   Z    zero all 6 legs
//   p    print state    ?    help
//
// Workflow: on boot all legs go to TRIM pose (fully extended outward — no IK,
// no SIGN involved, so any servo mount direction is OK).  Type 'h' to home
// current leg via IK. If a joint moves WRONG way, note which, flip the
// corresponding SIGN[][] entry below, re-upload.

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <Preferences.h>
#include <math.h>

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

struct LegGeom {
  float coxa_x, coxa_y, coxa_z;
  float axis_angle_rad;
};

struct IkResult {
  float coxa_rad, femur_rad, knee_rad;
  bool reachable;
};

const LegGeom GEOM[6] = {
  { +56.75f, -32.0f, 0.0f,        -PI / 4.0f },  // FR
  {   0.00f, -42.4f, 0.0f,        -PI / 2.0f },  // MR
  { -56.75f, -32.0f, 0.0f, -3.0f * PI / 4.0f },  // RR
  { +56.75f, +32.0f, 0.0f,        +PI / 4.0f },  // FL
  {   0.00f, +42.4f, 0.0f,        +PI / 2.0f },  // ML
  { -56.75f, +32.0f, 0.0f, +3.0f * PI / 4.0f },  // RL
};

constexpr float L2 = 55.0f;
constexpr float L3 = 89.5f;
constexpr float R_HOME = 55.0f;   // = L2. Foot horizontal offset from coxa pivot at L-pose rest.
constexpr float D_HOME = 89.5f;   // = L3. Foot depth below coxa pivot at L-pose rest.

// Per-joint sign table (+1 or -1). Loaded from NVS at boot — discovered by
// stage 2b (02b_sign_discovery). Default +1 for any joint not yet discovered.
int8_t SIGN[6][3];

// Kinematic joint angles at HOME pose. Computed at boot from the (R_HOME,
// D_HOME) target. Trim is treated as the REST pose (not geometric kinematic
// zero), so angleToUs subtracts these before converting to PWM delta. Result:
// when IK target == home pose, every servo sits exactly at trim — no movement.
float HOME_KIN_ANGLE[3] = {0.0f, 0.0f, 0.0f};

Adafruit_PWMServoDriver& driverFor(const LegSpec& leg) {
  return leg.board_addr == ADDR_RIGHT ? pwm_right : pwm_left;
}

uint16_t usToTicks(uint16_t us) {
  return (uint16_t)((uint32_t)us * PWM_FREQ_HZ * 4096UL / 1000000UL);
}

// --- State ---
Preferences prefs;
uint16_t trim_us[6][3];
float target_x[6], target_y[6], target_z[6];
uint8_t cur_leg = 0;

void loadTrims() {
  prefs.begin("hexa_trim", true);
  for (uint8_t l = 0; l < 6; ++l) {
    for (uint8_t j = 0; j < 3; ++j) {
      char k[16];
      snprintf(k, 16, "l%d_j%d", l, j);
      trim_us[l][j] = prefs.getUShort(k, SERVO_CTR_US);
    }
  }
  prefs.end();
}

void loadSigns() {
  prefs.begin("hexa_sign", true);
  for (uint8_t l = 0; l < 6; ++l) {
    for (uint8_t j = 0; j < 3; ++j) {
      char k[16];
      snprintf(k, 16, "l%d_j%d", l, j);
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
    return r;  // unreachable
  }

  // Knee-up config: femur tilts CCW from coxa-to-foot line in the vertical plane.
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
  // Trim represents the HOME pose (not geometric kinematic zero) — so subtract
  // home-pose angle first to get the delta that should drive the servo.
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
  if (!ik.reachable) {
    Serial.print("  UNREACHABLE: (");
    Serial.print(fx); Serial.print(", ");
    Serial.print(fy); Serial.print(", ");
    Serial.print(fz); Serial.println(")");
    return false;
  }
  cmdJointAngle(leg, 0, ik.coxa_rad);
  cmdJointAngle(leg, 1, ik.femur_rad);
  cmdJointAngle(leg, 2, ik.knee_rad);
  target_x[leg] = fx;
  target_y[leg] = fy;
  target_z[leg] = fz;
  return true;
}

void homePoseForLeg(uint8_t leg, float& x, float& y, float& z) {
  const LegGeom& g = GEOM[leg];
  x = g.coxa_x + R_HOME * cosf(g.axis_angle_rad);
  y = g.coxa_y + R_HOME * sinf(g.axis_angle_rad);
  z = g.coxa_z - D_HOME;
}

void computeHomeKinAngles() {
  // Geometry is symmetric across legs at home, so any leg's IK yields the
  // same kinematic angles. Use FR (leg 0) as the reference.
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
  float x, y, z;
  homePoseForLeg(leg, x, y, z);
  commandFoot(leg, x, y, z);
}

void homeAll() {
  for (uint8_t l = 0; l < 6; ++l) homeLeg(l);
}

void zeroLeg(uint8_t leg) {
  // Trim corresponds to home pose under the rest-pose-trim convention, so
  // command joints directly to trim AND set the tracked target to home.
  for (uint8_t j = 0; j < 3; ++j) cmdJointTrim(leg, j);
  homePoseForLeg(leg, target_x[leg], target_y[leg], target_z[leg]);
}

void zeroAll() {
  for (uint8_t l = 0; l < 6; ++l) zeroLeg(l);
}

// --- UI ---
void printState() {
  const LegSpec& L = LEGS[cur_leg];
  Serial.println();
  Serial.print("[Leg "); Serial.print(L.name); Serial.print("] target_body_mm = (");
  Serial.print(target_x[cur_leg]); Serial.print(", ");
  Serial.print(target_y[cur_leg]); Serial.print(", ");
  Serial.print(target_z[cur_leg]); Serial.println(")");
  IkResult ik = ikFootBody(cur_leg, target_x[cur_leg], target_y[cur_leg], target_z[cur_leg]);
  if (!ik.reachable) {
    Serial.println("  IK: UNREACHABLE");
    return;
  }
  Serial.print("  joint_deg: coxa="); Serial.print(ik.coxa_rad * 180.0f / PI);
  Serial.print(" femur=");            Serial.print(ik.femur_rad * 180.0f / PI);
  Serial.print(" knee=");             Serial.println(ik.knee_rad * 180.0f / PI);
  Serial.print("  trim_us:   coxa="); Serial.print(trim_us[cur_leg][0]);
  Serial.print(" femur=");            Serial.print(trim_us[cur_leg][1]);
  Serial.print(" knee=");             Serial.println(trim_us[cur_leg][2]);
  Serial.print("  sign:      coxa="); Serial.print(SIGN[cur_leg][0]);
  Serial.print(" femur=");            Serial.print(SIGN[cur_leg][1]);
  Serial.print(" knee=");             Serial.println(SIGN[cur_leg][2]);
  Serial.print("  servo_us:  coxa="); Serial.print(angleToUs(cur_leg, 0, ik.coxa_rad));
  Serial.print(" femur=");            Serial.print(angleToUs(cur_leg, 1, ik.femur_rad));
  Serial.print(" knee=");             Serial.println(angleToUs(cur_leg, 2, ik.knee_rad));
}

void printHelp() {
  Serial.println();
  Serial.println("Commands:");
  Serial.println("  1-6  select leg (FR/MR/RR/FL/ML/RL)");
  Serial.println("  w/s  +/- 1mm X    W/S  +/- 10mm    (X = forward/back, body frame)");
  Serial.println("  a/d  +/- 1mm Y    A/D  +/- 10mm    (Y = left/right)");
  Serial.println("  r/f  +/- 1mm Z    R/F  +/- 10mm    (Z = up/down)");
  Serial.println("  h    home current leg via IK     H    home all");
  Serial.println("  z    zero current leg (trim)     Z    zero all");
  Serial.println("  p    print state                 ?    help");
}

void nudge(float dx, float dy, float dz) {
  float nx = target_x[cur_leg] + dx;
  float ny = target_y[cur_leg] + dy;
  float nz = target_z[cur_leg] + dz;
  if (commandFoot(cur_leg, nx, ny, nz)) {
    Serial.print("  target=("); Serial.print(nx); Serial.print(", ");
    Serial.print(ny); Serial.print(", "); Serial.print(nz); Serial.println(")");
  }
}

void handleCommand(char c) {
  switch (c) {
    case '1': case '2': case '3': case '4': case '5': case '6':
      cur_leg = c - '1';
      Serial.print("Selected leg: "); Serial.println(LEGS[cur_leg].name);
      printState();
      break;
    case 'w': nudge(+1, 0, 0); break;
    case 's': nudge(-1, 0, 0); break;
    case 'a': nudge(0, +1, 0); break;
    case 'd': nudge(0, -1, 0); break;
    case 'r': nudge(0, 0, +1); break;
    case 'f': nudge(0, 0, -1); break;
    case 'W': nudge(+10, 0, 0); break;
    case 'S': nudge(-10, 0, 0); break;
    case 'A': nudge(0, +10, 0); break;
    case 'D': nudge(0, -10, 0); break;
    case 'R': nudge(0, 0, +10); break;
    case 'F': nudge(0, 0, -10); break;
    case 'h':
      Serial.print("Homing leg "); Serial.println(LEGS[cur_leg].name);
      homeLeg(cur_leg);
      printState();
      break;
    case 'H':
      Serial.println("Homing all legs.");
      homeAll();
      break;
    case 'z':
      Serial.print("Zeroing leg "); Serial.println(LEGS[cur_leg].name);
      zeroLeg(cur_leg);
      break;
    case 'Z':
      Serial.println("Zeroing all legs.");
      zeroAll();
      break;
    case 'p':
      printState();
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
  Serial.println("=== mini_hexapod stage 3: single-leg IK demo ===");

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

  for (uint8_t l = 0; l < 6; ++l) {
    homePoseForLeg(l, target_x[l], target_y[l], target_z[l]);
  }

  Serial.println("Loaded trims from NVS. Driving all joints to TRIM (kinematic zero).");
  Serial.println("Body should be on a stand. Holding 3s before commanding servos...");
  delay(3000);
  zeroAll();
  delay(500);

  Serial.println();
  Serial.println("All legs at kinematic zero (fully extended outward).");
  Serial.println("Type 'h' to drive FR to its home pose via IK; '?' for help.");
  printState();
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r' || c == ' ' || c == '\t') return;
    handleCommand(c);
  }
}
