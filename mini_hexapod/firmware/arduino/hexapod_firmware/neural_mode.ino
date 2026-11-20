// neural_mode.ino — menu 'N': the learned controller running ON the bot.
//
// The 21-input network in policy_weights.h (cloned from the gait dataset, see
// neural_core.h) is evaluated here at 50 Hz from a gait clock and the current
// command, and its 18 joint targets go straight to the servos through the usual
// trim / sign path. No host PC, no scaffold: the only hand-written motion left is
// the standing pose used while idle (the network never saw "stand still").
//
// Commands are text lines, the same words scripts/drive_serial.py --source console
// takes, so anything that can send a line of text (USB serial today, a BLE serial
// app or a gamepad bridge tomorrow) can drive the bot:
//   forward|back|left|right|fl|fr|bl|br [speed 0-1] [for SEC]     walk (default 5 s)
//   spin left|right [scale] [for SEC]      arc left|right [speed] [for SEC]
//   sweep [speed] [for SEC]                heading turns 0 -> 360 deg while walking (12 s)
//   up MM | down MM | wide MM | narrow MM  stance, persistent, ramped 20 mm/s
//   speed F        default speed fraction (trained band 0.40-0.85)
//   autorest SEC   squat + go limp after SEC idle (0 = off)
//   stop | home | rest | stats | help | q
#include "neural_core.h"
#include "web_link.h"

constexpr unsigned long NC_TICK_MS      = 20;      // one PCA9685 PWM frame
constexpr float NC_DT                   = 0.020f;
constexpr float NC_VEL_RAMP_S           = 0.4f;    // 0 -> full commanded velocity
constexpr float NC_FADE_S               = 0.5f;    // network <-> standing pose blend
constexpr float NC_STANCE_RAMP_M_S      = 0.020f;  // height / width changes
constexpr float NC_DEFAULT_DURATION_S   = 5.0f;
constexpr float NC_SWEEP_DEFAULT_S      = 12.0f;

struct NeuralCtl {
  float vel[3], vel_now[3];                 // commanded / ramped (vx, vy, wz)
  float dh, dw, dh_goal, dw_goal, dh_before_rest;
  float pitch, roll, pitch_goal, roll_goal;  // body tilt (0 unless the policy was trained with it)
  float gain;                               // 1 = network, 0 = standing pose
  float speed_frac;
  float move_end_s, hold_until_s, idle_since_s;
  bool  sweep; float sweep_speed, sweep_dur;
  bool  resting, rest_pending, paused, was_moving, gp_driving, gp_spin_hold, web_driving;
  unsigned long tele_ticks;
  uint8_t gp_prev_stick;
  float gp_stick_vel[2], gp_stick_t;        // last non-zero stick command (for spin-while-walking)
  float auto_rest_s;
  char  label[40];
  // stats
  unsigned long ticks, late_ticks, max_tick_us, max_net_us;
};

static NeuralCtl nc;
static void ncWake(float t);

// Print to the serial console AND mirror to the web page's log strip.
static void ncSay(const char* msg) {
  Serial.print("  "); Serial.println(msg);
  char m[128]; snprintf(m, sizeof(m), "M %s", msg); webSend(m);
}

static void ncHelp() {
  Serial.println("  forward|back|left|right|fl|fr|bl|br [speed 0-1] [for SEC]   walk (default 5 s)");
  Serial.println("  spin left|right [scale] [for SEC]     arc left|right [speed] [for SEC]");
  Serial.println("  sweep [speed] [for SEC]               heading 0->360 deg while walking (12 s)");
  Serial.println("  up MM | down MM | wide MM | narrow MM  stance (persistent)   speed F   autorest SEC");
  Serial.println("  stop | home | rest | stats | help | q");
  Serial.println("  phone: join WiFi 'hexapod' / 'hexapod123', open http://192.168.4.1/  (web controller)");
  Serial.println("         or Dabble gamepad over BLE, or any BLE serial app sending these words.");
  Serial.println("  wifi SSID PASS   also join your own network (saved); 'wifi' shows addresses");
}

static void ncPrintStance() {
  char m[96];
  snprintf(m, sizeof(m), "stance height %.0f mm (range %.0f..%.0f)  width %.0f mm", -nc.dh_goal * 1000.0f,
           -POLICY_DH_MAX * 1000.0f, -POLICY_DH_MIN * 1000.0f, nc.dw_goal * 1000.0f);
  ncSay(m);
}

static void ncRest(float t) {
  if (nc.resting || nc.rest_pending) return;
  nc.move_end_s = t; nc.sweep = false; nc.paused = false;
  nc.dh_before_rest = nc.dh_goal;
  nc.dh_goal = POLICY_DH_MAX;
  nc.dw_goal = nc_clamp_dw(nc.dh_goal, nc.dw_goal);
  nc.rest_pending = true;
  char m[96]; snprintf(m, sizeof(m), "v resting: squatting %.0f mm, then servos go limp", (POLICY_DH_MAX - nc.dh) * 1000.0f); ncSay(m);
}

static void ncStartMove(const float* vel, const char* label, float t, float dur) {
  for (int i = 0; i < 3; ++i) nc.vel[i] = vel[i];
  if (dur <= 0.0f) dur = NC_DEFAULT_DURATION_S;
  nc.move_end_s = t + dur;
  nc.sweep = false;
  nc.paused = false;
  strncpy(nc.label, label, sizeof(nc.label) - 1); nc.label[sizeof(nc.label) - 1] = 0;
  char m[96]; snprintf(m, sizeof(m), "> %s for %.1f s", label, dur); ncSay(m);
  if (nc.resting || nc.rest_pending) {            // stand back up first
    ncWake(t);
    float delay_s = nc.hold_until_s - t;
    nc.move_end_s += delay_s;
    snprintf(m, sizeof(m), "^ standing up (%.1f s), then the move", delay_s); ncSay(m);
  }
}

// "[speed] [for SEC] [SECs]" -> speed fraction (or -1) and duration (or -1).
static void ncParseTail(char** tok, int n, float* frac, float* dur) {
  *frac = -1.0f; *dur = -1.0f;
  for (int i = 0; i < n; ++i) {
    if (strcmp(tok[i], "for") == 0 && i + 1 < n) { *dur = atof(tok[i + 1]); ++i; continue; }
    size_t L = strlen(tok[i]);
    if (L > 1 && tok[i][L - 1] == 's' && isdigit((unsigned char)tok[i][0])) { *dur = atof(tok[i]); continue; }
    char* end; float v = strtof(tok[i], &end);
    if (end == tok[i]) { Serial.print("  ? unexpected token "); Serial.println(tok[i]); continue; }
    if (v <= 1.0f) *frac = v; else *dur = v;
  }
}

static float ncSpeedFrac(float frac) {
  if (frac < 0.0f) frac = nc.speed_frac;
  return nc_clampf(frac, POLICY_SPEED_MIN_FRAC, POLICY_SPEED_MAX_FRAC);
}

static void ncHandle(char* line, float t) {
  char* tok[8]; int n = 0;
  for (char* p = strtok(line, " \t"); p && n < 8; p = strtok(NULL, " \t")) tok[n++] = p;
  if (n == 0) return;
  for (char* q = tok[0]; *q; ++q) *q = tolower(*q);
  if (strcmp(tok[0], "wifi") != 0)                       // SSID / password keep their case
    for (int i = 1; i < n; ++i) for (char* q = tok[i]; *q; ++q) *q = tolower(*q);
  const char* v = tok[0];
  float frac, dur;
  if (!strcmp(v, "wifi")) { if (n >= 3) webSetWifi(tok[1], tok[2]); else webStatus(); return; }

  if (!strcmp(v, "help") || !strcmp(v, "?")) { ncHelp(); return; }
  if (!strcmp(v, "q") || !strcmp(v, "quit") || !strcmp(v, "exit") || !strcmp(v, "m")) {
    homeAll(); current_mode = MODE_MAIN; Serial.println("  leaving neural mode (legs homed)"); return;
  }
  if (!strcmp(v, "stop"))  { nc.move_end_s = t; nc.sweep = false; return; }
  if (!strcmp(v, "home"))  { nc.move_end_s = t; nc.sweep = false; nc.paused = true; nc.resting = nc.rest_pending = false; homeAll(); Serial.println("  home pose (any move resumes the controller)"); return; }
  if (!strcmp(v, "rest"))  { ncRest(t); return; }
  if (!strcmp(v, "stats")) {
    Serial.print("  BLE "); Serial.print(gp_connected ? "connected" : "not connected"); Serial.print(", gamepad frames "); Serial.println(gp_frames);
    webStatus();
    Serial.print("  ticks "); Serial.print(nc.ticks); Serial.print("  late "); Serial.print(nc.late_ticks);
    Serial.print("  max tick "); Serial.print(nc.max_tick_us / 1000.0f, 1); Serial.print(" ms  max network "); Serial.print(nc.max_net_us / 1000.0f, 2); Serial.println(" ms");
    ncPrintStance(); return;
  }
  if (!strcmp(v, "speed") && n > 1)    { nc.speed_frac = ncSpeedFrac(atof(tok[1])); Serial.print("  default speed "); Serial.print(nc.speed_frac, 2); Serial.println(" x max"); return; }
  if (!strcmp(v, "autorest") && n > 1) { nc.auto_rest_s = atof(tok[1]); Serial.print("  auto-rest after "); Serial.print(nc.auto_rest_s, 0); Serial.println(" s idle (0 = off)"); return; }
  if ((!strcmp(v, "up") || !strcmp(v, "down") || !strcmp(v, "wide") || !strcmp(v, "narrow")) && n > 1) {
    float mm = atof(tok[1]);
    if (!strcmp(v, "up"))        nc.dh_goal = nc_clamp_dh(nc.dh_goal - mm / 1000.0f);   // + dh = body lower
    else if (!strcmp(v, "down")) nc.dh_goal = nc_clamp_dh(nc.dh_goal + mm / 1000.0f);
    else if (!strcmp(v, "wide")) nc.dw_goal += mm / 1000.0f;
    else                         nc.dw_goal -= mm / 1000.0f;
    nc.dw_goal = nc_clamp_dw(nc.dh_goal, nc.dw_goal);
    nc.paused = false;
    if (nc.resting || nc.rest_pending) { nc.resting = nc.rest_pending = false; }
    ncPrintStance(); return;
  }
  if (!strcmp(v, "spin") || !strcmp(v, "arc")) {
    if (n < 2 || (tok[1][0] != 'l' && tok[1][0] != 'r')) { Serial.println("  ? spin/arc need left|right"); return; }
    float sign = tok[1][0] == 'l' ? 1.0f : -1.0f;
    ncParseTail(tok + 2, n - 2, &frac, &dur);
    float vel[3] = {0, 0, 0}; char label[40];
    if (!strcmp(v, "spin")) {
      float scale = frac < 0.0f ? 1.0f : nc_clampf(frac, 0.0f, 1.0f);
      vel[2] = sign * POLICY_MAX_YAW_RATE * scale;
      snprintf(label, sizeof(label), "spin %s", sign > 0 ? "left" : "right");
    } else {
      float speed = POLICY_MAX_SPEED * ncSpeedFrac(frac);
      float R = POLICY_MIN_TURN_RADIUS + 0.5f * (POLICY_MAX_TURN_RADIUS - POLICY_MIN_TURN_RADIUS);
      vel[0] = speed; vel[2] = sign * speed / R;
      snprintf(label, sizeof(label), "arc %s", sign > 0 ? "left" : "right");
    }
    ncStartMove(vel, label, t, dur); return;
  }
  if (!strcmp(v, "sweep")) {
    ncParseTail(tok + 1, n - 1, &frac, &dur);
    float f = ncSpeedFrac(frac); float speed = POLICY_MAX_SPEED * f;
    if (dur <= 0.0f) dur = NC_SWEEP_DEFAULT_S;
    float vel[3] = {speed, 0, 0}; char label[40];
    snprintf(label, sizeof(label), "heading sweep 0->360 %.2fx", f);
    ncStartMove(vel, label, t, dur);
    nc.sweep = true; nc.sweep_speed = speed; nc.sweep_dur = dur;
    return;
  }
  static const char* HEAD_NAMES[8] = {"forward", "back", "left", "right", "fl", "fr", "bl", "br"};
  static const float HEAD_RAD[8]   = {0.0f, NC_PI, NC_PI / 2, -NC_PI / 2, NC_PI / 4, -NC_PI / 4, 3 * NC_PI / 4, -3 * NC_PI / 4};
  for (int i = 0; i < 8; ++i) if (!strcmp(v, HEAD_NAMES[i])) {
    ncParseTail(tok + 1, n - 1, &frac, &dur);
    float f = ncSpeedFrac(frac); float speed = POLICY_MAX_SPEED * f;
    float vel[3] = {speed * cosf(HEAD_RAD[i]), speed * sinf(HEAD_RAD[i]), 0.0f}; char label[40];
    snprintf(label, sizeof(label), "%s %.2fx", HEAD_NAMES[i], f);
    ncStartMove(vel, label, t, dur); return;
  }
  Serial.print("  ? unknown command '"); Serial.print(v); Serial.println("' — 'help' lists the grammar");
}

// Stand up from rest / unpause so the next command can run.
static void ncWake(float t) {
  nc.paused = false;
  if (nc.resting || nc.rest_pending) {
    nc.resting = nc.rest_pending = false;
    if (nc.dh_goal == POLICY_DH_MAX) nc.dh_goal = nc.dh_before_rest;
    nc.dw_goal = nc_clamp_dw(nc.dh_goal, nc.dw_goal);
    nc.hold_until_s = t + fabsf(nc.dh - nc.dh_goal) / NC_STANCE_RAMP_M_S + 0.3f;
  }
}

// Dabble gamepad (ble_dabble.ino): stick = heading + speed, square/circle = spin
// left/right (combine with the stick for an arc), triangle/cross = body up/down
// 5 mm per press, start = wake + home, select = rest. Continuous: the held state
// is re-applied every tick, so the command stream is as live as the stick.
static void ncGamepad(float t) {
  bleDrain();
  if (!gp_connected && !gp_fresh) return;
  gp_fresh = false;
  uint8_t pressed = gp_buttons & ~gp_prev_buttons;
  if (gp_buttons != gp_prev_buttons || gp_stick != nc.gp_prev_stick) {        // log every change
    static const char* names[6] = {"start", "select", "triangle", "circle", "cross", "square"};
    Serial.print("  [pad] t="); Serial.print(t, 2); Serial.print(" buttons=0x"); Serial.print(gp_buttons, HEX);
    for (int b = 0; b < 6; ++b) if (gp_buttons & (1 << b)) { Serial.print(' '); Serial.print(names[b]); }
    Serial.print("  stick angle="); Serial.print((gp_stick >> 3) * 15); Serial.print(" r="); Serial.print(gp_stick & 7);
    Serial.print(" mode="); Serial.println(gp_mode);
  }
  gp_prev_buttons = gp_buttons; nc.gp_prev_stick = gp_stick;
  if (pressed & 0x01) { ncWake(t); homeAll(); Serial.println("  [pad] start: awake"); }                 // start
  if (pressed & 0x02) { ncRest(t); Serial.println("  [pad] select: rest"); }                             // select
  if (pressed & 0x04) { nc.dh_goal = nc_clamp_dh(nc.dh_goal - 0.005f); nc.dw_goal = nc_clamp_dw(nc.dh_goal, nc.dw_goal); ncWake(t); ncPrintStance(); }  // triangle
  if (pressed & 0x10) { nc.dh_goal = nc_clamp_dh(nc.dh_goal + 0.005f); nc.dw_goal = nc_clamp_dw(nc.dh_goal, nc.dw_goal); ncWake(t); ncPrintStance(); }  // cross
  float frac, heading, vel[3] = {0, 0, 0};
  bool stick = gpStick(&frac, &heading);
  bool spin = (gp_buttons & 0x28) != 0;                                                                  // square | circle
  if (stick) {
    float sp = POLICY_MAX_SPEED * frac; vel[0] = sp * cosf(heading); vel[1] = sp * sinf(heading);
    nc.gp_stick_vel[0] = vel[0]; nc.gp_stick_vel[1] = vel[1]; nc.gp_stick_t = t;
  } else if (spin && nc.gp_stick_t > 0.0f && (nc.gp_spin_hold || t - nc.gp_stick_t < 0.6f)) {
    // The app reports only one control at a time: a spin button press zeroes the
    // stick. Keep the last stick vector while the button is held -> arc turn.
    vel[0] = nc.gp_stick_vel[0]; vel[1] = nc.gp_stick_vel[1]; stick = true;
  }
  nc.gp_spin_hold = spin && stick;
  if (gp_buttons & 0x20) vel[2] += POLICY_MAX_YAW_RATE;                                                  // square
  if (gp_buttons & 0x08) vel[2] -= POLICY_MAX_YAW_RATE;                                                  // circle
  if (spin && stick) vel[2] *= 0.5f;                                                                     // arc: half yaw rate
  bool want = stick || (vel[2] != 0.0f);
  if (want) {
    if (nc.resting || nc.rest_pending || nc.paused) ncWake(t);
    for (int i = 0; i < 3; ++i) nc.vel[i] = vel[i];
    nc.sweep = false; nc.label[0] = 0;
    nc.move_end_s = t + 0.3f;             // refreshed every tick while held; ramps down 0.3 s after release
    nc.gp_driving = true;
  } else if (nc.gp_driving) {
    nc.gp_driving = false; nc.move_end_s = t;
  }
}

// Web dashboard (wifi_web.ino). Everything arrives together in one 20 Hz message,
// so stick + spin + stance buttons + tilt combine naturally. Held buttons ramp.
static void ncWeb(float t) {
  webPoll();
  if (web_has_line) { web_has_line = false; Serial.print("  [web] "); Serial.println(web_line); ncHandle(web_line, t); }
  if (web_has_cmd) {
    web_has_cmd = false; Serial.print("  [web] "); Serial.println(web_cmd);
    if (!strcmp(web_cmd, "stand"))      { ncWake(t); homeAll(); ncSay("stand: awake"); }
    else if (!strcmp(web_cmd, "rest"))  ncRest(t);
    else if (!strcmp(web_cmd, "stop"))  { nc.move_end_s = t; nc.sweep = false; nc.web_driving = false; }
    else if (!strcmp(web_cmd, "home"))  { nc.move_end_s = t; nc.sweep = false; nc.paused = true; nc.resting = nc.rest_pending = false; homeAll(); ncSay("home pose"); }
  }
  if (!web.fresh) {
    if (nc.web_driving && millis() - web.last_ms > 500) { nc.web_driving = false; nc.move_end_s = t; ncSay("web: control stream lost, stopping"); }
    return;
  }
  web.fresh = false;
  const float dtm = 0.05f;                                    // page sends at 20 Hz
  if (web.up) nc.dh_goal -= NC_STANCE_RAMP_M_S * dtm;
  if (web.dn) nc.dh_goal += NC_STANCE_RAMP_M_S * dtm;
  nc.dh_goal = nc_clamp_dh(nc.dh_goal);
  if (web.wd) nc.dw_goal += NC_STANCE_RAMP_M_S * dtm;
  if (web.nr) nc.dw_goal -= NC_STANCE_RAMP_M_S * dtm;
  nc.dw_goal = nc_clamp_dw(nc.dh_goal, nc.dw_goal);
  nc.pitch_goal = nc_clampf(web.ry, -1.0f, 1.0f) * POLICY_PITCH_MAX;   // stick up = nose up
  nc.roll_goal  = nc_clampf(web.rx, -1.0f, 1.0f) * POLICY_ROLL_MAX;    // stick right = right side down (roll+ = left up)
  bool stance_btn = web.up || web.dn || web.wd || web.nr || web.rx != 0.0f || web.ry != 0.0f;
  if (stance_btn && (nc.resting || nc.rest_pending || nc.paused)) ncWake(t);
  float mag = hypotf(web.lx, web.ly), vel[3] = {0, 0, 0};
  bool stick = mag > 0.1f;
  if (stick) {
    if (mag > 1.0f) mag = 1.0f;
    float top = nc_clampf(web.spd, POLICY_SPEED_MIN_FRAC, POLICY_SPEED_MAX_FRAC);
    float frac = POLICY_SPEED_MIN_FRAC + (top - POLICY_SPEED_MIN_FRAC) * (mag - 0.1f) / 0.9f;
    float sp = POLICY_MAX_SPEED * frac;
    vel[0] = sp * web.ly / mag; vel[1] = -sp * web.lx / mag;          // page: x right, y up; robot: vx fwd, vy left
  }
  float yaw = (web.sl ? 1.0f : 0.0f) - (web.sr ? 1.0f : 0.0f);
  vel[2] = yaw * POLICY_MAX_YAW_RATE * (stick ? 0.5f : 1.0f);
  bool want = stick || yaw != 0.0f;
  if (want) {
    if (nc.resting || nc.rest_pending || nc.paused) ncWake(t);
    for (int i = 0; i < 3; ++i) nc.vel[i] = vel[i];
    nc.sweep = false; nc.label[0] = 0;
    nc.move_end_s = t + 0.3f;
    nc.web_driving = true;
  } else if (nc.web_driving) { nc.web_driving = false; nc.move_end_s = t; }
}

static void ncTelemetry(float t) {
  if (++nc.tele_ticks % 10 != 0 || !webConnected()) return;   // 5 Hz
  const char* state = nc.resting ? "resting" : nc.rest_pending ? "squatting" : nc.paused ? "home" : (nc.gain > 0.0f ? "walking" : "standing");
  char m[160];
  snprintf(m, sizeof(m), "T %.3f %.3f %.2f %.0f %.0f %.1f %.1f %.2f %s %.1f %.2f %d",
           nc.vel_now[0], nc.vel_now[1], nc.vel_now[2], -nc.dh * 1000.0f, nc.dw * 1000.0f,
           nc.pitch * 57.2958f, nc.roll * 57.2958f, nc.gain, state, nc.max_tick_us / 1000.0f, nc.max_net_us / 1000.0f,
           WiFi.status() == WL_CONNECTED ? (int)WiFi.RSSI() : 0);
  webSend(m);
}

static void ncTick(float t) {
  unsigned long t_us0 = micros();
  ncWeb(t);
  ncGamepad(t);
  ncTelemetry(t);
  if (nc.label[0] && t >= nc.move_end_s) {
    char m[64]; snprintf(m, sizeof(m), "# %s done, idle", nc.label); ncSay(m);
    nc.label[0] = 0;
  }
  // Stance ramps (20 mm/s, 0.5 rad/s).
  float step = NC_STANCE_RAMP_M_S * NC_DT, tstep = 0.5f * NC_DT;
  nc.dh += nc_clampf(nc.dh_goal - nc.dh, -step, step);
  nc.dw += nc_clampf(nc.dw_goal - nc.dw, -step, step);
  nc.pitch += nc_clampf(nc_clampf(nc.pitch_goal, -POLICY_PITCH_MAX, POLICY_PITCH_MAX) - nc.pitch, -tstep, tstep);
  nc.roll  += nc_clampf(nc_clampf(nc.roll_goal, -POLICY_ROLL_MAX, POLICY_ROLL_MAX) - nc.roll, -tstep, tstep);

  bool moving = nc.hold_until_s <= t && t < nc.move_end_s;
  if (nc.was_moving && !moving) nc.idle_since_s = t;
  nc.was_moving = moving;
  nc.gain = moving ? fminf(1.0f, nc.gain + NC_DT / NC_FADE_S) : fmaxf(0.0f, nc.gain - NC_DT / NC_FADE_S);
  if (moving && nc.sweep) {
    float h = 2.0f * NC_PI * (t - (nc.move_end_s - nc.sweep_dur)) / nc.sweep_dur;
    nc.vel[0] = nc.sweep_speed * cosf(h); nc.vel[1] = nc.sweep_speed * sinf(h); nc.vel[2] = 0.0f;
  }
  const float scale[3] = {POLICY_MAX_SPEED, POLICY_MAX_SPEED, POLICY_MAX_YAW_RATE};
  for (int i = 0; i < 3; ++i) {
    float target = moving ? nc.vel[i] : 0.0f;
    float s = scale[i] * NC_DT / NC_VEL_RAMP_S;
    nc.vel_now[i] = nc_clampf(target, nc.vel_now[i] - s, nc.vel_now[i] + s);
  }
  bool idle = !moving && nc.gain <= 0.0f && nc.vel_now[0] == 0.0f && nc.vel_now[1] == 0.0f && nc.vel_now[2] == 0.0f;
  if (nc.rest_pending && idle && fabsf(nc.dh - nc.dh_goal) < 1e-4f) {
    relaxAll(); nc.rest_pending = false; nc.resting = true;
    ncSay("# resting (servos limp). Any move stands the bot up.");
  }
  if (nc.auto_rest_s > 0.0f && idle && !nc.resting && !nc.rest_pending && !nc.paused && t - nc.idle_since_s > nc.auto_rest_s) ncRest(t);
  if (nc.resting || nc.paused) return;

  float target[18], stance[18];
  nc_stance_tilt(nc.dh, nc.dw, nc.pitch, nc.roll, stance);
  if (nc.gain > 0.0f) {
    float cmd[9] = {nc.vel_now[0], nc.vel_now[1], nc.vel_now[2], nc.pitch, nc.roll, nc.dh, nc.dw, 0.0f, 0.0f};
    float obs[POLICY_OBS_DIM], pol[18];
    float phase = fmodf(t / POLICY_GAIT_PERIOD_S, 1.0f);
    nc_build_obs(phase, cmd, obs);
    unsigned long n0 = micros();
    nc_policy(obs, pol);
    unsigned long n_us = micros() - n0; if (n_us > nc.max_net_us) nc.max_net_us = n_us;
    for (int j = 0; j < 18; ++j) target[j] = nc.gain * pol[j] + (1.0f - nc.gain) * stance[j];
  } else {
    for (int j = 0; j < 18; ++j) target[j] = stance[j];
  }
  for (uint8_t sl = 0; sl < 6; ++sl) {
    uint8_t fl = SIM_TO_FW_LEG[sl];
    for (uint8_t j = 0; j < 3; ++j) cmdJointAngle(fl, j, target[sl * 3 + j]);
  }
  unsigned long us = micros() - t_us0;
  if (us > nc.max_tick_us) nc.max_tick_us = us;
  ++nc.ticks;
}

void runNeuralMode() {
  Serial.println();
  Serial.println("=== NEURAL CONTROLLER (policy running on the bot) ===");
  Serial.print("  network "); Serial.print(POLICY_OBS_DIM); Serial.print(" -> 256 -> 128 -> 64 -> "); Serial.print(POLICY_ACT_DIM);
  Serial.print(", gait period "); Serial.print(POLICY_GAIT_PERIOD_S, 2); Serial.print(" s, max speed "); Serial.print(POLICY_MAX_SPEED, 3); Serial.println(" m/s");
  memset(&nc, 0, sizeof(nc));
  nc.speed_frac = POLICY_SPEED_MAX_FRAC;
  nc.move_end_s = -1.0f;
  ncHelp();
  homeAll();
  delay(300);
  unsigned long t0 = millis(), next_tick = t0;
  static char line[128]; size_t len = 0;
  while (current_mode == MODE_NEURAL) {
    while (Serial.available()) {
      char ch = Serial.read();
      if (ch == '\r') continue;
      if (ch != '\n') { if (len < sizeof(line) - 1) line[len++] = ch; continue; }
      line[len] = 0; len = 0;
      ncHandle(line, (millis() - t0) * 0.001f);
      if (current_mode != MODE_NEURAL) return;
    }
    static char bline[96];
    bleDrain();
    if (bleTakeLine(bline, sizeof(bline))) {
      Serial.print("  [ble] "); Serial.println(bline);
      ncHandle(bline, (millis() - t0) * 0.001f);
      if (current_mode != MODE_NEURAL) return;
    }
    unsigned long now = millis();
    if ((long)(now - next_tick) < 0) { delay(1); continue; }
    if ((long)(now - next_tick) > (long)NC_TICK_MS) { ++nc.late_ticks; next_tick = now; }
    next_tick += NC_TICK_MS;
    ncTick((now - t0) * 0.001f);
  }
}
