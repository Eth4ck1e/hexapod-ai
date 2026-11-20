// ble_dabble.ino — Bluetooth LE link for the on-board controller.
//
// Advertises the Nordic UART service under the name "hexapod". Two kinds of
// traffic are accepted on the RX characteristic:
//   * Dabble GamePad frames (STEMpedia Dabble app, "Gamepad" module, joystick
//     mode): FF 01 <fn> 01 02 <buttons> <stick> 00 — fn 1 digital, 2 joystick,
//     3 accelerometer; buttons bit0 start, bit1 select, bit2 triangle, bit3 circle,
//     bit4 cross, bit5 square; stick = (angle/15 deg) << 3 | radius (0..7).
//   * Plain text lines (any BLE serial terminal app): the same words the neural
//     mode takes over USB — forward, spin left, up 10, rest ...
// The neural mode polls bleDrain() each tick; nothing here moves a servo.
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include "neural_core.h"   // speed band + NC_PI (the .ino files form one translation unit; the header is once-only)

static const char* BLE_NAME = "hexapod";
#define NUS_SERVICE "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define NUS_RX      "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"   // app -> board (write)
#define NUS_TX      "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"   // board -> app (notify)

// Gamepad state (latest Dabble frame). Dabble only sends on change, so this is
// held until the next frame or a disconnect.
volatile bool    gp_connected = false;
uint8_t          gp_buttons = 0, gp_prev_buttons = 0;
uint8_t          gp_mode = 0;              // 1 digital, 2 joystick, 3 accelerometer
uint8_t          gp_stick = 0;             // raw byte: angle/15 << 3 | radius
bool             gp_fresh = false;         // a frame arrived since the last poll
unsigned long    gp_frames = 0, gp_last_ms = 0;

static uint8_t   ble_ring[512];
static volatile uint16_t ble_head = 0, ble_tail = 0;
static BLECharacteristic* ble_tx = nullptr;
static char      ble_line[96]; static size_t ble_line_len = 0; static bool ble_line_ready = false;

static void blePush(const uint8_t* d, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    uint16_t nh = (ble_head + 1) % sizeof(ble_ring);
    if (nh == ble_tail) return;            // full: drop
    ble_ring[ble_head] = d[i]; ble_head = nh;
  }
}

class HexRxCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* c) override { blePush(c->getData(), c->getLength()); }
};
class HexServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer*) override { gp_connected = true; }
  void onDisconnect(BLEServer* s) override {
    gp_connected = false; gp_buttons = 0; gp_stick = 0; gp_fresh = true;   // -> stop
    s->startAdvertising();
  }
};

static void bleSend(const uint8_t* d, size_t n) {
  if (!ble_tx || !gp_connected) return;
  ble_tx->setValue((uint8_t*)d, n); ble_tx->notify();
}

void bleBegin() {
  BLEDevice::init(BLE_NAME);
  BLEServer* server = BLEDevice::createServer();
  server->setCallbacks(new HexServerCallbacks());
  BLEService* svc = server->createService(NUS_SERVICE);
  ble_tx = svc->createCharacteristic(NUS_TX, BLECharacteristic::PROPERTY_NOTIFY);
  ble_tx->addDescriptor(new BLE2902());
  BLECharacteristic* rx = svc->createCharacteristic(NUS_RX, BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR);
  rx->setCallbacks(new HexRxCallbacks());
  svc->start();
  BLEAdvertising* adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(NUS_SERVICE);
  adv->setScanResponse(true);
  BLEDevice::startAdvertising();
  Serial.print("BLE: advertising as '"); Serial.print(BLE_NAME); Serial.println("' (Dabble gamepad or any BLE serial app)");
}

static void dabbleFrame(uint8_t mod, uint8_t fn, uint8_t arglen, const uint8_t* pay) {
  if (mod == 0x01 && fn >= 1 && fn <= 3 && arglen == 2) {      // GamePad
    gp_buttons = pay[0]; gp_stick = pay[1]; gp_mode = fn;
    gp_fresh = true; ++gp_frames; gp_last_ms = millis();
  } else if (mod == 0x00 && fn == 0x03) {                        // board-id request -> "ESP32"
    static const uint8_t id[] = {0xFF, 0x00, 0x03, 0x01, 0x04, 0x06, 0x01, 0x05, 0x01, 0x00};
    bleSend(id, sizeof(id));
  }
}

// Byte-at-a-time Dabble parser; anything outside a frame is treated as text.
static void bleFeed(uint8_t b) {
  static uint8_t st = 0, mod, fn, argc, arglen, pay[4], pi;
  switch (st) {
    case 0:
      if (b == 0xFF) { st = 1; break; }
      if (b == '\n' || b == '\r') { if (ble_line_len) { ble_line[ble_line_len] = 0; ble_line_ready = true; ble_line_len = 0; } }
      else if (b >= 0x20 && b < 0x7F && ble_line_len < sizeof(ble_line) - 1 && !ble_line_ready) ble_line[ble_line_len++] = (char)b;
      break;
    case 1: mod = b; st = 2; break;
    case 2: fn = b; st = 3; break;
    case 3: argc = b; st = argc ? 4 : 6; break;
    case 4: arglen = b; pi = 0; st = arglen ? 5 : (--argc ? 4 : 6); break;
    case 5: if (pi < sizeof(pay)) pay[pi] = b; ++pi; if (pi >= arglen) st = (--argc ? 4 : 6); break;
    case 6: if (b == 0x00) dabbleFrame(mod, fn, arglen, pay); st = 0; break;
  }
}

// Pump received bytes through the parser. Call often.
void bleDrain() {
  while (ble_tail != ble_head) { uint8_t b = ble_ring[ble_tail]; ble_tail = (ble_tail + 1) % sizeof(ble_ring); bleFeed(b); }
}

// A complete text line arrived over BLE: copy it out (returns false if none).
bool bleTakeLine(char* out, size_t n) {
  if (!ble_line_ready) return false;
  strncpy(out, ble_line, n - 1); out[n - 1] = 0; ble_line_ready = false;
  return true;
}

// Main-menu hook: a gamepad frame while idling in the menu enters neural mode,
// so a powered-up bot needs no laptop at all — connect the phone and drive.
bool bleWantsNeural() { bleDrain(); return gp_fresh && gp_connected; }

// Joystick -> (speed fraction within the trained band, heading rad, 0 = forward).
bool gpStick(float* speed_frac, float* heading) {
  uint8_t radius = gp_stick & 0x07;
  if (gp_mode != 2 || radius == 0) return false;
  float angle_deg = (float)(gp_stick >> 3) * 15.0f;     // Dabble: 0 = right, 90 = up
  *heading = (angle_deg - 90.0f) * NC_PI / 180.0f;      // robot: 0 = forward, +90 = left
  *speed_frac = POLICY_SPEED_MIN_FRAC + (POLICY_SPEED_MAX_FRAC - POLICY_SPEED_MIN_FRAC) * (float)(radius - 1) / 6.0f;
  return true;
}
