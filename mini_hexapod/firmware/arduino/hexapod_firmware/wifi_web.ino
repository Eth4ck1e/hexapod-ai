// wifi_web.ino — WiFi access point + one-page web controller for the on-board policy.
//
// The board brings up its own network (SSID "hexapod", password "hexapod123",
// 192.168.4.1; any address resolves there, so a captive-portal sheet pops up on
// most phones) and can ALSO join a normal network (`wifi SSID PASS` in neural
// mode, stored in NVS) so a laptop on that network can drive and watch logs.
// HTTP serves web_page.h; a hand-rolled WebSocket (RFC 6455, one client, text
// frames) carries the control stream. Nothing here moves a servo: neural_mode.ino
// polls webPoll() and reads the `web` state each tick.
#include <DNSServer.h>
#include "mbedtls/sha1.h"
#include "mbedtls/base64.h"
#include "web_link.h"
#include "web_page.h"

static const char* AP_SSID = "hexapod";
static const char* AP_PASS = "hexapod123";

static WiFiServer web_server(80);
static WiFiClient web_http;                    // one request in flight
static WiFiClient web_ws;                      // the controller client
static DNSServer  web_dns;
static String     http_req; static unsigned long http_t0 = 0;
static uint8_t    ws_rx[320]; static size_t ws_len = 0;

WebState web = {};
char web_cmd[16]; bool web_has_cmd = false;
char web_line[96]; bool web_has_line = false;
unsigned long web_msgs = 0; static bool web_activity = false;

void webBegin() {
  WiFi.persistent(false);
  WiFi.mode(WIFI_AP_STA);
  WiFi.softAP(AP_SSID, AP_PASS);
  Preferences wp; wp.begin("wifi", true);
  String ssid = wp.getString("ssid", ""), pass = wp.getString("pass", ""); wp.end();
  if (ssid.length()) { WiFi.begin(ssid.c_str(), pass.c_str()); Serial.print("WiFi: joining '"); Serial.print(ssid); Serial.println("' (stored)"); }
  web_server.begin();
  web_dns.start(53, "*", WiFi.softAPIP());
  Serial.print("WiFi: AP '"); Serial.print(AP_SSID); Serial.print("' pass '"); Serial.print(AP_PASS);
  Serial.print("' -> http://"); Serial.print(WiFi.softAPIP()); Serial.println("/  (web controller)");
}

void webSetWifi(const char* ssid, const char* pass) {
  Preferences wp; wp.begin("wifi", false); wp.putString("ssid", ssid); wp.putString("pass", pass); wp.end();
  // A join already in progress rejects new credentials ("sta is connecting, cannot set
  // config"): restart the STA interface first.
  WiFi.setAutoReconnect(false); WiFi.disconnect(false, true);
  WiFi.mode(WIFI_AP); delay(300); WiFi.mode(WIFI_AP_STA); delay(300);
  WiFi.setAutoReconnect(true);
  WiFi.begin(ssid, pass);
  Serial.print("WiFi: joining '"); Serial.print(ssid); Serial.println("' (saved; 'wifi' alone shows the address once connected)");
}

// List nearby networks (blocks ~2 s; use while standing). Prints the SSID bytes exactly
// as broadcast, so names with curly apostrophes etc. can be copied for `wifi`.
void webScan() {
  Serial.println("  scanning...");
  // A pending join makes scanNetworks() fail (-2, "sta is connecting") and disconnect()
  // alone does not stop the driver's retry loop: drop the STA interface and bring it back.
  WiFi.setAutoReconnect(false);
  WiFi.disconnect(false, true);
  WiFi.mode(WIFI_AP); delay(400);
  WiFi.mode(WIFI_AP_STA); delay(400);
  int n = WiFi.scanNetworks(false, false, false, 300);
  for (int i = 0; i < n; ++i) {
    Serial.print("  "); Serial.print(WiFi.RSSI(i)); Serial.print(" dBm  ");
    Serial.print(WiFi.encryptionType(i) == WIFI_AUTH_OPEN ? "open   " : "secured");
    Serial.print("  \""); Serial.print(WiFi.SSID(i)); Serial.println("\"");
  }
  WiFi.scanDelete();
  Serial.print("  "); Serial.print(n); Serial.println(" networks");
  Preferences wp; wp.begin("wifi", true);
  String ssid = wp.getString("ssid", ""), pass = wp.getString("pass", ""); wp.end();
  WiFi.setAutoReconnect(true);
  if (ssid.length()) WiFi.begin(ssid.c_str(), pass.c_str());   // resume the stored join
}

void webStatus() {
  Serial.print("  WiFi AP "); Serial.print(AP_SSID); Serial.print(" @ "); Serial.print(WiFi.softAPIP());
  Serial.print(", "); Serial.print(WiFi.softAPgetStationNum()); Serial.print(" joined; STA ");
  if (WiFi.status() == WL_CONNECTED) { Serial.print(WiFi.SSID()); Serial.print(" @ "); Serial.print(WiFi.localIP()); Serial.print(" rssi "); Serial.print(WiFi.RSSI()); }
  else Serial.print("not connected");
  Serial.print("; page client "); Serial.print(web_ws.connected() ? "connected" : "none"); Serial.print(", msgs "); Serial.println(web_msgs);
}

static void wsSendRaw(const uint8_t* p, size_t n, uint8_t op) {
  if (!web_ws.connected()) return;
  uint8_t h[4]; size_t hl;
  h[0] = 0x80 | op;
  if (n < 126) { h[1] = (uint8_t)n; hl = 2; } else { h[1] = 126; h[2] = (uint8_t)(n >> 8); h[3] = (uint8_t)n; hl = 4; }
  web_ws.write(h, hl); if (n) web_ws.write(p, n);
}
void webSend(const char* s) { wsSendRaw((const uint8_t*)s, strlen(s), 0x1); }
bool webConnected() { return web_ws.connected(); }

static void webHandleMsg(char* m) {
  ++web_msgs;
  if (m[0] == 'S' && m[1] == ' ') {
    float v[11]; char* p = m + 2; char* e; int i;
    for (i = 0; i < 11; ++i) { v[i] = strtof(p, &e); if (e == p) break; p = e; }
    if (i != 11) return;
    web.lx = v[0]; web.ly = v[1]; web.rx = v[2]; web.ry = v[3];
    web.sl = v[4] > 0.5f; web.sr = v[5] > 0.5f; web.up = v[6] > 0.5f; web.dn = v[7] > 0.5f; web.wd = v[8] > 0.5f; web.nr = v[9] > 0.5f;
    web.spd = v[10]; web.last_ms = millis(); web.fresh = true;
    if (web.lx || web.ly || web.rx || web.ry || web.sl || web.sr || web.up || web.dn || web.wd || web.nr) web_activity = true;
  } else if (m[0] == 'C' && m[1] == ' ') { strncpy(web_cmd, m + 2, sizeof(web_cmd) - 1); web_cmd[sizeof(web_cmd) - 1] = 0; web_has_cmd = true; web_activity = true; }
  else if (m[0] == 'L' && m[1] == ' ') { strncpy(web_line, m + 2, sizeof(web_line) - 1); web_line[sizeof(web_line) - 1] = 0; web_has_line = true; web_activity = true; }
  else if (m[0] == 'P') webSend("P");
}

static void wsHandshake(const String& req) {
  WiFiClient& c = web_http;
  int k = req.indexOf("Sec-WebSocket-Key:");
  String key = req.substring(k + 18, req.indexOf("\r\n", k)); key.trim();
  String src = key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
  uint8_t sha[20]; mbedtls_sha1((const unsigned char*)src.c_str(), src.length(), sha);
  unsigned char b64[40]; size_t olen = 0; mbedtls_base64_encode(b64, sizeof(b64), &olen, sha, 20);
  c.print("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ");
  c.write(b64, olen); c.print("\r\n\r\n");
}

static void webServePage() {
  WiFiClient& c = web_http;
  size_t n = strlen(WEB_PAGE);
  c.print("HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nCache-Control: no-store\r\nConnection: close\r\nContent-Length: ");
  c.print(n); c.print("\r\n\r\n");
  c.write((const uint8_t*)WEB_PAGE, n);
  c.flush(); delay(2); c.stop();
}

// Call often (every tick, and while idling in the menu).
void webPoll() {
  web_dns.processNextRequest();
  if (!web_http.connected()) {
    WiFiClient c = web_server.accept();
    if (c) { web_http = c; http_req = ""; http_t0 = millis(); }
  }
  if (web_http.connected()) {
    while (web_http.available() && http_req.length() < 2048) { http_req += (char)web_http.read(); if (http_req.endsWith("\r\n\r\n")) break; }
    if (http_req.endsWith("\r\n\r\n")) {
      if (http_req.indexOf("Upgrade: websocket") >= 0 || http_req.indexOf("Upgrade: WebSocket") >= 0) {
        if (web_ws.connected()) web_ws.stop();           // one controller at a time: newest wins
        wsHandshake(http_req);
        web_ws = web_http; web_http = WiFiClient(); ws_len = 0;
        webSend("M page connected");
      } else { webServePage(); web_http = WiFiClient(); }
    } else if (millis() - http_t0 > 3000) { web_http.stop(); web_http = WiFiClient(); }
  }
  if (web_ws.connected()) {
    while (web_ws.available() && ws_len < sizeof(ws_rx)) ws_rx[ws_len++] = (uint8_t)web_ws.read();
    for (;;) {
      if (ws_len < 2) break;
      uint8_t op = ws_rx[0] & 0x0F; bool masked = ws_rx[1] & 0x80; size_t plen = ws_rx[1] & 0x7F, off = 2;
      if (plen == 126) { if (ws_len < 4) break; plen = ((size_t)ws_rx[2] << 8) | ws_rx[3]; off = 4; }
      else if (plen == 127) { web_ws.stop(); ws_len = 0; break; }
      if (masked) off += 4;
      if (off + plen > sizeof(ws_rx)) { web_ws.stop(); ws_len = 0; break; }
      if (ws_len < off + plen) break;
      uint8_t* pay = ws_rx + off;
      if (masked) { const uint8_t* mk = ws_rx + off - 4; for (size_t i = 0; i < plen; ++i) pay[i] ^= mk[i & 3]; }
      if (op == 0x1) { char msg[300]; memcpy(msg, pay, plen); msg[plen] = 0; webHandleMsg(msg); }
      else if (op == 0x9) wsSendRaw(pay, plen, 0xA);
      else if (op == 0x8) { wsSendRaw(pay, plen > 125 ? 0 : plen, 0x8); web_ws.stop(); ws_len = 0; break; }
      size_t tot = off + plen; memmove(ws_rx, ws_rx + tot, ws_len - tot); ws_len -= tot;
    }
  } else if (web.last_ms) {                       // client gone -> everything released
    web = WebState(); web.fresh = true;
  }
}

// Main-menu hook: a real control input from the page enters neural mode.
bool webWantsNeural() { webPoll(); bool a = web_activity; web_activity = false; return a; }
