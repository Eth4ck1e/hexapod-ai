// web_link.h — shared declarations between wifi_web.ino (owner) and neural_mode.ino (user).
// The Arduino build concatenates .ino files alphabetically, so the web state has to be
// declared here for neural_mode.ino to see it.
#pragma once
#include <WiFi.h>

struct WebState { float lx, ly, rx, ry; bool sl, sr, up, dn, wd, nr; float spd; unsigned long last_ms; bool fresh; };
extern WebState web;
extern char web_cmd[16]; extern bool web_has_cmd;
extern char web_line[96]; extern bool web_has_line;
extern unsigned long web_msgs;

void webBegin();
void webPoll();
void webSend(const char* s);
bool webConnected();
bool webWantsNeural();
void webSetWifi(const char* ssid, const char* pass);
void webStatus();
void webScan();
