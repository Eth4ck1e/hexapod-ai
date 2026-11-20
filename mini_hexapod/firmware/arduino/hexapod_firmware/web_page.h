// web_page.h — the phone/laptop dashboard served by wifi_web.ino (one static page, no deps).
// Two sticks + hold buttons; sends "S lx ly rx ry spinL spinR up down wide narrow speed" at
// 20 Hz over a WebSocket, "C stand|rest|stop|home" one-shots, "L <console line>" passthrough.
// Receives "T ..." telemetry and "M <text>" messages.
#pragma once
static const char WEB_PAGE[] PROGMEM = R"HTML(<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes"><title>hexapod</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#121820;color:#f4f1ea;font:14px/1.3 -apple-system,system-ui,sans-serif;touch-action:none;overflow:hidden;-webkit-user-select:none;user-select:none;-webkit-touch-callout:none}
#app{display:grid;grid-template-columns:1fr minmax(230px,auto) 1fr;grid-template-rows:auto 1fr auto auto;height:100vh;gap:6px;padding:8px}
header{grid-column:1/4;display:flex;justify-content:space-between;align-items:center;color:#9aa5b1}header b{color:#f4f1ea}
.stick{position:relative;width:min(40vh,30vw);height:min(40vh,30vw);border-radius:50%;background:#1c2530;border:2px solid #2e3a48;align-self:center;justify-self:center;grid-row:2}
.stick .knob{position:absolute;width:32%;height:32%;border-radius:50%;background:#f2a33c;left:34%;top:34%;box-shadow:0 2px 10px #0008}
.stick .lbl{position:absolute;top:-24px;width:100%;text-align:center;color:#9aa5b1;font-size:12px;letter-spacing:.04em}
.stick .cross{position:absolute;left:50%;top:8%;width:1px;height:84%;background:#2e3a48}.stick .cross2{position:absolute;top:50%;left:8%;height:1px;width:84%;background:#2e3a48}
#mid{grid-row:2;display:grid;grid-template-columns:1fr 1fr;gap:6px;align-content:center}
button{background:#263241;color:#f4f1ea;border:1px solid #3a4858;border-radius:10px;padding:13px 6px;font-size:15px;font-weight:600;touch-action:none}
button.on,button:active{background:#f2a33c;color:#121820}.wide{grid-column:1/3}.danger{border-color:#c6584f}.ok{border-color:#4fb0c6}
label{display:flex;justify-content:space-between;color:#9aa5b1;font-size:12px}input[type=range]{width:100%}
#tele{grid-column:1/4;display:grid;grid-template-columns:repeat(6,1fr);gap:6px;font-family:ui-monospace,Menlo,monospace;font-size:12px}
#tele div{background:#1c2530;border-radius:8px;padding:6px 8px}#tele span{display:block;color:#9aa5b1;font-size:10px;letter-spacing:.06em}
#log{grid-column:1/4;height:3.9em;overflow:hidden;font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#4fb0c6;white-space:pre}
</style></head><body><div id="app">
<header><b>hexapod</b><span>the policy runs on the bot · this page only sends commands</span><span id="conn">connecting…</span></header>
<div class="stick" id="walk"><div class="lbl">WALK · direction = heading · deflection = speed</div><div class="cross"></div><div class="cross2"></div><div class="knob"></div></div>
<div id="mid">
 <button class="wide ok" id="stand">STAND / WAKE</button>
 <button data-hold="up">RAISE ▲</button><button data-hold="dn">LOWER ▼</button>
 <button data-hold="wd">WIDER ◁ ▷</button><button data-hold="nr">NARROWER ▷ ◁</button>
 <button data-hold="sl">SPIN ⟲ LEFT</button><button data-hold="sr">SPIN ⟳ RIGHT</button>
 <div class="wide"><label><span>max speed</span><span id="spdv">0.85 ×</span></label><input type="range" id="spd" min="0.4" max="0.85" step="0.05" value="0.85"></div>
 <button id="stop" class="danger">STOP</button><button id="rest" class="danger">REST (limp)</button>
</div>
<div class="stick" id="tilt"><div class="lbl">TILT · up/down = pitch · left/right = roll</div><div class="cross"></div><div class="cross2"></div><div class="knob"></div></div>
<div id="tele"></div><div id="log"></div></div>
<script>
const st={lx:0,ly:0,rx:0,ry:0,sl:0,sr:0,up:0,dn:0,wd:0,nr:0,spd:0.85};
function stick(el,kx,ky){const knob=el.querySelector('.knob');let id=null;
 const set=(x,y)=>{st[kx]=x;st[ky]=y;knob.style.left=(34+x*34)+'%';knob.style.top=(34-y*34)+'%'};
 const mv=e=>{if(e.pointerId!==id)return;const r=el.getBoundingClientRect();let x=(e.clientX-r.left)/r.width*2-1,y=-((e.clientY-r.top)/r.height*2-1);const m=Math.hypot(x,y);if(m>1){x/=m;y/=m}set(x,y)};
 el.addEventListener('pointerdown',e=>{if(id!==null)return;id=e.pointerId;el.setPointerCapture(id);mv(e)});
 el.addEventListener('pointermove',mv);const up=e=>{if(e.pointerId!==id)return;id=null;set(0,0)};
 el.addEventListener('pointerup',up);el.addEventListener('pointercancel',up);}
stick(document.getElementById('walk'),'lx','ly');stick(document.getElementById('tilt'),'rx','ry');
document.querySelectorAll('[data-hold]').forEach(b=>{const k=b.dataset.hold;
 b.addEventListener('pointerdown',e=>{b.setPointerCapture(e.pointerId);st[k]=1;b.classList.add('on')});
 const off=()=>{st[k]=0;b.classList.remove('on')};b.addEventListener('pointerup',off);b.addEventListener('pointercancel',off);});
const conn=document.getElementById('conn'),tele=document.getElementById('tele'),log=document.getElementById('log');
let ws,alive=false,lines=[];
function connect(){ws=new WebSocket('ws://'+location.host+'/ws');
 ws.onopen=()=>{alive=true;conn.textContent='connected'};
 ws.onclose=()=>{alive=false;conn.textContent='reconnecting…';setTimeout(connect,1000)};ws.onerror=()=>ws.close();
 ws.onmessage=e=>{const m=e.data;if(m[0]=='T')showTele(m.split(' '));else if(m[0]=='M'){lines.unshift(m.slice(2));lines=lines.slice(0,3);log.textContent=lines.join('\n')}}}
connect();
const cmd=c=>{if(alive)ws.send('C '+c)};
document.getElementById('stand').onclick=()=>cmd('stand');document.getElementById('stop').onclick=()=>cmd('stop');document.getElementById('rest').onclick=()=>cmd('rest');
const spd=document.getElementById('spd');spd.oninput=()=>{st.spd=+spd.value;document.getElementById('spdv').textContent=spd.value+' ×'};
setInterval(()=>{if(alive&&ws.bufferedAmount<200)ws.send('S '+[st.lx,st.ly,st.rx,st.ry].map(v=>v.toFixed(3)).join(' ')+' '+[st.sl,st.sr,st.up,st.dn,st.wd,st.nr].join(' ')+' '+st.spd)},50);
function showTele(f){const c=(l,v)=>'<div><span>'+l+'</span>'+v+'</div>';
 tele.innerHTML=c('velocity m/s · rad/s',f[1]+' '+f[2]+' · '+f[3])+c('height · width mm',f[4]+' · '+f[5])+c('pitch · roll °',f[6]+' · '+f[7])+c('state',f[9]+' · net '+f[8])+c('tick · network ms (avg)',f[10]+' · '+f[11])+c('wifi rssi · late ticks',f[12]+' · '+(f[13]||0));}
</script></body></html>)HTML";
