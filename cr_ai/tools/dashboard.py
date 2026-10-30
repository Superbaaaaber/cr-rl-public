"""实时训练看板：python -m cr_ai.tools.dashboard [--port 8765] [--runs runs]

本地 HTTP 服务（只监听 127.0.0.1），每 15 秒读一次 runs/<run>/ 下的日志并刷新：
  train_log.jsonl（每 log_every 步）、val_log.jsonl（每 eval_every 步）、arena.jsonl（对战评测）、config.yaml、run_info.json（可选：总步数）、
  events.json（可选：[{step, label}]，配置改动的时间点，在每张曲线上画竖线）。
另外扫 runs/*/h2h.jsonl：单独跑的 400 局对打检查（plateau_check、breadth_check、check_340……），带 fit.txt 的一起显示。
只读文件、不碰训练进程；训练没在跑也能看历史。
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import yaml

from ..config import ROOT

RUNS = os.path.join(ROOT, 'runs')
SAMPLES_EST = 82.9e6


def _jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass                           # 正在写的最后一行
    return out


def _finite(o):
    """JSON 里不能有 NaN/Infinity（混合精度下偶尔出现的溢出步的梯度范数）→ None。"""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    return o


def list_runs():
    rs = []
    for d in glob.glob(os.path.join(RUNS, '*')):
        f = os.path.join(d, 'train_log.jsonl')
        if os.path.exists(f):
            rs.append((os.path.getmtime(f), os.path.basename(d)))
    return [n for _t, n in sorted(rs, reverse=True)]


CHECK_TITLES = {'plateau_check': '停滞检查（主卡组镜像）', 'breadth_check': '广度检查（随机卡组专项）', 'check_340': '3.4 亿检查（80/20 新配比）', 'gen_check': '通用性基线（双方都用随机卡组，对 big_248M）'}


def checks_data():
    """runs/*/h2h.jsonl：单独的对打检查。→ [{name, title, mtime, rows, fit}]，新的在前。"""
    out = []
    for f in glob.glob(os.path.join(RUNS, '*', 'h2h.jsonl')):
        d = os.path.dirname(f)
        n = os.path.basename(d)
        rows = [dict(a=os.path.basename(r.get('a', '')), b=r.get('b_name') or os.path.basename(r.get('b', '')), step=r.get('step'),
                     win=r.get('win', 0), loss=r.get('loss', 0), draw=r.get('draw', 0)) for r in _jsonl(f)]
        fit = None
        fp = os.path.join(d, 'fit.txt')
        if os.path.exists(fp):
            fit = open(fp, encoding='utf-8', errors='replace').read().strip()
        out.append(dict(name=n, title=CHECK_TITLES.get(n, n), mtime=os.path.getmtime(f), rows=rows, fit=fit))
    return sorted(out, key=lambda c: -c['mtime'])


def run_data(name):
    d = os.path.join(RUNS, os.path.basename(name))
    train = _jsonl(os.path.join(d, 'train_log.jsonl'))
    # 续训会从存档点重跑一段、或者回退到更早的存档（10-24 回退到 3.59 亿）：从后往前，比后写的步数小才留，
    # 被放弃的那段记成 (lo, hi]，下面评测里落在这段、存档又已经移走的点也不画
    keep, cut, lo = [], [], None
    for r in reversed(train):
        if lo is None or r['step'] < lo:
            keep.append(r)
            lo = r['step']
        elif cut and cut[-1][0] == lo:
            cut[-1][1] = max(cut[-1][1], r['step'])
        else:
            cut.append([lo, r['step']])
    train = keep[::-1]

    def dropped(step):
        return step is not None and any(a < step <= b for a, b in cut) and not os.path.exists(os.path.join(d, 'ppo_step%d.pt' % step))
    last_step = train[-1]['step'] if train else None
    val = {}
    for r in _jsonl(os.path.join(d, 'val_log.jsonl')):
        val[r['step']] = r
    arena = []
    for r in _jsonl(os.path.join(d, 'arena.jsonl')):
        ck = os.path.basename(r.get('ckpt', ''))
        step = r.get('step')
        if step is not None:
            pass
        elif 'step' in ck:
            try:
                step = int(ck.split('step')[1].split('.')[0])
            except ValueError:
                pass
        elif 'final' in ck:
            step = last_step
        if dropped(step):
            continue
        arena.append(dict(step=step, ckpt=ck, opp=r.get('opp'), deck=r.get('deck'), win_rate=r.get('win_rate'), games=r.get('games'),
                          win=r.get('win'), loss=r.get('loss'), draw=r.get('draw'),
                          crowns_for=r.get('avg_crowns_for'), crowns_against=r.get('avg_crowns_against')))
    # 固定考官（tools/gauntlet.py）：每个被测存档对每个考官的胜率；Elo 来自 gauntlet_elo.json（只用普通考官拟合）
    gaunt, gelo = [], []
    ge = {}
    gp = os.path.join(d, 'gauntlet_elo.json')
    if os.path.exists(gp):
        try:
            ge = json.load(open(gp, encoding='utf-8')).get('elo') or {}
        except ValueError:
            ge = {}
    seen_a = {}
    for r in _jsonl(os.path.join(d, 'gauntlet.jsonl')):
        if dropped(r.get('step')):
            continue
        gaunt.append(dict(step=r.get('step'), name=r.get('b_name') or r.get('b'), win_rate=r.get('win_rate'), games=r.get('games'),
                          win=r.get('win'), loss=r.get('loss'), special=not r.get('elo', True)))
        if r.get('step') is not None and r.get('a') in ge:
            seen_a[r['a']] = r['step']
    for a_, st in seen_a.items():
        gelo.append(dict(step=st, elo=ge[a_].get('elo'), ci=ge[a_].get('ci95')))
    gelo.sort(key=lambda x: x['step'])
    hardest = []
    lp = os.path.join(d, 'league.json')
    if os.path.exists(lp):
        try:
            hardest = json.load(open(lp, encoding='utf-8')).get('deck_hardest') or []
        except ValueError:
            hardest = []
    events = []
    ep = os.path.join(d, 'events.json')
    if os.path.exists(ep):
        try:
            events = sorted(json.load(open(ep, encoding='utf-8')), key=lambda e: e['step'])
        except (ValueError, KeyError, TypeError):
            events = []
    total, kind = None, 'bc'
    info = os.path.join(d, 'run_info.json')
    if os.path.exists(info):
        ri = json.load(open(info, encoding='utf-8'))
        total, kind = ri.get('total_steps'), ri.get('kind', 'bc')
    if total is None and os.path.exists(os.path.join(d, 'config.yaml')):
        bc = (yaml.safe_load(open(os.path.join(d, 'config.yaml'), encoding='utf-8')) or {}).get('bc') or {}
        if bc:
            total = int(bc.get('epochs', 1) * SAMPLES_EST * (1 - bc.get('val_frac', 0.02)) / bc.get('batch_size', 512))
    if kind == 'ppo':
        keep = ('step', 't', 'sps', 'actor_sps', 'actor_env_wait', 'actor_gpu', 'learner_train_s', 'learner_wait_s', 'lr', 'beta', 'shaping_c',
                'grad_norm', 'pg', 'vf', 'ent_card', 'ent_place', 'kl_bc_card', 'kl_bc_place', 'approx_kl', 'clipfrac', 'explained_var', 'warmup',
                'gns_G2', 'gns_trS', 'gns_batch',
                'win_all', 'win_bc_pool', 'win_bc_main', 'win_rule', 'n_episodes', 'ep_return', 'ep_len', 'crowns_for', 'crowns_against', 'env_errors',
                'ab_per_game', 'ab_ready_per_game', 'ab_use_rate', 'ab_p', 'ab_eps', 'ab_hero_used', 'ab_champ_used',
                'win_self', 'win_hist', 'win_bc_pool_by_main_deck', 'win_bc_pool_by_pool_deck', 'elo', 'n_snapshots', 'pfsp_min_winrate',
                'pfsp_mean_winrate', 'hist_id', 'n_exploiters', 'exp_mean_winrate', 'exp_min_winrate', 'target_winrate', 'target_step',
                'deck_tv', 'deck_min_winrate', 'deck_mean_winrate', 'deck_seen', 'n_self', 'n_hist', 'n_bc_main', 'n_bc_pool', 'n_rule')
    else:
        keep = ('step', 'samples_per_s', 'loader_wait', 'lr', 'grad_norm', 'loss_card', 'loss_place', 'loss_value',
                'card_top1_play', 'place_le1', 'place_exact', 'value_acc', 'play_rate_pred', 'play_rate_true', 'play_brier', 't')
    lm = os.path.getmtime(os.path.join(d, 'train_log.jsonl')) if train else None
    return dict(name=name, kind=kind, total=total, now=time.time(), last_write=lm,
                train=[{k: r.get(k) for k in keep} for r in train], val=[val[k] for k in sorted(val)], arena=arena,
                gauntlet=gaunt, gauntlet_elo=gelo, deck_hardest=hardest, events=events, checks=checks_data() if kind == 'ppo' else [])


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        b = body.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/':
            self._send(PAGE, 'text/html; charset=utf-8')
        elif u.path == '/api/runs':
            self._send(json.dumps(list_runs()), 'application/json')
        elif u.path == '/api/run':
            q = parse_qs(u.query)
            self._send(json.dumps(_finite(run_data(q.get('name', [''])[0]))), 'application/json')
        else:
            self.send_response(404)
            self.end_headers()


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>训练看板</title>
<style>
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
 --border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--good:#0ca30c;--warn:#fab219;--crit:#d03b3b;--wash:rgba(42,120,214,.08)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
 --grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--wash:rgba(57,135,229,.12)}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;
 --border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--wash:rgba(57,135,229,.12)}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:16px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:12px}
h1{font-size:18px;margin:0 8px 0 0;font-weight:650}
select,button,input{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:4px 8px}
button{cursor:pointer}
.status{display:inline-flex;align-items:center;gap:6px;color:var(--ink2)}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.ctl{display:flex;gap:12px;align-items:center;flex-wrap:wrap;color:var(--ink2)}
.prog{margin:4px 0 16px;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 14px}
.bar{height:8px;background:var(--grid);border-radius:4px;overflow:hidden;margin-top:8px}
.bar>i{display:block;height:100%;background:var(--s1);border-radius:4px}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;margin-bottom:16px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px 12px}
.tile .k{color:var(--ink2);font-size:12px}
.tile .v{font-size:22px;font-weight:650;margin-top:2px}
.tile .s{color:var(--muted);font-size:12px}
.sec{font-size:13px;color:var(--ink2);margin:18px 0 8px;font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:12px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px 12px 6px;position:relative;min-width:0}
.card h3{font-size:13px;margin:0;font-weight:600}
.card .sub{color:var(--muted);font-size:12px;margin:0 0 4px}
.legend{display:flex;gap:14px;font-size:12px;color:var(--ink2);margin:2px 0 2px}
.legend span{display:inline-flex;align-items:center;gap:5px}
.legend i{display:inline-block;width:14px;height:2px;border-radius:1px}
.legend i.dotk{width:8px;height:8px;border-radius:50%}
svg{display:block;width:100%;overflow:visible}
svg text{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:6px 8px;
 font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.12);white-space:nowrap;display:none;z-index:5}
.tip .row{display:flex;align-items:center;gap:6px}
.tip b{font-variant-numeric:tabular-nums}
.tip .x{color:var(--muted);margin-bottom:2px}
.tip i{display:inline-block;width:12px;height:2px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:12px;background:var(--surface);border:1px solid var(--border);border-radius:8px;overflow:hidden}
th,td{padding:6px 8px;text-align:right;border-bottom:1px solid var(--grid)}
th{color:var(--ink2);font-weight:600}
th:first-child,td:first-child{text-align:left}
.tablewrap{overflow-x:auto}
.empty{color:var(--muted);padding:30px 0;text-align:center}
.stale{opacity:.6;transition:opacity .2s}
.evl{color:var(--ink2);font-size:12px;margin:-8px 0 14px;display:flex;flex-wrap:wrap;gap:4px 16px}
.evl b{color:var(--ink);font-weight:600}
pre.fit{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:8px 10px;font-size:12px;overflow-x:auto;margin:6px 0 0}
@media (max-width:520px){.grid{grid-template-columns:1fr}.tile .v{font-size:18px}}
</style></head><body><div class="wrap">
<header>
  <h1>训练看板</h1>
  <select id="run" aria-label="选择训练"></select>
  <span class="status" id="status"></span>
  <span style="flex:1"></span>
  <div class="ctl">
    <label>平滑 <input id="smooth" type="range" min="0" max="0.98" step="0.02" value="0.8" style="vertical-align:middle;width:100px"></label>
    <label>范围 <select id="range"><option value="all">全部</option><option value="20000">最近 2 万步</option><option value="5000">最近 5 千步</option></select></label>
    <button id="theme" title="切换深浅色">◐</button>
  </div>
</header>
<div class="prog" id="prog"></div>
<div class="evl" id="evl"></div>
<div class="tiles" id="tiles"></div>
<div class="sec" id="sec1">准确率（训练：每 100 步的滑动值；验证：每 5000 步在验证局上算）</div>
<div class="grid" id="g-acc"></div>
<div class="sec" id="sec2">损失</div>
<div class="grid" id="g-loss"></div>
<div class="sec" id="sec3">对战评测与训练状态</div>
<div class="grid" id="g-sys"></div>
<div class="sec" id="sec4">验证记录</div>
<div class="tablewrap" id="vtable"></div>
</div>
<script>
const $=s=>document.querySelector(s);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let DATA=null, RUN=null;
const fmtPct=v=>v==null?'–':(v*100).toFixed(1)+'%';
const fmtNum=(v,d=3)=>v==null?'–':(+v).toFixed(d);
const fmtK=v=>v==null?'–':(v>=1000?(v/1000).toFixed(v>=1e5?0:1)+'k':String(Math.round(v)));
const fmtDur=s=>{if(s==null||!isFinite(s))return '–';s=Math.max(0,s);const h=Math.floor(s/3600),m=Math.round((s%3600)/60);return h?`${h} 小时 ${m} 分`:`${m} 分`};

const CIRC='①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮';
const fmtYi=v=>(v/1e8).toFixed(3)+' 亿';
// Wilson 95% 区间（w = 胜场，平局由调用方折成半胜）
const wilson=(w,n)=>{if(!n)return null;const z=1.96,p=w/n,dd=1+z*z/n,c=(p+z*z/(2*n))/dd,h=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n))/dd;return [c-h,c+h]};
let EVS=[];
function ema(arr,a){let m=null;return arr.map(v=>{if(v==null)return null;m=m==null?v:a*m+(1-a)*v;return m})}
function nice(lo,hi,n=4){if(lo===hi){lo-=1;hi+=1}const span=hi-lo,st=Math.pow(10,Math.floor(Math.log10(span/n))),
 e=span/n/st,step=st*(e>=5?10:e>=2?5:e>=1?2:1);const a=Math.floor(lo/step)*step,b=Math.ceil(hi/step)*step;const t=[];
 for(let v=a;v<=b+step*1e-9;v+=step)t.push(+v.toFixed(10));return {lo:a,hi:b,ticks:t}}
function el(tag,attrs,parent){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const k in attrs)e.setAttribute(k,attrs[k]);if(parent)parent.appendChild(e);return e}

// series: [{name,color,pts:[[x,y]],kind:'line'|'dots',raw:[[x,y]]?}]
function chart(host,{title,sub,series,yfmt=v=>fmtNum(v,2),xr,yr,ev=true}){
  const card=document.createElement('div');card.className='card';host.appendChild(card);
  const h=document.createElement('h3');h.textContent=title;card.appendChild(h);
  if(sub){const p=document.createElement('div');p.className='sub';p.textContent=sub;card.appendChild(p)}
  if(series.length>1){const lg=document.createElement('div');lg.className='legend';
    for(const s of series){const sp=document.createElement('span');const i=document.createElement('i');i.style.background=s.color;if(s.kind==='dots')i.className='dotk';
      sp.appendChild(i);sp.appendChild(document.createTextNode(s.name));lg.appendChild(sp)}card.appendChild(lg)}
  const W=Math.max(280,card.clientWidth-24),H=190,ml=48,mr=12,mt=8,mb=24,pw=W-ml-mr,ph=H-mt-mb;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,height:H,role:'img','aria-label':title},card);
  const all=series.flatMap(s=>(s.raw||[]).concat(s.pts).concat((s.ci||[]).filter(c=>c).flatMap(c=>[[0,c[0]],[0,c[1]]]))).filter(p=>p[1]!=null&&isFinite(p[1]));
  if(!all.length){const t=el('text',{x:W/2,y:H/2,'text-anchor':'middle'},svg);t.textContent='暂无数据';return}
  let ylo=Math.min(...all.map(p=>p[1])),yhi=Math.max(...all.map(p=>p[1]));
  if(yr){ylo=yr[0]!=null?Math.min(yr[0],ylo):ylo;yhi=yr[1]!=null?Math.max(yr[1],yhi):yhi}
  const ny=nice(ylo,yhi),x0=xr[0],x1=xr[1]>xr[0]?xr[1]:xr[0]+1;
  const X=x=>ml+(x-x0)/(x1-x0)*pw,Y=y=>mt+ph-(y-ny.lo)/(ny.hi-ny.lo)*ph;
  for(const t of ny.ticks){el('line',{x1:ml,x2:ml+pw,y1:Y(t),y2:Y(t),stroke:css('--grid'),'stroke-width':1},svg);
    const tx=el('text',{x:ml-6,y:Y(t)+4,'text-anchor':'end'},svg);tx.textContent=yfmt(t)}
  el('line',{x1:ml,x2:ml+pw,y1:mt+ph,y2:mt+ph,stroke:css('--axis'),'stroke-width':1},svg);
  const nx=nice(x0,x1,5);for(const t of nx.ticks){if(t<x0||t>x1)continue;const tx=el('text',{x:X(t),y:H-6,'text-anchor':'middle'},svg);tx.textContent=fmtK(t)}
  // 配置改动的时间点（events.json）：竖虚线 + 圈号，说明在页面顶部
  if(ev)for(const e of EVS){if(e.step<x0||e.step>x1)continue;const xx=X(e.step);
    el('line',{x1:xx,x2:xx,y1:mt,y2:mt+ph,stroke:css('--muted'),'stroke-width':1,'stroke-dasharray':'2 3'},svg);
    const t=el('text',{x:xx+2,y:mt+9,style:'fill:var(--ink2);font-size:10px'},svg);t.textContent=e.n}
  const path=pts=>{let d='',pen=false;for(const [x,y] of pts){if(y==null||!isFinite(y)){pen=false;continue}d+=(pen?'L':'M')+X(x).toFixed(1)+','+Y(y).toFixed(1);pen=true}return d};
  for(const s of series){
    if(s.raw)el('path',{d:path(s.raw),fill:'none',stroke:s.color,'stroke-width':1,'stroke-opacity':.22},svg);
    if(s.kind==='dots'){if(s.ci)s.pts.forEach((p,j)=>{const c=s.ci[j];if(p[1]==null||!c)return;
        el('line',{x1:X(p[0]),x2:X(p[0]),y1:Y(c[0]),y2:Y(c[1]),stroke:s.color,'stroke-width':1.5,'stroke-opacity':.45},svg)});
      for(const [x,y] of s.pts){if(y==null)continue;el('circle',{cx:X(x),cy:Y(y),r:4.5,fill:s.color,stroke:css('--surface'),'stroke-width':2},svg)}
      if(s.pts.length>1)el('path',{d:path(s.pts),fill:'none',stroke:s.color,'stroke-width':1.5,'stroke-dasharray':'3 3'},svg)}
    else el('path',{d:path(s.pts),fill:'none',stroke:s.color,'stroke-width':2,'stroke-linejoin':'round'},svg)
  }
  // 末端直接标注
  const labs=[];for(const s of series){const p=[...s.pts].reverse().find(p=>p[1]!=null);if(p)labs.push({p,y:Y(p[1])-6})}
  labs.sort((a,b)=>a.y-b.y);for(let i=1;i<labs.length;i++)if(labs[i].y-labs[i-1].y<13)labs[i].y=labs[i-1].y+13;   // 末端标注错开，不重叠
  for(const l of labs){const t=el('text',{x:Math.min(X(l.p[0])+4,ml+pw-2),y:l.y,'text-anchor':X(l.p[0])>ml+pw-40?'end':'start',style:'fill:var(--ink2);font-weight:600'},svg);t.textContent=yfmt(l.p[1])}
  // 悬停：竖线 + 所有序列在最近 x 的值
  const cross=el('line',{y1:mt,y2:mt+ph,stroke:css('--axis'),'stroke-width':1,visibility:'hidden'},svg);
  const hit=el('rect',{x:ml,y:mt,width:pw,height:ph,fill:'transparent'},svg);
  const tip=document.createElement('div');tip.className='tip';card.appendChild(tip);
  const xs=[...new Set(series.flatMap(s=>s.pts.map(p=>p[0])))].sort((a,b)=>a-b);
  const near=(pts,x)=>{let b=null,bd=1e18;for(const p of pts){const d=Math.abs(p[0]-x);if(d<bd&&p[1]!=null){bd=d;b=p}}return b};
  function show(ev){const r=svg.getBoundingClientRect(),sx=(ev.clientX-r.left)*W/r.width,x=x0+(sx-ml)/pw*(x1-x0);
    const xn=near(xs.map(v=>[v,0]),x);if(!xn)return;cross.setAttribute('x1',X(xn[0]));cross.setAttribute('x2',X(xn[0]));cross.setAttribute('visibility','visible');
    tip.textContent='';const hd=document.createElement('div');hd.className='x';hd.textContent='第 '+xn[0].toLocaleString()+' 步';tip.appendChild(hd);
    for(const s of series){const p=s.kind==='dots'?near(s.pts,xn[0]):s.pts.find(q=>q[0]===xn[0]);if(!p)continue;
      const row=document.createElement('div');row.className='row';const i=document.createElement('i');i.style.background=s.color;
      const b=document.createElement('b');b.textContent=yfmt(p[1]);const n=document.createElement('span');n.style.color='var(--ink2)';
      n.textContent=s.name+(s.kind==='dots'&&p[0]!==xn[0]?'（@'+fmtK(p[0])+'）':'');row.append(i,b,n);tip.appendChild(row)}
    tip.style.display='block';const cr=card.getBoundingClientRect();let lx=ev.clientX-cr.left+12;if(lx+tip.offsetWidth>cr.width)lx=ev.clientX-cr.left-tip.offsetWidth-12;
    tip.style.left=lx+'px';tip.style.top=(ev.clientY-cr.top+12)+'px'}
  hit.addEventListener('pointermove',show);hit.addEventListener('pointerleave',()=>{tip.style.display='none';cross.setAttribute('visibility','hidden')});
}

function tile(host,k,v,s){const t=document.createElement('div');t.className='tile';const a=document.createElement('div');a.className='k';a.textContent=k;
  const b=document.createElement('div');b.className='v';b.textContent=v;t.append(a,b);if(s){const c=document.createElement('div');c.className='s';c.textContent=s;t.appendChild(c)}host.appendChild(t)}

function render(){
  const d=DATA;if(!d)return;
  const tr=d.train,va=d.val,ar=d.arena.filter(r=>r.step!=null);
  const last=tr[tr.length-1];
  const lastStep=last?last.step:0;
  const rng=$('#range').value,xlo=rng==='all'?(tr.length?Math.max(0,tr[0].step-(tr.length>1?tr[1].step-tr[0].step:0)):0):Math.max(0,lastStep-(+rng));
  const T=tr.filter(r=>r.step>=xlo),V=va.filter(r=>r.step>=xlo),A=ar.filter(r=>r.step>=xlo);
  const a=+$('#smooth').value,xr=[xlo,Math.max(lastStep,xlo+1)];
  const age=d.last_write?d.now-d.last_write:null,running=age!=null&&age<300;
  const st=$('#status');st.textContent='';const dot=document.createElement('span');dot.className='dot';dot.style.background=running?css('--good'):css('--muted');
  st.append(dot,document.createTextNode(running?'训练中':(age==null?'无数据':'已停止（'+fmtDur(age)+'前最后更新）')));
  // 进度
  const recent=tr.slice(-10),sps=recent.length?recent.reduce((s,r)=>s+(r.samples_per_s??r.sps??0),0)/recent.length:null;
  const stepsPerS=recent.length>1?(recent[recent.length-1].step-recent[0].step)/Math.max(1,recent[recent.length-1].t-recent[0].t):null;
  const eta=d.total&&stepsPerS?(d.total-lastStep)/stepsPerS:null;
  const pg=$('#prog');pg.textContent='';const line=document.createElement('div');
  line.textContent=`第 ${lastStep.toLocaleString()} / ${d.total?d.total.toLocaleString():'?'} 步（${d.total?(lastStep/d.total*100).toFixed(1):'?'}%） · 预计还要 ${running?fmtDur(eta):'–'}`;
  const bar=document.createElement('div');bar.className='bar';const fill=document.createElement('i');fill.style.width=(d.total?Math.min(100,lastStep/d.total*100):0)+'%';bar.appendChild(fill);pg.append(line,bar);
  EVS=(d.events||[]).map((e,i)=>({step:e.step,label:e.label,n:CIRC[i]||String(i+1)}));
  const evl=$('#evl');evl.textContent='';
  if(EVS.length){const h=document.createElement('span');h.textContent='改动时间点（曲线上的竖虚线）：';evl.appendChild(h);
    for(const e of EVS){const sp=document.createElement('span');const b=document.createElement('b');b.textContent=e.n+' '+fmtYi(e.step);
      sp.append(b,document.createTextNode(' '+e.label+(e.step>lastStep?'（还没到）':'')));evl.appendChild(sp)}}
  if(d.kind==='ppo'){renderPPO(d,T,xr,a,sps);return}
  $('#sec1').textContent='准确率（训练：每 100 步的滑动值；验证：每 5000 步在验证局上算）';$('#sec2').textContent='损失';
  $('#sec3').textContent='对战评测与训练状态';$('#sec4').textContent='验证记录';
  // 指标卡
  const tl=$('#tiles');tl.textContent='';const lv=va[va.length-1],la=[...d.arena].reverse().find(r=>r.opp==='rule'&&r.deck!=='pool');
  tile(tl,'训练速度',sps?Math.round(sps).toLocaleString()+' 样本/s':'–',last?'等数据 '+fmtPct(last.loader_wait):'');
  tile(tl,'学习率',last?last.lr.toExponential(2):'–','梯度范数 '+(last?fmtNum(last.grad_norm,2):'–'));
  tile(tl,'牌头 top-1（验证）',fmtPct(lv&&lv.card_top1_play),lv?'@ '+fmtK(lv.step)+' 步':'');
  tile(tl,'位置 ≤1 格（验证）',fmtPct(lv&&lv.place_le1),lv?'完全命中 '+fmtPct(lv.place_exact):'');
  tile(tl,'价值头胜负（验证）',fmtPct(lv&&lv.value_acc),lv?'MSE '+fmtNum(lv.loss_value,3):'');
  tile(tl,'对规则 AI 胜率',la?fmtPct(la.win_rate):'–',la?`${la.win} 胜 ${la.loss} 负 ${la.draw} 平`+(la.step!=null?` · @${fmtK(la.step)} 步`:''):'尚未评测');
  const S1=css('--s1'),S2=css('--s2');
  const tser=(key)=>{const raw=T.map(r=>[r.step,r[key]]);const sm=ema(raw.map(p=>p[1]),a);return {name:'训练',color:S1,kind:'line',raw,pts:raw.map((p,i)=>[p[0],sm[i]])}};
  const vser=(key)=>({name:'验证',color:S2,kind:'dots',pts:V.map(r=>[r.step,r[key]])});
  const g1=$('#g-acc'),g2=$('#g-loss'),g3=$('#g-sys');g1.textContent=g2.textContent=g3.textContent='';
  chart(g1,{title:'牌头 top-1（出牌步）',sub:'在出牌选项里挑对是哪张牌/技能',series:[tser('card_top1_play'),vser('card_top1_play')],yfmt:fmtPct,xr});
  chart(g1,{title:'位置误差 ≤1 格',sub:'切比雪夫距离，只算出手牌的步',series:[tser('place_le1'),vser('place_le1')],yfmt:fmtPct,xr});
  chart(g1,{title:'价值头胜负预测',sub:'符号与该方终局胜负一致的比例（平局不算）',series:[tser('value_acc'),vser('value_acc')],yfmt:fmtPct,xr});
  chart(g2,{title:'牌头损失',series:[tser('loss_card'),vser('loss_card')],xr});
  chart(g2,{title:'位置头损失',series:[tser('loss_place'),vser('loss_place')],xr});
  chart(g2,{title:'价值头损失（MSE）',sub:'1.0 = 一直猜平局的水平',series:[tser('loss_value'),vser('loss_value')],xr});
  chart(g3,{title:'对规则 AI 胜率',sub:'主卡组，各存档点在引擎里采样对战',series:[{name:'对规则 AI',color:S1,kind:'dots',pts:A.filter(r=>r.opp==='rule'&&r.deck!=='pool').map(r=>[r.step,r.win_rate])}],yfmt:fmtPct,xr,yr:[0,1]});
  const sp=T.map(r=>[r.step,r.samples_per_s]);
  chart(g3,{title:'训练速度（样本/秒）',sub:'你打游戏时显卡被占用会明显下降',series:[{name:'样本/秒',color:S1,kind:'line',pts:sp}],yfmt:v=>fmtK(v),xr,yr:[0,null]});
  chart(g3,{title:'学习率',series:[{name:'学习率',color:S1,kind:'line',pts:T.map(r=>[r.step,r.lr])}],yfmt:v=>v.toExponential(1),xr,yr:[0,null]});
  // 验证表
  const vt=$('#vtable');vt.textContent='';
  if(!va.length){const e=document.createElement('div');e.className='empty';e.textContent='还没有验证记录';vt.appendChild(e)}
  else{const tb=document.createElement('table');const hd=['步数','牌头 top-1','位置 ≤1 格','位置完全命中','价值头胜负','牌头损失','位置头损失','价值头 MSE'];
    const tr0=document.createElement('tr');for(const h of hd){const th=document.createElement('th');th.textContent=h;tr0.appendChild(th)}tb.appendChild(tr0);
    for(const r of [...va].reverse()){const row=document.createElement('tr');
      for(const v of [r.step.toLocaleString(),fmtPct(r.card_top1_play),fmtPct(r.place_le1),fmtPct(r.place_exact),fmtPct(r.value_acc),fmtNum(r.loss_card),fmtNum(r.loss_place),fmtNum(r.loss_value)]){
        const td=document.createElement('td');td.textContent=v;row.appendChild(td)}tb.appendChild(row)}vt.appendChild(tb)}
}

function renderPPO(d,T,xr,a,sps){
  const S1=css('--s1'),S2=css('--s2'),S3=css('--s3');
  const last=T[T.length-1]||{};
  const E=d.arena.filter(r=>r.step!=null&&r.step>=xr[0]);
  const lastE=o=>[...d.arena].reverse().find(r=>r.opp===o);
  const tl=$('#tiles');tl.textContent='';
  const sm=key=>{const v=ema(T.map(r=>r[key]),a);for(let i=v.length-1;i>=0;i--)if(v[i]!=null)return v[i];return null};
  tile(tl,'吞吐',sps?Math.round(sps).toLocaleString()+' 步/s':'–',last.actor_sps?'采样 '+Math.round(last.actor_sps).toLocaleString()+' 步/s':'');
  for(const [o,n] of [['bc_main','对 BC 主卡组（评测）'],['bc_pool','对 BC 对手池（评测）'],['rule','对规则 AI（评测）']]){
    const e=lastE(o);tile(tl,n,e?fmtPct(e.win_rate):'–',e?`${e.win} 胜 ${e.loss} 负 ${e.draw} 平 · @${fmtK(e.step)}`:'等待首次评测')}
  tile(tl,'训练中胜率',fmtPct(sm('win_all')),'采样对局的结果（平滑后）');
  // 训练里最后一次有 BC 对局的步数：之后联赛 Elo 只按对历史快照的结果更新，没有锚点
  const lastBC=d.train.reduce((m,r)=>(r.n_bc_main||0)+(r.n_bc_pool||0)>0?r.step:m,null),noAnchor=lastBC!=null&&(last.step||0)>lastBC+2e6;
  const klOff=(()=>{let s0=null;for(const r of d.train){if(r.beta===0){if(s0==null)s0=r.step}else if(r.beta!=null)s0=null}return s0})();
  if(last.elo!=null)tile(tl,noAnchor?'联赛 Elo（无锚点，仅参考）':'Elo（BC = 0）',fmtNum(last.elo,0),
    noAnchor?`${fmtYi(lastBC)}后没有 BC 对局；强弱看固定考官`:`快照池 ${last.n_snapshots} 个`);
  {const e=lastE('pool:bc_pool');if(e)tile(tl,'其他卡组对 BC（评测）',fmtPct(e.win_rate),`${e.win} 胜 ${e.loss} 负 · @${fmtK(e.step)}`)}
  tile(tl,'离 BC 的距离（KL）',last.kl_bc_card!=null?fmtNum(last.kl_bc_card+(last.kl_bc_place||0),4):'–',
    klOff!=null?`约束已取消（${fmtYi(klOff)}起 β = 0），只作观察`:'约束系数 β '+(last.beta!=null?fmtNum(last.beta,3):'–'));
  tile(tl,'阶段',last.warmup?'热身 critic':'PPO 训练',last.explained_var!=null?'价值解释方差 '+fmtNum(last.explained_var,2):'');
  $('#sec1').textContent='胜率（常规评测：每 800 万步、每项 100–200 局、按概率采样，对 BC 早已饱和；强弱主要看固定考官和下面的单独检查）';
  $('#sec2').textContent='损失与约束';$('#sec3').textContent='吞吐与调度';$('#sec4').textContent='评测记录';
  const g1=$('#g-acc'),g2=$('#g-loss'),g3=$('#g-sys');g1.textContent=g2.textContent=g3.textContent='';
  const ev=o=>E.filter(r=>r.opp===o).map(r=>[r.step,r.win_rate]);
  const tw=key=>{const raw=T.map(r=>[r.step,r[key]]);const v=ema(raw.map(p=>p[1]),a);return raw.map((p,i)=>[p[0],p[1]==null?null:v[i]])};
  const evs=[{name:'对 BC 主卡组',color:S1,kind:'dots',pts:ev('bc_main')},{name:'对 BC 对手池',color:S2,kind:'dots',pts:ev('bc_pool')},
    {name:'对规则 AI',color:S3,kind:'dots',pts:ev('rule')}];
  if(ev('pool:bc_pool').length)evs.push({name:'其他卡组对 BC 对手池',color:css('--s4'),kind:'dots',pts:ev('pool:bc_pool')});
  chart(g1,{title:'评测胜率（对 BC）',sub:'每点 100–200 局；前两项是主卡组。取消 KL 约束后这里明显下降 = 策略在往不像人的方向漂',series:evs,yfmt:fmtPct,xr,yr:[0.3,1]});
  const isExp=T.some(r=>r.target_winrate!=null);
  if(isExp){
    chart(g1,{title:'剥削者对主模型的胜率',sub:'对手是主模型的冻结副本；起点约 50%，越高说明找到的漏洞越多（≥ 55% 的快照会加入主模型的联赛）',
      series:[{name:'每批（平滑）',color:S1,kind:'line',pts:tw('win_hist')},{name:'带衰减的累计',color:S2,kind:'line',pts:T.map(r=>[r.step,r.target_winrate])}],yfmt:fmtPct,xr,yr:[0.3,0.9]});
  }
  if(!isExp&&T.some(r=>r.elo!=null)){
    chart(g1,{title:noAnchor?'联赛 Elo（无锚点，仅参考）':'Elo（以 BC 为 0）',sub:noAnchor?`${fmtYi(lastBC)}起只对自己和历史快照打（80/20），这条线只反映相对漂移，不代表强弱`:'按对 BC 和历史快照的每局结果在线更新',
      series:[{name:'Elo',color:S1,kind:'line',pts:T.map(r=>[r.step,r.elo])}],yfmt:v=>fmtNum(v,0),xr});
    chart(g1,{title:'训练中胜率（联赛对手）',sub:'自我对弈理论上在 50% 附近；对历史快照越高说明在进步',series:[{name:'对历史快照',color:S1,kind:'line',pts:tw('win_hist')},
      {name:'自我对弈',color:S2,kind:'line',pts:tw('win_self')},{name:'PFSP 最难的快照',color:S3,kind:'line',pts:T.map(r=>[r.step,r.pfsp_min_winrate])}],yfmt:fmtPct,xr,yr:[0.2,1]});
    chart(g1,{title:'主卡组 vs 其他卡组（训练中，对 BC 对手池）',series:[{name:'主卡组',color:S1,kind:'line',pts:tw('win_bc_pool_by_main_deck')},
      {name:'其他卡组',color:S2,kind:'line',pts:tw('win_bc_pool_by_pool_deck')}],yfmt:fmtPct,xr,yr:[0.3,1]});
  }
  chart(g1,{title:'训练中胜率（按对手）',sub:'采样对局的结果，每批约 25 局，已平滑',series:[{name:'对 BC 主卡组',color:S1,kind:'line',pts:tw('win_bc_main')},
    {name:'对 BC 对手池',color:S2,kind:'line',pts:tw('win_bc_pool')},{name:'对规则 AI',color:S3,kind:'line',pts:tw('win_rule')}],yfmt:fmtPct,xr,yr:[0.3,1]});
  chart(g1,{title:'每局回报',sub:'胜负 ±1 + 塔血势能塑形 + 圣水溢出惩罚',series:[{name:'回报',color:S1,kind:'line',pts:tw('ep_return')}],xr});
  if(T.some(r=>r.ab_use_rate!=null)){
    chart(g1,{title:'英雄/冠军技能：每局发动次数',sub:'人类回放约每方 1–2 次（用英雄的方里 50–80% 用过）；10-28 查出 PPO 压到几乎 0',
      series:[{name:'每局发动',color:S1,kind:'line',pts:tw('ab_per_game')}],yfmt:v=>fmtNum(v,2),xr,yr:[0,null]});
    chart(g1,{title:'英雄/冠军技能：可以按时按的比例',sub:'发动次数 ÷ 有技能可以按的决策步数（每步 0.25 s）；BC 约 0.2–1%，10-28 的 PPO 约 0.01–0.08%',
      series:[{name:'按的比例',color:S1,kind:'line',pts:tw('ab_use_rate')},{name:'策略本身（不含下限）',color:S2,kind:'line',pts:tw('ab_p')},
        {name:'采样端下限 ε',color:S3,kind:'line',pts:T.map(r=>[r.step,r.ab_eps])}],yfmt:v=>(v*100).toFixed(2)+'%',xr,yr:[0,null]});
  }
  if(T.some(r=>r.ab_hero_used!=null||r.ab_champ_used!=null)){
    chart(g1,{title:'用过技能的方占比',sub:'这局有过可以按的时候、最后按过的方；人类回放里英雄 50–80%',
      series:[{name:'英雄',color:S1,kind:'line',pts:tw('ab_hero_used')},{name:'冠军等（单位名不带 Hero）',color:S2,kind:'line',pts:tw('ab_champ_used')}],yfmt:fmtPct,xr,yr:[0,1]});
  }
  {const G=(d.gauntlet||[]).filter(r=>r.step!=null&&r.step>=xr[0]);
   if(G.length){const pal=[S1,S2,S3,css('--s4'),'#8b5cf6','#d97706','#0891b2','#be185d'];
     const core=[...new Set(G.filter(r=>!r.special).map(r=>r.name))],spec=[...new Set(G.filter(r=>r.special).map(r=>r.name))];
     const ser=ns=>ns.map((n,i)=>{const R=G.filter(r=>r.name===n);return {name:n,color:pal[i%pal.length],kind:'dots',pts:R.map(r=>[r.step,r.win_rate]),
       ci:R.map(r=>wilson((r.win||0)+0.5*Math.max(0,(r.games||0)-(r.win||0)-(r.loss||0)),r.games))}});
     chart(g1,{title:'固定考官胜率',sub:'被测方用主卡组打冻结的考官；竖线 = 95% 区间。龙骑墓镜像，只作参考（目标是通用模型；主指标看右边的 metagen_248M）',series:ser(core),yfmt:fmtPct,xr,yr:[0.2,1]});
     if(spec.length)chart(g1,{title:'专项评测（同一个考官换卡组）',sub:'metagen = 双方都用主流卡组、配对换边（主指标，10-27 起，400 局）；gen = 双方都用随机拼凑的卡组（参考，10-27 起 200 局、配对换边）；meta = 被测方用龙骑墓打主流卡组；mg / rand 已停用；竖线 = 95% 区间（配对评测的实际误差更小）',series:ser(spec),yfmt:fmtPct,xr,yr:[0.2,1]});
     const ge=(d.gauntlet_elo||[]).filter(r=>r.step>=xr[0]);
     if(ge.length)chart(g1,{title:'固定考官 Elo',sub:'Bradley–Terry 拟合（第一个考官 = 0），只用普通考官；跨时间可比',series:[{name:'Elo',color:S1,kind:'dots',pts:ge.map(r=>[r.step,r.elo])}],yfmt:v=>fmtNum(v,0),xr});}}
  if(T.some(r=>r.deck_tv!=null))chart(g1,{title:'按卡组挑难的练',sub:'对各对手卡组的胜率（带衰减、向平均收缩）；偏离 = 对手卡组抽样分布偏离使用量的程度',
    series:[{name:'最难卡组的胜率',color:S1,kind:'line',pts:T.map(r=>[r.step,r.deck_min_winrate])},{name:'按使用量加权的平均胜率',color:S2,kind:'line',pts:T.map(r=>[r.step,r.deck_mean_winrate])},
      {name:'偏离',color:S3,kind:'line',pts:T.map(r=>[r.step,r.deck_tv])}],yfmt:fmtPct,xr,yr:[0,1]});
  const one=(key,name,col)=>({name,color:col||S1,kind:'line',pts:tw(key)});
  chart(g2,{title:'策略损失（PPO 截断目标）',series:[one('pg','策略损失')],yfmt:v=>fmtNum(v,3),xr});
  chart(g2,{title:'价值损失',series:[one('vf','价值 MSE')],yfmt:v=>fmtNum(v,3),xr});
  chart(g2,{title:'价值解释方差',sub:'越接近 1 越好',series:[one('explained_var','解释方差')],yfmt:v=>fmtNum(v,2),xr});
  chart(g2,{title:klOff!=null?'离 BC 策略的距离（KL，已不作约束）':'KL 到 BC 策略',sub:klOff!=null?`${fmtYi(klOff)}起约束已取消（和 OpenAI Five 一样）：这条线只用来观察离人类打法有多远，持续上升 = 在漂移`:'防止忘掉人类打法；系数逐步退火',
    series:[one('kl_bc_card','牌头'),one('kl_bc_place','位置头',S2)],yfmt:v=>fmtNum(v,4),xr});
  chart(g2,{title:'牌头熵',series:[one('ent_card','牌头熵')],yfmt:v=>fmtNum(v,3),xr});
  chart(g2,{title:'每批策略变化（approx KL）',sub:'过大说明一步走得太远',series:[one('approx_kl','approx KL')],yfmt:v=>fmtNum(v,4),xr});
  chart(g2,{title:'截断比例（clipfrac）',series:[one('clipfrac','clipfrac')],yfmt:fmtPct,xr});
  {const R=T.filter(r=>r.gns_G2!=null&&r.gns_trS!=null),N=50,pts=[];
   // 临界批量 B_noise = tr(Σ)/|G|²：单次估计噪声很大，分子分母各取最近 N 次更新的和再相除（McCandlish 等 2018）
   for(let i=N-1;i<R.length;i++){let a=0,b=0;for(let j=i-N+1;j<=i;j++){a+=R[j].gns_trS;b+=R[j].gns_G2}pts.push([R[i].step,b>0?a/b:null])}
   if(pts.length)chart(g2,{title:'梯度噪声尺度（临界批量）',sub:`最近 ${N} 次更新合起来估计。远大于每批数据量 = 每批的梯度主要是噪声（可以把学习率和批量按比例一起加大）；接近或小于 = 再加大批量就浪费`,
     series:[{name:'临界批量估计',color:S1,kind:'line',pts},{name:'每批数据量',color:S2,kind:'line',pts:R.map(r=>[r.step,r.gns_batch])}],yfmt:v=>fmtK(v),xr,yr:[0,null]});}
  chart(g3,{title:'吞吐（步/秒）',sub:'整体 = 采样与训练并行后的实际速度',series:[{name:'整体',color:S1,kind:'line',pts:T.map(r=>[r.step,r.sps])},
    {name:'采样',color:S2,kind:'line',pts:T.map(r=>[r.step,r.actor_sps])}],yfmt:v=>fmtK(v),xr,yr:[0,null]});
  chart(g3,{title:'采样进程时间分解',sub:'推理 = 等 GPU；等环境 = 等引擎进程',series:[{name:'推理',color:S1,kind:'line',pts:T.map(r=>[r.step,r.actor_gpu])},
    {name:'等环境',color:S2,kind:'line',pts:T.map(r=>[r.step,r.actor_env_wait])}],yfmt:fmtPct,xr,yr:[0,1]});
  chart(g3,{title:'学习进程每批耗时（秒）',series:[{name:'训练',color:S1,kind:'line',pts:T.map(r=>[r.step,r.learner_train_s])},
    {name:'等数据',color:S2,kind:'line',pts:T.map(r=>[r.step,r.learner_wait_s])}],yfmt:v=>fmtNum(v,1),xr,yr:[0,null]});
  chart(g3,{title:'调度系数',sub:'β：KL 约束；c：奖励塑形',series:[{name:'β',color:S1,kind:'line',pts:T.map(r=>[r.step,r.beta])},
    {name:'c',color:S2,kind:'line',pts:T.map(r=>[r.step,r.shaping_c])}],yfmt:v=>fmtNum(v,2),xr,yr:[0,null]});
  chart(g3,{title:'学习率',series:[{name:'学习率',color:S1,kind:'line',pts:T.map(r=>[r.step,r.lr])}],yfmt:v=>v.toExponential(1),xr,yr:[0,null]});
  const vt=$('#vtable');vt.textContent='';
  const mk=(hd,rows,title)=>{if(title){const h=document.createElement('div');h.className='sec';h.textContent=title;vt.appendChild(h)}
    const t=document.createElement('table');const r0=document.createElement('tr');for(const x of hd){const th=document.createElement('th');th.textContent=x;r0.appendChild(th)}t.appendChild(r0);
    for(const r of rows){const row=document.createElement('tr');for(const v of r){const td=document.createElement('td');td.textContent=v;row.appendChild(td)}t.appendChild(row)}
    const w=document.createElement('div');w.className='tablewrap';w.appendChild(t);vt.appendChild(w)};
  for(const c of d.checks||[]){
    const when=new Date(c.mtime*1000).toLocaleString('zh-CN',{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'});
    mk(['被测存档','对手','局数','胜率','95% 区间','胜','负','平'],c.rows.map(r=>{const n=r.win+r.loss+r.draw,w=r.win+0.5*r.draw,ci=wilson(w,n);
      return [r.a.replace('ppo_step','').replace('.pt',''),r.b,n,fmtPct(n?w/n:null),ci?`${fmtPct(ci[0])} – ${fmtPct(ci[1])}`:'–',r.win,r.loss,r.draw]}),
      `单独检查：${c.title}（${when} 更新；区间不含 50% 才算显著）`);
    if(c.fit){const pre=document.createElement('pre');pre.className='fit';pre.textContent=c.fit;vt.appendChild(pre)}}
  if(!d.arena.length){const e=document.createElement('div');e.className='empty';e.textContent='还没有评测记录（开训时先评一次初始模型）';vt.appendChild(e);return}
  {const h=document.createElement('div');h.className='sec';h.textContent='常规评测记录（对 BC）';vt.appendChild(h)}
  const tb=document.createElement('table');const hd=['步数','对手','局数','胜率','胜','负','平','平均冠数（我 : 敌）'];
  const tr0=document.createElement('tr');for(const h of hd){const th=document.createElement('th');th.textContent=h;tr0.appendChild(th)}tb.appendChild(tr0);
  const nm={bc_main:'BC 主卡组',bc_pool:'BC 对手池',rule:'规则 AI',random:'随机','pool:bc_pool':'其他卡组 vs BC 对手池'};
  for(const r of [...d.arena].reverse()){const row=document.createElement('tr');
    for(const v of [r.step!=null?r.step.toLocaleString():'–',nm[r.opp]||r.opp,r.games,fmtPct(r.win_rate),r.win,r.loss,r.draw,
                    r.crowns_for!=null?`${fmtNum(r.crowns_for,2)} : ${fmtNum(r.crowns_against,2)}`:'–']){
      const td=document.createElement('td');td.textContent=v;row.appendChild(td)}tb.appendChild(row)}
  vt.appendChild(tb);
  if((d.gauntlet||[]).length){const gel=new Map((d.gauntlet_elo||[]).map(r=>[r.step,r]));
    mk(['步数','考官','类别','局数','胜率','胜','负','该存档 Elo'],[...d.gauntlet].reverse().map(r=>{const e=gel.get(r.step);
      return [r.step!=null?r.step.toLocaleString():'–',r.name,r.special?'专项':'普通',r.games,fmtPct(r.win_rate),r.win,r.loss,e?`${fmtNum(e.elo,0)} ± ${fmtNum(e.ci,0)}`:'–']}),'固定考官记录')}
  if((d.deck_hardest||[]).length)mk(['卡组','胜率','局数（衰减后）','使用量','现在的抽样概率'],d.deck_hardest.map(r=>[r.cards.split('|').join('、'),fmtPct(r.winrate),fmtNum(r.games,1),fmtPct(r.usage),fmtPct(r.prob)]),
    '最难对付的对手卡组（按卡组挑难的练）')}

async function loadRuns(){const rs=await (await fetch('/api/runs')).json();const s=$('#run');const cur=s.value;s.textContent='';
  for(const r of rs){const o=document.createElement('option');o.value=r;o.textContent=r;s.appendChild(o)}
  s.value=cur&&rs.includes(cur)?cur:(rs[0]||'');}
async function refresh(){const name=$('#run').value;if(!name)return;document.body.classList.add('stale');
  try{DATA=await (await fetch('/api/run?name='+encodeURIComponent(name))).json();render()}catch(e){}
  document.body.classList.remove('stale')}
$('#run').addEventListener('change',refresh);$('#smooth').addEventListener('input',render);$('#range').addEventListener('change',render);
$('#theme').addEventListener('click',()=>{const r=document.documentElement,dark=r.dataset.theme?r.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;
  r.dataset.theme=dark?'light':'dark';render()});
let rt;addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(render,150)});
(async()=>{await loadRuns();await refresh();setInterval(async()=>{await loadRuns();await refresh()},15000)})();
</script></body></html>'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--runs', default=None)
    a = ap.parse_args()
    global RUNS
    if a.runs:
        RUNS = a.runs if os.path.isabs(a.runs) else os.path.join(ROOT, a.runs)
    srv = ThreadingHTTPServer(('127.0.0.1', a.port), H)
    print('训练看板：http://127.0.0.1:%d' % a.port, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    sys.exit(main())
