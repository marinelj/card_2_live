# Build the v9 transition: camera arc (path9) + room + the player as two inflated shells that turn in the air.
import sys, json, numpy as np
sys.path.insert(0, '/tmp/claude-0/tools')
from path9 import camera, RB, RA, PB, PA, sstep
from camspec import spec
from scipy.spatial.transform import Rotation as Rot
import player9 as P9
cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
N, NS, SH = cfg.get('n', 60), cfg.get('subs', 6), cfg.get('shutter', .6)
F = 1290.0; sB, sA = 5.16 / F, 6.6 / F
M0 = np.stack([RB[0], RB[1], RB[2]], 1)           # local -> world at the start: the clip camera's axes
M1 = np.stack([-RA[0], RA[1], -RA[2]], 1)         # at the end: the card camera's axes, seen from behind
rv = Rot.from_matrix(M1 @ M0.T).as_rotvec(); th = np.linalg.norm(rv); ax = rv / th
def Mspin(a): return Rot.from_rotvec(ax * a).as_matrix() @ M0
half = Mspin(th / 2); facing_half = -half[:, 2]
TH = th if facing_half[0] < 0 else th - 2 * np.pi    # turn towards the rim (-X), the natural 180 dunk
TH = TH if not cfg.get('flip') else (th - 2 * np.pi if TH == th else th)
print('spin', round(np.degrees(TH), 1), 'deg about', ax.round(2))
ESW = cfg.get('esw', .3)
def spin_u(e): return sstep(np.clip((e - cfg.get('s0', .04)) / (cfg.get('s1', .8) - cfg.get('s0', .04)), 0, 1))
def model(e):
    R, C, intr, hip = camera(e, cfg); M = Mspin(TH * spin_u(e)); return R, C, intr, hip, M
def yaw(v): return np.arctan2(v[0], v[2])
def clampM(M, C, hip, back, lim):
    """Turn the shell about the vertical so it never shows more than `lim` degrees off its own source view."""
    v = C - hip; n = M[:, 2] if back else -M[:, 2]
    d = (yaw(n) - yaw(v) + np.pi) % (2 * np.pi) - np.pi; L = np.radians(lim)
    dc = np.clip(d, -L, L)
    return Rot.from_euler('y', dc - d).as_matrix() @ M
def bview(M, C, hip):
    v = (C - hip) / np.linalg.norm(C - hip); return np.degrees(np.arccos(np.clip(np.dot(-M[:, 2], v), -1, 1)))
def mat(M, hip, s): return list(np.concatenate([M[:, 0] * s, [0], M[:, 1] * s, [0], M[:, 2] * s, [0], hip, [1]]))
FB, FA = P9.load('nb'), P9.load('na')
for f in FB: pass
gA = P9.kit_stats(FA[-1]); gB = P9.kit_stats(FB[0])
fb0, fb1 = cfg.get('fb', [0, 2.2]); fa0 = cfg.get('fa0', 2.0)
out = []
for i in range(N):
    e = i / (N - 1)
    es = [np.clip(e + (j / max(NS - 1, 1) - .5) * SH / (N - 1), 0, 1) for j in range(NS)] if NS > 1 else [e]
    subs = [model(x) for x in es]
    rel = min(1, e / .12, (1 - e) / .12) * cfg.get('relief', 1.0)
    LIM, X0, X1 = cfg.get('lim', 48), cfg.get('x0', 72), cfg.get('x1', 108)
    wbs = [sstep((bview(M, C, hip) - X0) / (X1 - X0)) for R, C, intr, hip, M in subs]
    wb = float(np.mean(wbs)); layers = []
    if wb < .999:
        x = fb0 + (fb1 - fb0) * min(1, e / ESW)
        im, a, h = P9.sample(FB, x); g = sstep(min(1, e / ESW))
        im = P9.regrade_kit(im, h, gB, gA, g * cfg.get('grade', 1.0))
        geo = P9.shell(im, a, h, f'f{i:03d}', False, rel)
        layers.append(dict(geo=geo, op=1 - wb, mats=[mat(clampM(M, C, hip, False, LIM), hip, sB) for R, C, intr, hip, M in subs]))
    if wb > .001:
        x = fa0 + (len(FA) - 1 - fa0) * np.clip((e - ESW * .6) / (1 - ESW * .6), 0, 1)
        im, a, h = P9.sample(FA, x)
        geo = P9.shell(im, a, h, f'b{i:03d}', True, rel)
        layers.append(dict(geo=geo, op=wb, mats=[mat(clampM(M, C, hip, True, LIM), hip, sA) for R, C, intr, hip, M in subs]))
    vis_f, vis_b = wb < .999, wb > .001
    k = float(sstep(e))
    out.append(dict(k=k, cams=[spec(R, C, intr) for R, C, intr, hip, M in subs], layers=layers))
    R, C, intr, hip, M = subs[len(subs) // 2]; v = (C - hip) / np.linalg.norm(C - hip)
    print(i, 'bv %.0f' % np.degrees(np.arccos(np.clip(np.dot(-M[:, 2], v), -1, 1))), 'front' if vis_f else '', 'back' if vis_b else '', flush=True)
open(cfg.get('out', f'{P9.W9}/t9.jsonl'), 'w').write('\n'.join(json.dumps(x) for x in out))
