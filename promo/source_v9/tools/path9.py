# v9 camera path: from the clip's last camera (B) to the card camera (A) in one big 3D arc around the player.
import sys, json, numpy as np
sys.path.insert(0, '/tmp/claude-0/tools'); from geo import CAM, F
from camspec import crop_intr, CROP
RB, CB = CAM['B']; RA, CA = CAM['A']
def at(cam, u, v, z):
    R, C = CAM[cam]; return C + R.T @ (np.array([(u - 960) / F, (v - 540) / F, 1.]) * z)
RIM = np.array([3.59, 3.05, 8.62])
PB = at('B', 1034, 929, 5.16); PA = at('A', 861, 514, 6.6)          # player's hip at the break point / card moment
def cyl(C, P): d = C - P; return np.hypot(d[0], d[2]), np.degrees(np.arctan2(d[0], d[2])), C[1]
dB, azB, hB = cyl(CB, PB); dA, azA, hA = cyl(CA, PA)
def ang(R):      # forward yaw/pitch and roll of an OpenCV rotation
    f = R[2]; yaw = np.degrees(np.arctan2(f[0], f[2])); pitch = np.degrees(np.arcsin(f[1]))
    x = R[0]; xr = np.array([-np.cos(np.radians(yaw)), 0, np.sin(np.radians(yaw))]); roll = np.degrees(np.arcsin(np.clip(np.dot(np.cross(xr, x), f), -1, 1)))
    return yaw, pitch, roll
def look(yaw, pitch, roll):
    y, p, r = np.radians([yaw, pitch, roll])
    f = np.array([np.sin(y) * np.cos(p), np.sin(p), np.cos(y) * np.cos(p)])
    x = np.array([-np.cos(y), 0, np.sin(y)]); d = np.cross(f, x)        # camera y (down) = f x right
    R = np.stack([x, d, f]); c, s = np.cos(r), np.sin(r)
    return np.stack([c * R[0] + s * R[1], -s * R[0] + c * R[1], R[2]])
def dirang(v): return np.degrees(np.arctan2(v[0], v[2])), np.degrees(np.arcsin(v[1] / np.linalg.norm(v)))
yB, pB, rB = ang(RB); yA, pA, rA = ang(RA)
oyB, opB = yB - dirang(PB - CB)[0], pB - dirang(PB - CB)[1]       # aim offsets from "look at the player"
oyA, opA = yA - dirang(PA - CA)[0], pA - dirang(PA - CA)[1]
iB, iA = crop_intr(*CROP['B']), crop_intr(*CROP['A'])
def sstep(x): x = np.clip(x, 0, 1); return x * x * x * (x * (x * 6 - 15) + 10)
def bump(e, c=.5, w=.5): return np.exp(-((e - c) / (w * .5)) ** 2) * (1 - np.exp(-(e / .08) ** 2)) * (1 - np.exp(-((1 - e) / .08) ** 2))
def camera(e, cfg):
    s = sstep(e)
    hip = PB + (PA - PB) * s; hip[1] = PB[1] + (PA[1] - PB[1]) * (1 - (1 - s) ** 2) + cfg.get('lift', .25) * np.sin(np.pi * s)
    az = azB + (azA - azB) * s + cfg.get('az', 45) * bump(e, cfg.get('azc', .45), cfg.get('azw', .9))
    d = np.exp(np.log(dB) + (np.log(dA) - np.log(dB)) * s) * (1 - cfg.get('dolly', .2) * bump(e, .5, .9))
    h = hB + (hA - hB) * s + cfg.get('crane', 1.2) * bump(e, .5, .9)
    C = hip + np.array([d * np.sin(np.radians(az)), 0, d * np.cos(np.radians(az))]); C[1] = h
    T = hip + cfg.get('aimrim', .35) * bump(e, .5, .9) * (RIM - hip)
    ly, lp = dirang(T - C)
    wb = 1 - sstep((e - cfg.get('o0', .22)) / cfg.get('o1', .4)); cen = 1 - cfg.get('center', .6) * bump(e, cfg.get('cc', .45), .7)
    oy = (oyB * wb + oyA * (1 - wb)) * cen; op = (opB * wb + opA * (1 - wb)) * cen
    R = look(ly + oy, lp + op, rB + (rA - rB) * s + cfg.get('roll', 4) * np.sin(np.pi * s) * bump(e, .5, 1.2))
    lf = np.log(iB['fx']) + (np.log(iA['fx']) - np.log(iB['fx'])) * s - cfg.get('wide', .18) * bump(e, .5, .9)
    intr = dict(fx=np.exp(lf), fy=np.exp(lf), cx=iB['cx'] + (iA['cx'] - iB['cx']) * s, cy=iB['cy'] + (iA['cy'] - iB['cy']) * s, W=720, H=1280)
    return R, C, intr, hip
if __name__ == '__main__':
    print('B: d %.2f az %.1f h %.2f | A: d %.2f az %.1f h %.2f' % (dB, azB, hB, dA, azA, hA))
    print('aim offsets B', round(oyB, 1), round(opB, 1), 'A', round(oyA, 1), round(opA, 1), 'roll', round(rB, 1), round(rA, 1))
    for c, R0, C0, i0 in (('B', RB, CB, iB), ('A', RA, CA, iA)):
        R, C, intr, _ = camera(0 if c == 'B' else 1, {}); print(c, 'pos err', np.abs(C - C0).max().round(4), 'rot err', np.abs(R - R0).max().round(4), 'fx', round(intr['fx'], 1), round(i0['fx'], 1))
