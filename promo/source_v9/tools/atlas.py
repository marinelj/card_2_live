# Bake per-plane textures for the 3D gym from the two calibrated stills (A: card moment, B: break point).
# Each plane gets two textures: texA (A first, gaps from B colour-matched to A) and texB (the reverse).
import sys, json, numpy as np, cv2
sys.path.insert(0, '/tmp/claude-0/tools'); from geo import proj, F
W9 = '/tmp/claude-0/w/v9'; G = json.load(open(f'{W9}/room.json'))
IM = {'A': cv2.imread(f'{W9}/plate/A.jpg').astype(np.float32), 'B': cv2.imread(f'{W9}/plateB.jpg').astype(np.float32)}
def poly_mask(cam, pts, grow):
    uv, z = proj(np.array(pts), cam); m = np.zeros((1080, 1920), np.uint8)
    if (z > .1).all(): cv2.fillConvexPoly(m, cv2.convexHull(uv.astype(np.int32)), 1)
    return cv2.dilate(m, np.ones((grow, grow), np.uint8))
def occluders(cam):
    bx, zc = 3.22, 8.62; m = np.zeros((1080, 1920), np.uint8)
    m |= poly_mask(cam, [[bx, y, z] for y in (2.85, 4.05) for z in (zc - 1.0, zc + 1.0)] + [[bx - .25, y, z] for y in (2.85, 4.05) for z in (zc - 1.0, zc + 1.0)], 9)
    t = np.linspace(0, 2 * np.pi, 24)
    m |= poly_mask(cam, [[3.59 + .3 * np.cos(a), y, 8.62 + .3 * np.sin(a)] for a in t for y in (2.5, 3.12)] + [[bx, 3.0, 8.62]], 9)
    m |= poly_mask(cam, [[x, y, z] for x in (2.9, 3.2) for y in (3.9, 9) for z in (8.5, 8.75)], 9)     # support beam behind the board
    return m.astype(bool)
pa = np.load(f'{W9}/plate/A_a.npy').astype(np.float32)
BAD = {'A': occluders('A') | cv2.dilate((pa > .2).astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool), 'B': occluders('B')}
def rimmask(cam):
    t = np.linspace(0, 2 * np.pi, 24)
    return poly_mask(cam, [[3.59 + .27 * np.cos(a), y, 8.62 + .27 * np.sin(a)] for a in t for y in (2.55, 3.1)], 7).astype(bool)
# the board itself is hidden only by the player and by the rim and net in front of it; A sees it edge-on, so B leads
BAD_BOARD = {'A': (BAD['A'] & ~occluders('A')) | rimmask('A'), 'B': rimmask('B')}
PERIOD = {'far': 8.3, 'end': 3.68}
def bake(name, p):
    O, U, V = map(np.array, (p['O'], p['U'], p['V'])); n = p['ppm']; w, h = int(p['su'] * n), int(p['sv'] * n)
    i, j = np.meshgrid((np.arange(w) + .5) / n, (np.arange(h) + .5) / n)
    P = O + i[..., None] * U + j[..., None] * V
    out = {}
    for c in 'AB':
        uv, z = proj(P, c); u, v = uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32)
        ok = (z > .3) & (u > 1) & (u < 1918) & (v > 1) & (v < 1078)
        bad = BAD_BOARD[c] if name == 'board' else BAD[c]
        ui, vi = np.clip(u.astype(int), 0, 1919), np.clip(v.astype(int), 0, 1079)
        ok &= ~bad[vi, ui]
        if name == 'board': ok &= (j > .02) | True
        col = cv2.remap(IM[c], u, v, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        out[c] = (col, ok)
    res = {}
    for a, b in (('A', 'B'), ('B', 'A')):
        ca, oa = out[a]; cb, ob = out[b]; both = oa & ob
        cb2 = cb.copy()
        if both.sum() > 500:     # per-channel linear colour transfer b -> a, fitted where both see the plane
            for k in range(3):
                x, y = cb[..., k][both], ca[..., k][both]
                g = np.clip(np.std(y) / max(np.std(x), 1e-3), .6, 1.6); cb2[..., k] = (cb[..., k] - x.mean()) * g + y.mean()
        if name == 'board' and a == 'A': oa, ca = ob, cb2; cb2 = cb2      # A's view of the board is too oblique
        have = oa | ob
        # soften the seam between the two sources
        wa = cv2.GaussianBlur(oa.astype(np.float32), (0, 0), 6) * oa
        img = np.where(have[..., None], ca * wa[..., None] + np.where(ob[..., None], cb2, ca) * (1 - wa[..., None]), 0)
        if name in PERIOD:       # repeat the wall's bays into the parts neither camera saw
            sh = int(round(PERIOD[name] * n)); src = img.copy(); hv = have.copy()
            for k in [1, -1, 2, -2, 3, -3, 4, -4]:
                m2 = np.roll(hv, k * sh, 1); s2 = np.roll(src, k * sh, 1)
                if k > 0: m2[:, :k * sh] = False
                else: m2[:, k * sh:] = False
                put = ~have & m2; img[put] = s2[put]; have = have | put
        if name == 'board':      # the board is symmetric: fill what the rim and net hide from the mirrored half
            mi, mh = img[:, ::-1], have[:, ::-1]; put = ~have & mh; img[put] = mi[put]; have = have | put
        hole = (~have).astype(np.uint8)
        if hole.any():          # paint unseen areas from their surroundings (at low resolution, then upsample)
            s = 4; sm = cv2.resize(img, (w // s, h // s), interpolation=cv2.INTER_AREA).clip(0, 255).astype(np.uint8)
            hm = cv2.resize(hole, (w // s, h // s), interpolation=cv2.INTER_NEAREST)
            fill = cv2.resize(cv2.inpaint(sm, hm, 9, cv2.INPAINT_TELEA), (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            img = np.where(hole[..., None] > 0, fill, img)
        cv2.imwrite(f'{W9}/tex/{name}_{a}.jpg', img.clip(0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 94])
        np.save(f'{W9}/tex/{name}_cov.npy', np.stack([out['A'][1], out['B'][1]]))
    print(name, w, h, 'A', round(out['A'][1].mean(), 3), 'B', round(out['B'][1].mean(), 3))
for k, p in G['planes'].items(): bake(k, p)
