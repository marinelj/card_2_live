# Bullet-time bridge from the clip's break point (B, 44455) to the card moment (A, 44448).
# The player is cut out as a rigid layer; the gym without him is lifted to 3D with Depth Anything V2.
# A virtual camera orbits around the player in one arc while the jump plays in slow motion:
# the first half uses B's angle of the jump, the second half A's angle of the same jump.
import numpy as np, cv2, sys, os, json, glob
from scipy.spatial.transform import Rotation as Rot
cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
CAM, SEQ = cfg.get('cam', '/tmp/claude-0/w/cam'), cfg.get('seq', '/tmp/claude-0/w/cam3')
OW, OH = 720, 1280
f = 960 / np.tan(np.radians(65 / 2))
def load_seq(d):
    fr, al, last = [], [], None
    for p in sorted(glob.glob(f'{SEQ}/{d}/*.jpg')):
        im = cv2.imread(p)[:, :, ::-1].astype(np.float32)
        if last is not None and np.abs(im - last).mean() < .5: continue      # drop 25->30 fps duplicates
        a = np.load(p.replace('.jpg', '_a.npy')).astype(np.float32)
        a = np.clip((a - .35) / .3, 0, 1); a = cv2.erode(a, np.ones((3, 3), np.uint8)); a = cv2.GaussianBlur(a, (5, 5), 0)
        fr.append(im); al.append(a); last = im
    return fr, al
FA, AA = load_seq('sa'); FB, AB = load_seq('sb')
def cents(al):
    ys_, xs_ = np.mgrid[0:1080, 0:1920]
    c = np.array([[(xs_ * a).sum() / a.sum(), (ys_ * a).sum() / a.sum()] for a in al])
    return np.array([c[max(0, i - 1):i + 2].mean(0) for i in range(len(c))])
CNA, CNB = cents(AA), cents(AB)
CNA[-1] = (CNA[-1] * 0 + np.array([(np.mgrid[0:1080, 0:1920][1] * AA[-1]).sum() / AA[-1].sum(), (np.mgrid[0:1080, 0:1920][0] * AA[-1]).sum() / AA[-1].sum()]))
A, B = FA[-1], FB[0]; mA, mB = AA[-1], AB[0]
dA, dB = np.load(f'{CAM}/A.npy'), np.load(f'{CAM}/B.npy')
def samp(d, x, y): return float(np.median(d[y-6:y+6, x-6:x+6]))
def metric(d, p1, z1, p2, z2):
    d1, d2 = samp(d, *p1), samp(d, *p2); a = (1/z1 - 1/z2) / (d1 - d2); b = 1/z1 - a * d1
    return 1 / np.clip(a * d + b, 1/80, None)
zA = metric(dA, (850, 470), 9.0, (1300, 420), 14.0); zB = metric(dB, (1040, 820), 3.2, (1750, 450), 10.0)
def clean_bg(im, z, a):
    hole = cv2.dilate((a > .05).astype(np.uint8), np.ones((31, 31), np.uint8))
    sm = cv2.resize(im, (960, 540)).astype(np.uint8); hm = cv2.resize(hole, (960, 540), interpolation=cv2.INTER_NEAREST)
    fill = cv2.resize(cv2.inpaint(sm, hm, 12, cv2.INPAINT_TELEA), (1920, 1080)).astype(np.float32)
    lz = np.log(z).astype(np.float32); lzs = cv2.resize(lz, (960, 540))
    # depth behind the player: inpaint log depth from the surroundings (the wall/floor he stands in front of)
    zl = cv2.resize(cv2.inpaint(((lzs - lzs.min()) / (lzs.max() - lzs.min()) * 65535).astype(np.uint16).astype(np.float32).astype(np.uint8) if False else (np.clip((lzs - lzs.min()) / (lzs.max() - lzs.min()), 0, 1) * 255).astype(np.uint8), hm, 12, cv2.INPAINT_TELEA), (1920, 1080)).astype(np.float32) / 255 * (lzs.max() - lzs.min()) + lzs.min()
    h = hole.astype(bool)
    im2 = im.copy(); im2[h] = fill[h]; lz2 = lz.copy(); lz2[h] = zl[h]
    return im2, np.exp(lz2)
bgA, zbA = clean_bg(A, zA, mA); bgB, zbB = clean_bg(B, zB, mB)
ys, xs = np.mgrid[0:1080, 0:1920].astype(np.float32)
def keep(z):
    lz = np.log(z); g = np.hypot(cv2.Sobel(lz, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(lz, cv2.CV_32F, 0, 1, ksize=3)) / 8
    return cv2.dilate((g > .03).astype(np.uint8), np.ones((5, 5), np.uint8)).reshape(-1) == 0
def cloud(im, z):
    k = keep(z); P = np.stack([(xs - 960) * z / f, (ys - 540) * z / f, z], -1).reshape(-1, 3)
    return P[k], im.reshape(-1, 3)[k], z.reshape(-1)[k]
CLA, CLB = cloud(bgA, zbA), cloud(bgB, zbB)
def centroid(a):
    w = a / a.sum(); return np.array([(xs * w).sum(), (ys * w).sum()])
qA, qB = centroid(mA), centroid(mB)
def height(a): r = np.where(a.max(1) > .5)[0]; return r.max() - r.min()
zpA = float(np.median(zA[mA > .5])); zpB = float(np.median(zB[mB > .5]))
pivA = np.array([(qA[0] - 960) * zpA / f, (qA[1] - 540) * zpA / f, zpA]); pivB = np.array([(qB[0] - 960) * zpB / f, (qB[1] - 540) * zpB / f, zpB])
s0, s1 = OW / 461, OW / 608                         # output px per source px: card crop of A, clip crop of B
pA1 = (qA - [620, 40]) * s0; pB0 = (qB - [632, 0]) * s1
TH, PH, SG = cfg.get('theta', 60), cfg.get('phi', 12), cfg.get('sign', 1)
def ease(u): return u * u * u * (u * (u * 6 - 15) + 10)
def scene_cam(piv, yaw, pitch):
    M = Rot.from_euler('YX', [np.radians(yaw), np.radians(pitch)]).as_matrix()
    return M, piv + M @ (np.zeros(3) - piv)
def render(cl, R, C, F, cx, cy):
    P, col, zs = cl
    X = (P - C) @ R; z = X[:, 2]; ok = z > .3
    zz = z[ok]; uf = F * X[ok, 0] / zz + cx; vf = F * X[ok, 1] / zz + cy
    sz = np.clip(np.ceil(F * zs[ok] / (f * zz) * cfg.get('splat', 2.0)), 1, 7).astype(np.int32); cc = col[ok]
    ins = (uf > -8) & (uf < OW + 8) & (vf > -8) & (vf < OH + 8)
    uf, vf, zz, cc, sz = uf[ins], vf[ins], zz[ins], cc[ins], sz[ins]
    us, vs, zl, cl_ = [], [], [], []
    for k in range(1, 8):
        g = sz == k
        if not g.any(): continue
        u0 = np.round(uf[g] - (k - 1) / 2).astype(np.int32); v0 = np.round(vf[g] - (k - 1) / 2).astype(np.int32)
        for dx in range(k):
            for dy in range(k): us.append(u0 + dx); vs.append(v0 + dy); zl.append(zz[g]); cl_.append(cc[g])
    u = np.concatenate(us); v = np.concatenate(vs); zz = np.concatenate(zl); cc = np.concatenate(cl_)
    m = (u >= 0) & (u < OW) & (v >= 0) & (v < OH); u, v, zz, cc = u[m], v[m], zz[m], cc[m]
    o = np.argsort(-zz, kind='stable'); idx = (v * OW + u)[o]
    img = np.zeros((OH * OW, 3), np.float32); hit = np.zeros(OH * OW, bool); img[idx] = cc[o]; hit[idx] = True
    img = img.reshape(OH, OW, 3); hit = cv2.morphologyEx(hit.reshape(OH, OW).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    return pushpull(img, hit), hit
def pushpull(img, mask):
    lv = [(img * mask[..., None], mask.astype(np.float32))]
    while min(lv[-1][1].shape) > 4: c, w = lv[-1]; lv.append((cv2.pyrDown(c), cv2.pyrDown(w)))
    c, w = lv[-1]; out = c / np.maximum(w, 1e-4)[..., None]
    for c, w in reversed(lv[:-1]):
        up = cv2.resize(out, (c.shape[1], c.shape[0])); out = np.where(w[..., None] > .2, c / np.maximum(w, 1e-4)[..., None], up)
    return np.where(mask[..., None], img, out)
HA, HB = height(mA), height(mB)
hA1, hB0 = HA * s0, HB * s1
def view(e):
    p = pB0 + (pA1 - pB0) * e                        # player's screen position follows one smooth path
    h = hB0 ** (1 - e) * hA1 ** e                     # and his size, so both scenes agree at the hand-over
    mB_, mA_ = h / HB, h / HA
    return p, mB_, mA_
def bg_frame(t):
    e = ease(t); p, mBm, mAm = view(e)
    RB, CB = scene_cam(pivB, SG * TH * e, -PH * e)
    RA, CA = scene_cam(pivA, SG * TH * (e - 1), PH * (1 - e))
    cB = p - mBm * (qB - [960, 540]); cA = p - mAm * (qA - [960, 540])
    ib, hb = render(CLB, RB, CB, f * mBm, *cB); ia, ha = render(CLA, RA, CA, f * mAm, *cA)
    w = np.clip((t - cfg.get('x0', .4)) / (cfg.get('x1', .6) - cfg.get('x0', .4)), 0, 1); w = w * w * (3 - 2 * w)
    wa = w * (ha + .1); wb = (1 - w) * (hb + .1)
    return (ia * wa[..., None] + ib * wb[..., None]) / (wa + wb)[..., None]
def sprite(frames, alphas, k, q, pos, mag, cn=None):
    i0 = int(np.floor(k)); i1 = min(i0 + 1, len(frames) - 1); fr = k - i0
    if cn is not None: q = cn[i0] * (1 - fr) + cn[i1] * fr      # pin the player's own centre to the camera path
    im = frames[i0] * (1 - fr) + frames[i1] * fr; al = alphas[i0] * (1 - fr) + alphas[i1] * fr
    M = np.array([[mag, 0, pos[0] - mag * q[0]], [0, mag, pos[1] - mag * q[1]]], np.float32)
    return cv2.warpAffine(im, M, (OW, OH), flags=cv2.INTER_LINEAR), cv2.warpAffine(al, M, (OW, OH), flags=cv2.INTER_LINEAR)
NB, NA = len(FB), len(FA)
PB_SPAN, A_FROM = cfg.get('bspan', 7), cfg.get('afrom', None)
def frame(t, sub=1, shutter=.02):
    bg = sum(bg_frame(min(1, max(0, t + (k - (sub - 1) / 2) * shutter))) for k in range(sub)) / sub
    e = ease(t); p, mBm, mAm = view(e)
    ph = t                                             # action phase: linear slow motion through the jump
    kB = ph * PB_SPAN; kA = (NA - 1) - (1 - ph) * PB_SPAN
    sb, ab = sprite(FB, AB, min(kB, NB - 1), qB, p, mBm, CNB); sa, aa = sprite(FA, AA, max(kA, 0), qA, p, mAm, CNA)
    w = np.clip((t - cfg.get('s0', .42)) / (cfg.get('s1', .58) - cfg.get('s0', .42)), 0, 1); w = w * w * (3 - 2 * w)
    out = bg
    out = out * (1 - ab[..., None] * (1 - w)) + sb * (ab[..., None] * (1 - w))
    out = out * (1 - aa[..., None] * w) + sa * (aa[..., None] * w)
    return out
if __name__ == '__main__':
    n = cfg.get('n', 45); od = cfg.get('out', 'bt'); os.makedirs(od, exist_ok=True)
    ts = cfg.get('ts') or [i / (n - 1) for i in range(n)]
    print('unique frames A', NA, 'B', NB, 'heights', HA, HB, 'pivots', pivA.round(2), pivB.round(2))
    for i, t in enumerate(ts):
        o = frame(t, cfg.get('sub', 1), cfg.get('shutter', .02))
        cv2.imwrite(f'{od}/{i + 1:03d}.jpg', np.clip(o, 0, 255)[:, :, ::-1].astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
