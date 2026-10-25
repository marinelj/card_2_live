# Player layers for the v9 transition: per output frame, a front shell (clip angle, 44455) and/or a back shell
# (card angle, 44448), each an inflated relief of the real cut-out, slow motion by optical-flow in-betweens.
import sys, os, json, glob, numpy as np, cv2
W9 = '/tmp/claude-0/w/v9'
INFL = 6.5
def load(d):
    out = []
    for f in sorted(glob.glob(f'{W9}/{d}/*.jpg')):
        im = cv2.imread(f); a = np.load(f.replace('.jpg', '_a.npy')).astype(np.float32); k = np.load(f.replace('.jpg', '_kp.npy'))
        a = np.clip((a - .3) / .4, 0, 1)
        hip = (k[11, :2] + k[12, :2]) / 2; out.append(dict(im=im, a=a, hip=hip, kp=k))
    if d == 'nb': out = refine_b(out)
    hs = np.array([o['hip'] for o in out])            # smooth the hip track (keypoints jitter)
    for i, o in enumerate(out): o['hip'] = hs[max(0, i - 1):i + 2].mean(0) if 0 < i < len(out) - 1 else hs[i]
    return out
def refine_b(out):
    """Clip-side frames: drop wall pixels that the person mask swallowed (e.g. between the raised arms),
    by differencing against the clean plate registered to each frame."""
    plate = cv2.imread(f'{W9}/plateB.jpg'); sift = cv2.SIFT_create(6000); g = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    kP, dP = sift.detectAndCompute(g(plate), None)
    for o in out:
        m = (cv2.dilate((o['a'] > .1).astype(np.uint8), np.ones((41, 41), np.uint8)) == 0).astype(np.uint8)
        kI, dI = sift.detectAndCompute(g(o['im']), m)
        mt = cv2.BFMatcher().knnMatch(dP, dI, k=2); good = [p for p, q in mt if p.distance < .7 * q.distance]
        H, _ = cv2.findHomography(np.float32([kP[p.queryIdx].pt for p in good]), np.float32([kI[p.trainIdx].pt for p in good]), cv2.RANSAC, 2.0)
        bg = cv2.warpPerspective(plate, H, (1920, 1080))
        d = cv2.GaussianBlur(np.abs(o['im'].astype(np.float32) - bg).max(2), (5, 5), 0)
        bgl = ((d < 22) & (o['a'] > .1)).astype(np.uint8)
        bgl = cv2.morphologyEx(bgl, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        sh_y = min(o['kp'][5, 1], o['kp'][6, 1])
        n, lab, st, cen = cv2.connectedComponentsWithStats(bgl)
        cut = np.zeros((1080, 1920), np.float32)
        for i in range(1, n):     # only big wall patches above the shoulders (the gap between the raised arms)
            if st[i, 4] > 1200 and cen[i][1] < sh_y: cut[lab == i] = 1
        cut = cv2.GaussianBlur(cv2.dilate(cut, np.ones((3, 3), np.uint8)), (5, 5), 0)
        o['a'] = o['a'] * (1 - cut)
    return out
def blue_w(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    return np.clip((hsv[..., 1] - 50) / 40, 0, 1) * np.clip(1 - np.abs(hsv[..., 0] - 118) / 22, 0, 1)
def kit_stats(f):
    lab = cv2.cvtColor(f['im'], cv2.COLOR_BGR2LAB).astype(np.float32); w = (blue_w(f['im']) > .6) & (f['a'] > .7)
    ys = np.mgrid[0:1080, 0:1920][0]; top = w & (ys < f['hip'][1] - 15); bot = w & (ys > f['hip'][1] + 15)
    return [(lab[r].mean(0), lab[r].std(0) + 1e-3) for r in (top, bot)]
def regrade_kit(im, hip, sB, sA, g):
    """Shift the jersey and the shorts (blue pixels above / below the hip) from the clip's grade to the card's."""
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32); out = lab.copy()
    ys = np.mgrid[0:1080, 0:1920][0].astype(np.float32); wt = np.clip((hip[1] + 20 - ys) / 40, 0, 1)[..., None]
    tr = [((lab - b[0]) / b[1] * a[1] + a[0]) for b, a in zip(sB, sA)]
    t = tr[0] * wt + tr[1] * (1 - wt); w = (blue_w(im) * g)[..., None]
    out = lab * (1 - w) + t * w
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
def lab_stats(im, a):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3); w = a.reshape(-1) > .7
    return lab[w].mean(0), lab[w].std(0)
def regrade(im, src, dst, g):      # Reinhard transfer in Lab, amount g
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    t = (lab - src[0]) / src[1] * dst[1] + dst[0]
    return cv2.cvtColor(np.clip(lab * (1 - g) + t * g, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
_raft = None; _flows = {}
def raft(i0, i1):
    """RAFT flow from i0 to i1 (BGR uint8, same size, multiple of 8)."""
    global _raft
    if _raft is None: torch.set_num_threads(4); _raft = raft_large(weights=Raft_Large_Weights.DEFAULT).eval()
    tt = lambda im: torch.from_numpy(im[:, :, ::-1].copy()).permute(2, 0, 1).float()[None] / 127.5 - 1
    with torch.no_grad(): f = _raft(tt(i0), tt(i1), num_flow_updates=16)[-1]
    return f[0].permute(1, 2, 0).numpy()
def pair_flow(f0, f1):
    key = (id(f0), id(f1))
    if key in _flows: return _flows[key]
    sh = f1['hip'] - f0['hip']; M = np.float32([[1, 0, -sh[0]], [0, 1, -sh[1]]])
    i1 = cv2.warpAffine(f1['im'], M, (1920, 1080), borderMode=cv2.BORDER_REPLICATE); a1 = cv2.warpAffine(f1['a'], M, (1920, 1080))
    m = (f0['a'] > .05) | (a1 > .05); ys, xs = np.nonzero(m)
    x0, x1 = max(0, xs.min() - 64), min(1920, xs.max() + 64); y0, y1 = max(0, ys.min() - 64), min(1080, ys.max() + 64)
    x1 = x0 + (x1 - x0) // 8 * 8; y1 = y0 + (y1 - y0) // 8 * 8
    fw = np.zeros((1080, 1920, 2), np.float32); bw = np.zeros_like(fw)
    fw[y0:y1, x0:x1] = raft(f0['im'][y0:y1, x0:x1], i1[y0:y1, x0:x1]); bw[y0:y1, x0:x1] = raft(i1[y0:y1, x0:x1], f0['im'][y0:y1, x0:x1])
    _flows[key] = (i1, a1, fw, bw); return _flows[key]
def tween(f0, f1, t):
    """In-between of two frames at time t, both re-anchored at their hip; returns image, alpha, hip."""
    if t <= 1e-3: return f0['im'], f0['a'], f0['hip']
    if t >= 1 - 1e-3: return f1['im'], f1['a'], f1['hip']
    i1, a1, fw, bw = pair_flow(f0, f1)
    ys, xs = np.mgrid[0:1080, 0:1920].astype(np.float32)
    m0x, m0y = xs - t * fw[..., 0], ys - t * fw[..., 1]; m1x, m1y = xs - (1 - t) * bw[..., 0], ys - (1 - t) * bw[..., 1]
    w0 = cv2.remap(f0['im'], m0x, m0y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    w1 = cv2.remap(i1, m1x, m1y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    wa0 = cv2.remap(f0['a'], m0x, m0y, cv2.INTER_LINEAR) * (1 - t); wa1 = cv2.remap(a1, m1x, m1y, cv2.INTER_LINEAR) * t
    s_ = wa0 + wa1 + 1e-4
    im = ((w0 * wa0[..., None] + w1 * wa1[..., None]) / s_[..., None]).clip(0, 255).astype(np.uint8)
    a = np.clip((s_ - .25) / .5, 0, 1)
    return im, a, f0['hip']
def sample(seq, x):      # x in source-frame units (float)
    i = int(np.clip(np.floor(x), 0, len(seq) - 1)); j = min(i + 1, len(seq) - 1)
    return tween(seq[i], seq[j], float(np.clip(x - i, 0, 1)) if j > i else 0.)
def shell(im, a, hip, name, back, relief, step=4):
    """Crop, inflate and mesh one cut-out. Local coords in source px relative to the hip; z toward the viewer side."""
    ys, xs = np.nonzero(a > .02); pad = 12
    x0, x1 = max(0, xs.min() - pad), min(1920, xs.max() + pad); y0, y1 = max(0, ys.min() - pad), min(1080, ys.max() + pad)
    x0 -= (x0 % step); y0 -= (y0 % step)
    crop = im[y0:y1, x0:x1]; ca = a[y0:y1, x0:x1]
    rgba = np.dstack([crop, (ca * 255).astype(np.uint8)]); cv2.imwrite(f'{W9}/pl/{name}.png', rgba)
    hm = (ca > .5).astype(np.uint8); d = cv2.distanceTransform(hm, cv2.DIST_L2, 5)
    h = cv2.GaussianBlur(np.sqrt(d) * INFL, (0, 0), 3) * relief
    gw, gh = (x1 - x0) // step + 1, (y1 - y0) // step + 1
    gx, gy = np.meshgrid(np.arange(gw) * step, np.arange(gh) * step)
    gx = np.minimum(gx, x1 - x0 - 1); gy = np.minimum(gy, y1 - y0 - 1)
    hz = h[gy, gx]; lx = gx + x0 - hip[0]; ly = gy + y0 - hip[1]
    if back: lx = -lx
    z = hz if back else -hz
    verts = np.stack([lx, ly, z], -1).reshape(-1)
    uvs = np.stack([gx / (x1 - x0), 1 - gy / (y1 - y0)], -1).reshape(-1)
    am = cv2.dilate((ca > .02).astype(np.uint8), np.ones((step * 2 + 1, step * 2 + 1), np.uint8))[gy, gx]
    idx = []
    for r in range(gh - 1):
        for c in range(gw - 1):
            if am[r, c] or am[r + 1, c + 1] or am[r, c + 1] or am[r + 1, c]:
                p00 = r * gw + c; p10 = p00 + 1; p01 = p00 + gw; p11 = p01 + 1
                idx += [p00, p11, p10, p00, p01, p11]
    json.dump(dict(img=f'pl/{name}.png', verts=np.round(verts, 2).tolist(), uvs=np.round(uvs, 4).tolist(), idx=[int(v) for v in idx]), open(f'{W9}/pl/{name}.json', 'w'))
    return f'pl/{name}.json'
if __name__ == '__main__':
    os.makedirs(f'{W9}/pl', exist_ok=True)
    FB, FA = load('nb'), load('na')
    print('front frames', len(FB), 'back frames', len(FA))
