# Rebuild the end wall texture from its one clean bay (next to the corner), repeated along the wall and up to the roof.
import cv2, numpy as np, shutil, os
T = '/tmp/claude-0/w/v9/tex'; n = 110; P = 3.79; Z0 = 0.45; RL, RP = 6.42, 0.855
for nm in 'AB':
    src = f'{T}/end_{nm}.orig.jpg'
    if not os.path.exists(src): shutil.copy(f'{T}/end_{nm}.jpg', src)
    t = cv2.imread(src); h, w = t.shape[:2]
    cx = lambda Z: (20 - Z) * n
    bay = t[:, int(cx(Z0 + P)):int(cx(Z0))].copy(); bw = bay.shape[1]
    plain = bay[:, 60:120].copy(); tmpl = np.concatenate([plain, bay[:, 60:]], 1)[:, -bw:].copy()
    r0, r1 = int((11.4 - 5.3) * n), int((11.4 - 4.3) * n); tmpl[r0:r1, 30:170] = tmpl[r0:r1, 200:340]   # an inpainting smudge
    Zs = 20 - (np.arange(w) + .5) / n; Ys = 11.4 - (np.arange(h) + .5) / n
    col = (bw - 1 - ((Zs - Z0) % P) * n).astype(np.float32)
    Ym = np.where(Ys < RL, Ys, RL - RP + ((Ys - RL) % RP))
    row = ((11.4 - Ym) * n).astype(np.float32)
    mx, my = np.meshgrid(col, row); out = cv2.remap(tmpl, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    keep = Zs < Z0 + .05; out[:, keep] = t[:, keep]
    cv2.imwrite(f'{T}/end_{nm}.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 94])
print('ok')
