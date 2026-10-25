# Clean background plate: fill the player's hole in a still with other frames of the same clip,
# registered by a homography fitted on background SIFT matches (player masked out).
import cv2, numpy as np, sys
ref, others, out = sys.argv[1], sys.argv[2:-1], sys.argv[-1]
def load(p):
    im = cv2.imread(p); a = np.load(p.replace('.jpg', '_a.npy')).astype(np.float32)
    m = cv2.dilate((a > .3).astype(np.uint8), np.ones((45, 45), np.uint8)); return im, m
R, mR = load(ref); sift = cv2.SIFT_create(8000)
g = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
kR, dR = sift.detectAndCompute(g(R), 1 - mR)
stack, valid = [], []
for p in others:
    I, mI = load(p); kI, dI = sift.detectAndCompute(g(I), 1 - mI)
    mt = cv2.BFMatcher().knnMatch(dI, dR, k=2); good = [a for a, b in mt if a.distance < .7 * b.distance]
    src = np.float32([kI[a.queryIdx].pt for a in good]); dst = np.float32([kR[a.trainIdx].pt for a in good])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    W = cv2.warpPerspective(I, H, (1920, 1080), flags=cv2.INTER_CUBIC)
    v = cv2.warpPerspective(1 - mI, H, (1920, 1080), flags=cv2.INTER_NEAREST) * cv2.warpPerspective(np.ones_like(mI), H, (1920, 1080), flags=cv2.INTER_NEAREST)
    err = np.abs(W.astype(float) - R).mean(2)[(v > 0) & (mR == 0)].mean()
    print(p, len(good), int(inl.sum()), 'bg err', round(err, 2)); stack.append(W.astype(np.float32)); valid.append(v.astype(bool))
S = np.stack(stack); V = np.stack(valid)
out_im = R.astype(np.float32).copy(); hole = mR.astype(bool)
# per-pixel median of the frames that see the background there
cnt = V.sum(0)
med = np.zeros_like(out_im)
for c in range(3):
    x = np.where(V, S[..., c], np.nan); med[..., c] = np.nan_to_num(np.nanmedian(x, 0))
fill = hole & (cnt > 0)
# soft blend at the hole edge
w = cv2.GaussianBlur(fill.astype(np.float32), (31, 31), 0)[..., None] * fill[..., None]
w = np.maximum(w, cv2.erode(fill.astype(np.uint8), np.ones((15, 15), np.uint8))[..., None])
out_im = out_im * (1 - w) + med * w
left = hole & (cnt == 0)
print('unfilled px', int(left.sum()))
cv2.imwrite(out, out_im.clip(0, 255).astype(np.uint8)); np.save(out.replace('.jpg', '_hole.npy'), left)
