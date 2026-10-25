# Finish the v9 transition frames: dissolve from / into the real footage at the two ends, write the bridge set.
import sys, glob, os, cv2, numpy as np
src, out = sys.argv[1], sys.argv[2]; K = int(sys.argv[3]) if len(sys.argv) > 3 else 6
def sstep(x): x = np.clip(x, 0, 1); return x * x * x * (x * (x * 6 - 15) + 10)
B = cv2.resize(cv2.imread('/tmp/claude-0/w/cam/B.jpg')[:, 632:1240], (720, 1280), interpolation=cv2.INTER_AREA).astype(np.float32)
A = cv2.resize(cv2.imread('/tmp/claude-0/w/cam/A.jpg')[40:860, 620:1081], (720, 1280), interpolation=cv2.INTER_AREA).astype(np.float32)
fs = sorted(glob.glob(src + '/*.jpg')); N = len(fs); os.makedirs(out, exist_ok=True)
for i, f in enumerate(fs):
    im = cv2.imread(f).astype(np.float32)
    wb = 1 - sstep(i / K); wa = sstep((i - (N - 1 - K)) / K)
    im = im * (1 - wb) + B * wb; im = im * (1 - wa) + A * wa
    cv2.imwrite(f'{out}/{i + 1:03d}.jpg', im.clip(0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 94])
print(N, 'frames')
