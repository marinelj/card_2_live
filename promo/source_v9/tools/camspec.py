# Helpers to build camera specs for room.html (OpenCV convention, output 720x1280).
import sys, json, numpy as np
sys.path.insert(0, '/tmp/claude-0/tools'); from geo import CAM, F
def crop_intr(x0, y0, wsrc):
    s = 720 / wsrc; return dict(fx=F * s, fy=F * s, cx=(960 - x0) * s, cy=(540 - y0) * s, W=720, H=1280)
CROP = {'B': (632, 0, 608), 'A': (620, 40, 461)}
def spec(R, C, intr): return dict(R=list(np.asarray(R).reshape(-1)), C=list(np.asarray(C, float)), **intr)
