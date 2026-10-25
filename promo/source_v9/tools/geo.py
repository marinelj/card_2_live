# Shared camera geometry for the v9 room model (room frame: floor corner of end wall X=0 and far wall Z=0, Y up).
import numpy as np
D = np.load('/tmp/claude-0/w/cam/calib.npy', allow_pickle=True).item()
F = D['f']; CX, CY = 960.0, 540.0
CAM = {'A': (D['RA'], D['CA']), 'B': (D['RB'], D['CB'])}
def proj(P, cam):
    R, C = CAM[cam] if isinstance(cam, str) else cam
    P = np.asarray(P, float); x = (P - C) @ R.T
    z = x[..., 2]
    return np.stack([F * x[..., 0] / z + CX, F * x[..., 1] / z + CY], -1), z
def ray(u, v, cam):
    R, C = CAM[cam] if isinstance(cam, str) else cam
    d = np.stack([(u - CX) / F, (v - CY) / F, np.ones_like(u)], -1) @ R
    return C, d / np.linalg.norm(d, axis=-1, keepdims=True)
