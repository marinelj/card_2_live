# Builds audio.wav: licensed Mixkit music + sound effects (see ASSETS.md), cued to the scene timings in scene.html.
import subprocess, numpy as np, wave, os
SR = 44100; DUR = 57.4; N = int(SR * DUR)
here = os.path.dirname(os.path.abspath(__file__))
def load(name, start=0.0, length=None):
    cmd = ['ffmpeg', '-v', 'error', '-ss', str(start)] + (['-t', str(length)] if length else []) + ['-i', os.path.join(here, 'audio', name), '-f', 'f32le', '-ac', '2', '-ar', str(SR), '-']
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32).reshape(-1, 2).copy()
def norm(x): return x / max(1e-6, np.abs(x).max())
def fades(x, fin=0.0, fout=0.0):
    n = len(x)
    if fin: k = min(n, int(fin * SR)); x[:k] *= np.linspace(0, 1, k)[:, None]
    if fout: k = min(n, int(fout * SR)); x[n - k:] *= np.linspace(1, 0, k)[:, None]
    return x
out = np.zeros((N, 2), np.float32)
def put(x, t, gain):
    i = int(t * SR); j = min(N, i + len(x))
    if j > i: out[i:j] += x[:j - i] * gain
# music: "Need for speed" from 9.35 s so the beat drops on STEP 01 (video 4.4 s)
music = fades(load('music_369_need-for-speed.mp3', 9.35, DUR), 0.8, 1.6)
duck = np.ones(N, np.float32)
for a, b in [(44.6, 45.8), (55.6, 57.0)]:   # dip under the landing on the card moment and the finale impact
    i, j = int(a * SR), int(b * SR); duck[i:j] = np.minimum(duck[i:j], 0.6)
out += music[:N] * 0.55 * duck[:len(music), None][:N]
S = lambda id, **k: norm(load(f'sfx_{id}.wav', **k))
for t in [4.4, 9.4, 15.4, 20.9, 25.9, 39.1, 49.6, 52.2]: put(S(1492), t - 0.35, 0.28)
put(S(2350), 0.25, 0.30)                                  # card swings in
put(S(1432), 7.45, 0.55)                                  # shutter
put(S(1107), 9.15, 0.22)                                  # card identified
put(S(166), 10.6, 0.25)                                   # scan sweep
for i in range(4): put(S(2568), 11.8 + i * .18, 0.18)     # field boxes
put(S(1107), 14.35, 0.25)                                 # text check passed
put(S(2573), 15.9, 0.18); put(S(166), 16.9, 0.22); put(S(1490), 19.0, 0.25)
put(fades(S(2537, length=2.7), 0.1, 0.3), 21.2, 0.20)     # typing the shot list
put(S(1107), 25.4, 0.22)                                  # rendered
# STEP 05, phase A: the whole licensed take plays and stops on the break point
put(S(3082), 26.3, 0.20)                                  # card appears
put(S(166), 26.6, 0.20)                                   # full clip opens
put(S(2568), 29.3, 0.22); put(S(1107), 29.35, 0.24)       # break point marked
put(S(2573), 31.7, 0.20)                                  # "never in this take"
# STEP 05, phase B: AI generates the bridge
put(S(2568), 33.2, 0.20); put(S(2568), 33.6, 0.16)        # mismatch chips
put(S(2633), 34.5, 0.30)                                  # AI generates the bridge
put(S(1490), 35.7, 0.22)                                  # bridge plays into the clip
put(S(1107), 37.7, 0.22)                                  # matched
# output video: licensed clip -> AI transition -> card moment
put(S(2573), 39.5, 0.16)                                  # clip starts in the phone
put(S(1490), 42.65, 0.26); put(S(166), 43.25, 0.18)       # AI transition: 3D camera move, whoosh on the turn
put(S(2350), 44.65, 0.34); put(S(2093), 44.75, 0.45); put(fades(S(462), 0.05, 1.2)[:int(1.6 * SR)], 44.8, 0.22)   # lands on the card moment
put(S(2633), 45.4, 0.30)                                  # pulled back into the card
for i in range(3): put(S(2568), 46.5 + i * .15, 0.16)    # stats
put(S(2573), 50.1, 0.18)
for i in range(3): put(S(2568), 50.3 + i * .25, 0.16)
put(S(2350), 52.45, 0.32)                                 # finale card swings in
put(S(1490), 53.55, 0.22)                                 # AI transition inside the card
put(S(2908), 55.63, 0.85); put(S(2093), 55.73, 0.55); put(fades(S(462), 0.05, 1.6), 55.78, 0.30)  # lands on the card moment
out *= np.minimum(1, np.maximum(0, (DUR - np.arange(N) / SR) / 0.5))[:, None]
pk = np.abs(out).max(); out = np.tanh(out / pk * 1.2) / np.tanh(1.2) * 0.89     # gentle soft-clip limiter
with wave.open(os.path.join(here, 'audio.wav'), 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((out * 32767).astype('<i2').tobytes())
print('ok, peak before limiter', round(float(pk), 2))
