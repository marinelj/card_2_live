import sys, numpy as np, torch
from PIL import Image
from transformers import pipeline
pipe = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf', device='cpu')
for src, dst in zip(sys.argv[1::2], sys.argv[2::2]):
    im = Image.open(src).convert('RGB')
    out = pipe(im)
    d = out['predicted_depth'].squeeze().numpy()           # relative inverse depth
    d = np.array(Image.fromarray(d).resize(im.size, Image.BICUBIC))
    np.save(dst, d.astype(np.float32))
    v = (d - d.min()) / (d.max() - d.min()); Image.fromarray((v * 255).astype(np.uint8)).save(dst.replace('.npy', '_vis.png'))
    print(dst, d.min(), d.max())
