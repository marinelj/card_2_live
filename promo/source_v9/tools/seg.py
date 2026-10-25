# Person masks with torchvision Mask R-CNN (COCO). Keeps the largest confident person per frame.
import sys, glob, numpy as np, torch, torchvision, cv2
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
w = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
m = maskrcnn_resnet50_fpn_v2(weights=w).eval(); torch.set_num_threads(4)
for d in sys.argv[1:]:
    for f in sorted(glob.glob(d + '/*.jpg')):
        im = cv2.imread(f)[:, :, ::-1].copy()
        x = torch.from_numpy(im).permute(2, 0, 1).float() / 255
        with torch.no_grad(): o = m([x])[0]
        keep = [i for i in range(len(o['labels'])) if o['labels'][i] == 1 and o['scores'][i] > .6]
        if not keep: print('no person', f); continue
        i = max(keep, key=lambda i: float(o['masks'][i, 0].sum()))
        a = o['masks'][i, 0].numpy()
        for j in range(len(o['labels'])):          # add the ball (COCO 'sports ball') when it touches the player
            if o['labels'][j] == 37 and o['scores'][j] > .4:
                b = o['masks'][j, 0].numpy()
                if (cv2.dilate((a > .5).astype(np.uint8), np.ones((41, 41), np.uint8)) * (b > .5)).any(): a = np.maximum(a, b)
        np.save(f.replace('.jpg', '_a.npy'), a.astype(np.float16)); print(f, round(float(o['scores'][i]), 3), int((a > .5).sum()))
