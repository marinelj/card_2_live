# COCO body keypoints (torchvision Keypoint R-CNN) for the largest person in each frame.
import sys, glob, numpy as np, torch, cv2
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
m = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).eval(); torch.set_num_threads(4)
for d in sys.argv[1:]:
    fs = sorted(glob.glob(d + '/*.jpg')) if not d.endswith('.jpg') else [d]
    for f in fs:
        im = cv2.imread(f)[:, :, ::-1].copy(); x = torch.from_numpy(im).permute(2, 0, 1).float() / 255
        with torch.no_grad(): o = m([x])[0]
        if not len(o['scores']): print('none', f); continue
        b = o['boxes']; i = int(((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) * (o['scores'] > .5)).argmax())
        k = o['keypoints'][i].numpy(); s = o['keypoints_scores'][i].numpy()
        np.save(f.replace('.jpg', '_kp.npy'), np.c_[k[:, :2], s]); print(f, 'ok')
