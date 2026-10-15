"""上下颌分类数据：每张 Roboflow 图做成多种远近、任意角度的样本，标签只有 lower / upper。
模拟诊所照片：整排牙、紧裁、半边、半边缩小放进模糊大背景；全部随机旋转 0~360°、宽高比抖动。
左右镜像不影响上下颌，训练时可以开。
用法: python prepare_arch.py [small]   -> dataset_arch[_small]/{train,val}/{lower,upper}/*.jpg"""
import glob, os, cv2, numpy as np, random
import sys
SMALL = len(sys.argv) > 1 and sys.argv[1] == 'small'   # 另加"小范围"样本：只裁牙弓里面那一小块（舌头或腭）加旁边几颗牙
np.random.seed(0); random.seed(0); OUT = 'dataset_arch' + ('_small' if SMALL else ''); LONG = 384

def rotate(img):
    h, w = img.shape[:2]; a = random.uniform(0, 360)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0); c, s = abs(M[0, 0]), abs(M[0, 1])
    nW, nH = int(h * s + w * c), int(h * c + w * s); M[0, 2] += nW / 2 - w / 2; M[1, 2] += nH / 2 - h / 2
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)   # 镜像补边，避免拉伸条纹

def finish(img):
    r = random.uniform(0.8, 1.25); h, w = img.shape[:2]; img = cv2.resize(img, (max(int(w * r), 8), h))
    img = rotate(img); h, w = img.shape[:2]; k = LONG / max(h, w)
    return cv2.resize(img, (max(int(w * k), 8), max(int(h * k), 8)), interpolation=cv2.INTER_AREA)

def crop_around(img, pts, lo, hi):
    h, w = img.shape[:2]; x0, y0 = pts.min(0); x1, y1 = pts.max(0); bw, bh = x1 - x0, y1 - y0
    mx, my = bw * random.uniform(lo, hi), bh * random.uniform(lo, hi)
    return img[int(max(y0 - my, 0)):int(min(y1 + my, h)), int(max(x0 - mx, 0)):int(min(x1 + mx, w))]

def zoom_out(img):
    h, w = img.shape[:2]; f = random.uniform(0.4, 1.2)
    t, b, l, r = [int(f * x * random.uniform(0.4, 1.6)) for x in (h, h, w, w)]
    big = cv2.GaussianBlur(cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_REPLICATE), (0, 0), 25)
    big[t:t + h, l:l + w] = img; return big

n = 0
for arch in ('lower', 'upper'):
    for src, split in (('train', 'train'), ('valid', 'val')):
        os.makedirs(f'{OUT}/{split}/{arch}', exist_ok=True)
        for lf in sorted(glob.glob(f'data/{arch}/{src}/labels/*.txt')):
            img = cv2.imread(lf.replace('/labels/', '/images/').replace('.txt', '.jpg')); h = img.shape[0]
            w = int(round(h * 4 / 3)); img = cv2.resize(img, (w, h))
            a = np.array(open(lf).read().split()[5:], float).reshape(16, 3); a[:, 0] *= w; a[:, 1] *= h
            vis = a[a[:, 2] > 0][:, :2]
            if len(vis) < 4: continue
            outs = [img, crop_around(img, vis, 0.1, 0.6)]
            for s in (slice(0, 8), slice(8, 16)):
                p = a[s][a[s][:, 2] > 0][:, :2]
                if len(p) >= 3:
                    c = crop_around(img, p, 0.15, 0.6); outs += [c, zoom_out(c)]
            if SMALL:
                cen = vis.mean(0); span = np.ptp(vis, 0).max()     # 牙点的重心落在牙弓里面
                for _ in range(3):
                    s = span * random.uniform(0.25, 0.55); cx, cy = cen + np.random.uniform(-0.25, 0.25, 2) * span
                    o = img[int(max(cy - s, 0)):int(min(cy + s, h)), int(max(cx - s, 0)):int(min(cx + s, w))]
                    if min(o.shape[:2]) > 16: outs.append(o)
            base = os.path.basename(lf)[:-4]
            for i, o in enumerate(outs):
                if o.size == 0: continue
                cv2.imwrite(f'{OUT}/{split}/{arch}/{base}_{i}.jpg', finish(o), [cv2.IMWRITE_JPEG_QUALITY, 90]); n += 1
for s in ('train', 'val'):
    print(s, {a: len(os.listdir(f'{OUT}/{s}/{a}')) for a in ('lower', 'upper')})
