"""半边牙弓数据：每个半边 = 一个目标，8 个点（0中切 1侧切 2尖牙 3-4前磨牙 5-7磨牙），类别 0=下颌 1=上颌。
三组对照：
  A = 只用半边裁剪图
  B = A + 把半边图缩小放进模糊大背景（模拟带脸的大视野）
  C = B + 整排牙原图（左右两个半边各算一个目标）
加参数 quad：分成四类 0=下颌0号侧 1=下颌8号侧 2=上颌0号侧 3=上颌8号侧。
  数据集里两侧的标注顺序 100% 一致：0号侧的牙弓里面永远在"磨牙->尖牙"的右手边，8号侧在左手边，
  所以认出侧别就知道舌/腭侧在哪边。训练时必须关掉左右镜像增强（镜像会把侧别弄反）。
用法: python prepare_half.py A|B|C [quad]"""
import glob, os, sys, cv2, numpy as np, random
V = sys.argv[1]; Q = len(sys.argv) > 2 and sys.argv[2] == 'quad'
OUT = f'dataset_{V}' + ('4' if Q else ''); random.seed(0)
SIDES = [slice(0, 8), slice(8, 16)]

def jitter(img, ks):
    r = random.uniform(0.8, 1.25); h, w = img.shape[:2]
    img = cv2.resize(img, (max(int(w * r), 8), h))
    return img, [np.column_stack([k[:, 0] * r, k[:, 1], k[:, 2]]) for k in ks]

def zoom_out(img, ks):
    h, w = img.shape[:2]; f = random.uniform(0.4, 1.0)
    t, b, l, r = [int(f * x * random.uniform(0.4, 1.6)) for x in (h, h, w, w)]
    big = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_REPLICATE)
    big = cv2.GaussianBlur(big, (0, 0), 25); big[t:t + h, l:l + w] = img
    out = []
    for k in ks:
        k = k.copy(); m = k[:, 2] > 0; k[m, 0] += l; k[m, 1] += t; out.append(k)
    return big, out

def write(img, ks, cls, name, split):
    img, ks = jitter(img, ks); h, w = img.shape[:2]; rows = []
    for si, k in enumerate(ks):
        k = k.copy(); out = (k[:, 0] < 0) | (k[:, 1] < 0) | (k[:, 0] >= w) | (k[:, 1] >= h); k[out] = 0
        vis = k[:, 2] > 0
        if vis.sum() < 3: continue
        pad = 0.25 * max(np.ptp(k[vis, 0]), np.ptp(k[vis, 1]), 20)
        x0, x1 = max(k[vis, 0].min() - pad, 0), min(k[vis, 0].max() + pad, w)
        y0, y1 = max(k[vis, 1].min() - pad, 0), min(k[vis, 1].max() + pad, h)
        row = [str(cls * 2 + si if Q else cls)] + [f'{v:.6f}' for v in ((x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h)]
        for x, y, v in k: row += ['0', '0', '0'] if v == 0 else [f'{x / w:.6f}', f'{y / h:.6f}', str(int(v))]
        rows.append(' '.join(row))
    if not rows: return 0
    cv2.imwrite(f'{OUT}/{split}/images/{name}.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    open(f'{OUT}/{split}/labels/{name}.txt', 'w').write('\n'.join(rows) + '\n'); return 1

for s in ('train', 'val'):
    os.makedirs(f'{OUT}/{s}/images', exist_ok=True); os.makedirs(f'{OUT}/{s}/labels', exist_ok=True)
n = 0
for cls, arch in [(0, 'lower'), (1, 'upper')]:
    for src, split in [('train', 'train'), ('valid', 'val')]:
        for lf in sorted(glob.glob(f'data/{arch}/{src}/labels/*.txt')):
            img = cv2.imread(lf.replace('/labels/', '/images/').replace('.txt', '.jpg')); h = img.shape[0]
            w = int(round(h * 4 / 3)); img = cv2.resize(img, (w, h))          # 拉回 4:3
            a = np.array(open(lf).read().split()[5:], float).reshape(16, 3); a[:, 0] *= w; a[:, 1] *= h
            ks = [a[s].copy() for s in SIDES]; base = f'{arch}_{os.path.basename(lf)[:-4]}'
            if V == 'C': n += write(img, ks, cls, base, split)
            for si in range(2):
                main = ks[si]; pts = main[main[:, 2] == 2]
                if len(pts) < 4: continue
                bx0, by0 = pts[:, :2].min(0); bx1, by1 = pts[:, :2].max(0)
                mw, mh = (bx1 - bx0) * random.uniform(0.2, 0.6), (by1 - by0) * random.uniform(0.15, 0.4)
                cx0, cy0 = int(max(bx0 - mw, 0)), int(max(by0 - mh, 0)); cx1, cy1 = int(min(bx1 + mw, w)), int(min(by1 + mh, h))
                crop = img[cy0:cy1, cx0:cx1]
                cks = []
                for k in ks:
                    k = k.copy(); m = k[:, 2] > 0; k[m, 0] -= cx0; k[m, 1] -= cy0
                    out = (k[:, 0] < 0) | (k[:, 1] < 0) | (k[:, 0] >= cx1 - cx0) | (k[:, 1] >= cy1 - cy0)
                    k[out] = 0; cks.append(k)   # 裁剪框外的点先清零，否则缩小补边后会落进模糊背景变成假点
                n += write(crop, cks, cls, f'{base}_h{si}', split)
                if V in ('B', 'C'):
                    zi, zk = zoom_out(crop, cks); n += write(zi, zk, cls, f'{base}_h{si}z', split)
open(f'{OUT}/data.yaml', 'w').write(f"""path: {os.path.abspath(OUT)}
train: train/images
val: val/images
kpt_shape: [8, 3]
flip_idx: [0, 1, 2, 3, 4, 5, 6, 7]
names:
""" + ('  0: lower_side0\n  1: lower_side1\n  2: upper_side0\n  3: upper_side1\n' if Q else '  0: lower_teeth\n  1: upper_teeth\n'))
print(V, 'images', n, 'train', len(os.listdir(f'{OUT}/train/images')), 'val', len(os.listdir(f'{OUT}/val/images')))
