"""上下颌分类器：用"推理时真实裁出来的图"训练（集群上一组一张卡）。
1. Roboflow 每张图（整排 + 两个半边）做成"诊所照片"（缩小、模糊背景、任意角度、镜像）
2. 用找牙模型 C4 跑 auto_half.landmarks，按推理时完全一样的三种范围裁图（arch_crops，补黑边成正方形）
3. 另加一份范围随机放大缩小 ±25% 的，防止只认死一个尺寸
4. 生成完接着训练分类器（train_cls.py）；MIX=1 时再并上原来的小范围数据 dataset_arch_small
生成结果打包存到 OUTD，被抢卡重投时直接解压，不用重新生成。
环境变量：ARM DATA(=roboflow_raw) MODEL EPOCHS IMGSZ DROOT OUTD VARIANTS(每张图做几张诊所照片，默认 3：整排、半边0、半边1) MIX WAIT_TAR"""
import os, sys, glob, random, tarfile, cv2, numpy as np
ROOT, OUTD, ARM = os.environ['DROOT'], os.environ['OUTD'], os.environ['ARM']
sys.path.insert(0, OUTD)
NAME = 'dataset_arch_real'; GEN = f'{ROOT}/{NAME}'; TAR = f'{OUTD}/{NAME}.tar'; NV = int(os.environ.get('VARIANTS', 3))

def clinic_like(img):
    """缩小放进背景、任意角度、镜像。背景 = 原图放大后狠狠模糊（只剩肤色/粉色调），不会出现重复的嘴。"""
    h, w = img.shape[:2]; a = random.uniform(0, 360)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0); c, s = abs(M[0, 0]), abs(M[0, 1])
    rw, rh = int(h * s + w * c), int(h * c + w * s); M[0, 2] += rw / 2 - w / 2; M[1, 2] += rh / 2 - h / 2
    rot = cv2.warpAffine(img, M, (rw, rh)); mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (rw, rh))
    f = random.uniform(0.3, 1.0); t_, b_, l_, r_ = [int(f * x * random.uniform(0.4, 1.6)) for x in (rh, rh, rw, rw)]
    H, W = rh + t_ + b_, rw + l_ + r_
    bg = cv2.resize(cv2.GaussianBlur(cv2.resize(img, (64, 48), interpolation=cv2.INTER_AREA), (0, 0), 2), (W, H))   # 先缩小再模糊再放大，快
    roi = bg[t_:t_ + rh, l_:l_ + rw]; m = (mask > 127)[..., None]; bg[t_:t_ + rh, l_:l_ + rw] = np.where(m, rot, roi)
    k = 2000 / max(H, W); bg = cv2.resize(bg, (int(W * k), int(H * k)))
    return cv2.flip(bg, 1)                      # 变回反光板原始的镜像

def half_crop(img, kp, side):
    """按标注点裁半边牙弓（医生的镜子常常只照到半边），四周留一些余量"""
    p = kp[side * 8:(side + 1) * 8]; p = p[p[:, 2] > 0][:, :2]
    if len(p) < 4: return None
    h, w = img.shape[:2]; x0, y0 = p.min(0); x1, y1 = p.max(0); mx, my = (x1 - x0) * random.uniform(0.3, 0.8), (y1 - y0) * random.uniform(0.3, 0.8)
    c = img[int(max(y0 - my, 0)):int(min(y1 + my, h)), int(max(x0 - mx, 0)):int(min(x1 + mx, w))]
    return c if min(c.shape[:2]) > 32 else None

def jitter(b):
    x0, y0, x1, y1 = b; cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0; s = random.uniform(0.75, 1.25)
    cx += random.uniform(-0.1, 0.1) * w; cy += random.uniform(-0.1, 0.1) * h
    return (cx - w * s / 2, cy - h * s / 2, cx + w * s / 2, cy + h * s / 2)

if os.environ.get('WAIT_TAR') == '1':        # 另一组负责生成，这组等它生成好直接用，不重复生成
    import time
    for i in range(240):
        if os.path.exists(TAR): break
        print(f'等另一组生成数据… {i} 分钟', flush=True); time.sleep(60)
if os.path.exists(TAR):
    print('★ 已有生成好的数据，直接解压', flush=True); tarfile.open(TAR).extractall(ROOT)
else:
    from ultralytics import YOLO
    from auto_half import landmarks, arch_crops
    det = YOLO(f'{OUTD}/C4.pt'); random.seed(0); np.random.seed(0); n = miss = 0
    for arch in ('lower', 'upper'):
        for src, split in (('train', 'train'), ('valid', 'val')):
            os.makedirs(f'{GEN}/{split}/{arch}', exist_ok=True)
            for p in sorted(glob.glob(f'{ROOT}/{os.environ["DATA"]}/{arch}/{src}/images/*.jpg'))[:int(os.environ.get('LIMIT', 10**9))]:
                im = cv2.imread(p); h = im.shape[0]; w = int(h * 4 / 3); im = cv2.resize(im, (w, h)); base = os.path.basename(p)[:-4]
                kp = np.array(open(p.replace('/images/', '/labels/')[:-4] + '.txt').read().split()[5:], float).reshape(16, 3)
                kp[:, 0] *= w; kp[:, 1] *= h
                srcs = [im] + [half_crop(im, kp, sd) for sd in (0, 1)]          # 整排 + 两个半边
                for v in range(NV):
                    src_im = srcs[v % len(srcs)]
                    if src_im is None: continue
                    cl = clinic_like(src_im); P = landmarks(det, cl)
                    if P is None: miss += 1; continue
                    boxes = P['arch_boxes'] + [jitter(b) for b in P['arch_boxes']]     # 坐标是翻转后的图
                    for j, c in enumerate(arch_crops(cv2.flip(cl, 1), boxes)):
                        cv2.imwrite(f'{GEN}/{split}/{arch}/{base}_{v}_{j}.jpg', c, [cv2.IMWRITE_JPEG_QUALITY, 90]); n += 1
                cnt = globals().get('cnt', 0) + 1; globals()['cnt'] = cnt
                if cnt % 100 == 0: print(f'{arch} {split} 已处理原图 {cnt} 张  裁出 {n} 张  没找到牙 {miss}', flush=True)   # 定时打印，免得被当成卡死
    print('生成完', n, '张  没找到牙', miss, flush=True)
    part = f"{TAR}.{os.environ.get('SLURM_JOB_ID', 'x')}.part"   # 两组同时生成时各写各的临时文件
    with tarfile.open(part, 'w') as t: t.add(GEN, arcname=NAME)
    os.replace(part, TAR)

data = NAME
if os.environ.get('MIX') == '1':              # 并上原来的小范围数据（软链接，不复制）
    data = 'dataset_arch_mix'
    for s in ('train', 'val'):
        for a in ('lower', 'upper'):
            d = f'{ROOT}/{data}/{s}/{a}'; os.makedirs(d, exist_ok=True)
            for src in (NAME, 'dataset_arch_small'):
                for fp in glob.glob(f'{ROOT}/{src}/{s}/{a}/*.jpg'):
                    dst = f'{d}/{src[-5:]}_{os.path.basename(fp)}'
                    if not os.path.exists(dst): os.symlink(fp, dst)
for s in ('train', 'val'):
    print(s, {a: len(os.listdir(f'{ROOT}/{data}/{s}/{a}')) for a in ('lower', 'upper')}, flush=True)
os.environ['DATA'] = data
exec(open(f'{OUTD}/train_cls.py').read())
