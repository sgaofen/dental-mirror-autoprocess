"""大一点的上下颌测试：Roboflow 验证集（训练没见过）各取 N 张，做成"拍得远、模糊背景、任意角度、镜像"的样子，
比较"检测模型自己判断"和"交给分类器"的上下颌准确率。
用法: python eval_arch_big.py 检测模型.pt [分类器1.pt ...]   环境变量 N（每类张数，默认 150）"""
import sys, os, glob, random, cv2, numpy as np
from ultralytics import YOLO
from auto_half import landmarks
N = int(os.environ.get('N', 150)); random.seed(0)

def clinic_like(img):
    h, w = img.shape[:2]; f = random.uniform(0.3, 1.0)
    t, b, l, r = [int(f * x * random.uniform(0.4, 1.6)) for x in (h, h, w, w)]
    big = cv2.GaussianBlur(cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_REFLECT_101), (0, 0), 15)
    big[t:t + h, l:l + w] = img; H, W = big.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), random.uniform(0, 360), 1.0)
    big = cv2.warpAffine(big, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)
    k = 3000 / max(H, W); big = cv2.resize(big, (int(W * k), int(H * k)))      # 放到相机原图那么大
    return cv2.flip(big, 1)                                                     # 变回反光板原始的镜像

tests = []
for arch in ('lower', 'upper'):
    fs = sorted(glob.glob(f'data/{arch}/valid/images/*.jpg')); random.shuffle(fs)
    for p in fs[:N]:
        im = cv2.imread(p); h = im.shape[0]; im = cv2.resize(im, (int(h * 4 / 3), h)); tests.append((arch, clinic_like(im)))
det = YOLO(sys.argv[1])
for cp in [None] + sys.argv[2:]:
    cm = YOLO(cp) if cp else None; ok = miss = 0
    for arch, im in tests:
        P = landmarks(det, im, cm=cm)
        if P is None: miss += 1; continue
        ok += P['arch'] == arch
    n = len(tests) - miss
    print(f'{os.path.basename(cp) if cp else "检测模型自己判断":<18} 上下颌对 {ok}/{n} = {ok / max(n, 1):.1%}   没找到牙 {miss}')
