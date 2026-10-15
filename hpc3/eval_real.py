"""真实考题：整套流程跑一批外部照片，记下每张的结果，并存小图方便抽查（集群上一张卡）。
环境变量：ARM 组名  DATA 数据目录名（里面直接放 jpg）  DROOT 解压位置  OUTD 输出根目录
输出：OUTD/runs/ARM/results.csv、OUTD/runs/ARM/thumbs.tar（处理后小图 + 标点小图）"""
import os, sys, glob, csv, tarfile, cv2, numpy as np
ROOT, OUTD, ARM, DATA = os.environ['DROOT'], os.environ['OUTD'], os.environ['ARM'], os.environ['DATA']
sys.path.insert(0, OUTD)
from ultralytics import YOLO
from dental_process import load, process
from auto_half import landmarks
det, cm = YOLO(f'{OUTD}/C4.pt'), YOLO(f'{OUTD}/archsmall.pt')
run = f'{OUTD}/runs/{ARM}'; th = f'{ROOT}/thumbs_{ARM}'; os.makedirs(run, exist_ok=True); os.makedirs(th, exist_ok=True)
files = sorted(glob.glob(f'{ROOT}/{DATA}/**/*.jpg', recursive=True)); rows = []
print('共', len(files), '张', flush=True)

def small(x, L=480):
    k = L / max(x.shape[:2]); return cv2.resize(x, (max(int(x.shape[1] * k), 1), max(int(x.shape[0] * k), 1)), interpolation=cv2.INTER_AREA)

for i, p in enumerate(files):
    name = os.path.relpath(p, f'{ROOT}/{DATA}').replace('/', '__')[:-4]
    try:
        img = load(p); P = landmarks(det, img, cm=cm)
    except Exception as e:
        rows.append(dict(name=name, found=0, err=str(e)[:80])); continue
    if P is None:
        rows.append(dict(name=name, found=0)); continue
    res, g = process(img, P)
    cv2.imwrite(f'{th}/{name}_out.jpg', small(res), [cv2.IMWRITE_JPEG_QUALITY, 80])
    d = img.copy(); r = max(img.shape[:2]) // 80
    for k, c in (('molar', (0, 0, 255)), ('anterior', (0, 255, 0)), ('lingual', (255, 0, 0))):
        cv2.circle(d, tuple(int(v) for v in P[k]), r, c, -1)
    cv2.imwrite(f'{th}/{name}_pts.jpg', small(d), [cv2.IMWRITE_JPEG_QUALITY, 80])
    rows.append(dict(name=name, found=1, arch=P['arch'], p_up=round(P['p_up'], 4), conf=round(P['conf'], 3), side=P['side']))
    if (i + 1) % 50 == 0: print(f'已处理 {i + 1}/{len(files)}', flush=True)
keys = ['name', 'found', 'arch', 'p_up', 'conf', 'side', 'err']
with open(f'{run}/results.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, keys); w.writeheader(); [w.writerow(r) for r in rows]
with tarfile.open(f'{run}/thumbs.tar', 'w') as t: t.add(th, arcname=f'thumbs_{ARM}')
print('找到牙', sum(r['found'] for r in rows), '/', len(rows), flush=True)
print('DONE', flush=True)
