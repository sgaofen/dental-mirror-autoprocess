"""口内反光板照片一键处理：翻转 -> 摆正 -> 裁剪 -> 调亮，并标出磨牙、尖牙。
用法:
  python run.py 输入(文件夹或图片...) -o 输出文件夹 [--debug]
  --debug  另存一张标了磨牙(红)、尖牙(绿)、牙弓里面方向(蓝)的原图，方便检查
每次运行都会在输出文件夹写 results.csv（每张一行，方便程序读取）；一张都没处理成功时退出码为 1。
"""
import argparse, csv, glob, os, sys, cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from ultralytics import YOLO
from dental_process import load, process
from auto_half import landmarks

ap = argparse.ArgumentParser(description='口内反光板照片一键处理')
ap.add_argument('inputs', nargs='+', help='图片文件，或装图片的文件夹')
ap.add_argument('-o', '--out', default='结果', help='输出文件夹（默认：结果）')
ap.add_argument('--det', default=f'{HERE}/models/C4.pt', help='找牙模型')
ap.add_argument('--cls', default=f'{HERE}/models/archsmall.pt', help='上下颌分类器')
ap.add_argument('--debug', action='store_true', help='另存标点图')
a = ap.parse_args()

EXT = ('.jpg', '.jpeg', '.png', '.heic', '.tif', '.tiff', '.bmp')
files = []
for p in a.inputs:
    files += sorted(f for f in glob.glob(os.path.join(p, '*')) if f.lower().endswith(EXT)) if os.path.isdir(p) else [p]
if not files: sys.exit('没找到图片')
os.makedirs(a.out, exist_ok=True)
det, cm = YOLO(a.det), YOLO(a.cls)
ARCH = {'lower': '下颌', 'upper': '上颌'}
rows = []; REVIEW = (0.2, 0.9)          # 上颌概率落在这个区间：建议人工复核（真实照片里的错误几乎都在这里）
for p in files:
    name = os.path.splitext(os.path.basename(p))[0]
    try:
        img = load(p)
    except Exception as e:
        print(f'{name}: 读不了这张图（{e}）'); rows.append(dict(file=p, status='unreadable')); continue
    L = landmarks(det, img, cm=cm)
    if L is None:
        print(f'{name}: 没找到镜子里的牙，跳过'); rows.append(dict(file=p, status='no_teeth')); continue
    res, g = process(img, L)
    outp = f'{a.out}/{name}_处理后.jpg'; cv2.imwrite(outp, res, [cv2.IMWRITE_JPEG_QUALITY, 92])
    review = REVIEW[0] < L['p_up'] < REVIEW[1]
    warn = '  ← 上下颌把握不大，建议人工看一眼' if review else ''
    rows.append(dict(file=p, status='ok', arch=L['arch'], p_up=round(L['p_up'], 4), det_conf=round(L['conf'], 3), gamma=round(g, 3),
                     needs_review=int(review), output=outp, width=res.shape[1], height=res.shape[0]))
    print(f'{name}: {ARCH[L["arch"]]}  找牙把握 {L["conf"]:.2f}  上颌概率 {L["p_up"]:.2f}{warn}')
    if a.debug:
        d = img.copy(); r = max(img.shape[:2]) // 120
        for k, c in (('molar', (0, 0, 255)), ('anterior', (0, 255, 0)), ('lingual', (255, 0, 0))):
            cv2.circle(d, tuple(int(v) for v in L[k]), r, c, -1)
        cv2.imwrite(f'{a.out}/{name}_标点.jpg', d, [cv2.IMWRITE_JPEG_QUALITY, 85])
with open(f'{a.out}/results.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, ['file', 'status', 'arch', 'p_up', 'det_conf', 'gamma', 'needs_review', 'output', 'width', 'height'])
    w.writeheader(); w.writerows(rows)
ok = sum(r['status'] == 'ok' for r in rows)
print(f'完成 {ok}/{len(rows)} 张，结果在 {os.path.abspath(a.out)}（汇总表 results.csv）')
sys.exit(0 if ok else 1)
