"""拼对比图：原图 / 医生手工 / 自动结果。用法: python make_sheet.py 原图目录 自动结果目录 输出.jpg"""
import cv2, numpy as np, glob, sys
from dental_process import load
rows = []
for a, b in [('21', '22'), ('23', '24'), ('25', '26'), ('27', '28')]:
    o = load(glob.glob(f'{sys.argv[1]}/*_{a}_145.jpg')[0]); d = load(glob.glob(f'{sys.argv[1]}/*_{b}_145.jpg')[0])
    m = cv2.imread(glob.glob(f'{sys.argv[2]}/*_{a}_145_auto.jpg')[0]); h = 420
    ims = [cv2.resize(x, (int(x.shape[1] * h / x.shape[0]), h), interpolation=cv2.INTER_AREA) for x in (o, d, m)]
    rows.append(np.hstack([np.pad(i, ((4, 4), (4, 4), (0, 0)), constant_values=255) for i in ims]))
wm = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 0), (0, wm - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
cv2.imwrite(sys.argv[3], np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
