"""口内反光板照片自动处理：镜像翻转 -> 按牙弓方向旋转 -> 裁剪 -> 曝光。
关键点（最后一颗磨牙、尖牙/前牙端、舌/腭侧一点、牙列框）来自 landmarks.json，
可由视觉大模型或训练好的检测模型自动给出。
用法: python3 dental_process.py 原图目录 landmarks.json 输出目录
"""
import cv2, numpy as np, sys, glob, json, os
from PIL import Image, ImageOps

# 固定规则（医生的大致习惯，取整，不按个别照片调）：
#   下颌门牙朝下、上颌门牙朝上，舌/腭侧朝牙弓里面；磨牙->尖牙连线偏离竖直约 40°；成品 4:3；亮度调到中等
TILT = {'lower': 40, 'upper': 40}
ASPECT = 4 / 3
PAD = 0.0                            # 留白已经在牙点外扩一颗牙宽里了
TARGET_L = 0.45

def load(p):
    return cv2.cvtColor(np.array(ImageOps.exif_transpose(Image.open(p)).convert('RGB')), cv2.COLOR_RGB2BGR)

def process(img, L):
    H, W = img.shape[:2]; f = W / L['pw']
    pts = {k: np.array(L[k], float) * f for k in ('molar', 'anterior', 'lingual')}
    x0, y0, x1, y1 = np.array(L['box'], float) * f
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    if 'pts' in L: corners = np.array(L['pts'], float) * f   # 模型给的是牙齿上的点：直接旋转这些点，不用先框成水平方框（斜着时会撑大）
    # 1. 反光板拍的是镜像：水平翻转
    img = cv2.flip(img, 1)
    fl = lambda p: np.array([W - p[0], p[1]])
    pts = {k: fl(v) for k, v in pts.items()}; corners = np.array([fl(c) for c in corners])
    # 2. 旋转：下颌牙弓呈 U（前牙朝下），上颌呈 ∩（前牙朝上），舌/腭侧朝牙弓内侧
    v = pts['anterior'] - pts['molar']; cur = np.degrees(np.arctan2(-v[1], v[0]))
    t = TILT[L['arch']]
    cands = [-90 + t, -90 - t] if L['arch'] == 'lower' else [90 - t, 90 + t]
    c0 = (W / 2, H / 2)
    for tgt in cands:
        M = cv2.getRotationMatrix2D(c0, tgt - cur, 1.0)
        R = lambda p: M @ np.array([p[0], p[1], 1.0])
        vx = R(pts['anterior'])[0] - R(pts['molar'])[0]
        mid = (R(pts['anterior']) + R(pts['molar'])) / 2
        if np.sign(R(pts['lingual'])[0] - mid[0]) == np.sign(vx): break
    cos, sin = abs(M[0, 0]), abs(M[0, 1]); nW, nH = int(H * sin + W * cos), int(H * cos + W * sin)
    M[0, 2] += nW / 2 - c0[0]; M[1, 2] += nH / 2 - c0[1]
    rot = cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    R = lambda p: M @ np.array([p[0], p[1], 1.0])
    allp = np.array([R(p) for p in list(corners) + [pts['molar'], pts['anterior']]])
    # 3. 裁剪：牙列外接框 + 留白，补成固定宽高比，牙列居中
    bx0, by0 = allp.min(0); bx1, by1 = allp.max(0)
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2; w, h = bx1 - bx0, by1 - by0
    w, h = w * (1 + 2 * PAD), h * (1 + 2 * PAD)
    if w / h < ASPECT: w = h * ASPECT
    else: h = w / ASPECT
    k = min(1.0, nW / w, nH / h); w, h = w * k, h * k   # 特写照片牙占满画面时，框会比画面还大，整体缩到放得下（否则居中会算反、裁成一条缝）
    cx = np.clip(cx, w / 2, nW - w / 2); cy = np.clip(cy, h / 2, nH - h / 2)
    crop = rot[int(cy - h / 2):int(cy + h / 2), int(cx - w / 2):int(cx + w / 2)]
    # 4. 曝光：gamma 拉到目标亮度，宁欠勿过（gamma 限制在 0.7~1.05）
    lum = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[..., 0].mean() / 255
    g = float(np.clip(np.log(TARGET_L) / np.log(max(lum, 1e-3)), 0.7, 1.05))
    lut = (np.linspace(0, 1, 256) ** g * 255).astype(np.uint8)
    return cv2.LUT(crop, lut), g

if __name__ == '__main__':
    src, lmf, out = sys.argv[1:4]; LM = json.load(open(lmf)); os.makedirs(out, exist_ok=True)
    for k, L in LM.items():
        p = glob.glob(f'{src}/*_{k}_145.jpg')[0]
        res, g = process(load(p), L)
        cv2.imwrite(f'{out}/{k}_auto.jpg', res, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(k, L['arch'], 'gamma', round(g, 2), 'size', res.shape[1], 'x', res.shape[0])
