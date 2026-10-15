"""只考模型：磨牙点、尖牙点离手标位置差几颗牙宽；牙弓里面朝哪边、上下颌认得对不对。
旋转裁剪是固定公式，不在这里打分。
用法: python eval_points.py 原图目录 模型1.pt [模型2.pt ...]   （环境变量 CLS=分类器.pt 时上下颌交给分类器）"""
import sys, glob, json, os, numpy as np
from ultralytics import YOLO
from dental_process import load
from auto_half import landmarks
LM = json.load(open('landmarks.json')); CM = YOLO(os.environ['CLS']) if os.environ.get('CLS') else None

def inside(L, f=1.0):
    m, a, l = (np.array(L[k], float) * f for k in ('molar', 'anterior', 'lingual'))
    v = a - m; return np.sign(v[0] * (l - m)[1] - v[1] * (l - m)[0])     # 牙弓里面在磨牙->尖牙连线的哪一侧

for mp in sys.argv[2:]:
    model = YOLO(mp); print(f'== {os.path.basename(mp)}')
    ok_all = 0
    for k, H in LM.items():
        img = load(glob.glob(f'{sys.argv[1]}/*_{k}_145.jpg')[0]); f = img.shape[1] / H['pw']
        P = landmarks(model, img, cm=CM)
        if P is None: print(f'  {k}: 没找到牙'); continue
        hm, ha = np.array(H['molar']) * f, np.array(H['anterior']) * f
        tooth = np.linalg.norm(ha - hm) / 4                                   # 尖牙到磨牙约 4 颗牙
        em = np.linalg.norm(np.array(P['molar']) - hm) / tooth; ea = np.linalg.norm(np.array(P['anterior']) - ha) / tooth
        arch_ok = P['arch'] == H['arch']; side_ok = inside(P) == inside(H, f)
        ok = arch_ok and side_ok and em < 1.5 and ea < 1.5; ok_all += ok
        print(f'  {k}: 磨牙差 {em:3.1f} 颗牙宽  尖牙差 {ea:3.1f} 颗牙宽  上下颌{"对" if arch_ok else "错"}  里面朝向{"对" if side_ok else "错"}  -> {"能摆对" if ok else "会摆错"}' + (f'  上颌概率 {P["p_up"]:.2f}' if P.get('p_up') is not None else ''))
    print(f'  能摆对 {ok_all}/4')
