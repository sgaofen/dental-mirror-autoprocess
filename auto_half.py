"""全自动（半边牙弓模型，每个半边 8 点：0中切 1侧切 2尖牙 3-4前磨牙 5-7磨牙）：
先粗找 -> 从高清原图裁出 -> 再细认 -> 磨牙点、尖牙点、舌腭侧 -> 交给 dental_process 翻转/旋转/裁剪/曝光。
用法: [CLS=上下颌分类器.pt] python auto_half.py 模型.pt 输出目录 图片...   """
import sys, os, cv2, numpy as np
from ultralytics import YOLO
from dental_process import load, process

def best(r, by_area=True):
    if len(r.boxes) == 0: return None, None
    xyxy = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
    sc = cf * ((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) if by_area else 1)
    return np.argsort(-sc), sc

def candidates(model, f):
    """第一遍粗找的候选区域（翻转后图像坐标）：模型在整张图上找到的牙弓"""
    H, W = f.shape[:2]; boxes = []
    r = model.predict(f, imgsz=1280, conf=0.05, verbose=False)[0]
    order, sc = best(r)
    if order is not None:
        xyxy = r.boxes.xyxy.cpu().numpy(); cls = r.boxes.cls.cpu().numpy(); i0 = order[0]
        g = cls // 2 if len(model.names) == 4 else cls
        keep = [i for i in order if g[i] == g[i0] and sc[i] > 0.3 * sc[i0]]   # 同一排牙的两个半边合起来
        boxes.append((*xyxy[keep, :2].min(0), *xyxy[keep, 2:].max(0)))
    # 试过把"最清晰区域"也当候选（镜子对焦处），在医生 4 张图上反而更差，已去掉
    out = []
    for x0, y0, x1, y1 in boxes:
        mx, my = (x1 - x0) * 0.3, (y1 - y0) * 0.3
        out.append((int(max(x0 - mx, 0)), int(max(y0 - my, 0)), int(min(x1 + mx, W)), int(min(y1 + my, H))))
    return out

def arch_crops(f, boxes, size=384):
    """上下颌分类器看的图：每个范围的框先按短边往外扩成正方形，从原图里多裁真实的周围（不补黑边、不切两头），
    扩到照片边缘没东西了才补边。推理和生成训练图都用这一个函数。"""
    H, W = f.shape[:2]; out = []
    for x0, y0, x1, y1 in boxes:
        cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) / 2
        X0, Y0, X1, Y1 = int(cx - s), int(cy - s), int(cx + s), int(cy + s)
        c = f[max(Y0, 0):min(Y1, H), max(X0, 0):min(X1, W)]
        if min(c.shape[:2]) < 16: continue
        c = cv2.copyMakeBorder(c, max(-Y0, 0), max(Y1 - H, 0), max(-X0, 0), max(X1 - W, 0), cv2.BORDER_REFLECT_101)
        out.append(cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA))
    return out

def arch_vote(cm, f, boxes):
    """专门的上下颌分类器：几个范围 x 4 个角度各判一次，取"是上颌"概率的平均。分类器不受左右镜像影响。"""
    ps = []; up = [k for k, v in cm.names.items() if v == 'upper'][0]
    for c in arch_crops(f, boxes):
        for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
            r = cm.predict(c if rot is None else cv2.rotate(c, rot), verbose=False)[0]
            ps.append(float(r.probs.data[up]))
    return float(np.mean(ps)) if ps else 0.5

def landmarks(model, img, kc=0.5, cm=None):
    H, W = img.shape[:2]; f = cv2.flip(img, 1)          # 反光板照片先翻回来（训练数据都是翻好的）
    # 每个候选区域从高清图里裁出来细认一遍，取模型把握最大的
    bestr = None
    for cx0, cy0, cx1, cy1 in candidates(model, f):
        r = model.predict(f[cy0:cy1, cx0:cx1], imgsz=640, conf=0.15, verbose=False)[0]
        if len(r.boxes) and (bestr is None or float(r.boxes.conf.max()) > float(bestr[0].boxes.conf.max())):
            bestr = (r, cx0, cy0)
    if bestr is None: return None
    r, cx0, cy0 = bestr
    order, _ = best(r, by_area=False)
    quad = len(model.names) == 4                       # 四类模型：0下颌0号侧 1下颌8号侧 2上颌0号侧 3上颌8号侧
    Kf = r.keypoints.data.cpu().numpy().copy(); Kf[..., 0] += cx0; Kf[..., 1] += cy0      # 翻转后图像坐标
    cls = r.boxes.cls.cpu().numpy().astype(int); conf = r.boxes.conf.cpu().numpy(); i = order[0]
    arch = cls // 2 if quad else cls
    S = Kf[i]
    mol = [j for j in (5, 6) if S[j, 2] > kc] or [j for j in (4, 5, 6, 7) if S[j, 2] > 0.25]
    ant = [j for j in (2,) if S[j, 2] > kc] or [j for j in (0, 1, 2, 3) if S[j, 2] > 0.25]
    if not mol or not ant: return None
    molar, anterior = S[mol, :2].mean(0), S[ant, :2].mean(0)
    # 同一排牙的两个半边都拿来定裁剪范围
    same = [j for j in order if arch[j] == arch[i] and conf[j] > 0.3 * conf[i]]
    good = np.vstack([Kf[j][Kf[j][:, 2] > kc][:, :2] for j in same])
    gi = S[S[:, 2] > kc][:, :2]
    step = np.median(np.linalg.norm(np.diff(gi, axis=0), axis=1)) if len(gi) > 1 else 50   # 相邻两颗牙的间距 ≈ 一颗牙宽
    v = anterior - molar; nrm = np.array([-v[1], v[0]]) / (np.linalg.norm(v) + 1e-6)     # 图像坐标(y朝下)里"磨牙->尖牙"的右手边
    if quad:
        # 训练数据里 0号侧的牙弓里面永远在右手边，8号侧在左手边：侧别直接给出舌/腭侧
        side = 1.0 if cls[i] % 2 == 0 else -1.0
    else:
        inc = [j for j in (0, 1) if S[j, 2] > 0.25 and np.linalg.norm(S[j, :2] - S[2, :2]) > 0.5 * step]
        mids = [j for j in (3, 4) if S[j, 2] > 0.25]
        if inc: side = np.sign(np.dot(S[inc, :2].mean(0) - anterior, nrm)) or 1.0
        elif mids: side = -np.sign(np.dot(S[mids, :2].mean(0) - (molar + anterior) / 2, nrm)) or 1.0
        else: side = 1.0
    lingual = (molar + anterior) / 2 + side * nrm * np.linalg.norm(v)
    # 裁剪范围：镜子里每颗认出的牙外扩一颗牙宽（圆形外扩，旋转后不会被撑大），其余切掉
    ring = 1.0 * step * np.array([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)])
    pts = (good[:, None, :] + ring[None]).reshape(-1, 2)
    # 上下颌分类器看的范围（翻转后坐标）：以镜子里认出的牙列为中心的正方形，边长 = 牙列长边的 1.3 倍和 1.6 倍。
    # 扫过 0.8~4 倍：太小只剩牙、看不到腭；太大会把镜子外面的牙（比如下颌照片里露出的上门牙）带进来，下颌被认成上颌
    cen, span = good.mean(0), np.ptp(good, 0).max()
    arch_boxes = [(cen[0] - s * span / 2, cen[1] - s * span / 2, cen[0] + s * span / 2, cen[1] + s * span / 2) for s in (1.3, 1.6)]
    p_up = None
    if cm is not None:
        p_up = arch_vote(cm, f, arch_boxes); arch = np.full_like(arch, int(p_up > 0.5))
    un = lambda p: np.array([W - p[..., 0], p[..., 1]]).T if p.ndim > 1 else np.array([W - p[0], p[1]])   # 翻回原图坐标
    K = Kf.copy(); K[..., 0] = W - K[..., 0]; P = un(pts)
    return dict(arch='lower' if arch[i] == 0 else 'upper', pw=W, molar=un(molar).tolist(), anterior=un(anterior).tolist(),
                lingual=un(lingual).tolist(), box=[*P.min(0), *P.max(0)], pts=P.tolist(), kp=K[i], conf=float(conf[i]),
                side=int(cls[i] % 2) if quad else -1, p_up=p_up, arch_boxes=arch_boxes)

if __name__ == '__main__':
    model = YOLO(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
    cm = YOLO(os.environ['CLS']) if os.environ.get('CLS') else None    # CLS=上下颌分类器.pt
    for p in sys.argv[3:]:
        img = load(p); name = os.path.splitext(os.path.basename(p))[0]; L = landmarks(model, img, cm=cm)
        if L is None: print(name, '没找到牙列'); continue
        res, g = process(img, L); cv2.imwrite(f'{out}/{name}_auto.jpg', res, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(name, L['arch'], round(L['conf'], 2))
