"""抽查拼图：每格 = 标点原图 + 处理后小图，上面写 编号 / 模型判断 / 上颌概率 [/ 文件名推测]。
用法: python make_review.py results.csv 小图目录 输出前缀 [筛选: all|disagree|sample:N] [每张拼图格数=20]
OMNI 的文件名推测：末位 4=下颌 5=上颌（只是推测，有错）"""
import sys, csv, os, random, cv2, numpy as np
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r['found'] == '1']
th, pre = sys.argv[2], sys.argv[3]; mode = sys.argv[4] if len(sys.argv) > 4 else 'all'; per = int(sys.argv[5]) if len(sys.argv) > 5 else 20

def guess(n):
    return {'4': 'lower', '5': 'upper'}.get(n[-1]) if n[-1] in '45' and n[:-1].isdigit() else None

if mode == 'disagree': rows = [r for r in rows if guess(r['name']) and guess(r['name']) != r['arch']]
elif mode.startswith('sample:'):
    random.seed(0); k = int(mode.split(':')[1]); rows = random.sample(rows, min(k, len(rows)))
ZH = {'lower': '下', 'upper': '上'}
def tile(i, r):
    a = cv2.imread(f"{th}/{r['name']}_pts.jpg"); b = cv2.imread(f"{th}/{r['name']}_out.jpg")
    fit = lambda x: np.full((200, 200, 3), 128, np.uint8) if x is None or min(x.shape[:2]) < 2 else cv2.resize(x, (max(int(x.shape[1] * 200 / x.shape[0]), 1), 200))
    a, b = fit(a), fit(b)   # 处理结果偶尔几乎为空（裁剪出了问题），用灰块代替
    t = np.hstack([a, np.full((200, 4, 3), 255, np.uint8), b]); t = cv2.resize(t, (560, 200))
    txt = f"#{i} pred={r['arch'][:2]} p_up={float(r['p_up']):.2f}" + (f" fn={guess(r['name'])[:2]}" if guess(r['name']) else '')
    cv2.rectangle(t, (0, 0), (560, 26), (0, 0, 0), -1); cv2.putText(t, txt, (6, 19), 0, 0.6, (0, 255, 255), 2)
    return t
os.makedirs(os.path.dirname(pre) or '.', exist_ok=True)
with open(pre + '_index.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['idx', 'name', 'arch', 'p_up'])
    for i, r in enumerate(rows): w.writerow([i, r['name'], r['arch'], r['p_up']])
for s in range(0, len(rows), per):
    ts = [tile(i, rows[i]) for i in range(s, min(s + per, len(rows)))]
    while len(ts) % 4: ts.append(np.full((200, 560, 3), 255, np.uint8))
    cv2.imwrite(f'{pre}_{s // per:03d}.jpg', np.vstack([np.hstack(ts[j:j + 4]) for j in range(0, len(ts), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(len(rows), '张 ->', (len(rows) + per - 1) // per, '张拼图')
