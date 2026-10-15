"""一组对照的训练 + 在公共验证集（dataset_C/val）上重新打分。环境变量: ARM DATA MODEL EPOCHS"""
import os, re, json, shutil
from ultralytics import YOLO
ARM, DATA, MODEL = os.environ['ARM'], os.environ['DATA'], os.environ['MODEL']
EPOCHS = int(os.environ.get('EPOCHS', 40)); ROOT = os.environ['DROOT']; OUTD = os.environ['OUTD']
COMMON = os.environ.get('COMMON', 'dataset_C')   # 公共验证集（四类模型用 dataset_C4）
for d in {DATA, COMMON}:      # data.yaml 里的 path 要写成解压后的绝对路径，相对路径会按当前目录解析
    y = f'{ROOT}/{d}/data.yaml'; t = re.sub(r'(?m)^path: .*$', f'path: {ROOT}/{d}', open(y).read()); open(y, 'w').write(t)
run = f'{OUTD}/runs/{ARM}'; last = f'{run}/weights/last.pt'
if os.path.exists(last):            # 被抢占后重投：接着上次的检查点继续
    # 检查点里记的数据路径是上个作业的临时目录（已被清掉），换成这次解压的位置
    import torch; ck = torch.load(last, map_location='cpu', weights_only=False)
    ck['train_args']['data'] = f'{ROOT}/{DATA}/data.yaml'; torch.save(ck, last)
    print('★ RESUME from', last, flush=True); YOLO(last).train(resume=True)
else:
    YOLO(f'{OUTD}/{MODEL}').train(data=f'{ROOT}/{DATA}/data.yaml', imgsz=640, epochs=EPOCHS, batch=32, device=0,
        workers=max(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)) - 2, 2),
        degrees=180, fliplr=float(os.environ.get('FLIPLR', 0.5)), flipud=0.0, scale=0.5, translate=0.15, mosaic=1.0, close_mosaic=5,
        hsv_v=0.5, hsv_s=0.6, patience=15, project=f'{OUTD}/runs', name=ARM, exist_ok=True, plots=False, seed=0)
best = f'{run}/weights/best.pt'
m = YOLO(best).val(data=f'{ROOT}/{COMMON}/data.yaml', split='val', imgsz=640, batch=32, device=0, plots=False,
                   project=f'{OUTD}/runs', name=f'{ARM}_commonval', exist_ok=True)
res = {'arm': ARM, 'data': DATA, 'model': MODEL,
       'box_map50': float(m.box.map50), 'pose_map50': float(m.pose.map50), 'pose_map50_95': float(m.pose.map)}
json.dump(res, open(f'{run}/commonval.json', 'w'), indent=1); print('COMMONVAL', json.dumps(res), flush=True)
print('DONE', flush=True)
