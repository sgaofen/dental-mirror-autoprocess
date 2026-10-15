"""上下颌分类器对照实验（集群上一组一张卡）。环境变量：ARM 组名 DATA 数据目录名 MODEL 初始权重 EPOCHS IMGSZ DROOT OUTD"""
import os, json
from ultralytics import YOLO
ARM, DATA, MODEL = os.environ['ARM'], os.environ['DATA'], os.environ['MODEL']
EPOCHS = int(os.environ.get('EPOCHS', 30)); IMGSZ = int(os.environ.get('IMGSZ', 320))
ROOT, OUTD = os.environ['DROOT'], os.environ['OUTD']; data = f'{ROOT}/{DATA}'
run = f'{OUTD}/runs/{ARM}'; last = f'{run}/weights/last.pt'
if os.path.exists(last):            # 被抢占后重投：检查点里的数据路径换成这次解压的位置，再接着练
    import torch; ck = torch.load(last, map_location='cpu', weights_only=False)
    ck['train_args']['data'] = data; torch.save(ck, last)
    print('★ RESUME from', last, flush=True); YOLO(last).train(resume=True)
else:
    YOLO(f'{OUTD}/{MODEL}').train(data=data, imgsz=IMGSZ, epochs=EPOCHS, batch=64, device=0,
        workers=max(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)) - 2, 2), fliplr=0.5, flipud=0.5, degrees=180,
        hsv_v=0.5, hsv_s=0.6, erasing=0.3, patience=10, project=f'{OUTD}/runs', name=ARM, exist_ok=True, plots=False, seed=0)
m = YOLO(f'{run}/weights/best.pt').val(data=data, split='val', imgsz=IMGSZ, batch=64, device=0, plots=False)
res = dict(arm=ARM, data=DATA, model=MODEL, top1=float(m.top1))
json.dump(res, open(f'{run}/commonval.json', 'w'), indent=1); print('COMMONVAL', json.dumps(res), flush=True)
print('DONE', flush=True)
