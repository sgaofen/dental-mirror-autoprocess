"""外部数据缩图：按 EXIF 方向转正，长边缩到 2000 像素，存成 jpg（流程里本来也会缩小，不影响结果）。
用法: python -I shrink_images.py 输入目录 输出目录 [文件名列表.txt]   （列表每行一个相对路径，只处理列出的）"""
import sys, os, glob
from PIL import Image, ImageOps
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
if len(sys.argv) > 3:
    files = [os.path.join(src, l.strip()) for l in open(sys.argv[3]) if l.strip()]
else:
    files = sorted(p for p in glob.glob(os.path.join(src, '**', '*'), recursive=True) if p.lower().endswith(EXT))
n = 0
for p in files:
    try:
        im = ImageOps.exif_transpose(Image.open(p)).convert('RGB')
    except Exception as e:
        print('读不了', p, e); continue
    k = 2000 / max(im.size)
    if k < 1: im = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
    out = os.path.join(dst, os.path.splitext(os.path.relpath(p, src))[0].replace(os.sep, '__') + '.jpg')
    im.save(out, quality=92); n += 1
print('缩好', n, '张 ->', dst)
