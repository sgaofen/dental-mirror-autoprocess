# 口内反光板照片自动处理 · 部署与操作手册

> 面向需要部署、调用或继续开发这套工具的人或 agent。原理和实验记录见 `说明.md`。

## 1. 它做什么

**输入：** 牙医用反光板（口镜）拍的咬合面照片原图（jpg/png/tif/bmp，任意方向、任意大小）。

**输出：** 每张照片输出一张处理好的成品图，与医生手工处理的习惯一致：

1. **翻转**：水平翻转，消除镜像。
2. **摆正**：下颌摆成 U 形，门牙朝下；上颌摆成 ∩ 形，门牙朝上；舌头或腭在牙弓里面。半边牙弓的"磨牙→尖牙"连线偏离竖直方向 40°。
3. **裁剪**：只留镜子里的牙列，四周外扩一颗牙宽，统一 4:3。
4. **调亮**：亮度调到中等。

同时给出磨牙、尖牙的位置，以及上颌/下颌的判断。

**流程：**

```
原图 ─▶ 找牙模型 models/C4.pt（YOLO11s-pose，半边牙弓 8 个关键点 + 4 类：上下颌 × 左右侧）
     ─▶ 上下颌分类器 models/archsmall.pt（YOLO11s-cls，判断牙弓里面是舌头还是腭）
     ─▶ 固定公式 dental_process.process（翻转、旋转、裁剪、gamma）
     ─▶ 成品图 + results.csv
```

## 2. 环境

已验证过的组合（macOS arm64，CPU 推理）：

| 组件 | 版本 |
|---|---|
| Python | 3.11 |
| ultralytics | 8.4.177（固定这个版本，模型就是用它训练的） |
| torch / torchvision | 2.14.1 / 0.29.1 |
| opencv-python | 5.0.0 |
| pillow | 12.3.0 |
| numpy | 2.4.6 |

训练和大批量评测是在 Linux + CUDA 集群上跑的，同样用 ultralytics 8.4.177。

以上组合已经在全新虚拟环境里从零安装验证过：医生 4 张照片全部处理成功，退出码 0；空文件夹退出码 1；从别的目录用绝对路径调用也正常。

- 不需要 GPU。有 CUDA 时 ultralytics 会自动使用。
- 没有图形界面的服务器，可以把 `opencv-python` 换成 `opencv-python-headless`。
- 推理时不联网。模型文件都在 `models/` 里，不会自动下载。

### 安装

```bash
cd 口内照片自动处理
```

```bash
python3.11 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

用 uv 的话：`uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt`

### 验证安装

```bash
.venv/bin/python -c "from ultralytics import YOLO; m=YOLO('models/C4.pt'); c=YOLO('models/archsmall.pt'); print(len(m.names), c.names)"
```

应该输出 `4 {0: 'lower', 1: 'upper'}`。

## 3. 命令行用法

```bash
.venv/bin/python run.py 输入 [输入 ...] -o 输出文件夹 [--debug] [--det 模型] [--cls 模型]
```

| 参数 | 含义 |
|---|---|
| `输入` | 一个或多个图片文件，或装图片的文件夹（只读第一层，不递归） |
| `-o / --out` | 输出文件夹，默认 `结果`，不存在会自动创建 |
| `--debug` | 另存 `原名_标点.jpg`：在原图上标磨牙（红）、尖牙（绿）、牙弓里面方向（蓝） |
| `--det` | 找牙模型，默认 `models/C4.pt` |
| `--cls` | 上下颌分类器，默认 `models/archsmall.pt` |

`run.py` 会把自己所在的目录加入 import 路径，所以可以在任意目录下用绝对路径调用。

### 输出

- `输出文件夹/原名_处理后.jpg`：成品图（JPEG 质量 92）。
- `输出文件夹/results.csv`：每张一行，便于程序读取：

| 列 | 含义 |
|---|---|
| `file` | 输入路径（原样记录） |
| `status` | `ok` / `no_teeth`（没找到镜子里的牙）/ `unreadable`（读不了） |
| `arch` | `lower` / `upper` |
| `p_up` | 分类器给出的上颌概率（0–1），大于 0.5 判上颌 |
| `det_conf` | 找牙模型的把握（0–1） |
| `gamma` | 调亮用的 gamma 值（0.7–1.05） |
| `needs_review` | 1 = 建议人工看一眼（`0.2 < p_up < 0.9`，真实照片上的错误几乎都落在这个区间） |
| `output` | 成品图路径 |
| `width`, `height` | 成品图尺寸 |

- **退出码：** 至少一张处理成功为 0；一张都没成功为 1；没找到图片时也会以非零退出。

## 4. Python 接口

```python
import sys; sys.path.insert(0, '/path/to/口内照片自动处理')
from ultralytics import YOLO
from dental_process import load, process
from auto_half import landmarks

det = YOLO('/path/to/口内照片自动处理/models/C4.pt')
cls = YOLO('/path/to/口内照片自动处理/models/archsmall.pt')

img = load('photo.jpg')             # BGR ndarray，已按 EXIF 方向转正
L = landmarks(det, img, cm=cls)     # 没找到牙时返回 None
if L is not None:
    out, gamma = process(img, L)    # out 是 BGR ndarray 成品图
```

`landmarks()` 返回的字典（坐标都是 `img` 原图像素坐标，未翻转）：

| 键 | 含义 |
|---|---|
| `arch` | `'lower'` / `'upper'`（传了 `cm` 时由分类器决定，否则由找牙模型决定） |
| `p_up` | 上颌概率；不传 `cm` 时为 `None` |
| `molar` | `[x, y]`，第一、二磨牙中心 |
| `anterior` | `[x, y]`，尖牙中心 |
| `lingual` | `[x, y]`，牙弓里面方向上的一个点，只用来决定左右 |
| `pts` | 裁剪用的点集（每颗认出的牙外扩一颗牙宽） |
| `box` | `pts` 的外接框 `[x0, y0, x1, y1]` |
| `kp` | 把握最大的半边牙弓的 8 个关键点 `(x, y, conf)`：0 中切牙、1 侧切牙、2 尖牙、3–4 前磨牙、5–7 磨牙 |
| `conf` | 找牙把握 |
| `side` | 0 / 1，半边牙弓的侧别（四类模型才有） |
| `pw` | 原图宽度（`process` 用它换算坐标） |
| `arch_boxes` | 分类器看的两个正方形区域（翻转后坐标） |

**也可以不用模型：** 自己提供 `arch`、`pw`、`molar`、`anterior`、`lingual`、`box`（或 `pts`），直接调用 `process()`。`landmarks.json` 就是这样的人工标注示例。

## 5. 可以调的参数

### 成品摆法（`dental_process.py` 开头）

```python
TILT = {'lower': 40, 'upper': 40}   # 磨牙->尖牙连线偏离竖直方向的角度
ASPECT = 4 / 3                       # 成品宽高比
PAD = 0.0                            # 额外留白（牙点已经外扩了一颗牙宽）
TARGET_L = 0.45                      # 成品平均亮度（0~1）
```

### 上下颌分类器看的范围（`auto_half.py` 里 `arch_boxes`）

以牙列为中心的正方形，边长是牙列长边的 1.3 倍和 1.6 倍。

- **改小：** 看不到腭，上颌的把握会下降。
- **改大：** 会把镜子外面的牙带进来，下颌容易被认成上颌。

### 复核阈值（`run.py` 里的 `REVIEW`）

默认 `(0.2, 0.9)`。

## 6. 回归测试

改动代码或模型以后，至少跑这两项：

```bash
CLS=models/archsmall.pt .venv/bin/python eval_points.py 医生原图文件夹 models/C4.pt
```

- **用途：** 在医生的 4 张照片上逐张检查磨牙点、尖牙点的误差（单位是颗牙宽），以及上下颌判断和牙弓里面朝向是否正确。目前是 4/4 全对。
- **需要的文件：** 医生原图，文件名形如 `*_21_145.jpg`、`*_23_145.jpg`、`*_25_145.jpg`、`*_27_145.jpg`，**不在包里**，需要向项目负责人索取。手工标注在 `landmarks.json`。

```bash
N=100 .venv/bin/python eval_arch_big.py models/C4.pt models/archsmall.pt
```

- **用途：** 用 Roboflow 验证集做成 200 张模拟诊所照片，测上下颌准确率。目前约 98.5%。这套测试偏简单，只用来确认没有变差。

**真实照片的复测：** `人工核对标签.csv` 是 OMNI、AlphaDent 两个公开数据集各 100 张的人工核对结果（是否咬合面、真实上下颌）。重新下载这两个数据集后，用 `run.py` 跑一遍，再和这个表对比即可。目前上下颌准确率约为 OMNI 91%、AlphaDent 95%。

## 7. 重新训练

**训练数据：** `data/` 里是 Roboflow 的 "Lower Occlusal"（991 张）和 "Upper Occlusal"（1108 张），作者 "Teeth Identification"，许可证 CC BY 4.0。每张图标了 16 个牙齿关键点。

### 找牙模型

```bash
.venv/bin/python prepare_half.py C quad
```

```bash
.venv/bin/yolo pose train data=dataset_C4/data.yaml model=yolo11s-pose.pt imgsz=640 epochs=60 batch=32 degrees=180 fliplr=0 flipud=0 scale=0.5 translate=0.15 mosaic=1.0 close_mosaic=5 hsv_v=0.5 hsv_s=0.6 patience=15
```

- **数据：** 第一条命令生成 `dataset_C4`，把每张图拆成半边牙弓，每半边 8 个点，分 4 类。
- **必须设 `fliplr=0`：** 左右镜像增强会把侧别学反。
- **`data.yaml`：** 里面的 `path:` 是生成时的绝对路径。把数据集挪到别处以后，要把这一行改成新位置。

### 上下颌分类器

```bash
.venv/bin/python prepare_arch.py small
```

```bash
.venv/bin/yolo classify train data=dataset_arch_small model=yolo11s-cls.pt imgsz=320 epochs=30 batch=64 fliplr=0.5 flipud=0.5 degrees=180 hsv_v=0.5 hsv_s=0.6 erasing=0.3 patience=10
```

**数据：** 第一条命令生成 `dataset_arch_small`，包含整排、紧裁、半边、缩小加模糊背景、牙弓内侧小块几类，全部随机旋转 0–360°。

### 训练的通用说明

- 训练完把 `runs/.../weights/best.pt` 复制到 `models/`，替换对应文件，然后跑第 6 节的回归测试。
- 预训练权重 `yolo11s-pose.pt`、`yolo11s-cls.pt` 第一次使用时会由 ultralytics 自动下载。
- `hpc3/` 里是在 UCI HPC3 集群上跑对照实验用的 Slurm 脚本，跟该集群的目录和调度方式绑定，换环境时只能参考。

## 8. 已知局限

1. **验证规模：** 只在医生的 4 张照片，以及两个公开数据集的各 100 张抽样上验证过。
2. **舌头抬起的下颌：** 露出舌底、舌系带、口底的下颌，有时会被判成上颌（真实照片里约 5–9%），是目前最主要的错误。大多数会被 `needs_review` 标出来，少数把握很高的会漏掉。
3. **不检查拍摄视角：** 正面、侧面照片也会被照常处理，输出没有意义。调用方要自己保证输入的是咬合面照片。
4. **四角拉丝：** 旋转后四角的空白用边缘像素补，可能有拉丝条纹。
5. **裁剪偏松：** 和医生相比，自动结果里的牙略小。
6. **HEIC：** 需要先转成 jpg。

## 9. 文件清单

| 路径 | 用途 |
|---|---|
| `run.py` | 命令行入口 |
| `dental_process.py` | 读图（`load`）+ 固定公式（`process`） |
| `auto_half.py` | 调用两个模型：找牙、判断上下颌（`landmarks`） |
| `models/C4.pt`、`models/archsmall.pt` | 正在用的两个模型（`*.json` 是训练时的验证分数） |
| `requirements.txt` | 依赖 |
| `README.md` | 本文件 |
| `说明.md` | 原理、实验记录、为什么这样设计 |
| `eval_points.py`、`eval_arch_big.py` | 回归测试 |
| `make_sheet.py` | 拼"原图 / 医生 / 自动"三列对比图 |
| `make_review.py` | 把批量评测结果拼成抽查图（读的是 `hpc3/eval_real.py` 的输出，不是 `run.py` 的 `results.csv`） |
| `shrink_images.py` | 外部数据缩图 |
| `prepare_half.py`、`prepare_arch.py` | 生成训练数据 |
| `landmarks.json` | 医生 4 张照片的人工标注 |
| `人工核对标签.csv` | OMNI、AlphaDent 抽查的人工标签 |
| `data/` | Roboflow 原始数据（CC BY 4.0） |
| `hpc3/` | 集群训练和评测脚本（仅供参考） |

**不在包里：** 病人照片（隐私）、外部数据集 OMNI 和 AlphaDent（体积大；OMNI 没写许可证）、生成好的训练数据（可以用 `prepare_*.py` 重新生成）。
