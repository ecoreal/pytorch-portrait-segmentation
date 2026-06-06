# 基于 PyTorch U-Net 的人像分割

[English](README.md) | [简体中文](README.zh-CN.md)

本项目是一个基于 PyTorch 的轻量级 U-Net 人像二值分割实现，提供从人像图片预测前景 mask 的训练与推理流程。项目适合作为人像分割、人体分割以及相关计算机视觉实验的基础实现。

## 项目特点

- 使用 U-Net 编码器-解码器结构完成二值图像分割。
- 内置 360 对样例图片与 mask，便于快速验证训练流程。
- 支持自定义数据路径、训练轮数、batch size、学习率、输入尺寸和 checkpoint 输出路径。
- 提供单张图片推理脚本，可输出二值人像 mask。
- 代码结构简洁，便于学习、实验和进一步扩展。

## 效果预览

| 原图 | Mask |
| --- | --- |
| ![source](picture/source360/003ff269-1cec-4ec5-8782-a9566afbe05f.png) | ![mask](picture/mask360/003ff269-1cec-4ec5-8782-a9566afbe05f.png) |

## 仓库结构

```text
.
├── README.md           # 英文项目文档
├── README.zh-CN.md     # 简体中文项目文档
├── dataset.py          # 数据加载与预处理
├── model.py            # U-Net 模型定义
├── train.py            # 训练入口
├── predict.py          # 单图推理入口
├── picture/
│   ├── source360/      # 样例输入图片
│   └── mask360/        # 样例二值 mask
├── DATASET.md          # 数据格式与使用说明
├── requirements.txt    # Python 依赖
└── LICENSE             # 源代码许可证
```

## 安装

环境要求：

- Python 3.9+
- PyTorch
- torchvision
- Pillow

克隆仓库并安装依赖：

```bash
git clone https://github.com/ecoreal/pytorch-portrait-segmentation.git
cd pytorch-portrait-segmentation
pip install -r requirements.txt
```

如果需要使用 CUDA，请根据官方 PyTorch 安装说明安装匹配 CUDA 版本的 `torch` 和 `torchvision`，再安装其余依赖。

## 数据格式

训练流程要求原图和 mask 分别存放在两个目录中，并使用相同的文件名主干：

```text
picture/source360/example.png
picture/mask360/example.png
```

支持的图片后缀包括 `.jpg`、`.jpeg`、`.png`、`.bmp` 和 `.webp`。Mask 会以灰度图方式读取，并在训练阶段转换为二值张量。

更多数据说明见 [DATASET.md](DATASET.md)。

## 训练

使用仓库内置样例数据训练：

```bash
python3 train.py --epochs 10 --batch-size 4 --output model.pth
```

使用自定义数据集训练：

```bash
python3 train.py \
  --image-dir /path/to/images \
  --mask-dir /path/to/masks \
  --epochs 30 \
  --batch-size 8 \
  --learning-rate 0.001 \
  --image-size 256 \
  --output checkpoints/portrait_unet.pth
```

保存的 checkpoint 包含：

- `model_state_dict`
- `image_size`
- `n_class`

## 推理

训练完成后，可对单张图片生成二值 mask：

```bash
python3 predict.py picture/source360/003ff269-1cec-4ec5-8782-a9566afbe05f.png model.pth --output output_mask.png
```

可选参数：

- `--image-size`：覆盖推理阶段的缩放尺寸。
- `--threshold`：前景概率阈值，默认值为 `0.5`。

## 局限性

- 本项目定位为轻量级 baseline，不是生产级分割系统。
- 内置样例数据规模较小，模型效果依赖更多训练数据、标注质量、数据增强和训练配置。
- 仓库未包含预训练 checkpoint，推理前需要先完成训练。

## 开发计划

- 增加训练集/验证集划分，并支持 Dice score、IoU 等分割指标。
- 增加数据增强以提升泛化能力。
- 增加批量推理和前景提取工具。
- 增加 ONNX 导出能力，便于部署。

## 许可证

源代码基于 MIT License 开源，详见 [LICENSE](LICENSE)。

数据格式与复用说明见 [DATASET.md](DATASET.md)。
