# 裂缝分割知识蒸馏系统

基于知识蒸馏的轻量化裂缝图像语义分割系统。使用 DeepLabV3+-ResNet101 作为教师模型，MobileNetV2 编码器 + U-Net 风格解码器作为学生模型，通过多源蒸馏损失实现模型压缩，在保持分割精度的同时大幅降低参数量与推理延迟，适配 Jetson Nano 等边缘设备部署。

## 项目结构

```
Knowledge-distillation/
	├── data_file/                      # 数据集
	├── model/                          # 模型权重文件
	├── results/                        # 输出结果
	└── src/                            # 代码文件
		├── config.py                   # 统一配置中心
		├── datasets.py                 # 数据集加载与校验
		├── student_model.py            # 轻量化学生模型
		├── get_teacher.py              # 教师模型创建与保存
		├── distillation_loss.py        # 多源蒸馏损失函数
		├── distillation_trainer.py     # 蒸馏训练主程序
		├── teacher_test.py             # 教师模型基准测试
		├── predict_crack.py            # 推理脚本（单张/批量）
		├── plot_history.py             # 训练历史可视化
		├── show.py                     # 数据集样本展示
		├── chinese_test.py             # 中文字体适配工具
		├── env_test.py                 # 运行环境检测
		├── requirements.txt            # Python 依赖
		└── README.md                   # 本文件
```

## 模块说明

### 配置模块

| 文件 | 功能 |
|------|------|
| `config.py` | 基于 `dataclass` 的统一配置中心，包含数据集路径、模型参数、训练超参、蒸馏参数、损失权重、硬件配置、学习率调度、数据增强、混合精度、可视化与日志等全部可配置项。所有模块通过 `Config()` 实例获取配置，支持 `get()` / `__getitem__()` 安全访问。 |

### 数据模块

| 文件 | 功能 |
|------|------|
| `datasets.py` | 裂缝分割数据集类 `CrackSegmentationDataset`，支持 `.jpg/.jpeg/.png` 混合格式。初始化时自动遍历数据目录，过滤损坏文件，验证图像-掩码对的完整性与尺寸一致性，将异常文件记录到 JSON 日志。数据加载时执行 Resize→ToTensor→二值化，并包含最多 3 次重试的容错机制。 |
| `show.py` | 数据集可视化工具，从指定目录随机抽取样本，展示原始图像、分割掩码与叠加效果（红色标记裂缝区域 + 边界轮廓），并提供数据集分布统计（文件数量、匹配率）。 |

### 模型模块

| 文件 | 功能 |
|------|------|
| `student_model.py` | 轻量化学生模型 `LightweightStudent`：MobileNetV2 前 14 层作为编码器，提取 stage 1/3/6/13 四级特征；瓶颈层使用 `SimpleBottleneck`（1x1 降维 + 3x3 卷积 + 残差连接）；解码器由 4 级 `DepthwiseSeparableConv` 块逐级上采样并与编码器跳跃连接融合，最终 1x1 卷积 + Sigmoid 输出。前向传播同时返回预测掩码与中间特征字典（`bottleneck`/`decoder4`/`decoder3`），供蒸馏损失计算。 |
| `get_teacher.py` | 教师模型工厂，创建 `smp.DeepLabV3Plus(encoder_name='resnet101')` 并冻结全部参数，加载 ImageNet 预训练权重。提供 `save_teacher_model()` 将权重持久化为 `.pth` 文件。 |

### 蒸馏损失模块

| 文件 | 功能 |
|------|------|
| `distillation_loss.py` | 实现 4 种蒸馏损失 + 1 个联合损失类： |

- `ResponseDistillationLoss` — 输出响应蒸馏，KL 散度对齐教师-学生 soft label
- `FeatureMapDistillationLoss` — 特征图蒸馏，L2 对齐中间层特征（支持 adaptor 与尺寸自适应插值）
- `EdgeAwareDistillationLoss` — 边缘感知注意力蒸馏，基于 Sobel 算子生成边缘掩码增强裂缝边界区域的注意力迁移
- `ContrastiveDistillationLoss` — 对比蒸馏，拉近教师-学生对应空间位置的特征表示
- `MultiLevelDistillationLoss` — 联合损失，整合 Dice + BCE 任务损失与上述 4 种蒸馏损失，支持 epoch 衰减策略动态调整响应/特征蒸馏权重

### 训练模块

| 文件 | 功能 |
|------|------|
| `distillation_trainer.py` | 蒸馏训练主程序 `KnowledgeDistillationTrainer`（约 900 行），核心流程：初始化教师/学生模型→加载 train/val/test 数据集→配置 AdamW 优化器 + CosineAnnealingLR→每个 epoch 执行蒸馏训练（前向传播→教师推理→提取中间特征→计算联合损失→反向传播）→验证评估（IoU/Precision/Recall/Dice/Accuracy）→保存检查点与最佳模型→训练结束后测试集评估→生成训练历史图与总结报告。支持检查点断点续训与 PyTorch 2.6+ 安全加载。 |

### 评估与推理模块

| 文件 | 功能 |
|------|------|
| `teacher_test.py` | 教师模型基准测试 `TeacherModelEvaluator`，在测试集上计算 IoU/Dice/Accuracy/Precision/Recall/F1/Specificity，生成指标分布直方图、箱线图、相关性热图、混淆矩阵、预测样本可视化与性能雷达图，输出 JSON 评估摘要与 Markdown 分析报告。 |
| `predict_crack.py` | 学生模型推理脚本 `CrackSegmentationInference`，支持单张图像与批量目录两种模式。预测后自动计算指标（如有真实掩码），生成四联可视化图（原图/真实掩码/预测/叠加比较，绿=正确、红=误检、蓝=漏检），批量模式输出 JSON 摘要。脚本顶部提供配置区域，无需修改核心代码即可切换模式。 |

### 可视化与工具模块

| 文件 | 功能 |
|------|------|
| `plot_history.py` | 从检查点文件提取训练历史，绘制 6 子图综合面板（总损失曲线/IoU 曲线/Precision-Recall/损失分量/PR 散点图/学习率调度曲线），支持命令行参数覆盖配置。 |
| `chinese_test.py` | Matplotlib 中文字体适配工具，自动检测系统中文字体（Windows/macOS/Linux），配置 `rcParams` 解决中文乱码与负号显示问题，清除字体缓存，提供测试函数验证显示效果。 |
| `env_test.py` | 运行环境检测脚本，打印操作系统、Python 版本、PyTorch 版本、CUDA 可用性、GPU 型号与显存、CPU 核心数、内存总量等关键信息。 |

## 执行流程

```
1. 环境准备        →  python env_test.py
2. 数据集校验      →  python datasets.py
3. 教师模型生成    →  python get_teacher.py
4. 教师模型评估    →  python teacher_test.py
5. 蒸馏训练        →  python distillation_trainer.py
6. 训练历史可视化  →  python plot_history.py
7. 推理预测        →  python predict_crack.py
```

### 详细步骤

**第一步：环境确认**

```bash
python env_test.py
```

输出当前环境信息（GPU、CUDA、内存等），确认硬件与依赖是否就绪。

**第二步：数据集准备与校验**

```bash
python datasets.py
```

自动遍历 `data_root` 下的 `train/val/test` 子目录，校验图像-掩码对完整性，输出有效样本数与损坏文件列表。也可运行 `show.py` 可视化数据集样本：

```bash
python show.py
```

**第三步：教师模型创建**

```bash
python get_teacher.py
```

创建 DeepLabV3+-ResNet101 并加载 ImageNet 预训练权重，将模型权重保存到 `model/teacher_deeplabv3p.pth`。

> **注意**：教师模型需要先在裂缝数据集上完成预训练（fine-tune），项目代码中的教师模型评估与蒸馏训练均假设已有预训练权重。若需自行训练教师模型，请使用 `segmentation_models_pytorch` 的训练接口在 Crack500 数据集上 fine-tune。

**第四步（可选）：教师模型基准测试**

```bash
python teacher_test.py
```

在测试集上评估教师模型性能，生成完整的性能分析报告与可视化结果，保存到 `results/teacher_performance/`。

**第五步：蒸馏训练**

```bash
python distillation_trainer.py
```

核心训练流程。训练过程中自动保存检查点（每 10 个 epoch）与最佳模型（按验证 IoU 筛选），训练结束后在测试集上评估并保存最终模型。输出目录结构：

```
../results/distillation_performance/
├── checkpoint/                         # 训练检查点
├── best_model_ing/                     # 训练中最佳模型
├── training_history.json               # 训练历史数据
├── training_summary.txt                # 训练总结报告
└── distillation_training_history.png   # 训练历史可视化
```

最终模型保存到 `model/student_final_model.pth`。

**支持断点续训**：在 `config.py` 中设置 `use_checkpoint=True` 并指定 `checkpoint_path` 即可从上次中断处恢复训练。

**第六步：训练历史可视化**

```bash
# 使用默认配置
python plot_history.py

# 指定检查点路径
python plot_history.py --checkpoint_path results/distillation_performance/checkpoint/checkpoint_epoch50.pth

# 指定输出目录
python plot_history.py --output_dir results/plot_history
```

从检查点提取训练历史，生成 6 子图可视化面板（损失曲线、IoU 曲线、PR 曲线、损失分量、PR 散点图、学习率调度）。

**第七步：推理预测**

在 `predict_crack.py` 顶部配置区域设置参数后运行：

```bash
# 单张图像模式
# 设置 INPUT_MODE = "single", SINGLE_IMAGE_PATH = "你的图像路径"
python predict_crack.py

# 批量模式
# 设置 INPUT_MODE = "batch", INPUT_IMAGE_DIR = "你的图像目录"
python predict_crack.py
```

结果保存到 `results/inference/`，包含可视化图像与 JSON 摘要。

## 数据集准备

本项目使用 **Crack500** 裂缝分割数据集，目录结构要求如下：

```
data_file/Crack_500/
├── train/
│   ├── images/          # 训练图像（.jpg/.jpeg/.png）
│   └── masks/           # 对应掩码（文件名格式：xxx_mask.png）
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

- 图像与掩码的对应关系：掩码文件名 = 图像文件名（去扩展名）+ `_mask.png`
- 掩码为灰度图像，二值化阈值 0.5
- 数据集路径可在 `config.py` 的 `data_root` 字段中修改

## 环境配置

### 系统要求

- Python 3.8+
- CUDA 12.9（与 torch 2.8.0+cu129 匹配）
- GPU 显存 >= 6GB（训练），>= 2GB（推理）
- 内存 >= 16GB

### 安装依赖

```bash
pip install -r requirements.txt
```

### 关键依赖说明

| 包 | 用途 |
|----|------|
| `torch` / `torchvision` | 深度学习框架与视觉模型 |
| `segmentation-models-pytorch` | 教师模型 DeepLabV3+ 的构建与预训练权重加载 |
| `torchsummary` | 模型结构与参数量可视化 |
| `numpy` / `scipy` | 数值计算与图像处理 |
| `matplotlib` / `seaborn` | 可视化绘图 |
| `Pillow` | 图像读取与预处理 |
| `tqdm` | 进度条 |
| `scikit-learn` | 混淆矩阵计算 |
| `pandas` | 指标数据整理 |
| `psutil` | 系统资源检测 |

## 关键性能指标

| 指标 | 教师模型 (DeepLabV3+-ResNet101) | 学生模型 (MobileNetV2-UNet) | 变化 |
|------|------|------|------|
| 参数量 | 54.8M | 1.82M | **-96.7%** |
| mIoU | 0.29 | 0.46 | **+60.7%** |
| 推理延迟 | 89ms | 28ms | **3.2x** |
| Jetson Nano FPS | — | 12.3 | — |

> 学生模型 mIoU 高于教师模型，是因为教师模型未经充分 fine-tune，而学生模型通过蒸馏获得了更好的裂缝分割能力。