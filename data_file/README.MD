
**文件路径**：`Knowledge-distillation/data_file/README.md`

```markdown
# 数据集说明 (Dataset Documentation)

本目录用于存放裂缝分割数据集。

## 数据集详情：Crack500

本项目使用的是**Crack500公开数据集**，专门针对无人机航拍道路裂缝检测任务构建。

### 数据集统计
- **总图像数**：500张高分辨率道路裂缝图像
- **图像分辨率**：原始高分辨率，统一缩放至512×512（训练时）
- **标注类型**：像素级语义分割掩码（二值图）
- **数据划分**：
  - 训练集：350张（70%）
  - 验证集：100张（20%）
  - 测试集：50张（10%）

## 目录结构

请严格按照以下结构组织数据（与`datasets.py`中`CrackSegmentationDataset`类对应）：

```bash
data_file/
└── Crack_500/                    # 数据集根目录（与config.py中data_root对应）
    ├── train/                    # 训练集（350张）
    │   ├── images/               # 原始图像文件夹
    │   │   ├── 001.jpg           # 支持.jpg/.jpeg/.png
    │   │   ├── 002.jpg
    │   │   └── ...
    │   └── masks/                # 对应掩码文件夹
    │       ├── 001_mask.png      # 必须是_mask.png结尾！
    │       ├── 002_mask.png
    │       └── ...
    │
    ├── val/                      # 验证集（100张）
    │   ├── images/
    │   └── masks/
    │
    └── test/                     # 测试集（50张）
        ├── images/
        └── masks/