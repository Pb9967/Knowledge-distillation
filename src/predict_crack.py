# distillation_inference.py - 知识蒸馏学生模型推理程序（脚本化版本）
import os
import sys
from pathlib import Path
import json
from typing import Optional, List, Tuple

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from student_model import LightweightStudent

# ==================== 配置区域（用户只需修改这里） ====================
# 模型配置
MODEL_PATH = "../model/student_final_model_0.pth"  # 学生模型权重路径
IMAGE_SIZE = 512  # 输入图像尺寸
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # 推理设备，None为自动检测，可设为'cuda'或'cpu'

# 推理模式配置（二选一）
INPUT_MODE = "single"  # 可选："single"（单张）或"batch"（批量）

# 单张图像模式配置（当INPUT_MODE = "single"时生效）
SINGLE_IMAGE_PATH = rf"D:\python_file\Design-2025\data_file\Crack500\test\images\20160329_094016.jpg"
SINGLE_MASK_PATH = rf"D:\python_file\Design-2025\data_file\Crack500\test\masks\20160329_094016_mask.png" # 设为None表示仅预测不评估

# 批量推理模式配置（当INPUT_MODE = "batch"时生效）
INPUT_IMAGE_DIR = "../data/test_images"
MASK_DIR = "../data/test_masks"  # 设为None表示仅预测不评估

# 输出配置
OUTPUT_DIR = "../results/inference"  # 结果保存目录
SAVE_VISUALIZATION = True  # 是否保存可视化结果
# ===================================================================


class CrackSegmentationInference:
    """裂缝分割推理类"""

    def __init__(self, model_path: str, device: str = None, image_size: int = 512):
        """
        初始化推理器

        Args:
            model_path: 训练好的学生模型权重文件路径 (.pth)
            device: 推理设备 ('cuda' or 'cpu')，默认为自动检测
            image_size: 输入图像大小，默认为512
        """
        self.model_path = model_path
        self.image_size = image_size

        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"使用设备: {self.device}")

        # 加载模型
        self.model = self._load_model()

        # 设置matplotlib中文显示
        self._setup_matplotlib_font()

        # 定义颜色映射
        self.colors = {
            'tp': [0, 1, 0],  # 绿色 - 正确检测
            'fp': [1, 0, 0],  # 红色 - 误检
            'fn': [0, 0, 1]  # 蓝色 - 漏检
        }

    def _load_model(self) -> LightweightStudent:
        """加载学生模型和权重"""
        print("\n正在加载学生模型...")

        # 初始化模型
        model = LightweightStudent(num_classes=1, input_size=self.image_size, use_pretrained=False)

        # 加载权重
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        try:
            # 尝试安全加载（PyTorch 2.0+）
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except:
            # 兼容模式
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # 处理不同的保存格式
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ 从检查点加载模型 (epoch: {checkpoint.get('epoch', 'unknown')})")
            else:
                model.load_state_dict(checkpoint)
                print("✓ 直接加载模型权重")
        else:
            raise ValueError("不支持的模型文件格式")

        model.to(self.device)
        model.eval()

        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params:,} (约 {total_params / 1e6:.2f}M)")

        return model

    def _setup_matplotlib_font(self):
        """设置matplotlib中文字体"""
        import platform

        # Windows系统字体配置
        if platform.system() == 'Windows':
            font_candidates = [
                'SimHei',  # 黑体
                'Microsoft YaHei',  # 微软雅黑
                'KaiTi',  # 楷体
                'FangSong'  # 仿宋
            ]

            for font in font_candidates:
                try:
                    plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f"✓ 设置中文字体: {font}")
                    return
                except:
                    continue

        # macOS系统字体配置
        elif platform.system() == 'Darwin':
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'AppleGothic']
            plt.rcParams['axes.unicode_minus'] = False

        # Linux系统字体配置
        else:
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

        print("⚠ 使用默认字体配置")

    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        预处理单个图像

        Args:
            image_path: 输入图像路径

        Returns:
            预处理后的Tensor和原始图像数组
        """
        # 读取图像
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)

        # 预处理
        from torchvision import transforms

        preprocess = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        # 转换为Tensor并添加batch维度
        tensor = preprocess(image).unsqueeze(0)

        return tensor, original_image

    def predict(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        执行预测

        Args:
            image_tensor: 预处理后的图像Tensor

        Returns:
            预测结果 (H, W) 二值掩码
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)

            # 前向传播
            output, _ = self.model(image_tensor)

            # 后处理
            output = torch.sigmoid(output)

            # 插值到输入尺寸
            if output.shape[-2:] != (self.image_size, self.image_size):
                output = torch.nn.functional.interpolate(
                    output, size=(self.image_size, self.image_size),
                    mode='bilinear', align_corners=True
                )

            # 转换为numpy并二值化
            pred_mask = output.cpu().numpy().squeeze()
            pred_mask = (pred_mask > 0.5).astype(np.float32)

            return pred_mask

    def evaluate_single_image(self,
                              image_path: str,
                              mask_path: Optional[str] = None,
                              output_dir: Optional[str] = None,
                              save_visualization: bool = True) -> dict:
        """
        评估单张图像

        Args:
            image_path: 输入图像路径
            mask_path: 真实掩码路径（可选，用于评估）
            output_dir: 输出目录（可选）
            save_visualization: 是否保存可视化结果

        Returns:
            评估结果字典
        """
        # 预处理
        image_tensor, original_image = self.preprocess_image(image_path)

        # 预测
        pred_mask = self.predict(image_tensor)

        # 结果字典
        result = {
            'image_path': image_path,
            'predicted_mask': pred_mask,
            'original_image': original_image,
            'has_ground_truth': mask_path is not None
        }

        # 如果有真实掩码，计算指标
        if mask_path and os.path.exists(mask_path):
            # 读取并处理真实掩码
            gt_mask = Image.open(mask_path).convert('L')
            gt_mask = gt_mask.resize((self.image_size, self.image_size),
                                     Image.NEAREST)
            gt_mask = np.array(gt_mask) / 255.0
            gt_mask = (gt_mask > 0.5).astype(np.float32)

            result['ground_truth'] = gt_mask
            metrics = self.compute_metrics(pred_mask, gt_mask)
            result['metrics'] = metrics

            print(f"\n图像: {os.path.basename(image_path)}")
            print(f"  IoU: {metrics['iou']:.4f}")
            print(f"  Dice: {metrics['dice']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")

        # 保存可视化
        if save_visualization and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            vis_path = os.path.join(output_dir, f"{Path(image_path).stem}_result.png")
            self.visualize_result(result, vis_path)
            result['visualization_path'] = vis_path

        return result

    def compute_metrics(self, pred: np.ndarray, target: np.ndarray) -> dict:
        """
        计算评估指标

        Args:
            pred: 预测掩码 (H, W)
            target: 真实掩码 (H, W)

        Returns:
            指标字典
        """
        pred_binary = (pred > 0.5).astype(np.float32)
        target_binary = (target > 0.5).astype(np.float32)

        epsilon = 1e-6

        # 计算TP, FP, TN, FN
        tp = np.sum(pred_binary * target_binary)
        fp = np.sum(pred_binary * (1 - target_binary))
        tn = np.sum((1 - pred_binary) * (1 - target_binary))
        fn = np.sum((1 - pred_binary) * target_binary)

        # 计算指标
        iou = tp / (tp + fp + fn + epsilon)
        dice = 2 * tp / (2 * tp + fp + fn + epsilon)
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)

        return {
            'iou': float(iou),
            'dice': float(dice),
            'precision': float(precision),
            'recall': float(recall),
            'accuracy': float(accuracy),
            'f1': float(f1)
        }

    def visualize_result(self, result: dict, save_path: str):
        """
        可视化结果并保存（修复尺寸不匹配问题）

        Args:
            result: 结果字典
            save_path: 保存路径
        """
        original_image = result['original_image']
        pred_mask = result['predicted_mask']

        # 确保图像在0-255范围内
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)

        # 调整图像尺寸以匹配掩码
        if original_image.shape[:2] != pred_mask.shape:
            from PIL import Image
            image_pil = Image.fromarray(original_image)
            image_resized = image_pil.resize((pred_mask.shape[1], pred_mask.shape[0]),
                                             Image.BILINEAR)
            original_image = np.array(image_resized)

        # 创建可视化图像
        n_cols = 4 if result['has_ground_truth'] else 2
        figsize = (16, 4) if result['has_ground_truth'] else (8, 4)
        fig, axes = plt.subplots(1, n_cols, figsize=figsize)

        if result['has_ground_truth']:
            if n_cols == 1:  # 防止单张图像时axes不是数组
                axes = [axes]
            else:
                axes = axes.flatten()

            gt_mask = result['ground_truth']
            metrics = result['metrics']

            # 原始图像（已调整尺寸）
            axes[0].imshow(original_image)
            axes[0].set_title('原始图像')
            axes[0].axis('off')

            # 真实掩码
            axes[1].imshow(gt_mask, cmap='gray')
            axes[1].set_title('真实掩码')
            axes[1].axis('off')

            # 预测结果
            axes[2].imshow(pred_mask, cmap='gray')
            axes[2].set_title(f'预测结果\nIoU: {metrics["iou"]:.4f}')
            axes[2].axis('off')

            # 叠加图
            overlay = self.create_overlay(original_image, pred_mask, gt_mask)
            axes[3].imshow(overlay)
            axes[3].set_title('预测比较\n(绿:正确, 红:误检, 蓝:漏检)')
            axes[3].axis('off')

        else:
            # 仅预测模式
            if n_cols == 1:  # 防止单张图像时axes不是数组
                axes = [axes]

            axes[0].imshow(original_image)
            axes[0].set_title('原始图像')
            axes[0].axis('off')

            axes[1].imshow(pred_mask, cmap='gray')
            axes[1].set_title('预测结果')
            axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        print(f"✓ 可视化结果已保存: {save_path}")

    def create_overlay(self, image: np.ndarray, pred_mask: np.ndarray,
                       gt_mask: np.ndarray) -> np.ndarray:
        """
        创建叠加图，显示正确检测、误检和漏检（修复尺寸不匹配问题）

        Args:
            image: 原始图像（已调整尺寸）
            pred_mask: 预测掩码
            gt_mask: 真实掩码

        Returns:
            叠加图像
        """
        overlay = image.copy()

        # 确保掩码是二值的且尺寸匹配
        pred_binary = (pred_mask > 0.5).astype(bool)
        gt_binary = (gt_mask > 0.5).astype(bool)

        # 验证尺寸一致性
        assert overlay.shape[:2] == pred_binary.shape, f"尺寸不匹配: 图像{overlay.shape[:2]} vs 掩码{pred_binary.shape}"

        # 计算TP, FP, FN
        tp = pred_binary & gt_binary
        fp = pred_binary & ~gt_binary
        fn = ~pred_binary & gt_binary

        # 应用颜色（确保是uint8类型）
        if overlay.dtype != np.uint8:
            overlay = (overlay * 255).astype(np.uint8)

        # 使用布尔索引应用颜色
        if np.any(tp):
            overlay[tp] = (np.array(self.colors['tp']) * 255).astype(np.uint8)
        if np.any(fp):
            overlay[fp] = (np.array(self.colors['fp']) * 255).astype(np.uint8)
        if np.any(fn):
            overlay[fn] = (np.array(self.colors['fn']) * 255).astype(np.uint8)

        return overlay

    def batch_predict(self,
                      input_dir: str,
                      output_dir: str,
                      mask_dir: Optional[str] = None,
                      save_visualization: bool = True) -> List[dict]:
        """
        批量预测

        Args:
            input_dir: 输入图像目录
            output_dir: 输出目录
            mask_dir: 真实掩码目录（可选）
            save_visualization: 是否保存可视化结果

        Returns:
            所有结果列表
        """
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []

        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f'*{ext}'))
            image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))

        if not image_files:
            raise ValueError(f"在 {input_dir} 中未找到支持的图像文件")

        print(f"\n发现 {len(image_files)} 张待处理图像")

        results = []
        os.makedirs(output_dir, exist_ok=True)

        pbar = tqdm(image_files, desc="处理图像")
        for img_path in pbar:
            # 构造对应的掩码路径
            mask_path = None
            if mask_dir:
                mask_name = img_path.stem + '_mask.png'
                mask_path = Path(mask_dir) / mask_name
                if not mask_path.exists():
                    mask_path = None

            try:
                result = self.evaluate_single_image(
                    str(img_path),
                    mask_path=str(mask_path) if mask_path else None,
                    output_dir=output_dir,
                    save_visualization=save_visualization
                )
                results.append(result)
            except Exception as e:
                print(f"⚠ 处理 {img_path.name} 时出错: {e}")
                continue

        # 保存结果摘要
        self.save_summary(results, output_dir)

        return results

    def save_summary(self, results: List[dict], output_dir: str):
        """
        保存预测结果摘要

        Args:
            results: 结果列表
            output_dir: 输出目录
        """
        summary = {
            'total_images': len(results),
            'model_path': self.model_path,
            'image_size': self.image_size,
            'device': str(self.device)
        }

        # 如果有评估指标，计算平均值
        if results and 'metrics' in results[0]:
            metrics_keys = ['iou', 'dice', 'precision', 'recall', 'f1']
            avg_metrics = {}

            for key in metrics_keys:
                values = [r['metrics'][key] for r in results if 'metrics' in r]
                if values:
                    avg_metrics[key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }

            summary['average_metrics'] = avg_metrics

            # 打印关键指标
            print("\n" + "=" * 60)
            print("批量预测完成！平均性能指标:")
            for key, stats in avg_metrics.items():
                print(f"  {key.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print("=" * 60)

        # 保存到JSON
        summary_path = os.path.join(output_dir, 'inference_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"✓ 结果摘要已保存: {summary_path}")

def run_inference():
    """运行推理（脚本化入口）"""
    print("知识蒸馏学生模型裂缝分割推理程序")

    # 创建推理器
    try:
        inferencer = CrackSegmentationInference(
            model_path=MODEL_PATH,
            device=DEVICE,
            image_size=IMAGE_SIZE
        )
    except Exception as e:
        print(f"✗ 初始化推理器失败: {e}")
        sys.exit(1)

    # 执行推理
    try:
        if INPUT_MODE == "single":
            # 单张图像模式
            print("\n【单张图像推理模式】")
            print(f"图像路径: {SINGLE_IMAGE_PATH}")

            if not os.path.exists(SINGLE_IMAGE_PATH):
                raise FileNotFoundError(f"图像文件不存在: {SINGLE_IMAGE_PATH}")

            inferencer.evaluate_single_image(
                image_path=SINGLE_IMAGE_PATH,
                mask_path=SINGLE_MASK_PATH,
                output_dir=OUTPUT_DIR,
                save_visualization=SAVE_VISUALIZATION
            )

        elif INPUT_MODE == "batch":
            # 批量模式
            print("\n【批量推理模式】")
            print(f"输入目录: {INPUT_IMAGE_DIR}")

            if not os.path.exists(INPUT_IMAGE_DIR):
                raise FileNotFoundError(f"输入目录不存在: {INPUT_IMAGE_DIR}")

            results = inferencer.batch_predict(
                input_dir=INPUT_IMAGE_DIR,
                output_dir=OUTPUT_DIR,
                mask_dir=MASK_DIR,
                save_visualization=SAVE_VISUALIZATION
            )

            print(f"\n✓ 成功处理 {len(results)} 张图像")

        else:
            raise ValueError(f"无效的INPUT_MODE: {INPUT_MODE}，必须为'single'或'batch'")

    except Exception as e:
        print(f"✗ 推理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n推理完成！")
    print(f"结果保存至: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    run_inference()
