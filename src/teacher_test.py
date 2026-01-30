# teacher_test.py
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
from tqdm import tqdm
import warnings
import time

warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from sklearn.metrics import confusion_matrix

from chinese_test import setup_matplotlib_font, ensure_chinese_font
from config import Config
from datasets import CrackSegmentationDataset

setup_matplotlib_font()
sns.set_style("whitegrid")


class TeacherModelEvaluator:
    """教师模型评估器（完整可视化版）"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        self.output_dir = config.get('teacher_result', '../results/teacher_performance')
        os.makedirs(self.output_dir, exist_ok=True)

        self.vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(self.vis_dir, exist_ok=True)

        self._init_model()
        self._init_datasets()

        self.evaluation_results = {
            'test_metrics': {},
            'per_image_results': [],
            'confusion_matrix': None,
            'inference_time': None,
            'prediction_samples': []
        }

        self.colors = {
            'tp': [0, 1, 0],
            'fp': [1, 0, 0],
            'fn': [0, 0, 1]
        }

    def _init_model(self):
        """初始化教师模型"""
        print("\n初始化DeepLabV3+教师模型...")
        self.model = smp.DeepLabV3Plus(
            encoder_name='resnet101',
            encoder_weights=None,
            classes=1,
            activation='sigmoid'
        )

        model_path = self.config.get('model_path', '../model/teacher_deeplabv3p.pth')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✓ 已加载教师模型权重: {model_path}")
        else:
            print(f"✗ 警告: 未找到模型权重文件 {model_path}")

        self.model = self.model.to(self.device)
        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型参数量: {total_params:,} (约{total_params / 1e6:.1f}M)")

    def _init_datasets(self):
        """初始化数据集"""
        data_root = self.config['data_root']
        image_size = self.config.get('image_size', 512)
        batch_size = self.config.get('batch_size', 4)

        print(f"\n加载数据集 from {data_root}...")
        self.test_dataset = CrackSegmentationDataset(
            data_root=data_root,
            split='test',
            image_size=image_size
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        print(f"测试集: {len(self.test_dataset)} 张图像")

    def compute_metrics(self, pred, target):
        """计算所有评估指标"""
        pred_binary = (pred > 0.5).float()
        target_binary = (target > 0.5).float()

        tp = (pred_binary * target_binary).sum()
        fp = (pred_binary * (1 - target_binary)).sum()
        tn = ((1 - pred_binary) * (1 - target_binary)).sum()
        fn = ((1 - pred_binary) * target_binary).sum()

        epsilon = 1e-6

        metrics = {
            'iou': tp / (tp + fp + fn + epsilon),
            'dice': 2 * tp / (2 * tp + fp + fn + epsilon),
            'accuracy': (tp + tn) / (tp + tn + fp + fn + epsilon),
            'precision': tp / (tp + fp + epsilon),
            'recall': tp / (tp + fn + epsilon),
            'specificity': tn / (tn + fp + epsilon) if (tn + fp) > 0 else 0
        }
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (
                    metrics['precision'] + metrics['recall'] + epsilon)

        return {k: v.item() if hasattr(v, 'item') else float(v) for k, v in metrics.items()}

    def evaluate_test_set(self):
        """在测试集上评估（完整版）"""
        print("\n开始在测试集上评估教师模型...\n")

        self.model.eval()
        metrics_accumulator = {
            'iou': [], 'dice': [], 'accuracy': [],
            'precision': [], 'recall': [], 'f1': [], 'specificity': []
        }

        total_inference_time = 0
        total_images = 0

        all_preds = []
        all_targets = []
        sample_predictions = []

        with torch.no_grad():
            for batch_idx, (images, masks, img_names) in enumerate(tqdm(self.test_loader, desc="测试集评估")):
                images = images.to(self.device)
                masks = masks.to(self.device)

                start_time = time.time()
                outputs = self.model(images)
                batch_time = time.time() - start_time

                total_inference_time += batch_time
                total_images += images.size(0)

                for i in range(images.size(0)):
                    output_i = outputs[i:i + 1]
                    mask_i = masks[i:i + 1]

                    if mask_i.shape[1] == 3:
                        mask_i = mask_i.mean(dim=1, keepdim=True)
                    mask_i = (mask_i > 0.5).float()

                    metrics = self.compute_metrics(output_i, mask_i)

                    result = {'image_name': img_names[i], **metrics}
                    self.evaluation_results['per_image_results'].append(result)

                    for key in metrics:
                        metrics_accumulator[key].append(metrics[key])

                    if len(all_preds) < 10000:
                        pred_binary = (output_i > 0.5).float().cpu().numpy().flatten()
                        target_binary = mask_i.cpu().numpy().flatten()

                        if len(pred_binary) > 1000:
                            idx = np.random.choice(len(pred_binary), 1000, replace=False)
                            all_preds.extend(pred_binary[idx])
                            all_targets.extend(target_binary[idx])
                        else:
                            all_preds.extend(pred_binary)
                            all_targets.extend(target_binary)

                    if batch_idx < 2 and i < 2:
                        sample_data = {
                            'image': images[i].cpu(),
                            'mask': masks[i].cpu(),
                            'pred': outputs[i].cpu(),
                            'name': img_names[i],
                            'metrics': metrics
                        }
                        sample_predictions.append(sample_data)

        avg_metrics = {}
        for key in metrics_accumulator:
            if len(metrics_accumulator[key]) > 0:
                avg_metrics[key] = {
                    'mean': np.mean(metrics_accumulator[key]),
                    'std': np.std(metrics_accumulator[key]),
                    'min': np.min(metrics_accumulator[key]),
                    'max': np.max(metrics_accumulator[key]),
                    'median': np.median(metrics_accumulator[key])
                }

        avg_inference_time = total_inference_time / total_images if total_images > 0 else 0
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

        self.evaluation_results['test_metrics'] = avg_metrics
        self.evaluation_results['inference_time'] = {
            'total_seconds': total_inference_time,
            'avg_per_image': avg_inference_time,
            'fps': fps
        }
        self.evaluation_results['prediction_samples'] = sample_predictions

        if len(all_preds) > 0:
            self.compute_confusion_matrix(all_preds, all_targets)

        self.print_results(avg_metrics, total_inference_time, avg_inference_time, fps)
        self.save_results()

        if sample_predictions:
            self.visualize_results(metrics_accumulator, sample_predictions)

        self.generate_report()

        return avg_metrics

    def print_results(self, avg_metrics, total_time, avg_time, fps):
        """打印评估结果"""
        print("\n教师模型测试集评估结果\n")

        print(f"\n1. 分割性能指标:")
        print("-" * 40)
        for key in ['iou', 'dice', 'accuracy', 'precision', 'recall', 'f1', 'specificity']:
            if key in avg_metrics:
                stats = avg_metrics[key]
                print(f"  {key.upper():<12} : {stats['mean']:.4f} ± {stats['std']:.4f}")

        print(f"\n2. 推理性能:")
        print("-" * 40)
        print(f"  总图像数: {len(self.test_dataset)}")
        print(f"  总推理时间: {total_time:.2f} 秒")
        print(f"  平均每张图像推理时间: {avg_time:.4f} 秒")
        print(f"  推理速度: {fps:.2f} FPS")

        print("\n3. 关键指标:")
        print("-" * 40)
        print(f"  mIoU: {avg_metrics.get('iou', {}).get('mean', 0):.4f}")
        print(f"  Dice系数: {avg_metrics.get('dice', {}).get('mean', 0):.4f}")
        print(f"  F1分数: {avg_metrics.get('f1', {}).get('mean', 0):.4f}")

        print("\n" + "=" * 60)

    def compute_confusion_matrix(self, all_preds, all_targets):
        """计算混淆矩阵"""
        print("\n计算混淆矩阵...")

        all_preds_int = np.round(all_preds).astype(int)
        all_targets_int = np.round(all_targets).astype(int)

        cm = confusion_matrix(all_targets_int, all_preds_int)
        self.evaluation_results['confusion_matrix'] = cm

        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            print(f"\n混淆矩阵统计（基于{len(all_preds)}个像素）:")
            print(f"  真阳性(TP): {tp:>8} (正确预测的裂缝像素)")
            print(f"  假阳性(FP): {fp:>8} (误检为裂缝的背景像素)")
            print(f"  假阴性(FN): {fn:>8} (漏检的裂缝像素)")
            print(f"  真阴性(TN): {tn:>8} (正确预测的背景像素)")
            print(f"  精确率: {precision:.4f}")
            print(f"  召回率: {recall:.4f}")
            print(f"  F1分数: {f1:.4f}")
            print(f"  特异度: {specificity:.4f}")
        else:
            print(f"混淆矩阵形状异常: {cm.shape}")

        self.plot_confusion_matrix(cm)

    def plot_confusion_matrix(self, cm):
        """绘制混淆矩阵热图"""
        ensure_chinese_font()
        plt.figure(figsize=(8, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['背景', '裂缝'],
                    yticklabels=['背景', '裂缝'])

        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.title('教师模型混淆矩阵（像素级）', fontsize=14, pad=20)

        cm_path = os.path.join(self.vis_dir, 'confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"混淆矩阵热图已保存到: {cm_path}")

    def save_results(self):
        """保存评估结果到文件"""
        print("\n保存评估结果...")

        df = pd.DataFrame(self.evaluation_results['per_image_results'])
        csv_path = os.path.join(self.output_dir, 'detailed_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"详细结果已保存到: {csv_path}")

        summary = {
            'model_info': {
                'name': 'DeepLabV3+ (ResNet101)',
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'input_size': self.config.get('image_size', 512)
            },
            'dataset_info': {
                'test_set_size': len(self.test_dataset),
                'data_root': self.config['data_root']
            },
            'evaluation_metrics': self.evaluation_results['test_metrics'],
            'inference_performance': self.evaluation_results['inference_time']
        }

        if self.evaluation_results['confusion_matrix'] is not None:
            summary['confusion_matrix'] = self.evaluation_results['confusion_matrix'].tolist()

        json_path = os.path.join(self.output_dir, 'evaluation_summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"评估摘要已保存到: {json_path}")

    def visualize_results(self, metrics_accumulator, sample_predictions):
        """生成所有可视化结果"""
        print("\n生成可视化结果...")

        self.plot_metrics_distribution(metrics_accumulator)
        self.plot_metrics_boxplot(metrics_accumulator)
        self.plot_metrics_correlation(metrics_accumulator)

        if sample_predictions:
            self.visualize_prediction_samples(sample_predictions)

        self.plot_metrics_radar()

        print("所有可视化结果已生成!")

    def plot_metrics_distribution(self, metrics_accumulator):
        """绘制指标分布直方图"""
        ensure_chinese_font()
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        metrics_to_plot = ['iou', 'dice', 'accuracy', 'precision', 'recall', 'f1']
        titles = ['IoU分布', 'Dice系数分布', '准确率分布',
                  '精确率分布', '召回率分布', 'F1分数分布']
        colors = ['skyblue', 'lightgreen', 'lightcoral',
                  'gold', 'violet', 'orange']

        for idx, (metric, title, color) in enumerate(zip(metrics_to_plot, titles, colors)):
            if metric in metrics_accumulator and len(metrics_accumulator[metric]) > 0:
                values = metrics_accumulator[metric]

                axes[idx].hist(values, bins=30, edgecolor='black', alpha=0.7, color=color)
                axes[idx].axvline(np.mean(values), color='red', linestyle='--', linewidth=2,
                                  label=f'均值: {np.mean(values):.4f}')
                axes[idx].axvline(np.median(values), color='green', linestyle=':', linewidth=2,
                                  label=f'中位数: {np.median(values):.4f}')

                axes[idx].set_xlabel(f'{metric.upper()} 值')
                axes[idx].set_ylabel('频数')
                axes[idx].set_title(title)
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)

        plt.suptitle('教师模型性能指标分布', fontsize=16, y=1.02)
        plt.tight_layout()

        dist_path = os.path.join(self.vis_dir, 'metrics_distribution.png')
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"指标分布图已保存到: {dist_path}")

    def plot_metrics_boxplot(self, metrics_accumulator):
        """绘制指标箱线图"""
        ensure_chinese_font()
        fig, ax = plt.subplots(figsize=(12, 6))

        metrics_to_plot = ['iou', 'dice', 'accuracy', 'precision', 'recall', 'f1']
        data = []
        labels = []

        for metric in metrics_to_plot:
            if metric in metrics_accumulator and len(metrics_accumulator[metric]) > 0:
                data.append(metrics_accumulator[metric])
                labels.append(metric.upper())

        if data:
            box = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)

            colors = ['lightblue', 'lightgreen', 'lightcoral',
                      'lightyellow', 'lightpink', 'lightgray']
            for patch, color in zip(box['boxes'], colors[:len(data)]):
                patch.set_facecolor(color)

            ax.set_ylabel('指标值')
            ax.set_title('教师模型性能指标箱线图', fontsize=14, pad=20)
            ax.grid(True, alpha=0.3)

            means = [np.mean(values) for values in data]
            for i, mean in enumerate(means):
                ax.text(i + 1, mean, f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

        box_path = os.path.join(self.vis_dir, 'metrics_boxplot.png')
        plt.tight_layout()
        plt.savefig(box_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"指标箱线图已保存到: {box_path}")

    def plot_metrics_correlation(self, metrics_accumulator):
        """绘制指标相关性热图"""
        ensure_chinese_font()
        metrics_df = pd.DataFrame(metrics_accumulator)
        correlation_matrix = metrics_df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                    square=True, cbar_kws={"shrink": 0.8})
        plt.title('教师模型性能指标相关性热图', fontsize=14, pad=20)

        corr_path = os.path.join(self.vis_dir, 'metrics_correlation.png')
        plt.tight_layout()
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"指标相关性热图已保存到: {corr_path}")

    def visualize_prediction_samples(self, sample_predictions):
        """可视化预测样本"""
        num_samples = min(4, len(sample_predictions))

        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for row_idx, sample in enumerate(sample_predictions[:num_samples]):
            image = sample['image'].permute(1, 2, 0).numpy()

            mask = sample['mask']
            if mask.dim() == 3:
                if mask.shape[0] == 3:
                    mask = mask.mean(dim=0)
                else:
                    mask = mask.squeeze(0)
            mask_np = mask.numpy()

            pred = sample['pred']
            if pred.dim() == 3:
                if pred.shape[0] == 3:
                    pred = pred.mean(dim=0)
                else:
                    pred = pred.squeeze(0)
            pred_np = pred.numpy()
            pred_binary = (pred_np > 0.5).astype(np.float32)

            mask_np = (mask_np > 0.5).astype(np.float32)

            axes[row_idx, 0].imshow(image)
            axes[row_idx, 0].set_title(f'原始图像\n{sample["name"]}')
            axes[row_idx, 0].axis('off')

            axes[row_idx, 1].imshow(mask_np, cmap='gray')
            axes[row_idx, 1].set_title('真实掩码')
            axes[row_idx, 1].axis('off')

            axes[row_idx, 2].imshow(pred_binary, cmap='gray')
            axes[row_idx, 2].set_title(f'预测掩码\nIoU: {sample["metrics"]["iou"]:.3f}')
            axes[row_idx, 2].axis('off')

            overlay = image.copy()
            height, width = mask_np.shape[:2]

            if mask_np.shape != (height, width):
                mask_np = mask_np.reshape(height, width)
            if pred_binary.shape != (height, width):
                pred_binary = pred_binary.reshape(height, width)

            tp_mask = (pred_binary > 0.5) & (mask_np > 0.5)
            if tp_mask.any():
                overlay[tp_mask] = self.colors['tp']

            fp_mask = (pred_binary > 0.5) & (mask_np < 0.5)
            if fp_mask.any():
                overlay[fp_mask] = self.colors['fp']

            fn_mask = (pred_binary < 0.5) & (mask_np > 0.5)
            if fn_mask.any():
                overlay[fn_mask] = self.colors['fn']

            axes[row_idx, 3].imshow(overlay)
            axes[row_idx, 3].set_title('预测比较\n(绿:正确, 红:误检, 蓝:漏检)')
            axes[row_idx, 3].axis('off')

        plt.suptitle('教师模型预测结果可视化', fontsize=16, y=1.02)
        plt.tight_layout()

        vis_path = os.path.join(self.vis_dir, 'prediction_samples.png')
        plt.savefig(vis_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"预测样本可视化已保存到: {vis_path}")

    def plot_metrics_radar(self):
        """绘制性能雷达图"""
        ensure_chinese_font()
        metrics = self.evaluation_results['test_metrics']

        radar_metrics = ['iou', 'dice', 'accuracy', 'precision', 'recall', 'f1']
        labels = [m.upper() for m in radar_metrics]

        values = []
        for m in radar_metrics:
            if m in metrics:
                values.append(metrics[m]['mean'])
            else:
                values.append(0)

        if len(values) > 0:
            values = values + [values[0]]
            labels = labels + [labels[0]]

            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=True)

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

            ax.plot(angles, values, 'o-', linewidth=2, color='royalblue', label='教师模型')
            ax.fill(angles, values, alpha=0.25, color='royalblue')

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels[:-1], fontsize=11)

            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)

            ax.set_title('教师模型性能雷达图', fontsize=14, pad=20)
            ax.grid(True)

            radar_path = os.path.join(self.vis_dir, 'performance_radar.png')
            plt.tight_layout()
            plt.savefig(radar_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"性能雷达图已保存到: {radar_path}")

    def generate_report(self):
        """生成性能分析报告"""
        report_path = os.path.join(self.output_dir, 'performance_report.md')

        metrics = self.evaluation_results['test_metrics']

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 教师模型性能分析报告\n")

            f.write("## 1. 评估概要\n")
            f.write(f"- **模型**: DeepLabV3+ with ResNet101\n")
            f.write(f"- **测试集大小**: {len(self.test_dataset)} 张图像\n")
            f.write(f"- **评估设备**: {self.device}\n\n")

            f.write("## 2. 性能指标\n")
            f.write("| 指标 | 均值 | 标准差 | 最小值 | 最大值 | 中位数 |\n")
            f.write("|------|------|--------|--------|--------|--------|\n")
            for metric_name in ['iou', 'dice', 'accuracy', 'precision', 'recall', 'f1']:
                if metric_name in metrics:
                    stats = metrics[metric_name]
                    f.write(f"| {metric_name.upper()} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                            f"{stats['min']:.4f} | {stats['max']:.4f} | {stats['median']:.4f} |\n")

            f.write("\n## 3. 推理性能\n")
            infer = self.evaluation_results['inference_time']
            f.write(f"- **总推理时间**: {infer['total_seconds']:.2f} 秒\n")
            f.write(f"- **单张图像平均推理时间**: {infer['avg_per_image']:.4f} 秒\n")
            f.write(f"- **推理速度**: {infer['fps']:.2f} FPS\n\n")

            f.write("## 4. 可视化结果\n")
            f.write("生成的可视化文件包括:\n")
            f.write("1. `metrics_distribution.png` - 指标分布直方图\n")
            f.write("2. `metrics_boxplot.png` - 指标箱线图\n")
            f.write("3. `metrics_correlation.png` - 指标相关性热图\n")
            f.write("4. `confusion_matrix.png` - 混淆矩阵热图\n")
            f.write("5. `prediction_samples.png` - 预测样本可视化\n")
            f.write("6. `performance_radar.png` - 性能雷达图\n\n")

            f.write("## 5. 结论与建议\n")
            f.write("### 优势:\n")
            f.write("1. 高精度分割性能\n")
            f.write("2. 稳定的裂缝检测能力\n")
            f.write("3. 强大的特征提取能力\n\n")

            f.write("### 改进方向:\n")
            f.write("1. 考虑模型轻量化以适应边缘部署\n")
            f.write("2. 优化推理速度\n")
            f.write("3. 增强对细小裂缝的检测能力\n")

        print(f"性能分析报告已保存到: {report_path}")


def main():
    """主函数（用于教师模型评估）"""
    config = Config()

    evaluator = TeacherModelEvaluator(config)

    print("\n" + "=" * 60)
    print("开始教师模型基准性能评估")
    print("=" * 60)

    avg_metrics = evaluator.evaluate_test_set()

    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)
    print(f"所有结果已保存到: {evaluator.output_dir}")
    print(f"可视化结果: {evaluator.vis_dir}")

    if avg_metrics:
        print("\n关键指标:")
        print(f"- mIoU: {avg_metrics.get('iou', {}).get('mean', 0):.4f}")
        print(f"- Dice系数: {avg_metrics.get('dice', {}).get('mean', 0):.4f}")
        print(f"- F1分数: {avg_metrics.get('f1', {}).get('mean', 0):.4f}")
        if 'inference_time' in evaluator.evaluation_results:
            print(f"- 推理速度: {evaluator.evaluation_results['inference_time']['fps']:.2f} FPS")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()