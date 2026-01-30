# plot_history.py
import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import Config
from chinese_test import setup_matplotlib

setup_matplotlib()


def load_checkpoint_safely(checkpoint_path, device='cpu'):
    """安全加载检查点（兼容PyTorch 2.6+）"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

    try:
        from torch.serialization import add_safe_globals
        import numpy
        add_safe_globals([numpy.dtype, numpy._core.multiarray.scalar, numpy.ndarray])

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        print(f"✓ 使用安全模式加载: {checkpoint_path}")
        return checkpoint
    except:
        print("⚠ 使用兼容模式加载（信任此文件）")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        return checkpoint


def extract_train_history(checkpoint):
    """从检查点中提取训练历史"""
    if 'train_history' not in checkpoint:
        raise ValueError("检查点中未找到训练历史！")

    history = checkpoint['train_history']

    for key in history.keys():
        if not isinstance(history[key], list):
            if isinstance(history[key], (int, float)):
                history[key] = [history[key]]
            else:
                history[key] = []

    required_keys = ['train_loss', 'val_loss', 'train_iou', 'val_iou']
    for key in required_keys:
        if key not in history:
            raise ValueError(f"训练历史缺少必要键: {key}")

    train_len = len(history['train_loss'])
    val_len = len(history['val_loss'])

    if train_len == 0:
        raise ValueError("训练历史为空！")

    if train_len != val_len:
        min_len = min(train_len, val_len)
        for key in required_keys:
            if len(history[key]) > min_len:
                history[key] = history[key][:min_len]

    optional_keys = ['train_precision', 'val_precision', 'train_recall', 'val_recall',
                     'response_loss', 'feature_loss', 'edge_loss', 'contrast_loss', 'task_loss']

    for key in optional_keys:
        if key not in history:
            history[key] = [0.0] * min_len

    print(f"✓ 成功提取训练历史: {min(train_len, val_len)} 轮")
    return history, min(train_len, val_len)


def generate_simulated_lr_history(num_epochs, initial_lr=5e-4, min_lr=1e-6):
    """生成模拟的学习率曲线（余弦退火）"""
    lr_history = []
    for epoch in range(num_epochs):
        if num_epochs > 1:
            lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * epoch / (num_epochs - 1)))
        else:
            lr = initial_lr
        lr_history.append(lr)
    return lr_history


def plot_training_history(history, output_dir, checkpoint_name, config=None):
    """绘制训练历史（增强版）"""
    if not history or not history.get('train_loss'):
        raise ValueError("训练历史为空，无法绘图")

    print("正在生成训练历史图...")
    os.makedirs(output_dir, exist_ok=True)

    num_epochs = len(history['train_loss'])
    epochs = range(1, num_epochs + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)

    if config:
        checkpoint_path = config.get('checkpoint_path')
        title = f"Training History - {Path(checkpoint_path).stem}" if checkpoint_path else f'Training History - {checkpoint_name}'
    else:
        title = f'Training History - {checkpoint_name}'

    fig.suptitle(title, fontsize=16, fontweight='bold')

    if history['train_loss'] and history['val_loss']:
        axes[0, 0].plot(epochs, history['train_loss'], label='Training Loss', linewidth=2.5,
                        marker='o', markersize=4, alpha=0.8)
        axes[0, 0].plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2.5,
                        marker='s', markersize=4, alpha=0.8)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Total Loss Curve', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')

    if history['train_iou'] and history['val_iou']:
        axes[0, 1].plot(epochs, history['train_iou'], label='Training IoU', linewidth=2.5,
                        marker='o', markersize=4, alpha=0.8)
        axes[0, 1].plot(epochs, history['val_iou'], label='Validation IoU', linewidth=2.5,
                        marker='s', markersize=4, alpha=0.8)

        if history['val_iou']:
            best_iou = max(history['val_iou'])
            best_epoch = history['val_iou'].index(best_iou) + 1
            axes[0, 1].scatter([best_epoch], [best_iou], color='red', s=100,
                               zorder=5, label=f'Best IoU: {best_iou:.3f}')
            axes[0, 1].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)

        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('IoU', fontsize=12)
        axes[0, 1].set_title('IoU Curve', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')

    if history['train_precision'] and history['val_precision']:
        axes[0, 2].plot(epochs, history['train_precision'], label='Training Precision',
                        linewidth=2.5, marker='o', markersize=4, alpha=0.8)
        axes[0, 2].plot(epochs, history['val_precision'], label='Validation Precision',
                        linewidth=2.5, marker='s', markersize=4, alpha=0.8)
        axes[0, 2].plot(epochs, history['train_recall'], label='Training Recall',
                        linewidth=2.5, marker='^', markersize=4, alpha=0.8, linestyle='--')
        axes[0, 2].plot(epochs, history['val_recall'], label='Validation Recall',
                        linewidth=2.5, marker='v', markersize=4, alpha=0.8, linestyle='--')
        axes[0, 2].set_xlabel('Epoch', fontsize=12)
        axes[0, 2].set_ylabel('Score', fontsize=12)
        axes[0, 2].set_title('Precision & Recall', fontsize=14, fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3, linestyle='--')

    if history['task_loss']:
        axes[1, 0].plot(epochs, history['task_loss'], label='Task Loss', linewidth=2.5,
                        marker='o', markersize=4, alpha=0.8)
        axes[1, 0].plot(epochs, history['response_loss'], label='Response Distill',
                        linewidth=2, marker='s', markersize=3, alpha=0.7)
        axes[1, 0].plot(epochs, history['feature_loss'], label='Feature Distill',
                        linewidth=2, marker='^', markersize=3, alpha=0.7)
        axes[1, 0].plot(epochs, history['edge_loss'], label='Edge Distill',
                        linewidth=2, marker='v', markersize=3, alpha=0.7)
        axes[1, 0].plot(epochs, history['contrast_loss'], label='Contrast Distill',
                        linewidth=2, marker='*', markersize=3, alpha=0.7)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Loss', fontsize=12)
        axes[1, 0].set_title('Loss Components', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3, linestyle='--')

    if history['val_precision'] and history['val_recall']:
        scatter = axes[1, 1].scatter(history['val_precision'], history['val_recall'],
                                     c=range(len(history['val_precision'])),
                                     cmap='viridis', s=60, alpha=0.7,
                                     edgecolors='black', linewidth=0.5)
        axes[1, 1].set_xlabel('Validation Precision', fontsize=12)
        axes[1, 1].set_ylabel('Validation Recall', fontsize=12)
        axes[1, 1].set_title('Precision-Recall Scatter', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, linestyle='--')
        plt.colorbar(scatter, ax=axes[1, 1], label='Epoch')

    initial_lr = config.get('learning_rate', 5e-4) if config else 5e-4
    min_lr = config.get('min_lr', 1e-6) if config else 1e-6

    axes[1, 2].plot(epochs, generate_simulated_lr_history(num_epochs, initial_lr, min_lr),
                    linewidth=2.5, color='darkorange', marker='o', markersize=4, alpha=0.8)
    axes[1, 2].set_xlabel('Epoch', fontsize=12)
    axes[1, 2].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 2].set_title('Simulated LR Schedule (Cosine Annealing)', fontsize=14, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3, linestyle='--')
    axes[1, 2].set_yscale('log')
    axes[1, 2].text(0.05, 0.95, f'Initial LR: {initial_lr:.1e}\nMin LR: {min_lr:.1e}',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_filename = f'training_history_{checkpoint_name}.png'
    output_path = os.path.join(output_dir, output_filename)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.2, facecolor='white')

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"✓ 图像已成功保存: {os.path.abspath(output_path)}")
        print(f"  文件大小: {os.path.getsize(output_path) / 1024:.2f} KB")
    else:
        print(f"✗ 错误：图像文件未成功创建！")

    plt.close(fig)

    history_json_path = os.path.join(output_dir, f'training_history_{checkpoint_name}.json')
    with open(history_json_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"✓ 训练历史已保存为JSON: {history_json_path}")


def plot_from_config(config):
    """从配置字典绘制训练历史"""
    if isinstance(config, dict):
        config_obj = Config()
        for key, value in config.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
        config = config_obj

    checkpoint_path = config.get('checkpoint_path',
                                 default='../knowledge_distillation_results/checkpoint_epoch30.pth')
    output_dir = config.get('plot_history', os.path.dirname(checkpoint_path))

    if not checkpoint_path:
        raise ValueError("配置中必须包含 'checkpoint_path'")

    if not os.path.exists(checkpoint_path):
        print(f"✗ 错误: 检查点文件不存在: {checkpoint_path}")
        return

    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    print("训练历史可视化工具")
    print(f"检查点路径: {checkpoint_path}")
    print(f"输出目录: {os.path.abspath(output_dir)}")
    print(f"检查点名称: {checkpoint_name}")

    try:
        device = config.get('device', 'cpu')
        checkpoint = load_checkpoint_safely(checkpoint_path, device=device)

        history, num_epochs = extract_train_history(checkpoint)

        print("\n训练历史摘要:")
        print(f"  总训练轮次: {num_epochs}")

        if history.get('val_iou'):
            best_iou = max(history['val_iou'])
            best_epoch = history['val_iou'].index(best_iou) + 1
            print(f"  最佳验证IoU: {best_iou:.4f} (第 {best_epoch} 轮)")

        if history.get('val_precision') and history.get('val_recall'):
            best_precision = max(history['val_precision'])
            best_recall = max(history['val_recall'])
            print(f"  最佳验证精确率: {best_precision:.4f}")
            print(f"  最佳验证召回率: {best_recall:.4f}")

        if history.get('train_loss'):
            print(f"  最终训练损失: {history['train_loss'][-1]:.4f}")
        if history.get('val_loss'):
            print(f"  最终验证损失: {history['val_loss'][-1]:.4f}")

        if history.get('task_loss'):
            avg_task_loss = np.mean(history['task_loss']) if history['task_loss'] else 0
            avg_response_loss = np.mean(history['response_loss']) if history['response_loss'] else 0
            avg_feature_loss = np.mean(history['feature_loss']) if history['feature_loss'] else 0
            avg_edge_loss = np.mean(history['edge_loss']) if history['edge_loss'] else 0
            avg_contrast_loss = np.mean(history['contrast_loss']) if history['contrast_loss'] else 0
            print(f"\n平均损失分量:")
            print(f"  任务损失: {avg_task_loss:.4f}")
            print(f"  响应蒸馏: {avg_response_loss:.4f}")
            print(f"  特征蒸馏: {avg_feature_loss:.4f}")
            print(f"  边缘蒸馏: {avg_edge_loss:.4f}")
            print(f"  对比蒸馏: {avg_contrast_loss:.4f}")

        plot_training_history(history, output_dir, checkpoint_name, config)

        print("=" * 70)
        print("✓ 训练历史可视化完成！")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ 程序执行失败: {e}")


def main():
    """使用配置字典方式"""
    config = Config()

    parser = argparse.ArgumentParser(description='从检查点绘制训练历史')
    parser.add_argument('--checkpoint_path', type=str, help='检查点文件路径，覆盖配置文件中的设置')
    parser.add_argument('--output_dir', type=str, help='输出目录，覆盖配置文件中的设置')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='设备类型，覆盖配置文件中的设置')

    args = parser.parse_args()

    if args.checkpoint_path:
        config['checkpoint_path'] = args.checkpoint_path
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.device:
        config['device'] = args.device

    checkpoint_path = config.get('checkpoint_path',
                                 default='../knowledge_distillation_results/checkpoint_epoch30.pth')
    if not os.path.exists(checkpoint_path):
        print(f"错误: 检查点文件不存在: {checkpoint_path}")
        return

    plot_from_config(config)


if __name__ == "__main__":
    main()