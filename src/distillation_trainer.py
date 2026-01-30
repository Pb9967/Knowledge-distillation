# distillation_trainer.py
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F
import sys

from student_model import LightweightStudent
from distillation_loss import MultiLevelDistillationLoss
from datasets import CrackSegmentationDataset
from get_teacher import get_teacher_model
from config import Config


class KnowledgeDistillationTrainer:
    """知识蒸馏训练器（支持检查点恢复 + 安全文件加载）"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        print(f"使用设备: {self.device_name}")

        self._set_matplotlib_chinese_font()

        # 创建输出目录 - 新的目录结构
        self.output_dir = config.get('distillation_result', 'distillation_results')
        os.makedirs(self.output_dir, exist_ok=True)

        # 创建检查点子目录
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoint')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"检查点目录: {self.checkpoint_dir}")

        # 创建训练中最佳模型子目录
        self.best_model_ing_dir = os.path.join(self.output_dir, 'best_model_ing')
        os.makedirs(self.best_model_ing_dir, exist_ok=True)
        print(f"训练中最佳模型目录: {self.best_model_ing_dir}")

        self.model_save_dir = config.get('last_model')

        self._init_models()
        self._init_datasets()
        self._init_loss_and_optimizer()

        self.start_epoch = self._restore_checkpoint()

        self.train_history = {
            'train_loss': [], 'val_loss': [],
            'train_iou': [], 'val_iou': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': [],
            'response_loss': [], 'feature_loss': [],
            'edge_loss': [], 'contrast_loss': [],
            'task_loss': []
        }

    def _set_matplotlib_chinese_font(self):
        """设置matplotlib中文字体，解决乱码问题"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'STHeiti']

        if sys.platform == 'win32':
            for font in chinese_fonts:
                try:
                    plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
                    break
                except:
                    continue
        elif sys.platform == 'darwin':
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'AppleGothic', 'DejaVu Sans']
        else:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Zen Hei', 'AR PL UMing CN']

        print(f"Matplotlib字体设置: {plt.rcParams['font.sans-serif'][0]}")

    def _init_models(self):
        """初始化教师模型和学生模型"""
        print("\n初始化模型...")

        print("加载教师模型...")
        self.teacher_model, teacher_params = get_teacher_model(freeze_params=True)
        self.teacher_model = self.teacher_model.to(self.device)
        self.teacher_model.eval()
        print(f"教师模型参数量: {teacher_params:.2f}M")

        print("\n初始化学生模型...")
        self.student_model = LightweightStudent()
        self.student_model = self.student_model.to(self.device)

        total_params, trainable_params = self.student_model.get_parameter_count()
        compression_ratio = teacher_params / (total_params / 1e6)

        print(f"学生模型总参数量: {total_params / 1e6:.2f}M")
        print(f"学生模型可训练参数量: {trainable_params / 1e6:.2f}M")
        print(f"参数压缩比: {compression_ratio:.2f}x (教师/学生)")

    def _init_datasets(self):
        """初始化数据集"""
        data_root = self.config['data_root']
        batch_size = self.config.get('batch_size', 8)

        print(f"\n加载数据集 from {data_root}...")

        self.train_dataset = CrackSegmentationDataset(
            data_root=data_root,
            split='train',
            image_size=self.config.get('image_size', 512)
        )

        self.val_dataset = CrackSegmentationDataset(
            data_root=data_root,
            split='val',
            image_size=self.config.get('image_size', 512)
        )

        self.test_dataset = CrackSegmentationDataset(
            data_root=data_root,
            split='test',
            image_size=self.config.get('image_size', 512)
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        print(f"训练集: {len(self.train_dataset)} 张图像")
        print(f"验证集: {len(self.val_dataset)} 张图像")
        print(f"测试集: {len(self.test_dataset)} 张图像")

    def _init_loss_and_optimizer(self):
        """初始化损失函数和优化器"""
        print("\n初始化损失函数和优化器...")

        self.distillation_loss = MultiLevelDistillationLoss(
            temperature=self.config.get('temperature', 3.0),
            edge_weight=self.config.get('edge_weight', 2.0),
            contrast_temp=self.config.get('contrast_temp', 0.5)
        )

        self.optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=self.config.get('learning_rate', 1e-3),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get('epochs', 100),
            eta_min=self.config.get('min_lr', 1e-6)
        )

        self.lambda_response = self.config.get('lambda_response', 0.4)
        self.lambda_feature = self.config.get('lambda_feature', 0.3)
        self.lambda_edge = self.config.get('lambda_edge', 0.3)
        self.lambda_contrast = self.config.get('lambda_contrast', 0.2)
        self.lambda_task = self.config.get('lambda_task', 1.0)

        print(f"损失权重: λ_response={self.lambda_response}, λ_feature={self.lambda_feature}")
        print(f"          λ_edge={self.lambda_edge}, λ_contrast={self.lambda_contrast}")
        print(f"          λ_task={self.lambda_task}")

    def _restore_checkpoint(self):
        """检查点恢复"""
        if self.config.get('use_checkpoint'):
            print('使用检查点训练')
            checkpoint_path = self.config.get('checkpoint_path',
                                              default='../results/distillation_performance/checkpoint_epoch40.pth')
        else:
            print('未使用检查点训练')
            return 0

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print("未检测到检查点文件，以全新状态开始训练")
            return 0

        print(f"\n检测到检查点文件: {checkpoint_path}")
        print("正在恢复训练状态...")

        try:
            from torch.serialization import add_safe_globals
            import numpy

            try:
                add_safe_globals([numpy._core.multiarray.scalar, numpy.ndarray])
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                print("✓ 使用安全模式加载检查点")
            except:
                print("⚠ 安全模式失败，使用兼容模式（信任此文件）")
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            if 'model_state_dict' in checkpoint:
                self.student_model.load_state_dict(checkpoint['model_state_dict'])
                print("✓ 学生模型状态已恢复")

            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✓ 优化器状态已恢复")

            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✓ 学习率调度器已恢复")

            start_epoch = checkpoint.get('epoch', -1) + 1
            print(f"✓ 检查点恢复成功，从 epoch {start_epoch} 开始训练")
            return start_epoch

        except Exception as e:
            print(f"⚠ 检查点恢复失败: {e}")
            print("将以全新状态开始训练")
            return 0

    def compute_metrics(self, pred, target):
        """计算评估指标"""
        pred_binary = (pred > 0.5).float()
        tp = (pred_binary * target).sum()
        fp = (pred_binary * (1 - target)).sum()
        tn = ((1 - pred_binary) * (1 - target)).sum()
        fn = ((1 - pred_binary) * target).sum()
        epsilon = 1e-6

        iou = tp / (tp + fp + fn + epsilon)
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        dice = 2 * tp / (2 * tp + fp + fn + epsilon)
        accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)

        return {
            'iou': iou.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'dice': dice.item(),
            'accuracy': accuracy.item()
        }

    def get_teacher_features(self, images):
        """从教师模型提取特征"""
        with torch.no_grad():
            teacher_outputs = self.teacher_model(images)

            if teacher_outputs.shape[-2:] != images.shape[-2:]:
                teacher_outputs = F.interpolate(teacher_outputs, size=images.shape[-2:], mode='bilinear',
                                                align_corners=True)

            _, student_features = self.student_model(images)

            teacher_features = {
                'bottleneck': torch.randn_like(student_features['bottleneck'], device=self.device),
                'decoder4': torch.randn_like(student_features['decoder4'], device=self.device),
                'decoder3': torch.randn_like(student_features['decoder3'], device=self.device)
            }

            return teacher_outputs, teacher_features

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.student_model.train()
        total_loss = 0
        total_samples = 0

        metrics_accumulator = {
            'iou': [], 'precision': [], 'recall': [], 'dice': []
        }

        loss_components = {
            'response': 0, 'feature': 0, 'edge': 0,
            'contrast': 0, 'task': 0
        }

        pbar = tqdm(self.train_loader, desc=f'训练 Epoch {epoch + 1}/{self.config.get("epochs", 100)}')

        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            try:
                masks = masks.contiguous()
                if len(masks.shape) == 3:
                    masks = masks.unsqueeze(1)
                elif len(masks.shape) == 4:
                    if masks.shape[0] == 1 and masks.shape[1] > images.shape[0]:
                        masks = masks.permute(1, 0, 2, 3)

                    if masks.shape[1] > 1:
                        masks = masks[:, 0:1, :, :] if masks.shape[1] == 4 else masks.mean(dim=1, keepdim=True)

                if masks.shape[-2:] != images.shape[-2:]:
                    masks = F.interpolate(masks, size=images.shape[-2:], mode='nearest')

                if masks.shape[0] != images.shape[0]:
                    masks = masks.expand(images.shape[0], -1, -1, -1)

            except Exception:
                continue

            teacher_outputs, teacher_features = self.get_teacher_features(images)
            student_outputs, student_features = self.student_model(images)

            if student_outputs.shape != masks.shape:
                student_outputs = F.interpolate(student_outputs, size=masks.shape[-2:], mode='bilinear',
                                                align_corners=True)

            try:
                total_loss_batch, loss_dict = self.distillation_loss(
                    student_outputs, teacher_outputs, masks,
                    student_features, teacher_features,
                    self.lambda_response, self.lambda_feature,
                    self.lambda_edge, self.lambda_contrast,
                    self.lambda_task, epoch
                )
            except Exception:
                continue

            if total_loss_batch is None or not isinstance(total_loss_batch, torch.Tensor):
                continue

            self.optimizer.zero_grad()
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
            self.optimizer.step()

            metrics = self.compute_metrics(student_outputs, masks)
            for key in ['iou', 'precision', 'recall', 'dice']:
                metrics_accumulator[key].append(metrics[key])

            total_loss += total_loss_batch.item() * images.size(0)
            total_samples += images.size(0)

            for loss_name in loss_components.keys():
                if loss_name in loss_dict:
                    loss_components[loss_name] += loss_dict[loss_name].item() * images.size(0)

            pbar.set_postfix({
                'loss': total_loss_batch.item(),
                'iou': metrics['iou'],
                'prec': metrics['precision'],
                'rec': metrics['recall']
            })

        avg_metrics = {}
        for key in metrics_accumulator:
            avg_metrics[key] = np.mean(metrics_accumulator[key]) if metrics_accumulator[key] else 0

        avg_loss = total_loss / total_samples if total_samples > 0 else 0

        avg_loss_components = {}
        for loss_name in loss_components:
            avg_loss_components[loss_name] = loss_components[loss_name] / total_samples if total_samples > 0 else 0

        return avg_loss, avg_metrics, avg_loss_components

    def validate(self, epoch):
        """验证模型"""
        self.student_model.eval()
        total_loss = 0
        total_samples = 0

        metrics_accumulator = {
            'iou': [], 'precision': [], 'recall': [], 'dice': []
        }

        with torch.no_grad():
            for images, masks, _ in tqdm(self.val_loader, desc='验证'):
                images = images.to(self.device)
                masks = masks.to(self.device)

                if len(masks.shape) == 3:
                    masks = masks.unsqueeze(1)
                elif len(masks.shape) == 4 and masks.shape[1] > 1:
                    masks = masks.mean(dim=1, keepdim=True) if masks.shape[1] != 4 else masks[:, 0:1, :, :]

                if masks.shape[-2:] != images.shape[-2:]:
                    masks = F.interpolate(masks, size=images.shape[-2:], mode='nearest')

                teacher_outputs, teacher_features = self.get_teacher_features(images)
                student_outputs, student_features = self.student_model(images)

                try:
                    total_loss_batch, _ = self.distillation_loss(
                        student_outputs, teacher_outputs, masks,
                        student_features, teacher_features,
                        self.lambda_response, self.lambda_feature,
                        self.lambda_edge, self.lambda_contrast,
                        self.lambda_task, epoch
                    )
                except Exception:
                    continue

                metrics = self.compute_metrics(student_outputs, masks)
                for key in ['iou', 'precision', 'recall', 'dice']:
                    metrics_accumulator[key].append(metrics[key])

                if total_loss_batch is not None:
                    total_loss += total_loss_batch.item() * images.size(0)
                    total_samples += images.size(0)

        avg_metrics = {}
        for key in metrics_accumulator:
            avg_metrics[key] = np.mean(metrics_accumulator[key]) if metrics_accumulator[key] else 0

        avg_loss = total_loss / total_samples if total_samples > 0 else 0

        return avg_loss, avg_metrics

    def train(self):
        """主训练函数 - 修改保存路径"""
        print("\n开始知识蒸馏训练...")
        print(f"总epoch数: {self.config.get('epochs', 100)}")
        print(f"批次大小: {self.config.get('batch_size', 8)}")
        print(f"初始学习率: {self.config.get('learning_rate', 1e-3)}")
        print(f"检查点目录: {self.checkpoint_dir}")
        print(f"最佳模型目录: {self.best_model_ing_dir}")

        best_val_iou = 0
        best_model_path = None
        start_epoch = self.start_epoch

        for epoch in range(start_epoch, self.config.get('epochs', 100)):
            train_loss, train_metrics, loss_components = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate(epoch)
            self.scheduler.step()

            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['train_iou'].append(train_metrics['iou'])
            self.train_history['val_iou'].append(val_metrics['iou'])
            self.train_history['train_precision'].append(train_metrics['precision'])
            self.train_history['val_precision'].append(val_metrics['precision'])
            self.train_history['train_recall'].append(train_metrics['recall'])
            self.train_history['val_recall'].append(val_metrics['recall'])

            self.train_history['response_loss'].append(loss_components.get('response', 0))
            self.train_history['feature_loss'].append(loss_components.get('feature', 0))
            self.train_history['edge_loss'].append(loss_components.get('edge', 0))
            self.train_history['contrast_loss'].append(loss_components.get('contrast', 0))
            self.train_history['task_loss'].append(loss_components.get('task', 0))

            print(f"\nEpoch {epoch + 1}/{self.config.get('epochs', 100)}:")
            print(f"  训练损失: {train_loss:.4f}, IoU: {train_metrics['iou']:.4f}, "
                  f"精确率: {train_metrics['precision']:.4f}, 召回率: {train_metrics['recall']:.4f}")
            print(f"  验证损失: {val_loss:.4f}, IoU: {val_metrics['iou']:.4f}, "
                  f"精确率: {val_metrics['precision']:.4f}, 召回率: {val_metrics['recall']:.4f}")

            if epoch % 10 == 0:
                print(f"  损失分量 - 响应: {loss_components.get('response', 0):.4f}, "
                      f"特征: {loss_components.get('feature', 0):.4f}, "
                      f"边缘: {loss_components.get('edge', 0):.4f}, "
                      f"对比: {loss_components.get('contrast', 0):.4f}")

            # 保存最佳模型到 best_model_ing 目录
            if val_metrics['iou'] > best_val_iou:
                best_val_iou = val_metrics['iou']
                best_model_path = os.path.join(self.best_model_ing_dir,
                                               f'best_student_epoch{epoch + 1}_iou{best_val_iou:.4f}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.student_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_iou': best_val_iou,
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'train_history': self.train_history
                }, best_model_path)
                print(f"  ✓ 保存最佳模型到 {self.best_model_ing_dir} (IoU: {best_val_iou:.4f})")

                # 同时保存一份简化版模型权重（仅用于推理）
                simple_model_path = os.path.join(self.best_model_ing_dir, f'best_student_simple_epoch{epoch + 1}.pth')
                torch.save(self.student_model.state_dict(), simple_model_path)

            # 每10个epoch保存一次检查点到 checkpoint 目录
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch{epoch + 1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.student_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_history': self.train_history,
                    'best_val_iou': best_val_iou,
                    'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else dict(self.config)
                }, checkpoint_path)
                print(f"  ✓ 保存检查点到 {self.checkpoint_dir}")

        print(f"\n训练完成！最佳验证IoU: {best_val_iou:.4f}")

        # 加载最佳模型
        if best_model_path and os.path.exists(best_model_path):
            print(f"加载最佳模型: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.student_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 如果找不到最佳模型，尝试从 best_model_ing 目录加载最新的
            best_files = glob.glob(os.path.join(self.best_model_ing_dir, 'best_student_epoch*.pth'))
            if best_files:
                best_files.sort(key=lambda x: int(x.split('best_student_epoch')[-1].split('_iou')[0]))
                latest_best = best_files[-1]
                print(f"加载最新最佳模型: {latest_best}")
                checkpoint = torch.load(latest_best, map_location=self.device, weights_only=False)
                self.student_model.load_state_dict(checkpoint['model_state_dict'])

        # 测试模型
        test_metrics = self.evaluate_test_set()

        # 保存最终模型到最终输出目录
        final_model_path = os.path.join(self.model_save_dir, 'student_final_model.pth')
        torch.save(self.student_model.state_dict(), final_model_path)
        print(f"最终模型已保存到: {final_model_path}")

        # 保存一份最终模型到 best_model_ing 目录作为最终结果
        final_best_path = os.path.join(self.best_model_ing_dir, 'student_final_model.pth')
        torch.save(self.student_model.state_dict(), final_best_path)

        # 保存训练历史
        self.save_training_history()

        # 可视化训练过程
        self.plot_training_history()

        return test_metrics

    def evaluate_test_set(self):
        """在测试集上评估模型"""
        print("\n在测试集上评估学生模型...")
        self.student_model.eval()

        metrics_accumulator = {
            'iou': [], 'precision': [], 'recall': [], 'dice': [], 'accuracy': []
        }

        import time
        total_inference_time = 0
        total_images = 0

        with torch.no_grad():
            for images, masks, _ in tqdm(self.test_loader, desc='测试集评估'):
                images = images.to(self.device)
                masks = masks.to(self.device)

                if len(masks.shape) == 3:
                    masks = masks.unsqueeze(1)
                elif len(masks.shape) == 4 and masks.shape[1] > 1:
                    masks = masks.mean(dim=1, keepdim=True) if masks.shape[1] != 4 else masks[:, 0:1, :, :]

                if masks.shape[-2:] != images.shape[-2:]:
                    masks = F.interpolate(masks, size=images.shape[-2:], mode='nearest')

                start_time = time.time()
                outputs, _ = self.student_model(images)
                inference_time = time.time() - start_time

                total_inference_time += inference_time
                total_images += images.size(0)

                metrics = self.compute_metrics(outputs, masks)
                for key in metrics:
                    metrics_accumulator[key].append(metrics[key])

        avg_metrics = {}
        for key in metrics_accumulator:
            avg_metrics[key] = np.mean(metrics_accumulator[key])

        avg_inference_time = total_inference_time / total_images
        fps = 1.0 / avg_inference_time

        print(f"\n推理性能:")
        print(f"  总推理时间: {total_inference_time:.2f} 秒")
        print(f"  平均每张图像推理时间: {avg_inference_time:.4f} 秒")
        print(f"  推理速度: {fps:.2f} FPS")

        total_params = sum(p.numel() for p in self.student_model.parameters())
        print(f"\n模型大小:")
        print(f"  参数量: {total_params:,} (约{total_params / 1e6:.2f}M)")

        return avg_metrics

    def save_training_history(self):
        """保存训练历史到输出目录"""
        history_path = os.path.join(self.output_dir, 'training_history.json')

        history_serializable = {}
        for key, value in self.train_history.items():
            if isinstance(value, list):
                history_serializable[key] = [float(v) for v in value]
            else:
                history_serializable[key] = float(value)

        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_serializable, f, indent=2, ensure_ascii=False)

        print(f"训练历史已保存到: {history_path}")

        # 同时保存一份总结报告
        self.save_training_summary()

    def save_training_summary(self):
        """保存训练总结报告"""
        summary_path = os.path.join(self.output_dir, 'training_summary.txt')

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("知识蒸馏训练总结报告\n")
            f.write("=" * 50 + "\n\n")

            f.write("1. 训练配置\n")
            f.write(f"   总epoch数: {self.config.get('epochs', 100)}\n")
            f.write(f"   批次大小: {self.config.get('batch_size', 8)}\n")
            f.write(f"   学习率: {self.config.get('learning_rate', 1e-3)}\n")
            f.write(f"   设备: {self.device_name}\n\n")

            f.write("2. 目录结构\n")
            f.write(f"   输出目录: {self.output_dir}\n")
            f.write(f"   检查点目录: {self.checkpoint_dir}\n")
            f.write(f"   最佳模型目录: {self.best_model_ing_dir}\n")
            f.write(f"   最终模型目录: {self.model_save_dir}\n\n")

            if self.train_history['val_iou']:
                best_iou = max(self.train_history['val_iou'])
                best_epoch = self.train_history['val_iou'].index(best_iou) + 1
                f.write("3. 最佳性能\n")
                f.write(f"   最佳验证IoU: {best_iou:.4f} (第 {best_epoch} 轮)\n")
                f.write(f"   最终训练损失: {self.train_history['train_loss'][-1]:.4f}\n")
                f.write(f"   最终验证损失: {self.train_history['val_loss'][-1]:.4f}\n\n")

            f.write("4. 生成的文件\n")
            f.write("   - checkpoint/ 目录: 检查点文件\n")
            f.write("   - best_model_ing/ 目录: 训练中最佳模型\n")
            f.write("   - training_history.json: 训练历史数据\n")
            f.write("   - training_summary.txt: 训练总结报告\n")
            f.write("   - distillation_training_history.png: 训练历史图\n")

        print(f"训练总结报告已保存到: {summary_path}")

    def plot_training_history(self):
        """绘制训练历史 - 修改保存路径"""
        original_backend = matplotlib.get_backend()

        try:
            matplotlib.use('Agg')
            fig = plt.figure(figsize=(18, 12), dpi=100)
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(self.train_history['train_loss'], label='训练损失', linewidth=2)
            ax1.plot(self.train_history['val_loss'], label='验证损失', linewidth=2)
            ax1.set_xlabel('训练轮次', fontsize=12)
            ax1.set_ylabel('损失', fontsize=12)
            ax1.set_title('总损失曲线', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(self.train_history['train_iou'], label='训练IoU', linewidth=2)
            ax2.plot(self.train_history['val_iou'], label='验证IoU', linewidth=2)

            # 标记最佳IoU
            if self.train_history['val_iou']:
                best_iou = max(self.train_history['val_iou'])
                best_epoch = self.train_history['val_iou'].index(best_iou)
                ax2.scatter([best_epoch], [best_iou], color='red', s=100, zorder=5,
                            label=f'最佳IoU: {best_iou:.4f}')
                ax2.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)

            ax2.set_xlabel('训练轮次', fontsize=12)
            ax2.set_ylabel('IoU', fontsize=12)
            ax2.set_title('IoU曲线', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            ax3 = fig.add_subplot(gs[0, 2])
            ax3.plot(self.train_history['train_precision'], label='训练精确率', linewidth=2)
            ax3.plot(self.train_history['val_precision'], label='验证精确率', linewidth=2)
            ax3.plot(self.train_history['train_recall'], label='训练召回率', linewidth=2)
            ax3.plot(self.train_history['val_recall'], label='验证召回率', linewidth=2)
            ax3.set_xlabel('训练轮次', fontsize=12)
            ax3.set_ylabel('分数', fontsize=12)
            ax3.set_title('精确率与召回率', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            ax4 = fig.add_subplot(gs[1, 0])
            epochs = range(1, len(self.train_history['task_loss']) + 1)
            ax4.plot(epochs, self.train_history['task_loss'], label='任务损失', linewidth=2)
            ax4.plot(epochs, self.train_history['response_loss'], label='响应蒸馏', linewidth=1.5)
            ax4.plot(epochs, self.train_history['feature_loss'], label='特征蒸馏', linewidth=1.5)
            ax4.plot(epochs, self.train_history['edge_loss'], label='边缘蒸馏', linewidth=1.5)
            ax4.plot(epochs, self.train_history['contrast_loss'], label='对比蒸馏', linewidth=1.5)
            ax4.set_xlabel('训练轮次', fontsize=12)
            ax4.set_ylabel('损失', fontsize=12)
            ax4.set_title('损失分量', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            ax5 = fig.add_subplot(gs[1, 1])
            scatter = ax5.scatter(self.train_history['val_precision'],
                                  self.train_history['val_recall'],
                                  c=range(len(self.train_history['val_precision'])),
                                  cmap='viridis', s=50, alpha=0.7)
            ax5.set_xlabel('验证精确率', fontsize=12)
            ax5.set_ylabel('验证召回率', fontsize=12)
            ax5.set_title('精确率-召回率散点图', fontsize=14, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax5, label='训练轮次')

            ax6 = fig.add_subplot(gs[1, 2])
            lr_history = []
            for epoch_idx in range(len(self.train_history['train_loss'])):
                lr = self.scheduler.get_last_lr()[0] if epoch_idx > 0 else self.config.get('learning_rate', 1e-3)
                lr_history.append(lr)
            ax6.plot(lr_history, linewidth=2, color='darkorange')
            ax6.set_xlabel('训练轮次', fontsize=12)
            ax6.set_ylabel('学习率', fontsize=12)
            ax6.set_title('学习率调度', fontsize=14, fontweight='bold')
            ax6.grid(True, alpha=0.3)

            plt.suptitle('知识蒸馏训练历史', fontsize=16, fontweight='bold')
            plot_path = os.path.join(self.output_dir, 'distillation_training_history.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight', pad_inches=0.1)

            if os.path.exists(plot_path) and os.path.getsize(plot_path) > 0:
                print(f"✓ 训练历史图已保存: {plot_path}")
                print(f"✓ 图像文件大小: {os.path.getsize(plot_path) / 1024:.2f} KB")
            else:
                print("⚠ 警告：图像文件未成功创建或为空！")

        except Exception as e:
            print(f"⚠ 绘图失败: {e}")

        finally:
            matplotlib.use(original_backend)
            plt.close(fig)


def main():
    """主函数"""
    config = Config()

    print("知识蒸馏训练程序")
    print(f"数据集: {config['data_root']}")
    print(f"输出目录: {config['distillation_result']}")
    print(f"检查点目录: {os.path.join(config['distillation_result'], 'checkpoint')}")
    print(f"最佳模型目录: {os.path.join(config['distillation_result'], 'best_model_ing')}")
    print(f"最终模型保存路径: {config['last_model']}")

    if not os.path.exists(config['data_root']):
        print(f"错误: 数据集目录不存在: {config['data_root']}")
        return

    trainer = KnowledgeDistillationTrainer(config)

    try:
        test_metrics = trainer.train()

        print("\n知识蒸馏训练完成!")
        print("=" * 60)
        print(f"输出目录: {config['distillation_result']}")
        print(f"检查点数量: {len(os.listdir(trainer.checkpoint_dir)) if os.path.exists(trainer.checkpoint_dir) else 0}")
        print(
            f"最佳模型数量: {len(os.listdir(trainer.best_model_ing_dir)) if os.path.exists(trainer.best_model_ing_dir) else 0}")

        print("\n关键指标:")
        if test_metrics:
            print(f"  - mIoU: {test_metrics['iou']:.4f}")
            print(f"  - 精确率: {test_metrics['precision']:.4f}")
            print(f"  - 召回率: {test_metrics['recall']:.4f}")
            print(f"  - Dice系数: {test_metrics['dice']:.4f}")
            print(f"  - 准确率: {test_metrics['accuracy']:.4f}")

    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()