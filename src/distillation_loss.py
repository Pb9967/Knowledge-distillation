# distillation_loss.py - 知识蒸馏损失函数定义
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResponseDistillationLoss(nn.Module):
    """输出响应蒸馏损失 - KL散度"""

    def __init__(self, temperature=3.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits):
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        return loss


class FeatureMapDistillationLoss(nn.Module):
    """特征图蒸馏损失 - L2对齐"""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, student_features, teacher_features, adaptor=None):
        if adaptor is not None:
            student_features = adaptor(student_features)

        if student_features.shape[-2:] != teacher_features.shape[-2:]:
            student_features = F.interpolate(
                student_features,
                size=teacher_features.shape[-2:],
                mode='bilinear',
                align_corners=True
            )

        loss = F.mse_loss(student_features, teacher_features, reduction=self.reduction)
        return loss


class EdgeAwareDistillationLoss(nn.Module):
    """边缘感知注意力蒸馏损失"""

    def __init__(self, edge_weight=2.0):
        super().__init__()
        self.edge_weight = edge_weight

    def generate_edge_mask(self, gt_mask, kernel_size=3):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=gt_mask.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=gt_mask.device).view(1, 1, 3, 3)

        grad_x = F.conv2d(gt_mask, sobel_x, padding=1)
        grad_y = F.conv2d(gt_mask, sobel_y, padding=1)
        edge_mask = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        edge_mask = (edge_mask - edge_mask.min()) / (edge_mask.max() - edge_mask.min() + 1e-8)
        edge_mask = (edge_mask > 0.1).float()
        return edge_mask

    def generate_attention_map(self, features):
        attention_map = torch.norm(features, p=2, dim=1, keepdim=True)
        min_val, max_val = attention_map.min(), attention_map.max()
        if max_val - min_val > 1e-8:
            attention_map = (attention_map - min_val) / (max_val - min_val)
        else:
            attention_map = torch.zeros_like(attention_map)
        return attention_map

    def forward(self, student_features, teacher_features, gt_mask=None):
        student_attention = self.generate_attention_map(student_features)
        teacher_attention = self.generate_attention_map(teacher_features)

        if student_attention.shape[-2:] != teacher_attention.shape[-2:]:
            if student_attention.shape[2] < teacher_attention.shape[2]:
                student_attention = F.interpolate(
                    student_attention,
                    size=teacher_attention.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )
            else:
                teacher_attention = F.interpolate(
                    teacher_attention,
                    size=student_attention.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )

        if gt_mask is not None:
            edge_mask = self.generate_edge_mask(gt_mask)
            if edge_mask.shape[-2:] != teacher_attention.shape[-2:]:
                edge_mask = F.interpolate(edge_mask, size=teacher_attention.shape[-2:], mode='nearest')
            edge_enhanced = 1.0 + self.edge_weight * edge_mask
            teacher_attention = teacher_attention * edge_enhanced

        loss = F.mse_loss(student_attention, teacher_attention)
        return loss


class ContrastiveDistillationLoss(nn.Module):
    """对比蒸馏损失"""

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_features, teacher_features, gt_mask=None):
        B, C, H, W = student_features.shape
        student_features_flat = student_features.view(B, C, -1).permute(0, 2, 1)
        teacher_features_flat = teacher_features.view(B, C, -1).permute(0, 2, 1)

        student_features_norm = F.normalize(student_features_flat, dim=-1)
        teacher_features_norm = F.normalize(teacher_features_flat, dim=-1)

        similarity = torch.bmm(student_features_norm, teacher_features_norm.transpose(1, 2))
        similarity = similarity / self.temperature

        labels = torch.arange(H * W, device=student_features.device).unsqueeze(0).repeat(B, 1)
        loss = F.cross_entropy(similarity.view(-1, H * W), labels.view(-1))

        return loss


class MultiLevelDistillationLoss(nn.Module):
    """多层次蒸馏损失 - 整合所有蒸馏损失"""

    def __init__(self, temperature=3.0, edge_weight=2.0, contrast_temp=0.5):
        super().__init__()
        self.response_loss = ResponseDistillationLoss(temperature)
        self.feature_loss = FeatureMapDistillationLoss()
        self.edge_loss = EdgeAwareDistillationLoss(edge_weight)
        self.contrast_loss = ContrastiveDistillationLoss(contrast_temp)
        self.bce_loss = nn.BCELoss()

    def _dice_loss(self, pred, target):
        smooth = 1.0

        if pred.shape[-2:] != target.shape[-2:]:
            if target.shape[2] > pred.shape[2]:
                pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=True)
            else:
                target = F.interpolate(target, size=pred.shape[-2:], mode='nearest')

        if pred.shape[1] != target.shape[1]:
            if target.shape[1] > 1:
                target = target.mean(dim=1, keepdim=True)
            else:
                pred = pred.mean(dim=1, keepdim=True)

        if pred.shape != target.shape:
            target = target.view_as(pred)

        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

        return 1 - dice

    def forward(self, student_outputs, teacher_outputs, targets,
                student_features=None, teacher_features=None,
                lambda_response=0.4, lambda_feature=0.3,
                lambda_edge=0.3, lambda_contrast=0.2,
                lambda_task=1.0, epoch=None):
        losses = {}

        dice_loss = self._dice_loss(student_outputs, targets)
        bce_loss = self.bce_loss(student_outputs, targets)
        losses['task'] = dice_loss + bce_loss

        if lambda_response > 0 and teacher_outputs is not None:
            eps = 1e-7
            student_logits = torch.log(student_outputs / (1 - student_outputs + eps) + eps)
            teacher_logits = torch.log(teacher_outputs / (1 - teacher_outputs + eps) + eps)
            losses['response'] = self.response_loss(student_logits, teacher_logits)

        if lambda_feature > 0 and student_features and teacher_features:
            feature_loss = 0
            for layer in ['bottleneck', 'decoder4', 'decoder3']:
                if layer in student_features and layer in teacher_features:
                    layer_loss = self.feature_loss(
                        student_features[layer],
                        teacher_features[layer]
                    )
                    feature_loss += layer_loss

            if isinstance(feature_loss, torch.Tensor):
                losses['feature'] = feature_loss / 3.0

        if lambda_edge > 0 and student_features and teacher_features:
            if 'bottleneck' in student_features and 'bottleneck' in teacher_features:
                losses['edge'] = self.edge_loss(
                    student_features['bottleneck'],
                    teacher_features['bottleneck'],
                    targets
                )

        if lambda_contrast > 0 and student_features and teacher_features:
            if 'bottleneck' in student_features and 'bottleneck' in teacher_features:
                losses['contrast'] = self.contrast_loss(
                    student_features['bottleneck'],
                    teacher_features['bottleneck'],
                    targets
                )

        total_loss = lambda_task * losses['task']

        if epoch is not None:
            decay_factor = max(0, 1.0 - epoch / 100)
            lambda_response_dyn = lambda_response * decay_factor
            lambda_feature_dyn = lambda_feature * decay_factor
        else:
            lambda_response_dyn = lambda_response
            lambda_feature_dyn = lambda_feature

        if 'response' in losses:
            total_loss += lambda_response_dyn * losses['response']
        if 'feature' in losses:
            total_loss += lambda_feature_dyn * losses['feature']
        if 'edge' in losses:
            total_loss += lambda_edge * losses['edge']
        if 'contrast' in losses:
            total_loss += lambda_contrast * losses['contrast']

        losses['total'] = total_loss
        return total_loss, losses


def test_distillation_loss():
    """测试蒸馏损失函数"""
    batch_size = 2
    channels = 3
    height = width = 16

    student_output = torch.sigmoid(torch.randn(batch_size, 1, height, width))
    teacher_output = torch.sigmoid(torch.randn(batch_size, 1, height, width))
    target = torch.randint(0, 2, (batch_size, 1, height, width)).float()

    student_features = {
        'bottleneck': torch.randn(batch_size, 256, height // 16, width // 16),
        'decoder4': torch.randn(batch_size, 128, height // 8, width // 8),
        'decoder3': torch.randn(batch_size, 64, height // 4, width // 4)
    }

    teacher_features = {
        'bottleneck': torch.randn(batch_size, 256, height // 16, width // 16),
        'decoder4': torch.randn(batch_size, 128, height // 8, width // 8),
        'decoder3': torch.randn(batch_size, 64, height // 4, width // 4)
    }

    print("测试蒸馏损失函数...")

    response_loss_fn = ResponseDistillationLoss(temperature=3.0)
    response_loss = response_loss_fn(
        torch.log(student_output / (1 - student_output + 1e-7) + 1e-7),
        torch.log(teacher_output / (1 - teacher_output + 1e-7) + 1e-7)
    )
    print(f"响应蒸馏损失: {response_loss.item():.4f}")

    feature_loss_fn = FeatureMapDistillationLoss()
    feature_loss = feature_loss_fn(student_features['bottleneck'], teacher_features['bottleneck'])
    print(f"特征图蒸馏损失: {feature_loss.item():.4f}")

    edge_loss_fn = EdgeAwareDistillationLoss(edge_weight=2.0)
    edge_loss = edge_loss_fn(student_features['bottleneck'], teacher_features['bottleneck'], target)
    print(f"边缘感知注意力蒸馏损失: {edge_loss.item():.4f}")

    contrast_loss_fn = ContrastiveDistillationLoss(temperature=0.5)
    contrast_loss = contrast_loss_fn(student_features['bottleneck'], teacher_features['bottleneck'])
    print(f"对比蒸馏损失: {contrast_loss.item():.4f}")

    multi_loss_fn = MultiLevelDistillationLoss()
    total_loss, losses = multi_loss_fn(
        student_output, teacher_output, target,
        student_features, teacher_features,
        lambda_response=0.4, lambda_feature=0.3,
        lambda_edge=0.3, lambda_contrast=0.2,
        lambda_task=1.0
    )

    print(f"\n多层次蒸馏损失:")
    for loss_name, loss_value in losses.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")


if __name__ == "__main__":
    test_distillation_loss()