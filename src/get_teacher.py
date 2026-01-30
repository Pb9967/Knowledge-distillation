# get_teacher.py
import os
import torch
import segmentation_models_pytorch as smp


def get_teacher_model(freeze_params=True):
    """获取并返回DeepLabV3+教师模型"""
    teacher = smp.DeepLabV3Plus(
        encoder_name='resnet101',
        encoder_weights='imagenet',
        classes=1,
        activation='sigmoid'
    )

    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    print(f"教师模型创建成功（参数冻结，评估模式）")

    total_params = sum(p.numel() for p in teacher.parameters())
    print(f"参数量: {total_params / 1e6:.2f}M")
    print(f"模型结构: DeepLabV3Plus-ResNet101")

    return teacher, total_params / 1e6


def save_teacher_model(teacher, save_path='../model/teacher_deeplabv3p.pth'):
    """保存教师模型权重"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(teacher.state_dict(), save_path)
    print(f"教师模型权重已保存至: {save_path}")
    print(f"文件大小: {os.path.getsize(save_path) / 1e6:.2f} MB")


if __name__ == '__main__':
    teacher, params = get_teacher_model(freeze_params=True)

    dummy_input = torch.randn(1, 3, 512, 512)
    teacher.eval()
    with torch.no_grad():
        output = teacher(dummy_input)

    print(f"\n测试结果:")
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")
    print(f"Sigmoid输出验证: 所有值应在[0, 1]范围内: {torch.all((output >= 0) & (output <= 1))}")

    save_teacher_model(teacher)