# config.py 统一配置中心
from dataclasses import dataclass, fields
import torch


@dataclass
class Config:
    # 数据集路径配置
    data_root: str = '../data_file/Crack_500'
    teacher_result: str = '../results/teacher_performance'
    teacher_model_path: str = '../model/teacher_deeplabv3p.pth'
    checkpoint_path: str = None
    use_checkpoint: bool = True
    distillation_result: str = '../results/distillation_performance'
    plot_history: str = '../results/plot_history'
    distillation_model: str = '../model/distillation_model'
    last_model: str = '../model'

    # 模型参数
    image_size: int = 512
    num_classes: int = 1

    # 训练参数
    epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    min_lr: float = 1e-6
    num_workers: int = 4

    # 蒸馏参数
    temperature: float = 4.0
    edge_weight: float = 2.0
    contrast_temp: float = 0.5

    # 损失权重
    lambda_response: float = 0.6
    lambda_feature: float = 0.4
    lambda_edge: float = 0.3
    lambda_contrast: float = 0.2
    lambda_task: float = 0.8

    # 硬件配置
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 检查点配置
    checkpoint_interval: int = 10
    resume_from: str = None

    # 学习率调度配置
    scheduler_type: str = 'cosine'
    step_size: int = 10
    gamma: float = 0.1
    patience: int = 5

    # 模型保存配置
    save_interval: int = 5

    # 数据增强配置
    use_augmentation: bool = True
    rotation_range: int = 30
    flip_prob: float = 0.5

    # 混合精度训练配置
    use_amp: bool = True

    # 可视化配置
    plot_format: str = 'png'
    dpi: int = 300

    # 日志配置
    log_level: str = 'INFO'
    log_to_file: bool = True

    def get(self, key, default=None):
        """安全获取配置值"""
        if not hasattr(self, key):
            return default

        value = getattr(self, key)
        if value is None and default is not None:
            return default

        return value

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __contains__(self, key):
        return hasattr(self, key)


if __name__ == '__main__':
    config = Config()
    fields_to_print = [
        ('data_root', config.get('data_root')),
        ('teacher_result', config.get('teacher_result')),
        ('teacher_model_path', config.get('teacher_model_path')),
        ('image_size', config.get('image_size')),
        ('num_classes', config.get('num_classes')),
        ('epochs', config.get('epochs')),
        ('batch_size', config.get('batch_size')),
        ('learning_rate', config.get('learning_rate')),
        ('weight_decay', config.get('weight_decay')),
        ('num_workers', config.get('num_workers')),
        ('temperature', config.get('temperature')),
        ('edge_weight', config.get('edge_weight')),
        ('contrast_temp', config.get('contrast_temp')),
        ('lambda_response', config.get('lambda_response')),
        ('lambda_feature', config.get('lambda_feature')),
        ('lambda_edge', config.get('lambda_edge')),
        ('lambda_contrast', config.get('lambda_contrast')),
        ('lambda_task', config.get('lambda_task')),
        ('device', config.get('device')),
        ('checkpoint_interval', config.get('checkpoint_interval')),
        ('resume_from', config.get('resume_from')),
        ('checkpoint_path', config.get('checkpoint_path', default='mine_path'))
    ]

    for name, value in fields_to_print:
        print(f'{name}: {value}')

    print('checkpoint: ', config.get('use_checkpoint'))