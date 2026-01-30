# student_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积块"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class SimpleBottleneck(nn.Module):
    """轻量级瓶颈：1x1降维 + 3x3卷积 + 残差连接"""

    def __init__(self, in_channels=96, out_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class LightweightStudent(nn.Module):
    """CNN-Transformer混合轻量化学生模型：MobileNetV2编码器 + 渐进式解码器"""

    def __init__(self, num_classes=1, input_size=512, use_pretrained=True):
        super().__init__()
        self.input_size = input_size
        self.use_pretrained = use_pretrained

        try:
            from torchvision.models import MobileNet_V2_Weights
            if use_pretrained:
                mobilenet_full = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            else:
                mobilenet_full = mobilenet_v2(weights=None)
        except (ImportError, AttributeError):
            try:
                mobilenet_full = mobilenet_v2(pretrained=use_pretrained)
            except TypeError:
                mobilenet_full = mobilenet_v2()

        self.encoder = mobilenet_full.features[:14]

        dummy_input = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            dummy_output = self.encoder(dummy_input)
            encoder_out_channels = dummy_output.shape[1]
        self.bottleneck = SimpleBottleneck(in_channels=encoder_out_channels, out_channels=256)

        self.decoder4 = self._make_decoder_block(256 + 96, 128)
        self.decoder3 = self._make_decoder_block(128 + 32, 64)
        self.decoder2 = self._make_decoder_block(64 + 24, 32)
        self.decoder1 = self._make_decoder_block(32 + 16, 16)

        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
            nn.Sigmoid()
        )

        self.upsample_2x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.stage_indices = [1, 3, 6, 13]

    def _make_decoder_block(self, in_channels, out_channels):
        """构建解码器块"""
        return nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        enc_features = {}
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.stage_indices:
                enc_features[i] = x

        bottleneck_out = self.bottleneck(x)

        x = self.upsample_2x(bottleneck_out)
        enc_feat13 = enc_features[13]
        enc_feat13_up = F.interpolate(enc_feat13, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc_feat13_up], dim=1)
        decoder4_out = self.decoder4(x)

        x = self.upsample_2x(decoder4_out)
        enc_feat6 = enc_features[6]
        enc_feat6_up = F.interpolate(enc_feat6, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc_feat6_up], dim=1)
        decoder3_out = self.decoder3(x)

        x = self.upsample_2x(decoder3_out)
        enc_feat3 = enc_features[3]
        enc_feat3_up = F.interpolate(enc_feat3, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc_feat3_up], dim=1)
        decoder2_out = self.decoder2(x)

        x = self.upsample_2x(decoder2_out)
        enc_feat1 = enc_features[1]
        enc_feat1_up = F.interpolate(enc_feat1, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc_feat1_up], dim=1)
        decoder1_out = self.decoder1(x)

        output = self.final_conv(decoder1_out)

        if output.shape[-2:] != (H, W):
            output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=True)

        intermediate_features = {
            'bottleneck': bottleneck_out,
            'decoder4': decoder4_out,
            'decoder3': decoder3_out,
        }

        return output, intermediate_features

    def get_parameter_count(self):
        """统计模型参数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


def get_student_model(input_size=512, use_pretrained=True):
    """工厂函数：获取学生模型实例"""
    model = LightweightStudent(num_classes=1, input_size=input_size, use_pretrained=use_pretrained)
    total_params, trainable_params = model.get_parameter_count()

    print("学生模型创建成功")
    print(f"是否使用预训练权重: {'是' if use_pretrained else '否'}")
    print(f"总参数量: {total_params:,} (约 {total_params / 1e6:.2f}M)")
    print(f"可训练参数量: {trainable_params:,} (约 {trainable_params / 1e6:.2f}M)")

    return model


if __name__ == "__main__":
    print("开始学生模型单元测试...")

    for use_pretrained in [True, False]:
        print(f"测试模式: {'使用预训练权重' if use_pretrained else '不使用预训练权重'}")

        model = get_student_model(input_size=512, use_pretrained=use_pretrained)
        model.eval()

        test_cases = [
            (1, 3, 512, 512),
            (2, 3, 512, 512),
            (4, 3, 512, 512),
        ]

        for batch_shape in test_cases:
            dummy_input = torch.randn(*batch_shape)
            output, features = model(dummy_input)

            print(f"\n输入形状: {dummy_input.shape}")
            print(f"输出形状: {output.shape}")
            print(f"输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
            print(f"中间特征: {list(features.keys())}")
            for key, value in features.items():
                print(f"  {key}: {value.shape}")

            assert output.shape[0] == batch_shape[0], "批次大小不匹配"
            assert output.shape[1] == 1, "输出通道数应为1"
            assert output.shape[2] == batch_shape[2], f"高度不匹配: {output.shape[2]} vs {batch_shape[2]}"
            assert output.shape[3] == batch_shape[3], f"宽度不匹配: {output.shape[3]} vs {batch_shape[3]}"

            print("✓ 尺寸检查通过")

    print("\n所有测试通过！模型输出尺寸正确。")