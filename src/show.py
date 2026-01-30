import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def setup_matplotlib_chinese_font():
    """设置matplotlib中文字体"""
    import platform

    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 添加中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

    # 针对不同操作系统设置字体
    if platform.system() == 'Windows':
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
        for font in chinese_fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
                break
            except:
                continue
    elif platform.system() == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'AppleGothic', 'DejaVu Sans']
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Zen Hei', 'AR PL UMing CN']

    print(f"Matplotlib字体设置: {plt.rcParams['font.sans-serif'][0]}")


def load_image_and_mask(image_path, mask_path, target_size=(512, 512)):
    """加载图像和对应的mask"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size, Image.BILINEAR)
    image_array = np.array(image)

    # 加载mask
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize(target_size, Image.NEAREST)
    mask_array = np.array(mask) / 255.0  # 归一化到0-1

    return image_array, mask_array


def random_select_and_display(image_dir, mask_dir, num_samples=6, save_path=None):
    """
    从数据集中随机挑选图片并展示原始图像与mask

    Args:
        image_dir: 图像文件夹路径
        mask_dir: mask文件夹路径
        num_samples: 要展示的样本数量
        save_path: 图片保存路径（可选）
    """
    # 设置中文字体
    setup_matplotlib_chinese_font()

    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []

    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])

    if not image_files:
        print(f"在 {image_dir} 中未找到图像文件")
        return

    # 确保有足够的样本
    num_samples = min(num_samples, len(image_files))

    # 随机选择样本
    selected_files = random.sample(image_files, num_samples)

    # 创建图形
    fig = plt.figure(figsize=(15, 5 * num_samples))
    gs = gridspec.GridSpec(num_samples, 3, width_ratios=[1, 1, 1], height_ratios=[1] * num_samples)

    for i, img_file in enumerate(selected_files):
        # 构建图像和mask路径
        img_path = os.path.join(image_dir, img_file)

        # 构建mask文件名（假设mask文件名与图像文件名相同但扩展名可能不同）
        img_name, img_ext = os.path.splitext(img_file)
        mask_path = None

        # 尝试多种可能的mask文件名格式
        possible_mask_names = [
            f"{img_name}.png",  # Crack500格式
            f"{img_name}_mask.png",
            f"{img_name}_label.png",
            f"{img_name}_seg.png",
            f"{img_name}.jpg",
        ]

        for mask_name in possible_mask_names:
            test_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(test_path):
                mask_path = test_path
                break

        if mask_path is None:
            print(f"警告: 未找到 {img_file} 对应的mask文件")
            continue

        # 加载图像和mask
        try:
            image_array, mask_array = load_image_and_mask(img_path, mask_path)

            # 显示原始图像
            ax1 = plt.subplot(gs[i, 0])
            ax1.imshow(image_array)
            ax1.set_title(f'原始图像\n{img_file}', fontsize=12)
            ax1.axis('off')

            # 显示mask（灰度）
            ax2 = plt.subplot(gs[i, 1])
            ax2.imshow(mask_array, cmap='gray', vmin=0, vmax=1)
            ax2.set_title(f'分割掩码\n{os.path.basename(mask_path)}', fontsize=12)
            ax2.axis('off')

            # 显示叠加效果
            ax3 = plt.subplot(gs[i, 2])

            # 创建叠加图像（红色表示裂缝区域）
            overlay = image_array.copy()

            # 将mask转换为布尔数组
            mask_bool = mask_array > 0.5

            if mask_bool.any():
                # 将裂缝区域标记为红色（透明度0.5）
                overlay[mask_bool] = [1.0, 0.5, 0.5]  # 浅红色

                # 创建边界轮廓用于更好地可视化
                from scipy import ndimage
                struct = ndimage.generate_binary_structure(2, 2)
                eroded = ndimage.binary_erosion(mask_bool, structure=struct, iterations=1)
                boundaries = mask_bool & ~eroded

                # 将边界标记为红色
                overlay[boundaries] = [1.0, 0, 0]  # 亮红色

            ax3.imshow(overlay)
            ax3.set_title(f'叠加效果\n(红色:裂缝区域)', fontsize=12)
            ax3.axis('off')

            # 添加分隔线
            if i < num_samples - 1:
                line_y = i + 1 - 0.02
                plt.annotate('', xy=(0, line_y), xytext=(1, line_y),
                             xycoords='figure fraction',
                             arrowprops=dict(arrowstyle='-', color='gray', linewidth=1))

        except Exception as e:
            print(f"处理 {img_file} 时出错: {e}")
            continue

    plt.suptitle(f'数据集随机样本展示 (共{num_samples}个样本)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    # 保存或显示
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        print(f"图片已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_dataset_distribution(image_dir, mask_dir):
    """分析数据集分布情况"""
    print("分析数据集分布...")

    # 统计图像和mask数量
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []

    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])

    mask_files = []
    for ext in image_extensions:
        mask_files.extend([f for f in os.listdir(mask_dir) if f.lower().endswith(ext)])

    print(f"图像数量: {len(image_files)}")
    print(f"Mask数量: {len(mask_files)}")

    # 检查匹配情况
    matched_pairs = 0
    for img_file in image_files[:10]:  # 只检查前10个以节省时间
        img_name, _ = os.path.splitext(img_file)
        mask_found = False

        for mask_ext in ['.png', '_mask.png', '_label.png']:
            mask_file = img_name + mask_ext
            if os.path.exists(os.path.join(mask_dir, mask_file)):
                mask_found = True
                break

        if mask_found:
            matched_pairs += 1

    if len(image_files) > 0:
        match_rate = matched_pairs / min(10, len(image_files))
        print(f"图像-Mask匹配率(样本): {match_rate * 100:.1f}%")

    return len(image_files), len(mask_files)


def main():
    """主函数"""
    # ==================== 配置区域 ====================
    # 请根据实际情况修改以下路径
    dataset_config = {
        # Crack500数据集示例路径
        'crack500': {
            'image_dir': '../data_file/Crack_500/test/images',
            'mask_dir': '../data_file/Crack_500/test/masks',
        },
        # 其他数据集路径示例
        'deepcrack': {
            'image_dir': '../data_file/DeepCrack-master/dataset/train_img',
            'mask_dir': '../data_file/DeepCrack-master/dataset/train_lab',
        },
        # 自定义数据集
        'custom': {
            'image_dir': '../data/images',
            'mask_dir': '../data/masks',
        }
    }

    # 选择要使用的数据集配置
    dataset_name = 'crack500'  # 修改为要使用的数据集
    num_samples_to_show = 2  # 要显示的样本数量

    # 输出图片保存路径（可选）
    output_path = '../results/dataset_samples.png'
    # =================================================

    if dataset_name not in dataset_config:
        print(f"错误: 数据集配置 '{dataset_name}' 不存在")
        print(f"可用配置: {list(dataset_config.keys())}")
        return

    config = dataset_config[dataset_name]
    image_dir = config['image_dir']
    mask_dir = config['mask_dir']

    print(f"数据集: {dataset_name}")
    print(f"图像目录: {image_dir}")
    print(f"Mask目录: {mask_dir}")

    # 检查目录是否存在
    if not os.path.exists(image_dir):
        print(f"错误: 图像目录不存在: {image_dir}")
        return

    if not os.path.exists(mask_dir):
        print(f"错误: Mask目录不存在: {mask_dir}")
        return

    # 分析数据集分布
    analyze_dataset_distribution(image_dir, mask_dir)

    # 随机选择并显示样本
    print(f"\n随机选择 {num_samples_to_show} 个样本进行展示...")
    random_select_and_display(
        image_dir=image_dir,
        mask_dir=mask_dir,
        num_samples=num_samples_to_show,
        save_path=output_path
    )

    print(f"\n完成！")


if __name__ == "__main__":
    main()