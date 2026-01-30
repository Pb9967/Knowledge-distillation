# chinese_test.py - 中文字体显示测试程序
import os
import sys
import shutil
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def setup_matplotlib_font():
    """设置中文字体，避免中文显示为方框"""
    try:
        font_list = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
        chinese_keywords = ['simhei', 'msyh', 'microsoft yahei', 'pingfang', 'heiti',
                            'stsong', 'simsun', 'kai', 'fangsong', 'stkaiti']

        chinese_fonts = [
            font_path for font_path in font_list
            if any(keyword in os.path.basename(font_path).lower() for keyword in chinese_keywords)
        ]

        if chinese_fonts:
            try:
                matplotlib.font_manager.fontManager.addfont(chinese_fonts[0])
                font_prop = matplotlib.font_manager.FontProperties(fname=chinese_fonts[0])
                font_name = font_prop.get_name()
                plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'Arial', 'Helvetica']
            except Exception:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial', 'Helvetica']
        else:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial', 'Helvetica']

    except Exception:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

    plt.rcParams['axes.unicode_minus'] = False


def ensure_chinese_font():
    """确保中文字体正确设置（用于每个绘图函数前调用）"""
    current_fonts = plt.rcParams.get('font.sans-serif', [])
    chinese_check = any('simhei' in font.lower() or 'microsoft yahei' in font.lower()
                        for font in current_fonts)

    if not chinese_check:
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass


def setup_matplotlib():
    """强制配置matplotlib，解决中文乱码和空白图问题"""
    cache_dir = matplotlib.get_cachedir()
    if cache_dir and os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print(f"✓ 已清除matplotlib缓存: {cache_dir}")
        except:
            pass

    matplotlib.use('Agg')
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['axes.unicode_minus'] = True
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['figure.figsize'] = (16, 10)

    if sys.platform == 'win32':
        font_candidates = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/msyhbd.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
        ]

        available_fonts = [Path(font).stem for font in font_candidates if os.path.exists(font)]
        if available_fonts:
            available_fonts.extend(['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'])
            plt.rcParams['font.sans-serif'] = available_fonts
            print(f"✓ 使用字体: {available_fonts[0]}")
        else:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']
            print("⚠ 未找到系统字体，使用DejaVu Sans")

    elif sys.platform == 'darwin':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'AppleGothic', 'PingFang SC',
                                           'Hiragino Sans GB', 'DejaVu Sans', 'Helvetica']
        print("✓ macOS字体配置完成")
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Zen Hei',
                                           'AR PL UMing CN', 'Noto Sans CJK', 'Liberation Sans']
        print("✓ Linux字体配置完成")

    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.default'] = 'regular'

    print(f"✓ 当前后端: {matplotlib.get_backend()}")
    print(f"✓ 字体列表: {plt.rcParams['font.sans-serif'][:3]}")


def test_chinese_display():
    """测试中文字符显示"""
    print("=" * 60)
    print("开始测试中文字符显示")
    print("=" * 60)

    # 设置中文字体
    setup_matplotlib_font()
    ensure_chinese_font()

    # 创建测试数据
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.random.normal(0, 0.1, 100) + np.sin(x) * 0.5

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.suptitle('中文字符显示测试', fontsize=18, fontweight='bold')

    # 子图1: 基础中文测试
    axes[0, 0].plot(x, y1, label='正弦曲线', linewidth=2)
    axes[0, 0].plot(x, y2, label='余弦曲线', linewidth=2)
    axes[0, 0].set_title('三角函数曲线', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('时间 (秒)', fontsize=12)
    axes[0, 0].set_ylabel('振幅', fontsize=12)
    axes[0, 0].legend(loc='best')
    axes[0, 0].grid(True, alpha=0.3)

    # 子图2: 中文标注测试
    categories = ['苹果', '香蕉', '橙子', '葡萄', '西瓜']
    values = np.random.randint(10, 100, 5)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    axes[0, 1].bar(categories, values, color=colors, edgecolor='black')
    axes[0, 1].set_title('水果销售量统计', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('水果种类', fontsize=12)
    axes[0, 1].set_ylabel('销售量 (公斤)', fontsize=12)

    # 添加数值标签
    for i, v in enumerate(values):
        axes[0, 1].text(i, v + 1, str(v), ha='center', fontsize=10)

    # 子图3: 中文散点图
    x_scatter = np.random.randn(50)
    y_scatter = x_scatter * 0.5 + np.random.randn(50) * 0.3

    scatter = axes[1, 0].scatter(x_scatter, y_scatter,
                                 c=np.random.rand(50),
                                 cmap='viridis',
                                 s=100,
                                 alpha=0.7,
                                 edgecolor='black')
    axes[1, 0].set_title('随机数据散点图', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('特征 X', fontsize=12)
    axes[1, 0].set_ylabel('特征 Y', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)

    # 添加颜色条
    plt.colorbar(scatter, ax=axes[1, 0], label='颜色值')

    # 子图4: 中文箱线图
    data_box = [np.random.normal(i, 1, 100) for i in range(5)]
    labels_box = ['算法A', '算法B', '算法C', '算法D', '算法E']

    box = axes[1, 1].boxplot(data_box, labels=labels_box, patch_artist=True)

    # 设置箱线图颜色
    colors_box = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
    for patch, color in zip(box['boxes'], colors_box):
        patch.set_facecolor(color)

    axes[1, 1].set_title('算法性能比较', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('算法名称', fontsize=12)
    axes[1, 1].set_ylabel('性能得分', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存测试图像
    output_dir = '../test_results'
    os.makedirs(output_dir, exist_ok=True)

    test_image_path = os.path.join(output_dir, 'chinese_display_test.png')
    plt.savefig(test_image_path, dpi=150, bbox_inches='tight')

    # 显示当前字体配置
    current_fonts = plt.rcParams['font.sans-serif']
    print(f"\n当前使用的字体: {current_fonts[0] if current_fonts else '未设置'}")
    print(f"负号显示设置: {plt.rcParams['axes.unicode_minus']}")

    # 检查文件是否成功保存
    if os.path.exists(test_image_path):
        file_size = os.path.getsize(test_image_path) / 1024
        print(f"✓ 测试图像已保存: {test_image_path}")
        print(f"✓ 文件大小: {file_size:.2f} KB")

        # 显示测试成功信息
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.6, '中文字体测试成功!',
                 fontsize=24, fontweight='bold',
                 ha='center', va='center',
                 transform=plt.gca().transAxes)
        plt.text(0.5, 0.4, f'字体: {current_fonts[0] if current_fonts else "默认"}',
                 fontsize=16, ha='center', va='center',
                 transform=plt.gca().transAxes)
        plt.text(0.5, 0.2, '检查测试图像以验证中文显示',
                 fontsize=14, ha='center', va='center',
                 transform=plt.gca().transAxes)
        plt.axis('off')

        success_image_path = os.path.join(output_dir, 'chinese_test_success.png')
        plt.savefig(success_image_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"✓ 成功图像已保存: {success_image_path}")
        print("\n" + "=" * 60)
        print("中文字体显示测试完成!")
        print("请检查以下文件验证中文显示效果:")
        print(f"1. {test_image_path}")
        print(f"2. {success_image_path}")
        print("=" * 60)
    else:
        print("✗ 测试图像保存失败!")

    # 显示当前所有可用的中文字体
    print("\n系统检测到的中文字体:")
    try:
        font_list = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
        chinese_fonts = []
        for font_path in font_list:
            font_name = os.path.basename(font_path).lower()
            if any(keyword in font_name for keyword in ['simhei', 'msyh', 'yahei', 'pingfang', 'heiti']):
                chinese_fonts.append(font_path)

        if chinese_fonts:
            for i, font in enumerate(chinese_fonts[:5]):  # 只显示前5个
                print(f"  {i + 1}. {os.path.basename(font)}")
            if len(chinese_fonts) > 5:
                print(f"  还有 {len(chinese_fonts) - 5} 个中文字体...")
        else:
            print("  未检测到中文字体")
    except Exception as e:
        print(f"  字体检测失败: {e}")


def main():
    """主程序入口"""
    print("中文字体显示测试程序")
    print("版本: 1.0")
    print("功能: 测试matplotlib中文字体显示功能")
    print("-" * 50)

    # 检查matplotlib版本
    print(f"Matplotlib版本: {matplotlib.__version__}")
    print(f"Python版本: {sys.version.split()[0]}")
    print(f"操作系统: {sys.platform}")

    try:
        # 执行测试
        test_chinese_display()

        # 测试配置函数
        print("\n" + "=" * 50)
        print("测试setup_matplotlib配置函数...")
        setup_matplotlib()

    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    # 运行主程序
    exit_code = main()
    print(f"\n程序执行完成，退出码: {exit_code}")
    sys.exit(exit_code)