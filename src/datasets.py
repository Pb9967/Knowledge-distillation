# datasets.py
import os
import json
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
from config import Config


class CrackSegmentationDataset(Dataset):
    """裂缝分割数据集类（支持 .jpg/.png/.jpeg 混合格式）"""

    def __init__(self, data_root, split='train', transform=None, image_size=512):
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.images_dir = os.path.join(data_root, split, 'images')
        self.masks_dir = os.path.join(data_root, split, 'masks')

        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.endswith('_mask.png')
        ])

        self._filter_and_validate_files()
        print(f"✓ {split}集加载完成: {len(self.image_files)} 个有效样本")

    def _get_mask_filename(self, img_filename):
        """根据图像文件扩展名生成对应的掩码文件名"""
        lower_name = img_filename.lower()
        if lower_name.endswith('.jpg') or lower_name.endswith('.jpeg'):
            return img_filename[:lower_name.rfind('.')] + '_mask.png'
        elif lower_name.endswith('.png'):
            return img_filename[:lower_name.rfind('.')] + '_mask.png'
        else:
            raise ValueError(f"不支持的文件格式: {img_filename}")

    def _filter_and_validate_files(self):
        """过滤损坏的图像文件，验证图像-掩码对完整性"""
        print(f"正在检查 {self.split} 集文件完整性...")
        valid_files = []
        corrupted_files = []

        for img_file in tqdm(self.image_files, desc=f"检查{self.split}集"):
            img_path = os.path.join(self.images_dir, img_file)
            mask_name = self._get_mask_filename(img_file)
            mask_path = os.path.join(self.masks_dir, mask_name)

            if not os.path.exists(mask_path):
                corrupted_files.append((img_file, "缺失对应掩码文件"))
                continue

            try:
                with Image.open(img_path) as img:
                    img.verify()
                with Image.open(mask_path) as mask:
                    mask.verify()

                img = Image.open(img_path)
                mask = Image.open(mask_path)
                if img.size != mask.size:
                    print(f"⚠ 尺寸不匹配: {img_file} (图像: {img.size}, 掩码: {mask.size})")
                    continue

                valid_files.append(img_file)
            except Exception as e:
                corrupted_files.append((img_file, str(e)))
                print(f"⚠ 发现损坏文件: {img_file} - {e}")

        self.image_files = valid_files
        print(f"✓ {self.split} 集有效文件: {len(valid_files)} 个")

        if corrupted_files:
            print(f"⚠ 已排除 {len(corrupted_files)} 个损坏文件")
            log_path = os.path.join(self.data_root, f'{self.split}_corrupted_files.json')
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(corrupted_files, f, indent=2, ensure_ascii=False)

        if len(self.image_files) == 0:
            raise RuntimeError(f"{self.split} 集中没有有效的图像文件！")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_name = self._get_mask_filename(img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        max_retry = 3
        for attempt in range(max_retry):
            try:
                with Image.open(img_path) as img:
                    image = ImageOps.exif_transpose(img.convert('RGB'))

                with Image.open(mask_path) as mask:
                    mask = ImageOps.exif_transpose(mask.convert('L'))

                resize_transform = transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=transforms.InterpolationMode.BILINEAR
                )
                image = resize_transform(image)

                mask_resize = transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=transforms.InterpolationMode.NEAREST
                )
                mask = mask_resize(mask)

                image = transforms.ToTensor()(image)
                mask = transforms.ToTensor()(mask)
                mask = (mask > 0.5).float()

                if image.shape != mask.shape:
                    if mask.shape[0] == 1:
                        mask = mask.repeat(3, 1, 1)
                    elif mask.shape[0] == 3:
                        mask = mask.mean(dim=0, keepdim=True)

                if image.shape[-2:] != (self.image_size, self.image_size):
                    image = torch.nn.functional.interpolate(
                        image.unsqueeze(0),
                        size=(self.image_size, self.image_size)
                    ).squeeze(0)
                    mask = torch.nn.functional.interpolate(
                        mask.unsqueeze(0),
                        size=(self.image_size, self.image_size)
                    ).squeeze(0)

                return image, mask, img_name

            except Exception as e:
                print(f"[WARNING] 加载失败: {img_name} (尝试 {attempt + 1}/{max_retry}) - {e}")
                if attempt == max_retry - 1:
                    return (torch.zeros(3, self.image_size, self.image_size),
                            torch.zeros(1, self.image_size, self.image_size),
                            f"CORRUPTED_{img_name}")

        return (torch.zeros(3, self.image_size, self.image_size),
                torch.zeros(1, self.image_size, self.image_size),
                f"CORRUPTED_{img_name}")


if __name__ == '__main__':
    config = Config()
    data_root = config.get('data_root')

    print("验证数据集格式支持和归一化")
    for split in ['train', 'val', 'test']:
        try:
            dataset = CrackSegmentationDataset(data_root, split=split, image_size=512)
            print(f"\n{split}集: {len(dataset)} 个样本")

            if len(dataset) > 0:
                img, mask, name = dataset[0]
                print(f"  样本 '{name}': img={img.shape}, mask={mask.shape}")
                assert img.shape == mask.shape, f"尺寸不匹配: {img.shape} vs {mask.shape}"
                assert img.shape[-2:] == (512, 512), f"目标尺寸错误: {img.shape}"
                print(f"  ✓ 格式支持和归一化成功")
            else:
                print(f"  ⚠ 数据集为空")
        except Exception as e:
            print(f"⚠ {split}集加载失败: {e}")

    print("验证完成！")