import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T
import json


class FaceDS(Dataset):
    def __init__(self, image_dir: str, anno_file: str, transform=None):
        """
        image_dir: 图像文件夹路径
        anno_file: JSON 格式注释文件，格式应为 {filename: [[x1, y1, x2, y2], ...]}
        """
        self.image_dir = Path(image_dir)
        self.anno_file = anno_file

        # 加载 JSON 标注
        with open(anno_file, 'r') as f:
            ann = json.load(f)

        # 构建记录 [(img_path, boxes, frame_id)]
        self.recs = [
            (self.image_dir / fn, ann[fn], int(Path(fn).stem))
            for fn in ann
        ]
        self.recs.sort(key=lambda x: x[2])

        # Transformations
        self.transform = transform if transform else T.Compose([
            T.Resize((112, 112)),
            T.ToTensor(),
        ])

        self.gallery_aug = T.Compose([
            T.RandomHorizontalFlip(p=1.0),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ])

        self.probe_aug = T.Compose([
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ])

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        img_path, boxes, fr = self.recs[i]
        img = Image.open(img_path).convert("RGB")

        # 原图用于 gallery，probe 做变换
        gallery_img = self.transform(self.gallery_aug(img))
        probe_img = self.transform(self.probe_aug(img))

        # target 结构
        tgt = {
            "boxes":  torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "location": "cam",
            "frame": fr
        }

        return gallery_img, probe_img
