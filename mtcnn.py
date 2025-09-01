import os
import os.path as osp
import random
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from collections import defaultdict
from xml.etree import ElementTree as ET


class ClosedSetFaceDetection(Dataset):
    def __init__(self, image_dir, xml_dir=None, transform=None):
        """
        image_dir: Path to the directory containing images
        xml_dir: Path to the XML file containing annotations (optional)
        transform: Optional transformation to apply to the images
        """
        super().__init__()

        # Initialize paths
        self.image_dir = image_dir
        self.xml_dir = xml_dir

        # Basic transformation
        self.transform = transform if transform else T.Compose([
            T.Resize((112, 112)),
            T.ToTensor(),
        ])

        # Augmentations for gallery and probe
        self.gallery_aug = T.Compose([
            T.RandomHorizontalFlip(p=1.0),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ])

        self.probe_aug = T.Compose([
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ])

        # Load all images
        imgs = sorted(glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True))
        self.samples = [img for img in imgs if osp.isfile(img)]

        # Optional: Parse XML annotations (not used in __getitem__ currently)
        if xml_dir and osp.isfile(xml_dir):
            self.annotations = self.parse_annotation_file(xml_dir, image_dir)
        else:
            self.annotations = {}

    def parse_annotation_file(self, annotation_path, images_dir, margin=4, marginH=30):
        """
        Parse the XML annotations and extract bounding boxes for faces
        """
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        annotations_dict = defaultdict(list)

        for frame_tag in root.findall('frame'):
            frame_number_str = frame_tag.get('number')
            if not frame_number_str:
                continue
            image_filename = f"{frame_number_str}.jpg"

            for person_tag in frame_tag.findall('person'):
                left_eye_tag = person_tag.find('leftEye')
                right_eye_tag = person_tag.find('rightEye')

                if left_eye_tag is not None and right_eye_tag is not None:
                    lx = int(left_eye_tag.get('x'))
                    ly = int(left_eye_tag.get('y'))
                    rx = int(right_eye_tag.get('x'))
                    ry = int(right_eye_tag.get('y'))
                    t = max(lx, rx) - min(lx, rx)

                    xmin = min(lx, rx) - 3 - t/2
                    xmax = max(lx, rx) + 3 + t/2
                    ymin = min(ly, ry) - 1 - t
                    ymax = max(ly, ry) + 5 + 1.7 * t

                    xmin = max(0, xmin)
                    ymin = max(0, ymin)

                    annotations_dict[image_filename].append([xmin, ymin, xmax, ymax])

        return dict(annotations_dict)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        img = Image.open(path).convert("RGB")

        gallery_img = self.transform(img)
        probe_img = self.transform(self.probe_aug(img))

        return gallery_img, probe_img
