import sys
import os.path as osp
root = osp.join(osp.dirname(osp.abspath(__file__)), "..")
sys.path.append(root)

from datetime import datetime
import pandas as pd
import torch
import torchvision as tv
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms as T
from facenet_pytorch import MTCNN
from rich.progress import track
from isp.pipeline import Pipeline
from privacy.data import ClosedSetFaceDetection, FaceDS
from privacy.backbone import pretrained_backbone
from privacy.metric import CMC
from torchvision.transforms.functional import to_pil_image

# =============== PECAM PRIVACY TRANSFORM MODULE ===============
class PECAMTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# =============== FACE COUNTING FOR PRIVACY TRANSFORM EVALUATION ===============
@torch.no_grad()
def count_faces_after_pecam(dataset, pecam_transform, camera):
    loader = DataLoader(dataset(), batch_size=1, num_workers=4)
    mtcnn = MTCNN(keep_all=False, device='cuda')
    total, detected_raw, detected_priv = 0, 0, 0
    for img, _ in track(loader, description="PECAM Face Counting"):
        img = img.cuda()
        total += 1

        pil_raw = to_pil_image(img[0].cpu())
        boxes_raw, _ = mtcnn.detect(pil_raw)

        # pecam_out = pecam_transform(camera.forward(img)).clamp(0, 1)[0].cpu()
        pecam_out = pecam_transform(img).clamp(0, 1)[0].cpu()
        pil_priv = to_pil_image(pecam_out)
        boxes_priv, _ = mtcnn.detect(pil_priv)

        if boxes_raw is not None and len(boxes_raw) > 0:
            detected_raw += 1
        if boxes_priv is not None and len(boxes_priv) > 0:
            detected_priv += 1

    print(f"[SUMMARY] Faces detected on raw images: {detected_raw}/{total}")
    print(f"[SUMMARY] Faces detected on PECAM images: {detected_priv}/{total}")


# =============== FACE COUNTING FOR ENHANCEMENT EVALUATION ===============
@torch.no_grad()
def count_faces_after_enhancement(dataset, enhancer, camera):
    from sklearn.metrics import precision_score, recall_score, f1_score
    loader = DataLoader(dataset(), batch_size=1, num_workers=4)
    mtcnn = MTCNN(keep_all=False, device='cuda')

    total = 0
    protect_success = 0

    y_true = []  # 1 = 图像需要保护（原图有人脸）
    y_pred = []  # 1 = 成功保护（增强图未检测到脸）

    for img, info in track(loader, description="Evaluating Enhancement Protection"):
        img = img.cuda()
        total += 1

        # 判断原图是否有人脸（=是否需要保护）
        pil_raw = to_pil_image(img[0].cpu())
        boxes_raw, _ = mtcnn.detect(pil_raw)
        raw_has_face = boxes_raw is not None and len(boxes_raw) > 0

        # 判断增强图是否检测到人脸
        enhanced = enhancer(camera.forward(img)).clamp(0, 1)[0].cpu()
        pil_enh = to_pil_image(enhanced)
        boxes_enh, _ = mtcnn.detect(pil_enh)
        enh_has_face = boxes_enh is not None and len(boxes_enh) > 0

        # 是否需要保护
        y_true.append(int(raw_has_face))

        # 是否成功保护（raw 有脸，enh 没有）
        if raw_has_face and not enh_has_face:
            y_pred.append(1)
            protect_success += 1
        else:
            y_pred.append(0)

    print(f"[SUMMARY] Protection success: {protect_success}/{sum(y_true)}")

    P = precision_score(y_true, y_pred, zero_division=0)
    R = recall_score(y_true, y_pred, zero_division=0)
    F1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"[PROTECTION METRICS] Precision: {P:.4f}, Recall: {R:.4f}, F1 Score: {F1:.4f}")


# =============== PERSON COUNTING FOR UTILITY EVALUATION ===============
@torch.no_grad()
def detect_person_count(image_tensor, model, threshold=0.5):
    preds = model([image_tensor])[0]
    labels = preds["labels"]
    scores = preds["scores"]
    person_mask = (labels == 1) & (scores > threshold)
    return person_mask.sum().item()

@torch.no_grad()
def count_person_after_pecam(dataset, pecam_transform, camera):
    loader = DataLoader(dataset(), batch_size=1, num_workers=4)
    person_detector = tv.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().cuda()

    total_images = 0
    total_person_raw = 0
    total_person_priv = 0

    for img, _ in track(loader, description="PECAM Person Counting"):
        img = img.cuda()
        total_images += 1

        person_count_raw = detect_person_count(img[0], person_detector)
        total_person_raw += person_count_raw

        # pecam_out = pecam_transform(camera.forward(img)).clamp(0, 1)[0]
        pecam_out = pecam_transform(img).clamp(0, 1)[0]
        person_count_priv = detect_person_count(pecam_out, person_detector)
        total_person_priv += person_count_priv

    print(f"[SUMMARY] Persons detected on raw images: {total_person_raw} over {total_images} images")
    print(f"[SUMMARY] Persons detected on PECAM images: {total_person_priv} over {total_images} images")

    if total_person_raw > 0:
        utility = total_person_priv / total_person_raw
        print(f"[UTILITY] PECAM Utility (person detection preservation ratio): {utility:.4f}")
        if utility < 0.7:
            print("⚠️ Warning: PECAM transform reduces person detection significantly, consider weakening transform.")
    else:
        print("⚠️ Warning: No persons detected in raw images, cannot compute utility.")

# =============== EVALUATION ===============
@torch.no_grad()
def evaluate_privacy_backbone(d_class, pecam_transform, camera, backbone):
    normalize = tv.transforms.Normalize([0.5] * 3, [0.5] * 3)
    loader = DataLoader(d_class(), batch_size=32, num_workers=8)
    gallery, probe = [], []
    for a, b in track(loader, description="Feature extraction"):
        a, b = a.cuda(), b.cuda()
        # b = pecam_transform(camera.forward(b)).clamp(0, 1)
        b = pecam_transform(b).clamp(0, 1)
        gallery.append(backbone(normalize(a)))
        probe.append(backbone(normalize(b)))
    gallery = torch.cat(gallery, dim=0)
    probe = torch.cat(probe, dim=0)
    label = torch.arange(gallery.shape[0], device=gallery.device)
    return gallery, probe, label

# =============== MAIN PIPELINE ===============
def main():
    camera = Pipeline.load("../checkpoints/default.pt")
    pecam_transform = PECAMTransform().cuda().eval()
    pecam_transform.load_state_dict(torch.load("../checkpoints/pecam_generator2_final.pt"))  # <<< 加载训练好的transform

    # dataset_loader = lambda: ClosedSetFaceDetection(
    #     image_dir='/home/hossein/P2E_S4_C3.1',
    #     xml_dir='/home/hossein/groundtruth/P2E_S4_C3.1.xml'
    # )

    dataset_loader = lambda: FaceDS(
            image_dir='/home/hossein/P2E_S5_C1.1',
            anno_file="/home/hossein/face_annotations.json"
        )

    # print(">>> Counting faces under PECAM privacy transform:")
    # count_faces_after_pecam(dataset_loader, pecam_transform, camera)
    print("\n>>> Evaluating face protection under Enhancement module:")
    count_faces_after_enhancement(dataset_loader, pecam_transform, camera)


    print("\n>>> Counting persons under PECAM privacy transform (utility check):")
    count_person_after_pecam(dataset_loader, pecam_transform, camera)

    backbone = pretrained_backbone("arcface_ir18").cuda().eval()
    gallery, probe, label = evaluate_privacy_backbone(dataset_loader, pecam_transform, camera, backbone)
    acc = CMC(gallery, probe, label).cmc_curve()[1][0]
    print(f"\n[SUMMARY] Recognition accuracy under PECAM privacy transform: {acc:.4f}")

if __name__ == "__main__":
    main()
