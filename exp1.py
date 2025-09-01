import sys
import os.path as osp
from sklearn.metrics import precision_score, recall_score, f1_score


root = osp.join(
    osp.dirname((osp.abspath(__file__))),
    "..",
)
sys.path.append(root)

from datetime import datetime
import pandas as pd
import torch
import torchvision as tv
from torch.utils.data import DataLoader
from torch import nn, optim
from datetime import datetime
from facenet_pytorch import MTCNN


from rich.progress import track
from isp.pipeline import Pipeline

from privacy.data import ClosedSetCelebA, ClosedSetLFW, ClosedSetFaceDetection, FaceDS
from privacy.backbone import pretrained_backbone
from privacy.metric import CMC

from unet import UNet
import torchvision.transforms as T


def linear_classifier(gallery, probe, label):
    out_channels, in_channels = gallery.shape
    f = nn.Linear(in_channels, out_channels).cuda()
    g = nn.CrossEntropyLoss()
    opt = optim.SGD(f.parameters(), 5)
    best_acc = 0
    for _ in range(200):
        pred = f(gallery)
        with torch.no_grad():
            pred2 = f(probe)
        loss = g(pred, label)
        test_acc = (pred2.max(dim=1)[1] == label).float().mean().item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        best_acc = max(test_acc, best_acc)
    return best_acc


@torch.no_grad()
def evaluate(d_class, t_func, backbone):
    normalize = tv.transforms.Normalize([0.5] * 3, [0.5] * 3)
    loader = DataLoader(d_class(), batch_size=64, num_workers=8)
    gallery, probe = [], []
    for a, b in track(loader):
        a, b = a.cuda(), b.cuda()
        b = t_func(b)
        gallery.append(backbone(normalize(a)))
        probe.append(backbone(normalize(b)))
    gallery = torch.cat(gallery, dim=0)
    probe = torch.cat(probe, dim=0)
    label = torch.arange(gallery.shape[0], device=gallery.device)
    return gallery, probe, label


from torchvision.transforms.functional import to_pil_image

@torch.no_grad()
def count_faces_after_enhancement(dataset, enhancer, camera):
    from sklearn.metrics import precision_score, recall_score, f1_score
    loader = DataLoader(dataset(), batch_size=1, num_workers=4)
    mtcnn = MTCNN(keep_all=False, device='cuda')

    total = 0
    protect_success = 0

    y_true = []  # 1 = 图像需要保护（原图有人脸）
    y_pred = []  # 1 = 成功保护（增强图未检测到脸）

    for img, info in track(loader, description="Evaluating Protection"):
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

        # 是否成功保护（raw 有，enh 没）
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

@torch.no_grad()
def count_person_after_enhancement(dataset, enhancer, camera, threshold=0.5):
    """
    统计 PECAM/Enhancement 变换前后，图像中可检测到的人数。
    - dataset: 数据集构造函数
    - enhancer: 已训练的增强/隐私保护模型
    - camera: ISP pipeline
    - threshold: 检测置信度阈值
    """
    loader = DataLoader(dataset(), batch_size=1, num_workers=4)
    # 这里使用 torchvision 内置的 person detector
    person_detector = tv.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().cuda()

    total_images = 0
    total_person_raw = 0
    total_person_enh = 0

    for img, _ in track(loader, description="Evaluating Person Utility"):
        img = img.cuda()
        total_images += 1

        # 原图检测
        preds_raw = person_detector([img[0]])[0]
        labels, scores = preds_raw["labels"], preds_raw["scores"]
        raw_person_count = ((labels == 1) & (scores > threshold)).sum().item()
        total_person_raw += raw_person_count

        # 增强图检测
        enhanced = enhancer(camera.forward(img)).clamp(0, 1)[0]
        preds_enh = person_detector([enhanced])[0]
        labels, scores = preds_enh["labels"], preds_enh["scores"]
        enh_person_count = ((labels == 1) & (scores > threshold)).sum().item()
        total_person_enh += enh_person_count

    print(f"[SUMMARY] Persons detected on raw images: {total_person_raw} over {total_images} images")
    print(f"[SUMMARY] Persons detected on enhanced images: {total_person_enh} over {total_images} images")

    if total_person_raw > 0:
        utility = total_person_enh / total_person_raw
        print(f"[UTILITY] Person detection preservation ratio: {utility:.4f}")
        if utility < 0.7:
            print("⚠️ Warning: Enhancement reduces person detection significantly, consider weakening transform.")
    else:
        print("⚠️ Warning: No persons detected in raw images, cannot compute utility.")
   


def main():
    # Load optimized ISP parameters
    camera = Pipeline.load("../checkpoints/default.pt")

    # Load trained image enhancer
    ckpt = "../checkpoints/UNet.pt"
    enhancer = UNet()
    enhancer.load_state_dict(torch.load(ckpt))
    enhancer.eval().cuda()

    ## count face
    # x = lambda: ClosedSetFaceDetection(
    #     image_dir='/home/hossein/P2E_S4_C3.1',
    #     xml_dir='/home/hossein/groundtruth/P2E_S4_C3.1.xml',
    # )
    x = lambda: FaceDS(
            image_dir='/home/hossein/P2E_S5_C1.1',
            anno_file="/home/hossein/face_annotations.json"
        )

    print(">>> Counting detectable faces after enhancement in P2E_S5_C1.1 dataset:")
    count_faces_after_enhancement(x, enhancer, camera)
    
    print("\n>>> Counting detectable persons after enhancement (utility check):")
    count_person_after_enhancement(x, enhancer, camera)


    exit()
    # Experiment Configs
    # You can arbitrarily comment or uncomment some lines to enable or disable some experiments.
    models = [
        # "facenet",
        "arcface_ir18",
        # "arcface_irse50",
        # "arcface_ir152",
        # "magface_ir18",
        # "magface_ir50",
        # "magface_ir100",
        # "adaface_ir18",
        # "adaface_ir50",
        # "adaface_ir100",
    ]
    datasets = [
        # ("CelebA", ClosedSetCelebA),
        # ("LFW", ClosedSetLFW),
        ("P2ES5", lambda: FaceDS(
            image_dir='/home/hossein/P2E_S5_C1.1',
            anno_file="/home/hossein/face_annotations.json"
        )),
        ("FaceDet", lambda: ClosedSetFaceDetection(
        image_dir='/home/hossein/P2E_S4_C3.1',
        xml_dir='/home/hossein/groundtruth/P2E_S4_C3.1.xml',
        ))
    ]
    transforms = [
        # ("Raw", lambda o: o),
        # ("Captured", camera.forward),
        ("Enhanced", lambda o: enhancer(camera.forward(o)).clamp(0, 1)),
    ]
    classifiers = [
        ("Nearest", lambda g, p, l: CMC(g, p, l).cmc_curve()[1][0]),
        ("Linear", lambda g, p, l: linear_classifier(g, p, l)),
    ]
    # Expected computation time: 30 minutes for 1 repeat time
    repeat = 10

    # Start evaluation
    table = []
    for d_name, d_class in datasets:
        for c_name, classifer in classifiers:
            for t_name, t_func in transforms:
                tic = datetime.now()
                row = [d_name, c_name, t_name]
                for m in models:
                    backbone = pretrained_backbone(m).cuda().eval()
                    avg_acc = []
                    for _ in range(repeat):
                        gallery, probe, label = evaluate(d_class, t_func, backbone)
                        acc = classifer(gallery, probe, label)
                        avg_acc.append(acc)
                    avg_acc = sum(avg_acc) / len(avg_acc)
                    row.append(avg_acc)
                table.append(row)
                toc = datetime.now()
                print(toc - tic)

    df = pd.DataFrame(table, columns=["Dataset", "Image Type", "Classifier"] + models)
    df["Average"] = df.iloc[:, 3 : 3 + len(models)].mean(axis=1)
    df = df.reset_index(drop=True)
    print(df.head())
    df.to_csv("../results/1.csv", index=False)


if __name__ == "__main__":
    main()
