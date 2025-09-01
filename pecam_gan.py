import sys
import os.path as osp
root = osp.join(osp.dirname(osp.abspath(__file__)), "..")
sys.path.append(root)

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.utils import save_image
from facenet_pytorch import InceptionResnetV1
from rich.progress import track

from isp.pipeline import Pipeline
from privacy.data import ClosedSetFaceDetection

# ======== PECAM Transform (Generator) ========
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

# ======== Recovery Decoder ========
class RecoveryDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1), nn.Sigmoid()
        )
    def forward(self, encoded):
        return self.decoder(encoded)

# ======== Discriminator ========
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1)

# ======== Training Script ========
def main():
    device = torch.device('cuda')
    camera = Pipeline.load("../checkpoints/default.pt")

    # Dataset
    dataset = ClosedSetFaceDetection(
        image_dir='/home/hossein/P2E_S4_C3.1',
        xml_dir='/home/hossein/groundtruth/P2E_S4_C3.1.xml'
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8)

    # Models
    generator = PECAMTransform().to(device)
    recovery_decoder = RecoveryDecoder().to(device)
    discriminator = Discriminator().to(device)
    recognizer = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Optimizers and Losses
    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_RD = optim.Adam(recovery_decoder.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()
    criterion_feat = nn.MSELoss()

    lambda_l1 = 30
    lambda_feat = 10

    # Training loop
    for epoch in range(30):
        for img, _ in track(loader, description=f"Epoch {epoch}"):
            img = img.to(device)
            real_img = img
            # print(f"Input img min/max: {img.min().item():.3f}/{img.max().item():.3f}")
            # real_img = camera.forward(img)
            # print(f"Output real_img min/max: {real_img.min().item():.3f}/{real_img.max().item():.3f}")
            # exit()
            # =========== Discriminator Step ===========
            optimizer_D.zero_grad()
            with torch.no_grad():
                priv_img_D = generator(real_img).detach()
            d_real = discriminator(real_img)
            d_fake = discriminator(priv_img_D)
            real_label = torch.ones_like(d_real, device=device)
            fake_label = torch.zeros_like(d_fake, device=device)
            loss_D = criterion_GAN(d_real, real_label) + criterion_GAN(d_fake, fake_label)
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            # =========== Generator Step ===========
            optimizer_G.zero_grad()
            encoded_features_G = generator.encoder(real_img)
            priv_img = generator.decoder(encoded_features_G)
            d_fake_for_g = discriminator(priv_img)
            loss_G_GAN = criterion_GAN(d_fake_for_g, real_label)
            loss_G_L1 = criterion_L1(priv_img, real_img)
            with torch.no_grad():
                feat_real = recognizer(real_img)
            feat_priv = recognizer(priv_img)
            loss_G_feat = criterion_feat(feat_priv, feat_real)
            loss_G = loss_G_GAN + lambda_l1 * loss_G_L1 + lambda_feat * loss_G_feat
            loss_G.backward(retain_graph=True)
            optimizer_G.step()

            # =========== Recovery Decoder Step ===========
            optimizer_RD.zero_grad()
            with torch.no_grad():
                encoded_features_RD = generator.encoder(real_img).detach()
            recovery_img = recovery_decoder(encoded_features_RD.detach())
            loss_RD_L1 = criterion_L1(recovery_img, real_img)
            loss_RD_L1.backward()
            optimizer_RD.step()

            # ===== Save visuals for monitoring =====
            save_image(real_img.clamp(0, 1), f"../pecam_samples/original_epoch{epoch}.png", nrow=4)
            save_image(priv_img.clamp(0, 1), f"../pecam_samples/private_epoch{epoch}.png", nrow=4)
            save_image(recovery_img.clamp(0, 1), f"../pecam_samples/recovery_epoch{epoch}.png", nrow=4)


            print(f"[Epoch {epoch}] G_GAN: {loss_G_GAN.item():.4f}, G_L1: {loss_G_L1.item():.4f}, G_feat: {loss_G_feat.item():.4f}, RD_L1: {loss_RD_L1.item():.4f}, D: {loss_D.item():.4f}")

        # Periodic checkpointing
        if epoch % 10 == 0:
            torch.save(generator.state_dict(), f"../checkpoints/pecam_generator2_epoch{epoch}.pt")
            torch.save(recovery_decoder.state_dict(), f"../checkpoints/pecam_recovery2_decoder_epoch{epoch}.pt")

    torch.save(generator.state_dict(), "../checkpoints/pecam_generator2_final.pt")
    torch.save(recovery_decoder.state_dict(), "../checkpoints/pecam_recovery_decoder2_final.pt")
    print("Training completed and models saved.")

if __name__ == '__main__':
    main()
