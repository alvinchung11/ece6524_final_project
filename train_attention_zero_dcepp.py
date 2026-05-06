from pathlib import Path
import torch
import torch.nn as nn
import random
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pytorch_msssim import ssim

# zero dce++ snapshot
ZERO_DCEPP_DIR = Path("external/Zero-DCE_extension/Zero-DCE++").resolve()
SNAPSHOT_PATH = ZERO_DCEPP_DIR / "snapshots_Zero_DCE++" / "Epoch99.pth"

TRAIN_LOW_DIR = Path("data/LOL/lol_dataset/our485/low")
TRAIN_HIGH_DIR = Path("data/LOL/lol_dataset/our485/high")

SAVE_DIR = Path("snapshots")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

from attention_zero_dcepp import AttentionZeroDCEPP

def gamma_adjust(image, gamma):

    # Convert image to HSV
    hsv_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    
    # Scale intensity values to [0, 1]
    scaled_intensity = hsv_image[:, :, 2] / 255

    gamma_intensity = np.power(scaled_intensity, gamma)           # Apply gamma correction

    gamma_intensity = gamma_intensity * 255.0                       # Scale image back to [0, 255]
    gamma_intensity = np.clip(gamma_intensity, 0, 255)              # Clip to range [0, 255]
    gamma_intensity = np.round(gamma_intensity).astype(np.uint8)    # Convert to integers

    hsv_image[:, :, 2] = gamma_intensity

    output = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)

    return output

def local_darken(image, gamma=3.0, rng=None):
    if rng is None:
        rng = random
        
    output = image.copy()

    h, w, _ = image.shape

    rect_w = rng.randint(w // 4, w // 2)
    rect_h = rng.randint(h // 4, h // 2)

    x1 = rng.randint(0, w - rect_w)
    y1 = rng.randint(0, h - rect_h)

    x2 = x1 + rect_w
    y2 = y1 + rect_h

    patch = output[y1:y2, x1:x2]
    dark_patch = gamma_adjust(patch, gamma)

    output[y1:y2, x1:x2] = dark_patch

    return output


# LOL custum dataset class
class LOLDataset(Dataset):
    def __init__(
        self,
        low_dir,
        high_dir,
        mode="mixed",
        local_prob=0.25,
        gamma_range=(2.0, 5.0),
        random_darken=True
    ):
        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir)

        self.mode = mode
        self.local_prob = local_prob
        self.gamma_range = gamma_range

        self.image_paths = sorted(list(self.high_dir.glob("*.png")))
        self.transform = transforms.ToTensor()
        self.random = random_darken

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        rng = random.Random(index) if self.random is False else random
        high_path = self.image_paths[index]
        low_path = self.low_dir / high_path.name

        high_image = Image.open(high_path).convert("RGB")

        if self.mode == "lol_low":
            low_image = Image.open(low_path).convert("RGB")

        elif self.mode == "local_darken":
            gamma = rng.uniform(*self.gamma_range)

            low_image = np.array(high_image)
            low_image = local_darken(low_image, gamma=gamma, rng=rng)
            low_image = Image.fromarray(low_image)

        elif self.mode == "mixed":
            use_local_darken = rng.random() < self.local_prob

            if use_local_darken:
                gamma = rng.uniform(*self.gamma_range)

                low_image = np.array(high_image)
                low_image = local_darken(low_image, gamma=gamma, rng=rng)
                low_image = Image.fromarray(low_image)
            else:
                low_image = Image.open(low_path).convert("RGB")

        else:
            raise ValueError("mode must be 'lol_low', 'local_darken', or 'mixed'")

        low_tensor = self.transform(low_image)
        high_tensor = self.transform(high_image)

        return low_tensor, high_tensor
    

# load attention zero-dce++ model
def load_attention_zero_dcepp(device):
    attention_zero_dcepp_model = AttentionZeroDCEPP(scale_factor=1).to(device)
    snapshot = torch.load(SNAPSHOT_PATH, map_location=device)
    attention_zero_dcepp_model.load_state_dict(snapshot, strict=False)
    return attention_zero_dcepp_model


# validation loss for early stopping
def validate(zero_dcepp_model, val_dataloader, l1_loss, curve_map, device):
    zero_dcepp_model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for low_images, high_images in val_dataloader:
            low_images = low_images.to(device)
            high_images = high_images.to(device)

            output = zero_dcepp_model(low_images)
            enhanced_images = output[0]

            loss = enhancement_loss(l1_loss, enhanced_images, high_images, curve_map)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    zero_dcepp_model.train()

    return avg_val_loss


def enhancement_loss(l1_loss, enhanced_images, high_images, curve_map):
    l1 = l1_loss(enhanced_images, high_images)
    ssim_loss = 1 - ssim(
        enhanced_images,
        high_images,
        data_range=1.0,
        size_average=True
    )

    dx = torch.abs(curve_map[:, :, :, 1:] - curve_map[:, :, :, :-1]).mean()
    dy = torch.abs(curve_map[:, :, 1:, :] - curve_map[:, :, :-1, :]).mean()
    curve_smoothness = dx + dy

    return l1 + 0.1 * ssim_loss + 0.05 * curve_smoothness


# fine tune zero-dce++ model with spacial attention layer on LOL dataset
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    split_dataset = LOLDataset(TRAIN_LOW_DIR, TRAIN_HIGH_DIR,mode="lol_low")
    val_size = int(0.15 * len(split_dataset))
    train_size = len(split_dataset) - val_size

    generator = torch.Generator().manual_seed(6524)

    train_index_subset, val_index_subset = random_split(range(len(split_dataset)), [train_size, val_size], generator=generator)

    train_indices = train_index_subset.indices
    val_indices = val_index_subset.indices

    train_base_dataset = LOLDataset(
        TRAIN_LOW_DIR,
        TRAIN_HIGH_DIR,
        mode="mixed",
        local_prob=0.25,
        gamma_range=(2.0, 5.0),
        random_darken=True
    )
    train_dataset = Subset(train_base_dataset, train_indices)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True
    )

    val_base_dataset = LOLDataset(
        TRAIN_LOW_DIR,
        TRAIN_HIGH_DIR,
        mode="mixed",
        local_prob=0.1,
        gamma_range=(2.0, 5.0),
        random_darken=False
    )
    val_dataset = Subset(val_base_dataset, val_indices)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False
    )

    attention_zero_dcepp_model = load_attention_zero_dcepp(device)
    attention_zero_dcepp_model.train()

    for name, param in attention_zero_dcepp_model.named_parameters():
        if "spatial_attention" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    l1_loss = nn.L1Loss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, attention_zero_dcepp_model.parameters()), lr=1e-3)

    max_epochs = 150
    patience = 5
    delta = 1e-4

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    best_model_path = SAVE_DIR / "attention_zero_dcepp_best.pth"

    for epoch in range(max_epochs):
        total_train_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{max_epochs}")

        for low_images, high_images in progress_bar:
            low_images = low_images.to(device)
            high_images = high_images.to(device)

            output = attention_zero_dcepp_model(low_images)
            enhanced_images = output[0]
            curve_map = output[1]

            loss = enhancement_loss(l1_loss, enhanced_images, high_images, curve_map)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = validate(attention_zero_dcepp_model, val_dataloader, l1_loss, curve_map, device)

        print(
            f"Epoch {epoch + 1}: "
            f"train loss = {avg_train_loss:.6f}, "
            f"val loss = {avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss - delta:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0

            torch.save(attention_zero_dcepp_model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping.")
            print(f"Best model saved to: {best_model_path}")
            break

if __name__ == "__main__":
    train()