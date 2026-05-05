import sys
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image, make_grid

# import zero dce++ model.py
ZERO_DCEPP_DIR = Path("external/Zero-DCE_extension/Zero-DCE++")
SNAPSHOT_PATH = ZERO_DCEPP_DIR / "snapshots_Zero_DCE++" / "Epoch99.pth"
sys.path.append(str(ZERO_DCEPP_DIR))

import model

INPUT_DIR = Path("data/LOL/lol_dataset/eval15/low")
OUTPUT_DIR = Path("data/predicted/zero_dcepp")
COMPARISON_DIR = Path("data/predicted/zero_dcepp_comparisons")

# load zero-dce++ model and snapshot
def load_zero_dcepp(device):
    zero_dcepp = model.enhance_net_nopool(scale_factor=1).to(device)
    snapshot = torch.load(SNAPSHOT_PATH, map_location=device)
    zero_dcepp.load_state_dict(snapshot, strict=False)
    zero_dcepp.eval()
    return zero_dcepp


# enchange image with zero-dce++ model
def enhance_image(zero_dcepp, image_path, output_path, comparison_path, device):
    image = Image.open(image_path).convert("RGB")
    x = transforms.ToTensor()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = zero_dcepp(x)

    enhanced = output[0]
    enhanced = enhanced.clamp(0, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)

    save_image(enhanced, output_path)

    comparison = make_grid(
        [x.squeeze(0).cpu(), enhanced.squeeze(0).cpu()],
        nrow=2,
        padding=10
    )
    save_image(comparison, comparison_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zero_dcepp = load_zero_dcepp(device)

    image_paths = list(INPUT_DIR.glob("*.png"))

    for image_path in sorted(image_paths):
        output_path = OUTPUT_DIR / image_path.name
        comparison_path = COMPARISON_DIR / image_path.name

        enhance_image(zero_dcepp, image_path, output_path, comparison_path, device)

    print(f"Saved enhanced images to: {OUTPUT_DIR}")
    print(f"Saved comparisons to: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()