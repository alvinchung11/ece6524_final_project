import sys
from pathlib import Path

import torch
import torch.nn as nn

# import zero dce++ model.py
ZERO_DCEPP_DIR = Path("external/Zero-DCE_extension/Zero-DCE++")
sys.path.append(str(ZERO_DCEPP_DIR))

import model

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        attention_input = torch.cat([avg_pool, max_pool], dim=1)
        attention_map = self.attention(attention_input)

        return x + self.alpha * x * attention_map


class AttentionZeroDCEPP(model.enhance_net_nopool):
    def __init__(self, scale_factor=1):
        super().__init__(scale_factor=scale_factor)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):                
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x4 = self.spatial_attention(x4)

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        enhance_image = self.enhance(x, x_r)
        return enhance_image, x_r