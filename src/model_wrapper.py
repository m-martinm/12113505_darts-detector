from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
# from ultralytics import YOLO


class CordCNN(torch.nn.Module):
    def __init__(self, input_channels=1):
        super(CordCNN, self).__init__()
        in_dim = input_channels + 2
        self.features = nn.Sequential(
            # 1. Block with dilation (Increased receptive field)
            nn.Conv2d(in_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 2. Block with dilation (Increased receptive field)
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 3. Block without dilation
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inputs should be tensors in (B, C, H, W) format"""
        # https://arxiv.org/pdf/1807.03247
        batch_size, c, h, w = x.size()

        # Create coordinate grid [-1, 1]
        y_coords = torch.linspace(-1, 1, h, device=x.device)
        x_coords = torch.linspace(-1, 1, w, device=x.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Expand and Cat
        xx = xx.expand(batch_size, 1, h, w)
        yy = yy.expand(batch_size, 1, h, w)
        x = torch.cat([x, xx, yy], dim=1)

        # Forward pass through the network
        x = self.features(x)
        x = self.regressor(x)

        return x

    def load_weights(
        self, weight_path: Path, device: torch.device = torch.device("cpu")
    ):
        state_dict = torch.load(weight_path, map_location=device)
        self.load_state_dict(state_dict)

    def infer(self, img: np.ndarray) -> np.ndarray:
        assert len(img.shape) == 2, "Input image must be grayscale"
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        self.eval()

        with torch.no_grad():
            return self.forward(x).squeeze(0).cpu().numpy()
