import random

import numpy as np
import torch
import watermark
import watermark.config as config
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class WatermarkDataset(Dataset):
    def __init__(self, watermarks: list[watermark.Watermark], image_paths: list[str]):
        self.image_paths = image_paths

        self.input_transform = transforms.Compose(
            [
                transforms.RandomChoice(watermarks),
                transforms.Resize(config.training_settings.SIZE_OF_IMAGE),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )

        self.target_transform = transforms.Compose(
            [
                transforms.Resize(config.training_settings.SIZE_OF_IMAGE),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")

        return self.input_transform(image), self.target_transform(image)

    def show_sample(self):
        sample_idx = random.randint(0, len(self))
        input_image_tensor, target_image_tensor = self[sample_idx]
        input_image, target_image = transforms.ToPILImage()(
            input_image_tensor
        ), transforms.ToPILImage()(target_image_tensor)

        return Image.fromarray(
            np.hstack((np.array(input_image), np.array(target_image)))
        )
