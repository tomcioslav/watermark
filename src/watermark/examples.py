import watermark
import watermark.config as config
import watermark.data.dataset as dataset
from torch.utils.data import DataLoader

train_images_paths = list(config.paths.RAW.glob("train/*"))
test_images_paths = list(config.paths.RAW.glob("test/*"))
logo_paths = list(config.paths.LOGOS.glob("*"))

full_logo_train_dataset = dataset.WatermarkDataset(
    watermarks=[
        watermark.LogoWatermark(
            logo_path=logo_path, percentage_size=15, intensity=125, randomize=True
        )
        for logo_path in logo_paths
    ],
    image_paths=train_images_paths,
)
full_logo_test_dataset = dataset.WatermarkDataset(
    watermarks=[
        watermark.LogoWatermark(
            logo_path=logo_path, percentage_size=15, intensity=125, randomize=True
        )
        for logo_path in logo_paths
    ],
    image_paths=test_images_paths,
)

sample_logo_train_dataset = dataset.WatermarkDataset(
    watermarks=[
        watermark.LogoWatermark(
            logo_path=logo_path, percentage_size=15, intensity=125, randomize=True
        )
        for logo_path in logo_paths
    ],
    image_paths=train_images_paths[:100],
)
sample_logo_test_dataset = dataset.WatermarkDataset(
    watermarks=[
        watermark.LogoWatermark(
            logo_path=logo_path, percentage_size=15, intensity=125, randomize=True
        )
        for logo_path in logo_paths
    ],
    image_paths=test_images_paths[:20],
)

full_logo_train_dataloader = DataLoader(
    dataset=full_logo_train_dataset,
    batch_size=config.training_settings.SIZE_OF_BATCH,
    shuffle=True,
    num_workers=16,
)
full_logo_test_dataloader = DataLoader(
    dataset=full_logo_test_dataset,
    batch_size=config.training_settings.SIZE_OF_BATCH,
    shuffle=False,
    num_workers=16,
)

sample_logo_train_dataloader = DataLoader(
    dataset=sample_logo_train_dataset,
    batch_size=config.training_settings.SIZE_OF_BATCH,
    shuffle=True,
    num_workers=16,
)
sample_logo_test_dataloader = DataLoader(
    dataset=sample_logo_test_dataset,
    batch_size=config.training_settings.SIZE_OF_BATCH,
    shuffle=False,
    num_workers=16,
)
