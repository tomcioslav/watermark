import math
import random
from abc import ABC, abstractmethod

import watermark.config as config
import watermark.utils as utils
from PIL import Image, ImageDraw, ImageFont


class Watermark(ABC):
    def __init__(self, randomize: bool):
        self.randomize = randomize
        self.watermark = self.create_watermark()

    @abstractmethod
    def create_watermark() -> Image.Image:
        pass

    def get_watermark(self):

        if not self.randomize:
            return self.watermark
        else:
            rotation_angle = random.randint(0, 359)
            randomized_watermark = self.watermark.copy().rotate(rotation_angle)
            horizontal_shift = random.randint(-300, 300)
            vertical_shift = random.randint(-300, 300)

            return randomized_watermark.transform(
                (1000, 1000),
                Image.AFFINE,
                (1, 0, horizontal_shift, 0, 1, vertical_shift),
            )

    def add_to_image(self, image: Image.Image) -> Image.Image:
        resized_watermark = self.get_watermark().resize(size=image.size)
        watermarked_image = Image.alpha_composite(image, resized_watermark)
        return watermarked_image


class LogoWatermark(Watermark):
    def __init__(
        self,
        logo_path: str,
        percentage_size: int,
        intensity: int,
        randomize: bool = False,
    ):
        self.logo = Image.open(logo_path).convert("L")
        self.percentage_size = percentage_size
        self.intensity = intensity
        super().__init__(randomize)

    def create_watermark(self):
        target_logo_size = int(math.sqrt(self.percentage_size / 100) * 1000)
        inv_logo = self.logo.point(lambda p: self.intensity if p < 175 else 0)
        watermark = Image.new("RGB", self.logo.size, (255, 255, 255))
        watermark.putalpha(inv_logo)
        watermark = utils.expand_watermark_to_square(
            image=watermark, size=max(self.logo.size)
        ).resize(
            (
                target_logo_size,
                target_logo_size,
            )
        )
        watermark = utils.expand_watermark_to_square(
            image=watermark, size=config.watermark_settings.WATERMARK_SIZE
        )

        return watermark


class TextWatermark(Watermark):
    def __init__(
        self, text: str, percentage_size: int, intensity: int, randomize: bool = False
    ):
        self.text = text
        self.percentage_size = percentage_size
        self.intensity = intensity

        super().__init__(randomize)

    def create_watermark(self) -> Image.Image:
        watermark = Image.new("RGBA", (1000, 1000), (255, 255, 255, 0))
        d = ImageDraw.Draw(watermark)

        font_size = 1
        font = ImageFont.truetype(
            str(config.paths.FONTS / "DroidSans.ttf"),
            font_size,
        )
        text_left, text_top, text_right, text_bottom = d.textbbox(
            (0, 0), self.text, font
        )
        while (text_right - text_left) * (
            text_bottom - text_top
        ) * 100 / self.percentage_size < 1000000:
            font_size += 1

            font = ImageFont.truetype(
                str(config.paths.FONTS / "DroidSans.ttf"),
                font_size,
            )
            text_left, text_top, text_right, text_bottom = d.textbbox(
                (0, 0), self.text, font
            )

        d.text(
            ((1000 - text_right) / 2, (1000 - text_bottom) / 2),
            self.text,
            fill=(255, 255, 255, self.intensity),
            font=font,
        )
        return watermark


class RectangleWatermark(Watermark):
    def __init__(
        self,
        offset_top_percentage: int,
        offset_bottom_percentage: int,
        offset_left_percentage: int,
        offset_right_percentage: int,
        intensity: int,
        randomize: bool = False,
    ):
        self.offset_top_percentage = offset_top_percentage
        self.offset_bottom_percentage = offset_bottom_percentage
        self.offset_left_percentage = offset_left_percentage
        self.offset_right_percentage = offset_right_percentage
        self.intensity = intensity

        super().__init__(randomize)

    def create_watermark(
        self,
    ) -> Image.Image:

        watermark = Image.new(
            "RGBA",
            (
                (100 - self.offset_left_percentage - self.offset_right_percentage) * 10,
                (100 - self.offset_top_percentage - self.offset_bottom_percentage) * 10,
            ),
            (255, 255, 255, self.intensity),
        )
        watermark = utils.expand_watermark_to_square(
            image=watermark, size=config.watermark_settings.WATERMARK_SIZE
        )

        return watermark
