from PIL import Image


class WatermarkApplier:
    def apply(self, image: Image.Image, watermark: Image.Image) -> Image.Image:
        image = image.copy()
        watermark = watermark.copy()

        resized_watermark = watermark.resize(size=image.size)
        watermarked = Image.alpha_composite(image, resized_watermark)
        return watermarked
