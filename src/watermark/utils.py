from PIL import Image


def expand_watermark_to_square(image: Image.Image, size: int) -> Image.Image:
    image_width, image_height = image.size
    full_square_watermark = Image.new(
        "RGBA",
        (
            size,
            size,
        ),
        (255, 255, 255, 0),
    )
    full_square_watermark.paste(
        image,
        (
            (size - image_width) // 2,
            (size - image_height) // 2,
        ),
    )
    return full_square_watermark
