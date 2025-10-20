import math
from typing import Optional, Tuple
from PIL import Image


MAX_ASPECT_RATIO = 200
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384


# Modified version from https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
def _smart_resize(orig_height: int, orig_width: int, resize_factor: int, min_pixels: Optional[int] = None, max_pixels: int = None) -> Tuple[int, int]:
        """
        Rescales the image so that the following conditions are met:
        1. Both dimensions (height and width) are divisible by 'factor'.
        2. The total number of pixels of the resized image is within the range ['min_pixels', 'max_pixels'].
        3. The aspect ratio of the image is maintained as closely as possible.
        """

        if orig_height < resize_factor or orig_width < resize_factor:
            raise ValueError(f"The height ({orig_height}) and width ({orig_width}) of the image must be larger than the factor ({resize_factor}).")
        
        min_pixels = min_pixels if min_pixels is not None else (IMAGE_MIN_TOKEN_NUM * resize_factor ** 2)
        max_pixels = max_pixels if max_pixels is not None else (IMAGE_MAX_TOKEN_NUM * resize_factor ** 2)
        assert max_pixels >= min_pixels, "The max_pixels of the image must be greater than or equal to min_pixels."

        if max(orig_height, orig_width) / min(orig_height, orig_width) > MAX_ASPECT_RATIO:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {MAX_ASPECT_RATIO}, got {max(orig_height, orig_width) / min(orig_height, orig_width)}."
            )
        
        resized_height = round(float(orig_height) / resize_factor) * resize_factor
        resized_width = round(float(orig_width) / resize_factor) * resize_factor

        if resized_height * resized_width > max_pixels:
            max_pixels_scale_factor = math.sqrt((orig_height * orig_width) / max_pixels)
            resized_height = math.floor(float(orig_height) / max_pixels_scale_factor / resize_factor) * resize_factor
            resized_width = math.floor(float(orig_width) / max_pixels_scale_factor / resize_factor) * resize_factor
        elif resized_height * resized_width < min_pixels:
            min_pixels_scale_factor = math.sqrt(min_pixels / (orig_height * orig_width))
            resized_height = math.ceil(orig_height * min_pixels_scale_factor / resize_factor) * resize_factor
            resized_width = math.ceil(orig_width * min_pixels_scale_factor / resize_factor) * resize_factor
        
        return resized_height, resized_width


def get_resize_dims_for_qwen2_5_vl(orig_height: int, orig_width: int) -> Tuple[int, int]:
    return _smart_resize(orig_height, orig_width, resize_factor=28)


def resize_image_for_qwen2_5_vl(image: Image.Image) -> Image.Image:
    new_height, new_width = get_resize_dims_for_qwen2_5_vl(image.height, image.width)
    return image.resize((new_width, new_height))


def resize_point_for_qwen2_5_vl(point: Tuple[int, int], orig_height: int, orig_width: int) -> Tuple[int, int]:
    resized_height, resized_width = get_resize_dims_for_qwen2_5_vl(orig_height, orig_width)

    point_x, point_y = point

    resized_point_x = round(float(point_x) / orig_width * resized_width)
    resized_point_y = round(float(point_y) / orig_height * resized_height)

    return (resized_point_x, resized_point_y)


def resize_point_back_to_original_for_qwen2_5_vl(resized_point: Tuple[int, int], orig_height: int, orig_width: int) -> Tuple[int, int]:
    resized_height, resized_width = get_resize_dims_for_qwen2_5_vl(orig_height, orig_width)

    resized_point_x, resized_point_y = resized_point

    orig_point_x = round(float(resized_point_x) / resized_width * orig_width)
    orig_point_y = round(float(resized_point_y) / resized_height * orig_height)

    return (orig_point_x, orig_point_y)
