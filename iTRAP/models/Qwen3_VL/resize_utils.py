from typing import Tuple

QWEN3_VL_INTERNAL_SIZE = 1000


def resize_point_for_qwen3_vl(orig_point: Tuple[int, int], orig_height: int, orig_width: int) -> Tuple[int, int]:
    orig_point_x, orig_point_y = orig_point

    resized_point_x = round(orig_point_x / orig_width * QWEN3_VL_INTERNAL_SIZE)
    resized_point_y = round(orig_point_y / orig_height * QWEN3_VL_INTERNAL_SIZE)

    return (resized_point_x, resized_point_y)


def resize_point_back_to_original_for_qwen3_vl(resized_point: Tuple[int, int], orig_height: int, orig_width: int) -> Tuple[int, int]:
    resized_point_x, resized_point_y = resized_point

    orig_point_x = round(resized_point_x / QWEN3_VL_INTERNAL_SIZE * orig_width)
    orig_point_y = round(resized_point_y / QWEN3_VL_INTERNAL_SIZE * orig_height)

    return (orig_point_x, orig_point_y)
