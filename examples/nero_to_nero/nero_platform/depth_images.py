"""深度编码范围与界面伪彩色，与设备厂商无关。"""

import cv2
import numpy as np

DEPTH_MAX_METERS = 10.0


def depth_preview(depth: np.ndarray) -> np.ndarray:
    """固定 0～10 m 伪彩色，零值黑色；仅用于界面，绝不替代原始深度。"""
    values = depth[..., 0] if depth.ndim == 3 else depth
    height, width = values.shape
    ratio = min(1.0, 640 / width, 480 / height)
    values = cv2.resize(
        values, (max(1, round(width * ratio)), max(1, round(height * ratio))), interpolation=cv2.INTER_NEAREST
    )
    normalized = (np.clip(values, 0, DEPTH_MAX_METERS) / DEPTH_MAX_METERS * 255).astype(np.uint8)
    image = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    image[values <= 0] = 0
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
