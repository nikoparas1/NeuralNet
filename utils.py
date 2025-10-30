from pathlib import Path
import ipywidgets as w
from PIL import Image
import numpy as np

WEIGHTS_DIR = Path("weights")


def list_models(dirpath: Path = WEIGHTS_DIR):
    return sorted(p.name for p in dirpath.glob("*.npz"))


def model_dropdown(dirpath: Path = WEIGHTS_DIR) -> w.Dropdown:
    return w.Dropdown(options=list_models(dirpath), description="model")


def refresh_dropdown(dd: w.Dropdown, dirpath: Path = WEIGHTS_DIR):
    dd.options = list_models(dirpath)


def canvas_to_mnist(img28, thresh=10, target=20):
    a = img28 if img28.ndim == 2 else img28.reshape(28, 28)

    if a.dtype == np.float32:
        if a.max() <= 1.0:  # [0,1] → [0,255]
            a_u8 = (a * 255.0).astype(np.uint8)
        else:
            a_u8 = np.clip(a, 0, 255).astype(np.uint8)
    else:
        a_u8 = a.astype(np.uint8)

    mask = a_u8 > thresh
    if not mask.any():
        return np.zeros((1, 784), dtype=np.float32)

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = a_u8[y0:y1, x0:x1]

    h, w = crop.shape
    scale = target / max(h, w)
    newh = max(1, int(round(h * scale)))
    neww = max(1, int(round(w * scale)))
    crop_res = np.array(Image.fromarray(crop).resize((neww, newh), Image.BILINEAR))

    canvas = np.zeros((28, 28), dtype=np.uint8)
    top = (28 - newh) // 2
    left = (28 - neww) // 2
    canvas[top : top + newh, left : left + neww] = crop_res

    return canvas.reshape(1, 784).astype(np.float32) / 255.0
