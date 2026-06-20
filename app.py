import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import matplotlib
import rasterio
from rasterio.io import MemoryFile
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import EuroSATResNet

app = FastAPI(title="EuroSAT Multispectral API")

# Allow the Next.js dashboard (or any client) to call this API directly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# EuroSAT folder names are sorted alphabetically in dataset.py, so the model's
# output indices map to this order.
CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

EUROSAT_MEAN = [934.04, 1032.94, 1114.26, 2260.64]
EUROSAT_STD = [593.97, 395.56, 330.55, 1148.44]

device = torch.device("cpu")
model = EuroSATResNet()
model.load_state_dict(torch.load("resnet18_4channel_v1.pth", map_location=device))
model.eval()


def read_tiff_bands(image_bytes: bytes):
    """Read Sentinel-2 bands from a .tiff: 4=Red, 3=Green, 2=Blue, 8=NIR."""
    with MemoryFile(image_bytes) as memfile:
        with memfile.open() as dataset:
            r = dataset.read(4).astype(np.float32)
            g = dataset.read(3).astype(np.float32)
            b = dataset.read(2).astype(np.float32)
            nir = dataset.read(8).astype(np.float32)
    return r, g, b, nir


def stretch_to_uint8(channel: np.ndarray) -> np.ndarray:
    """Percentile contrast stretch a single band to 0-255 for display."""
    p2, p98 = np.percentile(channel, (2, 98))
    if p98 <= p2:
        p2, p98 = float(channel.min()), float(channel.max() + 1e-6)
    stretched = np.clip((channel - p2) / (p98 - p2), 0.0, 1.0)
    return stretched


def array_to_png_b64(arr_uint8: np.ndarray, upscale: int = 4) -> str:
    """Encode an HxWx3 uint8 array to a base64 PNG data URL, nearest-neighbor upscaled."""
    img = Image.fromarray(arr_uint8, mode="RGB")
    if upscale > 1:
        img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_true_color_rgb(r, g, b) -> np.ndarray:
    """Derive a true-color RGB image (0-1 float) from the visible bands."""
    rgb = np.stack(
        [stretch_to_uint8(r), stretch_to_uint8(g), stretch_to_uint8(b)], axis=-1
    )
    return rgb  # float 0-1, HxWx3


def build_ndvi(r, nir):
    """NDVI = (NIR - Red) / (NIR + Red), returns (ndvi_map, mean_ndvi)."""
    denom = nir + r
    denom[denom == 0] = 1e-6
    ndvi = (nir - r) / denom
    ndvi = np.clip(ndvi, -1.0, 1.0)
    return ndvi, float(np.mean(ndvi))


def colorize_ndvi(ndvi: np.ndarray) -> np.ndarray:
    """Map NDVI [-1,1] to an RdYlGn colormap (red=bare, green=healthy)."""
    normalized = (ndvi + 1.0) / 2.0  # -> 0..1
    colormap = matplotlib.colormaps["RdYlGn"]
    colored = colormap(normalized)[:, :, :3]  # drop alpha
    return (colored * 255).astype(np.uint8)


def health_label(mean_ndvi: float) -> str:
    if mean_ndvi >= 0.6:
        return "Very healthy vegetation"
    if mean_ndvi >= 0.4:
        return "Healthy vegetation"
    if mean_ndvi >= 0.2:
        return "Sparse / stressed vegetation"
    if mean_ndvi >= 0.0:
        return "Bare soil / minimal vegetation"
    return "Water / non-vegetated"


def build_gradcam(input_tensor: torch.Tensor, rgb_float: np.ndarray, class_idx: int) -> np.ndarray:
    """Run Grad-CAM for the predicted class and overlay it on the RGB image."""
    target_layers = [model.engine.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(rgb_float.astype(np.float32), grayscale_cam, use_rgb=True)
    return visualization  # HxWx3 uint8


@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(
            status_code=400,
            detail="Only .tiff / .tif multispectral files are supported.",
        )

    image_bytes = await file.read()

    try:
        r, g, b, nir = read_tiff_bands(image_bytes)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"Could not read the .tiff bands (expected 4=R,3=G,2=B,8=NIR): {exc}",
        )

    # Build the 4-channel input tensor the model was trained on.
    img_array = np.stack([r, g, b, nir], axis=0)
    img_tensor = torch.from_numpy(img_array).float()
    normalizer = T.Normalize(mean=EUROSAT_MEAN, std=EUROSAT_STD)
    input_tensor = normalizer(img_tensor).unsqueeze(0)

    # Classification + confidence
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    predicted_idx = int(predicted_idx.item())
    predicted_class = (
        CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else str(predicted_idx)
    )

    # Imagery outputs
    rgb_float = build_true_color_rgb(r, g, b)
    rgb_png = array_to_png_b64((rgb_float * 255).astype(np.uint8))

    ndvi_map, mean_ndvi = build_ndvi(r, nir)
    ndvi_png = array_to_png_b64(colorize_ndvi(ndvi_map))

    gradcam_img = build_gradcam(input_tensor, rgb_float, predicted_idx)
    gradcam_png = array_to_png_b64(gradcam_img)

    return {
        "filename": filename,
        "predicted_class_index": predicted_idx,
        "predicted_class": predicted_class,
        "confidence": round(float(confidence.item()), 4),
        "vegetation": {
            "mean_ndvi": round(mean_ndvi, 4),
            "health_label": health_label(mean_ndvi),
        },
        "images": {
            "rgb": rgb_png,
            "ndvi": ndvi_png,
            "gradcam": gradcam_png,
        },
    }
