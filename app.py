import numpy as np
import torch
import cv2
import torchvision.transforms as T
from fastapi import FastAPI, UploadFile, File
from model import EuroSATResNet
import rasterio
from rasterio.io import MemoryFile

app = FastAPI(title="EuroSAT Multispectral API")

EUROSAT_MEAN = [934.04, 1032.94, 1114.26, 2260.64]
EUROSAT_STD = [593.97, 395.56, 330.55, 1148.44]

device = torch.device("cpu")
model = EuroSATResNet()
model.load_state_dict(torch.load("resnet18_4channel_v1.pth", map_location=device))
model.eval()

def prepare_and_transform(image_bytes: bytes, filename: str) -> torch.Tensor:
    #Handle .tiff
    if filename.lower().endswith(('.tif', '.tiff')):
        with MemoryFile(image_bytes) as memfile:
            with memfile.open() as dataset:
                r = dataset.read(4)
                g = dataset.read(3)
                b = dataset.read(2)
                nir = dataset.read(8)
                img_array = np.stack([r, g, b, nir], axis=0)
                
        img_tensor = torch.from_numpy(img_array).float()
        
    #Handle (.jpg, .png)
    else:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_np = cv2.resize(img_np, (64, 64), interpolation=cv2.INTER_AREA)
        
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        
        h, w = img_np.shape[0], img_np.shape[1]
        neutral_nir = EUROSAT_MEAN[3] / 10.0
        blank_nir = np.full((h, w, 1), neutral_nir, dtype=img_np.dtype)
        img_np = np.concatenate((img_np, blank_nir), axis=2)
            
        img_tensor = torch.from_numpy(img_np).float()
        img_tensor = img_tensor * 10.0 
        img_tensor = img_tensor.permute(2, 0, 1)
        
    # Apply standard normalization to both
    normalizer = T.Normalize(mean=EUROSAT_MEAN, std=EUROSAT_STD)
    final_tensor = normalizer(img_tensor)
    
    return final_tensor.unsqueeze(0)

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    #Pass the filename so we know how to process it
    input_tensor = prepare_and_transform(image_bytes, file.filename)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_idx = torch.max(outputs, 1)

    return {
        "filename": file.filename, 
        "predicted_class_index": predicted_idx.item()
    }