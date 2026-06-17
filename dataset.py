
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import rasterio
import numpy as np

class EuroSATDataset(Dataset):
    def __init__(self, valid_image_paths, dataset_mean, dataset_std):
        #We're defining this class to make a proper dataset for dataloader,counting mean and std only on train data
        self.image_path = valid_image_paths
        self.normalize = T.Normalize(mean=dataset_mean, std=dataset_std)

        unique_classes = sorted(list(set([p.parent.name for p in self.image_path])))

        self.class_to_idx = {class_name: idx  for idx, class_name in enumerate(unique_classes)}
        print(f"Found {len(unique_classes)} classes: {self.class_to_idx}")

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img_path = self.image_path[idx]

        with rasterio.open(img_path) as dataset:
            r = dataset.read(4)
            g = dataset.read(3)
            b = dataset.read(2)
            nir = dataset.read(8)

        img_array = np.stack([r, g, b, nir], axis=0)
        img_tensor = torch.from_numpy(img_array).float()
        img_tensor = self.normalize(img_tensor)

        label_str = img_path.parent.name
        label_idx = self.class_to_idx[label_str]

        return img_tensor, label_idx
