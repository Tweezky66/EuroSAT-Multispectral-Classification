
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio

class Validation:
    def __init__(self, dir_path, expected_shape=(64, 64)):
        self.dir_path = Path(dir_path)
        self.expected_shape = expected_shape
        self.valid_images = []

    def tree_structure(self, current_path=None, indent=""):
        if current_path is None:
            current_path = self.dir_path

        if current_path.is_dir():
            print(indent + current_path.name)
            for child in current_path.iterdir():
                if child.is_dir():
                    self.tree_structure(child, indent + "  ")

    def data_report(self):
        print("Scanning...\n")

        total_samples = 0
        widths = []
        heights = []
        bad_format = []
        corrupted = []

        #Reset the valid images list in case we run the report twice
        self.valid_images = []

        for image_path in self.dir_path.rglob("*.tif"):
            if image_path.is_file():
                if image_path.parent.parent != self.dir_path:
                    continue

                total_samples += 1

                try:
                    with rasterio.open(image_path) as dataset:
                        w, h = dataset.width, dataset.height

                        widths.append(w)
                        heights.append(h)

                        #if image resolution is not valid then skip,using this allows to pass only valid data in dataloader later
                        if (w, h) != self.expected_shape:
                            bad_format.append(image_path)
                        else:
                            self.valid_images.append(image_path)

                except Exception:
                    corrupted.append(image_path)
                    continue

        print("Data Report:\n")
        print(f"Total samples: {total_samples}")
        print(f"Valid samples: {len(self.valid_images)}")
        print(f"Corrupted files: {len(corrupted)}")
        print(f"Invalid Format (Not {self.expected_shape[0]}x{self.expected_shape[1]}): {len(bad_format)}")

        if bad_format:
            print("First 3 invalid files:", [p.name for p in bad_format[:3]])

        if widths and heights:
            print("\nWidth statistic:")
            print(f"Min: {min(widths)} px | max: {max(widths)} px | mean: {np.mean(widths):.2f} px")
            print("Height statistic:")
            print(f"min: {min(heights)} px | max: {max(heights)} px | mean: {np.mean(heights):.2f} px\n")

        self.random_plot()

    def random_plot(self):
        if self.valid_images:
            random_sample = random.choice(self.valid_images)

            with rasterio.open(random_sample) as dataset:
              #In Sentinel-2 red green and blue channels are 2,3,4
                r = dataset.read(4)
                g = dataset.read(3)
                b = dataset.read(2)
              #normalization colors  for matplotlib
                rgb = np.dstack((r, g, b))
                rgb = rgb / np.max(rgb)

            plt.figure(figsize=(6,6))
            plt.imshow(rgb)
            plt.title(f"Random TIF (RGB View): {random_sample.parent.name}")
            plt.axis('off')
            plt.show()
        else:
            print("No valid images found to plot")

    def generate_nir_mask(self, tif_path, n_std=1.0):
        path = Path(tif_path)
        if not path.is_file():
            print(f"Error: File not found at {path}")
            return None

        try:
            with rasterio.open(path) as dataset:
                #next 4 lines is defining how each image in class is sensetive to NIR to adapt for each image for better model perfomance
                nir_band = dataset.read(8)
                img_mean = np.mean(nir_band)
                img_std = np.std(nir_band)

                adaptive_threshold = img_mean + (n_std * img_std)

                print(f"Image stats: min brightness: {np.min(nir_band)} | max: {np.max(nir_band)} | mean: {np.mean(nir_band):.0f}")
                print(f"Adaptive threshold set to: {adaptive_threshold:.0f} (Using n={n_std})")
                nir_mask = nir_band > adaptive_threshold

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(nir_band, cmap='gray')
            axes[0].set_title("Raw NIR Band")
            axes[0].axis('off')

            axes[1].imshow(nir_mask, cmap='binary_r')
            axes[1].set_title(f"Mask mean + ({n_std}*std)")
            axes[1].axis('off')

            plt.tight_layout()
            plt.show()

            return nir_mask

        except Exception as e:
            print(f"Error processing {path}: {e}")
            return None

    def random_nir_mask(self, n_std=1.0):
        if self.valid_images:
            random_sample = random.choice(self.valid_images)
            print(f"Generating mask for: {random_sample.name} (category: {random_sample.parent.name})")
            return self.generate_nir_mask(random_sample, n_std=n_std)
        else:
            print("No valid .tif files found.")
            return None

    def compute_dataset_stats(self, custom_image_list=None):
        print("\nComputing global mean and std for RGB + NIR channels...")

        target_images = custom_image_list if custom_image_list is not None else self.valid_images

        if not target_images:
            print("No valid images found to process!")
            return None, None

        channel_sum = np.zeros(4, dtype=np.float64)
        channel_sum_squared = np.zeros(4, dtype=np.float64)
        total_pixels = 0

        for idx, image_path in enumerate(target_images):
            if idx > 0 and idx % 5000 == 0:
                print(f"Processed {idx} / {len(target_images)} images...")

            with rasterio.open(image_path) as dataset:
                r = dataset.read(4)
                g = dataset.read(3)
                b = dataset.read(2)
                nir = dataset.read(8)

                img = np.stack([r, g, b, nir], axis=0).astype(np.float64)

                pixels_in_image = img.shape[1] * img.shape[2]
                total_pixels += pixels_in_image

                #axis=(1, 2) means we sum across the width and height, leaving an array of 4 totals
                channel_sum += img.sum(axis=(1, 2))

                channel_sum_squared += (img ** 2).sum(axis=(1, 2))

        global_mean = channel_sum / total_pixels

        global_variance = (channel_sum_squared / total_pixels) - (global_mean ** 2)

        global_std = np.sqrt(global_variance)

        print("\n Global dataset statistics:")
        print(f"Bands calculated: red, green, blue, NIR")
        print(f"Mean: {np.round(global_mean, 2).tolist()}")
        print(f"std:  {np.round(global_std, 2).tolist()}")

        return global_mean, global_std

