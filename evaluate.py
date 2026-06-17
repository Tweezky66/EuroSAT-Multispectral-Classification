import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class Evaluation:
    def __init__(self, y_true, y_pred, class_names):
        self.y_true = y_true
        self.y_pred = y_pred
        self.class_names = class_names

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_true, self.y_pred, normalize='true')
        fig, ax = plt.subplots(figsize=(10, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(cmap='Blues', ax=ax, xticks_rotation='vertical', values_format='.2f')
        plt.title("EuroSAT 4-channel ResNet confusion matrix", fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()

    def print_confidence_interval(self, confidence_level=0.95):
        n = len(self.y_true)
        correct_predictions = sum(1 for true, pred in zip(self.y_true, self.y_pred) if true == pred)
        p = correct_predictions / n
        z = 1.96
        margin_of_error = z * np.sqrt(p * (1-p) / n)
        
        lower_bound = (p - margin_of_error) * 100
        upper_bound = (p + margin_of_error) * 100
        accuracy_pct = p * 100
        
        print(f"Test Set Size (n): {n} images")
        print(f"Estimate accuracy: {accuracy_pct:.2f}%")
        print(f"95% confidence interval: [{lower_bound:.2f}%, {upper_bound:.2f}%]")
        print(f"Margin of error is: ±{margin_of_error * 100:.2f}%")

    def class_report(self):
        cr = classification_report(self.y_true, self.y_pred, target_names=self.class_names, digits=2)
        print(cr)

    def attention_heatmap(self, model, input_tensor, true_label):
        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            pred_name = self.class_names[pred_idx]
            true_name = self.class_names[true_label]

        target_layers = [model.engine.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(true_label)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        rgb_image = input_tensor[0, :3, :, :].cpu().numpy().transpose(1, 2, 0)
        p2, p98 = np.percentile(rgb_image, (2, 98))
        rgb_image = np.clip(rgb_image, p2, p98)
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

        visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(visualization)
        ax[0].set_title(f"Grad-CAM Attention: {self.class_names[true_label]}", fontsize=14)
        ax[0].axis('off')

        ax[1].imshow(rgb_image)
        ax[1].set_title(f"Original Photo: {self.class_names[true_label]}", fontsize=14)
        ax[1].axis('off')
        plt.tight_layout()
        plt.show()

    def full_eval_report(self, model, input_tensor, true_label):
        self.plot_confusion_matrix()
        print("\n classification report:")
        self.class_report()
        print("\nintervals of confidence:")
        self.print_confidence_interval()
        print("\nheatmap:")
        self.attention_heatmap(model, input_tensor, true_label)
