
import torch
import torch.nn as nn
import torchvision.models as models

class EuroSATResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.engine = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        for block in [self.engine.layer1, self.engine.layer2]:
            for param in block.parameters():
                param.requires_grad = False


        old_conv = self.engine.conv1

        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        #Copy original RGB weights
        new_conv.weight.data[:, :3, :, :] = old_conv.weight.data

        #Average the RGB weights to create the NIR channel head-start
        nir_weights = old_conv.weight.data.mean(dim=1, keepdim=True)
        new_conv.weight.data[:, 3:4, :, :] = nir_weights

        self.engine.conv1 = new_conv


        self.engine.fc = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        return self.engine(x)
