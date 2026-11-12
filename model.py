import torch
import torch.nn as nn
from torchvision.models import resnet18
from transformers import ViTConfig, ViTModel


class CNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.projection(torch.flatten(x, 1))


class ViTBranch(nn.Module):
    def __init__(self):
        super().__init__()

        config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            layer_norm_eps=1e-12,
            qkv_bias=True
        )

        self.backbone = ViTModel(config, add_pooling_layer=True)
        self.projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = self.backbone(pixel_values=x).last_hidden_state[:, 0]
        return self.projection(x)


class BehavioralBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)


class HCAFViTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNBranch()
        self.vit = ViTBranch()
        self.behavioral = BehavioralBranch()

        self.fusion = nn.Sequential(
            nn.Linear(640, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Linear(128, 2)

    def forward(self, image, behavioral):
        features = torch.cat([
            self.cnn(image),
            self.vit(image),
            self.behavioral(behavioral)
        ], dim=1)

        return self.classifier(self.fusion(features))