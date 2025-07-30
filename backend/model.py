import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from facenet_pytorch import MTCNN
from tqdm import tqdm

class GRUClassifier(nn.Module):
    def __init__(self, feature_size=512, hidden_size=256, num_classes=2):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove classification layer

        self.gru = nn.GRU(input_size=feature_size, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # x: [B, T, 3, H, W]
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        feats = self.resnet(x)  # [B*T, 512]
        feats = feats.view(B, T, -1)  # [B, T, 512]
        _, hidden = self.gru(feats)
        out = self.classifier(hidden[-1])  # last hidden state
        return out
    