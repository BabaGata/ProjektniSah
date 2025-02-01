import torch
import torch.nn as nn
from constants import *

class SimplifiedChessOpeningClassifier(nn.Module):
    def __init__(self, num_classes, max_moves=MAX_MOVES):
        super(SimplifiedChessOpeningClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Adjusted for input size
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # No sequence dimension, so x should have shape (batch_size, channels, height, width)
        batch_size, channels, height, width = x.shape
        if channels != 1:
            raise ValueError(f"Expected 1 channel, but got {channels} channels.")

        # Convolution layers
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        # Flatten and pass through fully connected layers
        x = x.view(batch_size, -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x