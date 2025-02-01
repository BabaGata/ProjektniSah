import torch
import torch.nn as nn
from constants import *

class ChessOpeningClassifier(nn.Module):
    def __init__(self, num_classes, max_moves=MAX_MOVES):
        super(ChessOpeningClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.lstm_input_size = self._calculate_lstm_input_size()
        self.rnn = nn.LSTM(self.lstm_input_size, 128, batch_first=True)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _calculate_lstm_input_size(self):
        dummy_input = torch.zeros(1, 1, 8, 8)
        x = self.relu(self.conv1(dummy_input))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)

        return x.view(1, -1).size(1)

    def forward(self, x):
        if x.dim() == 4:
            batch_size, seq_len, height, width = x.shape
        elif x.dim() == 5:
            batch_size, seq_len, channels, height, width = x.shape
            if channels != 1:
                raise ValueError(f"Expected 1 channel, but got {channels} channels.")
            x = x.squeeze(2)
        else:
            raise ValueError(f"Expected input tensor with 4 or 5 dimensions, but got {x.dim()} dimensions.")

        x = x.view(batch_size * seq_len, 1, height, width)

        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)

        features_dim = x.shape[1] * x.shape[2] * x.shape[3]

        x = x.view(batch_size, seq_len, features_dim)
        output, (hidden, cell) = self.rnn(x)
        x = hidden[-1]

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x