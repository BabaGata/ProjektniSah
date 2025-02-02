import torch
import numpy as np
from constants import *

class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_moves=MAX_MOVES):
        self.data = df["matrices"].values
        self.labels = df["encoded_eco"].values
        # self.opening_ply = df["opening_ply"].values
        self.max_moves = max_moves

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        matrices = self.data[idx]
        # opening_ply = int(self.opening_ply[idx])
        trim_moves = 2 * OPENING_MOVES
        end_index = min(trim_moves + self.max_moves, len(matrices))
        matrices = matrices[trim_moves:end_index]

        seq_len = len(matrices)

        if seq_len < self.max_moves:
            padding_shape = (self.max_moves - seq_len, 8, 8)
            padding = np.zeros(padding_shape, dtype=np.int32)
            matrices = np.concatenate([matrices, padding], axis=0)

        return torch.tensor(matrices, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)