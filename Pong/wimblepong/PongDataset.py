import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class PongDataset(Dataset):
    """Pong images dataset."""

    def __init__(self, numpydata):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.pong_frames = numpydata

    def __len__(self):
        return len(self.pong_frames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        y = self.pong_frames[:][1:2]
        frame = self.pong_frames[:][0]
        #frame = frame.astype('float').reshape(-1, 3)
        sample = (y, frame)

        return sample
