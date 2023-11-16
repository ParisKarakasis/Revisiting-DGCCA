import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import glob
import PIL


# Multiview Dateset
class ViewDataset(Dataset):
    def __init__(self, v1, v2, v3):
        self.v1 = torch.tensor(v1).float()
        self.v2 = torch.tensor(v2).float()
        self.v3 = torch.tensor(v3).float()
        self.data_len = v1.shape[0]

    def __getitem__(self, index):
        return self.v1[index], self.v2[index], self.v3[index], index

    def __len__(self):
        return self.data_len


# Get a dataloader
def get_dataloader(view1, view2, view3, batchsize, shuffle):
    dataset = ViewDataset(view1, view2, view3)

    # Dataloader
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle)

    return data_loader


