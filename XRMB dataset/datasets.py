from torch.utils.data import Dataset

class TensorDataset (Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __len__(self):
        return self.data_tensor.size()[0]

    def __getitem__(self, idx):
        return self.data_tensor[idx] , idx
