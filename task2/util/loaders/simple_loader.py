from torch.utils.data import Dataset, DataLoader
import  os
class Data(Dataset):
  def __init__(self, path_to_data, MEI):
    self.path_to_data = path_to_data
    self.MEI = MEI

    self.X = []
    self.y = []

    with open(os.path.join(path_to_data, MEI + '_output.txt'), 'r') as f:
      self.X = [l.strip() for l in f]
    with open(os.path.join(path_to_data, MEI + '_sentences.txt'), 'r') as g:
      self.y = [m.strip() for m in g]

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]