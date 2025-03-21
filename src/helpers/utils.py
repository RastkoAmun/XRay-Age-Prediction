from PIL import Image
from torch.utils.data import Dataset
import os
import torch

class CustomImageDataset(Dataset):
  def __init__(self, root_dir, labels, transform=None):
    self.root_dir = root_dir
    self.labels = labels
    self.transform = transform
    self.image_files = os.listdir(root_dir)

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    img_id, age, gender = self.labels.iloc[idx]
    img_path = os.path.join(self.root_dir, str(img_id) + '.png')
    image = Image.open(img_path)
    if self.transform:
      image = self.transform(image)
    age = torch.tensor(age, dtype=torch.float32)

    return image, age, gender, img_id