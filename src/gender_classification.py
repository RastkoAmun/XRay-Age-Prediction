import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import pandas as pd

from helpers.utils import CustomImageDataset

class GenderCNN(nn.Module):
    def __init__(self):
        super(GenderCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.MaxPool2d(2), # 256x344 -> 128x172

            nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 128x172 -> 64x86

            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 64x86 -> 32x43
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 43, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # 0: female, 1: male
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Function to convert labels into "Male" or "Female"
def label_to_str(label): 
    if isinstance(label, int): # Convert 0 -> Female, 1 -> Male
        return "Male" if label == 1 else "Female"
    if torch.is_tensor(label): # If scalar -> good, if not get first element
        val = label.item() if label.dim() == 0 else label[0].item()
        return "Male" if val == 1 else "Female"
    if isinstance(label, bool): # Convert False -> Female, True -> Male
        return "Male" if label else "Female"
    return str(label)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

csv_file = "data/boneage-training-dataset.csv"
img_dir = "data/processed/training-set"
dataset = CustomImageDataset(root_dir = img_dir, labels = pd.read_csv(csv_file), transform = transform)
subset_dataset = Subset(dataset, range(600)) # Take only 600 data for now
#dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)
dataloader = DataLoader(subset_dataset, batch_size = 32, shuffle = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GenderCNN().to(device)
print("Model architecture:")
print(model)

# Train model
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for i, batch in enumerate(dataloader):
        images, _, genders, _ = batch
        images = images.to(device)
        genders = genders.long().to(device)  # 0 or 1

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, genders)  # CrossEntropyLoss

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        if i % 200 == 199:
            avg_loss = running_loss / total_samples
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {avg_loss:.4f}")
            running_loss = 0.0
            total_samples = 0

print("Finished Training")


# Evaluate model
total_correct = 0
total_samples = 0

test_csv = "data/boneage-test-dataset.csv"
test_img_dir = "data/processed/test-set"
test_dataset = CustomImageDataset(root_dir = test_img_dir, labels = pd.read_csv(test_csv), transform = transform)
test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

model.eval()
with torch.no_grad():
    for batch in test_dataloader:
        images, genders, img_ids = batch # Ignore age
        images = images.to(device)

        outputs = model(images)
        # Apply Softmax to convert logits to probabilities (for each class)
        probs = nn.Softmax(dim = 1)(outputs)
        # Get predicted class by finding the index with maximum probability (Male 1 or Female 0)
        preds = torch.argmax(probs, dim = 1)

        genders_int = genders.long() # False -> 0, True -> 1

        correct = (preds.cpu() == genders_int).sum().item()
        total_correct += correct
        total_samples += images.size(0)

        # Get the maximum probability for each sample in the batch
        batch_pred_prob = probs.max(dim = 1)[0]

        pred_labels = [label_to_str(p) for p in preds]
        true_labels = [label_to_str(g) for g in genders]
        print("Batch Predictions:", pred_labels)
        print("Batch True genders:", true_labels)
        print("Batch prediction probabilities:", batch_pred_prob.tolist())

    overall_accuracy = total_correct / total_samples * 100
    print(f"Overall accuracy on {total_samples} samples: {overall_accuracy:.2f}%")
