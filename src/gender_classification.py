import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from helpers.utils import CustomImageDatasetForGender


def evaluate_model(model, dataloader, device):
    # Evaluate the model, print classification report, return F1 score
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, _, genders, _ in dataloader:
            images = images.to(device)
            labels = genders.cpu().numpy()
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    print(classification_report(all_labels, all_preds, target_names=["Female", "Male"]))
    print("Predicted class counts:", Counter(all_preds))

    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return macro_f1


def calculate_training_accuracy(model, dataloader, device):
    # Calculate training accuracy
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, _, genders, _ in dataloader:
            images = images.to(device)
            labels = genders.to(device)

            outputs = model(images)
            #preds = (outputs > 0.5).cpu()
            preds = torch.argmax(outputs, dim=1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 86 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc(x)
        return x  # shape: (batch_size, 2)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def imshow(img, true_label, pred_label):
    # Display the image with labels
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"True: {'Female' if true_label==0 else 'Male'}\nPred: {'Female' if pred_label==0 else 'Male'}")
    plt.axis('off')


# Main
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

csv_file = "data/boneage-training-dataset.csv"
img_dir = "data/processed/training-set"
df = pd.read_csv(csv_file)
filtered_df = df[df['boneage'] >= 165].reset_index(drop=True) # Extract data above 165
#Second_filtered_df = df[df['boneage'] >= 150].reset_index(drop=True)

print(len(filtered_df))
#print(len(Second_filtered_df))

#sub_dataset = CustomImageDatasetForGender(root_dir=img_dir, labels=Second_filtered_df , transform=transform)
sub_dataset = CustomImageDatasetForGender(root_dir=img_dir, labels=filtered_df, transform=transform)
dataloader = DataLoader(sub_dataset, batch_size=4, shuffle=False)

# splitting into training and testing dataset

train_df, test_df = train_test_split(filtered_df, test_size=0.1, random_state=42)
#train_df, test_df = train_test_split(Second_filtered_df, test_size=0.1, random_state=42)
print("# of male training dataset")
print(train_df['male'].value_counts())
print("\n# of male in testing dataset")
print(test_df['male'].value_counts())

train_dataset = CustomImageDatasetForGender(root_dir=img_dir, labels=train_df, transform=transform)
test_dataset = CustomImageDatasetForGender(root_dir=img_dir, labels=test_df, transform=transform)

# Handle class imbalance
gender_labels = [int(label) for _, _, label, _ in train_dataset]  # 0: Female, 1: Male
class_counts = torch.bincount(torch.tensor(gender_labels, dtype=torch.long))
class_weights = 1.0 / class_counts.float()
sample_weights = [class_weights[label] for label in gender_labels]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, shuffle=False)
#train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

#num_male = 1487  # male=True
#num_female = 464   # male=False

best_combined_score = 0.0
best_model_state = None
num_epochs = 50

class_weights = torch.tensor([1.2, 1.0], dtype=torch.float32).to(device)
criterion = FocalLoss(gamma=1.0, weight=class_weights).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    # Train the model
    model.train()
    running_loss = 0.0

    for images, _, genders, _ in train_loader:
        images = images.to(device)
        labels = genders.long().to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_acc = calculate_training_accuracy(model, train_loader, device)
    test_acc = calculate_training_accuracy(model, test_loader, device)

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")
    macro_f1 = evaluate_model(model, test_loader, device)
    combined_score = (macro_f1 + test_acc) / 2

    if combined_score > best_combined_score:
        best_combined_score = combined_score
        best_model_state = model.state_dict()
    
    print(f"Macro F1-score: {macro_f1:.4f} | Combined Score: {combined_score:.4f}")
    print("-" * 60)

if best_model_state is not None:
    torch.save(best_model_state, "balanced_model.pt")
    print(f"Best model saved with Combined Score: {best_combined_score:.4f}")


model.load_state_dict(torch.load("balanced_model.pt"))
model.eval()

all_preds = []
all_labels = []
wrong_images = []
wrong_preds = []
wrong_true = []

with torch.no_grad():
    for images, _, labels, _ in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy().astype(int))

        for i in range(len(preds)):
            # Collect misclassified
            if preds[i] != labels[i]:
                wrong_images.append(images[i].cpu())
                wrong_preds.append(preds[i].item())
                wrong_true.append(labels[i].item())

cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
#print (tn)
#print (fp)
#print (fn)
#print (tp)

classes = ["Female", "Male"]

# Display confusion matrix
steelblue_cmap = LinearSegmentedColormap.from_list("steelblue_map", ["#f0f8ff", "steelblue"])
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=steelblue_cmap)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 6))
for i in range(min(10, len(wrong_images))):
    plt.subplot(2, 5, i+1)
    imshow(wrong_images[i], wrong_true[i], wrong_preds[i])
plt.suptitle("Misclassified Test Images", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
