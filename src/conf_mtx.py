import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

from gender_classification import GenderCNN
from helpers.utils import CustomImageDataset  # Adjust the import if needed

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # File paths for CSV and images.
    csv_file = "data/boneage-training-dataset.csv"
    img_dir = "data/processed/training-set"
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load labels and process the "male" column.
    df_labels = pd.read_csv(csv_file)
    df_labels["male"] = df_labels["male"].astype(int)
    
    # Create the dataset and use a subset of 600 samples.
    dataset = CustomImageDataset(root_dir=img_dir, labels=df_labels, transform=transform)
    test_dataset = Subset(dataset, range(600))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Instantiate and (optionally) load your trained model.
    model = GenderCNN().to(device)
    # If you have saved weights, uncomment and adjust the next line:
    # model.load_state_dict(torch.load("my_trained_model.pth", map_location=device))
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, _, genders, _ in test_loader:
            images = images.to(device)
            genders = genders.to(device)
            
            outputs = model(images)  # Output shape: [batch_size, 2]
            predicted = torch.argmax(outputs, dim=1)
            
            y_true.extend(genders.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Compute the confusion matrix.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    print("Confusion Matrix (rows: actual, columns: predicted):")
    print(cm)
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")
    
    # Calculate metrics.
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', pos_label=1)
    recall = recall_score(y_true, y_pred, average='binary', pos_label=1)
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=1)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    # Visualize the confusion matrix.
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Female (0)", "Male (1)"],
                yticklabels=["Female (0)", "Male (1)"])
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()