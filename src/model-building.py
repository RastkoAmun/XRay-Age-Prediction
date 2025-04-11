import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from helpers.utils import CustomImageDataset
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.optim as optim



labels_df = pd.read_csv('data/boneage-training-dataset.csv')
training_labels, testing_labels = train_test_split(labels_df, train_size=0.95, test_size=0.05)

transformer = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5,), (0.5,))
])

# Using custom dataset to load images 
training_dataset = CustomImageDataset(
  root_dir='data/processed/training-set', labels=training_labels, transform=transformer
)
testing_dataset = CustomImageDataset(
  root_dir='data/processed/training-set', labels=testing_labels, transform=transformer
)

print("Training size: ", len(training_dataset))
print("Test size: ", len(testing_dataset))
batch_size = 32

# prepared dataloader for neural network (note it is using batch size of 3, just for this sample)
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

class BoneAgeModel(nn.Module):
  def __init__(self):
    super(BoneAgeModel, self).__init__()

    self.cnn = nn.Sequential(
      #First conv block
      nn.Conv2d(1, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32), # Normalize 
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      #Second conv block 
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64), 
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      #Third conv block 
      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128), 
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      #Forth conv block
      nn.Conv2d(128, 256, kernel_size=3, padding=1),
      nn.BatchNorm2d(256), 
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

    )
    feature_size = 256 * 16 * 21

    self.fc_layers = nn.Sequential(
      nn.Flatten(),
          
      nn.Linear(feature_size, 512),  
      nn.BatchNorm1d(512),
      nn.ReLU(),
      nn.Dropout(0.3), 
            
      nn.Linear(512, 256),
      nn.BatchNorm1d(256),
      nn.ReLU(),
      nn.Dropout(0.3), 
            
      nn.Linear(256, 1) 
    ) 


  def forward(self, x):
    x = self.cnn(x)
    x = self.fc_layers(x)
  
    return x
  
def train_model(model, training_dataloader, testing_dataloader, num_epochs):
   # Use both MSE and MAE
    criterion_mse = nn.MSELoss() # MSE penalizes larger error more severely 
    criterion_mae = nn.L1Loss() 

    #Adam optimizer with weigth decay (L2)
    #https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    optimizer  = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor = 0.5, patience=3
    )

    best_val_loss = float('inf')
    best_model_weights = None 

    train_losses = []
    val_losses = []
    val_maes = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, ages, _, _ in training_dataloader:
            images = images.float()
            ages = ages.float()

            optimizer.zero_grad() # clear gradients from the previous batch 

            #forward pass 
            predictions = model(images).squeeze()
            
            loss_mse = criterion_mse(predictions, ages)
            loss_mae = criterion_mae(predictions, ages)
            loss = loss_mse + loss_mae

            # Bakcward pass 
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(training_dataloader) 
        train_losses.append(train_loss)
        
        #validation phase 
        model.eval()
        val_loss = 0.0
        mae_total = 0.0

        with torch.no_grad():
            for images, ages, _, _ in testing_dataloader:
                images = images.float()
                ages = ages.float()

                # Forward pass 
                prediction = model(images).squeeze()
                
                mae = criterion_mae(prediction, ages) 
                val_loss += mae.item()

                mae_total += torch.abs(prediction - ages).sum().item() 

        val_loss /= len(testing_dataloader)
        val_losses.append(val_loss)

        mae_months = mae_total / len(testing_dataloader.dataset)
        val_maes.append(mae_months)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}, MAE: {mae_months:.2f} months")

        #Update scheduler
        scheduler.step(val_loss)

        if val_loss < best_val_loss: 
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()
            print(f"New best model saved!")
    
    model.load_state_dict(best_model_weights)
    return model

model = BoneAgeModel()
model = train_model(model, training_dataloader, testing_dataloader, num_epochs=20)

model.eval()
test_predictions = []
epoch_val_loss = 0
mae_total = 0

with torch.no_grad():
    for images, ages, _,img_ids in testing_dataloader:
        images = images.float()
        ages = ages.float()

        predictions = model(images).squeeze()

        for i in range(len(img_ids)):
            test_predictions.append([img_ids[i].item(), ages[i].item(), predictions[i].item()])
            loss = torch.abs(predictions[i] - ages[i])
            epoch_val_loss += loss.item()
            mae_total += loss.item()

            print(f"{img_ids[i].item():>6} {ages[i].item():>8} {predictions[i].item():>8.2f}")

val_loss = epoch_val_loss / len(testing_dataloader.dataset)
mae_months = mae_total / len(testing_dataloader.dataset)
print(f"\n Validation Loss: {val_loss:.4f}")
print(f"Mean Absolute Error: {mae_months:.2f} months")

submission_df = pd.DataFrame(test_predictions, columns=["id", "real", "prediction"])  
submission_df.to_csv("submission.csv", index=False)
print("Submission file saved!")