import os
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Character to index mapping
char_to_idx = {chr(i): i-65 for i in range(65, 91)}  # 'A' to 'Z'
char_to_idx.update({str(i): i+26 for i in range(10)})  # '0' to '9'
idx_to_char = {v: k for k, v in char_to_idx.items()}

class CaptchaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_filenames = os.listdir(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.data_dir, image_filename)
        image = Image.open(image_path).convert('RGB')
        
        label = image_filename.split('_')[-1].split('.')[0]
        
        if self.transform:
            image = self.transform(image)
        
        label_tensor = torch.zeros(6, 36)  # Assuming 6 characters max, 36 possible characters
        for i, char in enumerate(label):
            label_tensor[i, char_to_idx[char]] = 1
        
        return image, label_tensor

class EnhancedMinimalCNN(nn.Module):
    def __init__(self):
        super(EnhancedMinimalCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 17 * 50, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6 * 36)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x.view(-1, 6, 36)

class MinimalCNN(nn.Module):
    def __init__(self):
        super(MinimalCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 35 * 100, 6 * 36)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x.view(-1, 6, 36)

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.classifier = nn.Linear(3 * 70 * 200, 6 * 36)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x.view(-1, 6, 36)

class TinyMLP(nn.Module):
    def __init__(self):
        super(TinyMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3 * 70 * 200, 128),
            nn.ReLU(),
            nn.Linear(128, 6 * 36)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x).view(-1, 6, 36)

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(2)
            correct += (predicted == labels.max(2)[1]).sum().item()
            total += labels.size(0) * labels.size(1)
    return correct / total

# Data preparation
train_transform = transforms.Compose([
    transforms.Resize((70, 200)),  # Maintain aspect ratio
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ], p=0.5),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    ], p=0.2),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((70, 200)),  # Maintain aspect ratio
    transforms.ToTensor(),
])

data_dir = './dataset'
train_dataset = CaptchaDataset(data_dir=data_dir, transform=train_transform)
test_dataset = CaptchaDataset(data_dir=data_dir, transform=test_transform)

# Split dataset: 95% train, 5% test
train_size = int(0.95 * len(train_dataset))
test_size = len(test_dataset) - train_size
train_dataset, _ = random_split(train_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = MinimalCNN().to(device)
# model = TinyMLP().to(device)
# model = LinearModel().to(device)
model = EnhancedMinimalCNN().to(device)

# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Learning rate scheduler
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# # Training loop
# num_epochs = 10
# best_accuracy = 0
# for epoch in range(num_epochs):
#     train_loss = train(model, train_loader, criterion, optimizer, device)
#     test_accuracy = evaluate(model, test_loader, device)
    
#     print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
#     scheduler.step(train_loss)
    
#     if test_accuracy > best_accuracy:
#         best_accuracy = test_accuracy
#         torch.save(model.state_dict(), 'models/enhancedMinimalCNN.pth')
#         print(f"New best model saved with accuracy: {best_accuracy:.4f}")

#     if optimizer.param_groups[0]['lr'] < 1e-6:
#         print("Learning rate too small. Stopping training.")
#         break

# print(f"Best test accuracy: {best_accuracy:.4f}")

# Load the best model and test
if os.path.exists('models/enhancedMinimalCNN.pth'):
    model.load_state_dict(torch.load('models/enhancedMinimalCNN.pth'))
    print("Loaded the best model.")
else:
    print("No saved model found. Using the model from the last epoch.")

model.eval()
total_time = 0
num_samples = 0
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        start_time = time.time()
        outputs = model(images)
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000
        total_time += inference_time
        num_samples += images.size(0)
        
        _, predicted = outputs.max(2)
        true_labels = labels.max(2)[1]
        
        # Calculate accuracy for this batch
        correct_predictions += (predicted == true_labels).all(dim=1).sum().item()
        total_predictions += predicted.size(0)
        
        for i in range(5):
            pred_str = ''.join([idx_to_char[idx.item()] for idx in predicted[i]])
            true_str = ''.join([idx_to_char[idx.item()] for idx in true_labels[i]])
            print(f"Predicted: {pred_str}, Actual: {true_str}")
        
        print(f"Batch inference time: {inference_time:.2f} ms")
        # break # Optional: break after one batch if you only want to test a small sample

average_inference_time = total_time / num_samples
overall_accuracy = correct_predictions / total_predictions * 100

print(f"\nAverage inference time per sample: {average_inference_time:.2f} ms")
print(f"Overall accuracy: {overall_accuracy:.2f}%")