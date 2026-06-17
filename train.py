import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

from data_audit import Validation
from dataset import EuroSATDataset
from model import EuroSATResNet

def main():
    print("Starting EuroSAT pipeline")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Step 1 is to validate images providing
    DATA_PATH = "/content/drive/MyDrive/Datasets/EuroSATallBands" 
    validator = Validation(DATA_PATH, expected_shape=(64, 64))
    validator.data_report()
    
    #Step 2 split the data
    all_clean_files = validator.valid_images
    train_val_files, test_files = train_test_split(all_clean_files, test_size=0.15, random_state=42)
    train_files, val_files = train_test_split(train_val_files, test_size=(0.15 / 0.85), random_state=42)
    
    #Calculate mean/std only on training data to prevent data leakage
    mean, std = validator.compute_dataset_stats(custom_image_list=train_files)

    #Step 3 is to create datasets and dataloaders
    train_dataset = EuroSATDataset(train_files, dataset_mean=mean, dataset_std=std)
    val_dataset = EuroSATDataset(val_files, dataset_mean=mean, dataset_std=std)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    #Step 4 Initialize Model & Optimizer
    model = EuroSATResNet().to(device)
    
    base_lr = 0.001

    #Define optimizers layers learning rates to pretrain underperfomed layer & safe a computational power 
    optimizer = torch.optim.Adam([
        {"params": model.engine.layer1.parameters(), "lr": base_lr / 100},
        {"params": model.engine.layer2.parameters(), "lr": base_lr / 100},

        {"params": model.engine.layer3.parameters(), "lr": base_lr / 10},
        {"params": model.engine.layer4.parameters(), "lr": base_lr / 10},

        {"params": model.engine.conv1.parameters(), "lr": base_lr},
        {"params": model.engine.fc.parameters(), "lr": base_lr},
    ])

    criterion = nn.CrossEntropyLoss()

    #Step 5 training loop
    epochs = 20
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predictions = torch.max(outputs, 1)
            train_correct += torch.sum(predictions == labels.data)
            
        #Validation phase
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predictions = torch.max(outputs, 1)
                val_correct += torch.sum(predictions == labels.data)
                
        #Print metrics
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct.double() / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct.double() / len(val_loader.dataset)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}\n")

    #Step 6 Save the model
    MODELS_DIR = Path('/content/drive/MyDrive/EuroSAT_Project/models')
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / 'resnet18_4channel_v1.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Training complete.Model saved to: {save_path}")

if __name__ == "__main__":
    main()
