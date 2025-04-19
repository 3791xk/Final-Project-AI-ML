import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn import DeepCNN
from data.imagenet_loader import create_imagenet_dataloaders
from datetime import datetime

class CNNTrainer:
    def __init__(self, data_dir, num_classes=10, batch_size=32, img_size=64, device=None):
        self.data_dir = data_dir
        self.num_classes = num_classes - 1  # Adjust for zero-based indexing
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepCNN(num_classes=self.num_classes).to(self.device)
        self.model = nn.DataParallel(self.model)
        # Set the loss function and optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Data loaders
        print(f"Loading data from {data_dir}...", flush=True)
        self.train_loader, self.val_loader, self.test_loader = create_imagenet_dataloaders(base_data_dir=data_dir, batch_size=self.batch_size, img_size=self.img_size)
        print("Data loaded successfully.", flush=True)

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded CNN model from {path}")

    def update_optimizer(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(trainable_params, lr=0.001, weight_decay=1e-4)

    def train(self, num_epochs=25, mode='normal'):
        print(f"Training in {mode} mode...", flush=True)
        if mode == 'fine_tune':
            self.model.activate_fine_tune(self.num_classes)
            self.update_optimizer()

        best_acc = 0.0
        best_model_wts = self.model.state_dict()

        for epoch in range(num_epochs):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch+1}/{num_epochs}", flush=True)
            print('-' * 10)
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    loader = self.train_loader
                else:
                    self.model.eval()
                    loader = self.val_loader

                running_loss = 0.0
                running_correct_bits = 0
                running_total_bits = 0

                for inputs, labels in loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        logits = self.model(inputs)
                        loss = self.criterion(logits, labels)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)

                    # Multi-label predictions and accuracy (Hamming)
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    running_correct_bits += (preds == labels).sum().item()
                    running_total_bits += labels.numel()

                epoch_loss = running_loss / len(loader.dataset)
                epoch_acc = running_correct_bits / running_total_bits
                print(f'{phase} Loss: {epoch_loss:.4f} Hamming Acc: {epoch_acc:.4f}')

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = self.model.state_dict()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished epoch {epoch+1}", flush=True)

        self.model.load_state_dict(best_model_wts)
        print(f'Best val Hamming Acc: {best_acc:.4f}', flush=True)

    def test(self):
        self.model.eval()
        running_correct_bits = 0
        running_total_bits = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self.model(inputs)

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                running_correct_bits += (preds == labels).sum().item()
                running_total_bits += labels.numel()

        test_acc = running_correct_bits / running_total_bits
        print(f'Test Hamming Acc: {test_acc:.4f}', flush=True)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}', flush=True)

# QUick test to see if the trainer works
if __name__ == '__main__':
    trainer = CNNTrainer(data_dir='data/sample_data', num_classes=10, batch_size=32, img_size=64)
    trainer.train(num_epochs=5, mode='normal')
    trainer.save('cnn_normal.pth')
    # Now fine tune:
    trainer.train(num_epochs=5, mode='fine_tune')
    trainer.save('cnn_finetuned.pth')
