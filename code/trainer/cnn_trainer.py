import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn import DeepCNN
from data.data_loaders import create_dataloaders
from data.imagenet_loader import create_imagenet_dataloaders

class CNNTrainer:
    def __init__(self, data_dir, num_classes=10, batch_size=32, img_size=64, device=None):
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepCNN(num_classes=self.num_classes).to(self.device)
        # Set the loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # Load the data loaders to ensure that we use the same data
        #self.train_loader, self.val_loader, self.test_loader = create_dataloaders(self.data_dir, batch_size=self.batch_size, img_size=self.img_size)
        self.train_loader, self.val_loader, self.test_loader = create_imagenet_dataloaders(base_data_dir=data_dir, batch_size=self.batch_size, img_size=self.img_size)

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded CNN model from {path}")
    
    # Update optimizer based on the current trainable parameters (for fine-tuning)
    def update_optimizer(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(trainable_params, lr=0.001)

    def train(self, num_epochs=25, mode='normal'):
        # Enable fine-tuning mode and freeze all layers except the final FC layer
        if mode == 'fine_tune':
            self.model.activate_fine_tune()
            self.update_optimizer()
        
        best_acc = 0.0
        best_model_wts = self.model.state_dict()
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    dataloader = self.train_loader
                else:
                    self.model.eval()
                    dataloader = self.val_loader
                
                running_loss = 0.0
                running_corrects = 0
                
                # Iterate over data for the epoch
                for inputs, labels in dataloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Set gradients to zero
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        # If in training phase, backpropagate and optimize
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    # Sum up the loss
                    running_loss += loss.item() * inputs.size(0)
                    # Sum up how many correct predictions correctly classified
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # Save the best model based on validation accuracy
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = self.model.state_dict()
            
            print()
        
        # Load the best model weights from the training
        self.model.load_state_dict(best_model_wts)
        print(f'Best val Acc: {best_acc:.4f}')
    
    def test(self):
        self.model.eval()
        running_corrects = 0
        # Iterate over data for testing
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
        test_acc = running_corrects.double() / len(self.test_loader.dataset)
        print(f'Test Accuracy: {test_acc:.4f}')
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')

# QUick test to see if the trainer works
if __name__ == '__main__':
    trainer = CNNTrainer(data_dir='data/sample_data', num_classes=10, batch_size=32, img_size=64)
    trainer.train(num_epochs=5, mode='normal')
    trainer.save('cnn_normal.pth')
    # Now fine tune:
    trainer.train(num_epochs=5, mode='fine_tune')
    trainer.save('cnn_finetuned.pth')
