import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn import DeepCNN
from data.imagenet_loader import create_imagenet_dataloaders, create_dogs_loader
from datetime import datetime
import numpy as np

class CNNTrainer:
    def __init__(self, data_dir, num_classes=10, batch_size=32, img_size=64, device=None, percentage=1.0):
        self.data_dir = data_dir
        self.num_classes = num_classes - 1  # Adjust for zero-based indexing
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepCNN(num_classes=self.num_classes).to(self.device)
        self.model = nn.DataParallel(self.model)
        self.percentage = percentage
        
        # Data loaders
        print(f"Loading data from {data_dir}...", flush=True)
        self.train_loader, self.val_loader, self.test_loader = create_dogs_loader(
            base_data_dir=data_dir, batch_size=self.batch_size, 
            img_size=self.img_size, percent_used=percentage)
        print("Data loaded successfully.", flush=True)
        
        # compute pos_weight per class from the train split
        subset = self.train_loader.dataset.samples
        # all_samps = subset.dataset
        # idxs = subset.indices

        # labels_np = np.stack([all_samps[i][1] for i in idxs], axis=0)  # shape [N, C]
        labels_np = np.stack([sample[1] for sample in subset], axis=0)
        pos = labels_np.sum(axis=0)
        neg = labels_np.shape[0] - pos
        pos_weight = (neg / (pos + 1e-6)).astype(np.float32)
        
        pos_weight_tensor = torch.from_numpy(pos_weight).to(self.device)
        
        # Set the loss function and optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    def load_checkpoint(self, path):
        original_num_classes = 200 
        self.model = DeepCNN(num_classes=original_num_classes).to(self.device)
        self.model = nn.DataParallel(self.model)

        # Load the state dict
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        if not isinstance(self.model, nn.DataParallel):
            self.model = nn.DataParallel(self.model)

        print(f"Loaded CNN model from {path}")
        self.update_optimizer()

    def update_optimizer(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(trainable_params, lr=0.001, weight_decay=1e-4)

    def train(self, num_epochs=25, mode='normal'):
        print(f"Training in {mode} mode, with {len(self.train_loader.dataset)} datapoints", flush=True)
        if mode == 'fine_tune':
            self.model.module.activate_fine_tune(self.num_classes)
            # Send the model to the device again after modifying it
            self.model = self.model.to(self.device)
            self.update_optimizer()

        best_loss = float('inf')
        best_model_wts = self.model.state_dict()

        for epoch in range(num_epochs):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch+1}/{num_epochs}", flush=True)
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                loader = self.train_loader if phase=='train' else self.val_loader

                running_loss = 0.0
                running_sample_acc = 0.0

                for inputs, labels in loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    if phase == 'train':
                        self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        logits = self.model(inputs)
                        loss = self.criterion(logits, labels)

                        if phase == 'train':
                            loss.backward()
                            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.optimizer.step()
                    running_loss += loss.item() * inputs.size(0)

                    # per‑sample accuracy
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    # for each sample: fraction of bits correct, then sum
                    sample_acc_batch = (preds.eq(labels).float().mean(dim=1)).sum().item()
                    running_sample_acc += sample_acc_batch

                epoch_loss = running_loss / len(loader.dataset)
                epoch_acc  = running_sample_acc / len(loader.dataset)
                print(f'{phase} Loss: {epoch_loss:.4f} Sample-Acc: {epoch_acc:.4f}')

                if phase=='val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = self.model.state_dict()
                    save_path = f'cnn_model_finetune_{self.percentage}.pth' if mode == 'fine_tune' else 'cnn_model.pth'
                    torch.save(self.model.state_dict(), save_path)

            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished epoch {epoch+1}", flush=True)
            print('-' * 10)
        self.model.load_state_dict(best_model_wts)
        print("Training complete. Best val loss: {:.4f}".format(best_loss), flush=True)


    def test(self):
        self.model.eval()
        running_sample_acc = 0.0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self.model(inputs)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                running_sample_acc += (preds.eq(labels).float().mean(dim=1)).sum().item()

        test_acc = running_sample_acc / len(self.test_loader.dataset)
        print(f'Test Sample-Acc: {test_acc:.4f}', flush=True)

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
