import random
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms

# This transforms the data somewhat randomly to augment the data
# As well as normalizing the data and converting it to a tensor.
def get_train_transforms(img_size=64):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

# This applies similar transformations to the test data,
# but without the random transformations.
def get_test_transforms(img_size=64):
    return transforms.Compose([
        transforms.Resize(img_size + 16),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])


def create_dataloaders(data_dir, batch_size=32, val_split=0.2, test_split=0.1, img_size=64):
    # First create the full dataset and shuffle it
    full_dataset = ImageFolder(root=data_dir)
    total_size = len(full_dataset)
    indices = list(range(total_size))
    random.shuffle(indices)
    
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create separate dataset instances for each split
    train_dataset = ImageFolder(root=data_dir, transform=get_train_transforms(img_size))
    # Note we apply the test transforms to the validation and test sets
    val_dataset = ImageFolder(root=data_dir, transform=get_test_transforms(img_size))
    test_dataset = ImageFolder(root=data_dir, transform=get_test_transforms(img_size))
    
    # Use Subset to creat the splits
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    # Create the dataloaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader