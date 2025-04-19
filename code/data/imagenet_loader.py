import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np

def load_wnid_map(map_file):
    wnid_to_class = {}
    with open(map_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                wnid, class_id, class_name = parts
                wnid_to_class[wnid] = (int(class_id), class_name)
    return wnid_to_class

def build_samples_xml(image_dir, annotation_dir, wnid_to_class):
    """
    Parse Pascal‑style XML annotations, map each image to a one-hot label vector,
    and return a list of (image_path, one_hot_label, filename, class_name).
    """
    if not wnid_to_class:
        raise ValueError("wnid_to_class map is empty.")

    # Determine total number of classes
    num_classes = max(cid for cid, _ in wnid_to_class.values())

    samples = []
    for ann_file in os.listdir(annotation_dir):
        if not ann_file.endswith('.xml'):
            continue

        xml_path = os.path.join(annotation_dir, ann_file)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            filename = root.find('filename').text
            obj = root.find('object')
            if obj is None:
                continue

            wnid = obj.find('name').text
            if wnid not in wnid_to_class:
                continue

            class_id, class_name = wnid_to_class[wnid]
            image_path = os.path.join(image_dir, f"{filename}.JPEG")
            if not os.path.exists(image_path):
                continue

            # Build one-hot label
            one_hot = np.zeros(num_classes, dtype=np.float32)
            one_hot[class_id - 1] = 1.0

            samples.append((image_path, one_hot, filename, class_name))

        except Exception as e:
            print(f"Error parsing {ann_file}: {e}")

    print(f"Loaded {len(samples)} validation samples.")
    return samples


def build_samples_file(imagesets_dir, image_root_dir, wnid_to_class):
    img_dict = {}  # image_path -> set of class_ids
    img_info = {}  # image_path -> (basename, wnid)

    # Determine the number of classes from the map
    if not wnid_to_class:
        raise ValueError("wnid_to_class map is empty.")
    num_classes = max(cid for cid, _ in wnid_to_class.values()) if wnid_to_class else 0
    if num_classes <= 0:
        print("Warning: Could not determine number of classes from wnid_to_class map. Assuming 200 based on loop range.")
        num_classes = 200 

    for i in range(1, num_classes + 1):  # Iterate based on determined number of classes
        list_file = os.path.join(imagesets_dir, f'train_{i}.txt')
        if not os.path.exists(list_file):
            continue

        class_id = i
        count = 0

        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                rel_path, label = parts
                if label != '1':
                    continue  # Only include positive samples

                full_path = os.path.join(image_root_dir, rel_path + '.JPEG')
                if not os.path.exists(full_path):
                    continue

                if full_path not in img_dict:
                    img_dict[full_path] = set()
                img_dict[full_path].add(class_id)
                count += 1

                # Store basename and wnid for later use
                if full_path not in img_info:
                    wnid = rel_path.split('/')[1]
                    img_info[full_path] = (os.path.basename(rel_path), wnid)
        
        print(f"Loaded {count} samples for class {i}")

    # Now build final samples list
    samples = []
    for full_path, class_ids in img_dict.items():
        basename, wnid = img_info[full_path]
        class_names = [wnid_to_class.get(wnid, (cid, 'unknown'))[1] for cid in class_ids]
        
        # Create one-hot encoded label
        one_hot_label = np.zeros(num_classes, dtype=np.float32)
        for cid in class_ids:
            if 1 <= cid <= num_classes:
                one_hot_label[cid - 1] = 1.0  # Adjust for 0-based indexing
            else:
                print(f"Warning: Class ID {cid} out of range [1, {num_classes}] for image {basename}")

        samples.append((full_path, one_hot_label, basename, class_names))
        
    print(f"Loaded {len(samples)} training samples.")
    return samples


class ImageNetDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, filename, class_name = self.samples[idx]  # label is now one-hot numpy array
        image = Image.open(img_path)
        # Ensure image has the correct shape
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label  # label is the one-hot encoded numpy array
        

def create_imagenet_dataloaders(base_data_dir, batch_size=32, img_size=244, test_split=0.1, val_split=0.1):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_folder = 'train'
    val_folder = 'val'
    map_dir = os.path.join(base_data_dir, 'map_det.txt')
    wnid_to_class = load_wnid_map(map_dir)
    
    # Load map and all training samples
    train_dir = os.path.join(base_data_dir, 'Data/DET', train_folder)
    train_ann_dir = os.path.join(base_data_dir, 'Annotations/DET', train_folder)
    imagesets_dir = os.path.join(base_data_dir, 'ImageSets/DET')
    all_samples = build_samples_file(imagesets_dir, train_dir, wnid_to_class)

    # Split training samples into train, val, and test sets
    total     = len(all_samples)
    val_size  = int(val_split  * total)
    test_size = int(test_split * total)
    train_size = total - val_size - test_size
    train_ds, val_ds, test_ds = random_split(
        all_samples,
        [train_size, val_size, test_size]
    )

    # Build val dataset from val folder
    # val_dir = os.path.join(base_data_dir, 'Data/DET', val_folder)
    # val_ann_dir = os.path.join(base_data_dir, 'Annotations/DET', val_folder)
    # val_samples = build_samples_xml(val_dir, val_ann_dir, wnid_to_class)

    # Wrap all in Dataset + DataLoader
    train_loader = DataLoader(ImageNetDataset(train_ds, transform=train_transform), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(ImageNetDataset(val_ds, transform=eval_transform), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(ImageNetDataset(test_ds, transform=eval_transform), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader

