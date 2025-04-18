import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

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
    samples = []

    for i, ann_file in enumerate(os.listdir(annotation_dir)):
        if not ann_file.endswith('.xml'):
            continue

        xml_path = os.path.join(annotation_dir, ann_file)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            filename = root.find('filename').text
            object_elem = root.find('object')
            if object_elem is None:
                continue

            wnid = object_elem.find('name').text
            if wnid not in wnid_to_class:
                continue

            class_id, class_name = wnid_to_class[wnid]
            image_path = os.path.join(image_dir, f"{filename}.JPEG")
            if not os.path.exists(image_path):
                continue

            samples.append((image_path, class_id, filename, class_name))

        except Exception as e:
            print(f"Error parsing {ann_file}: {e}")

    print(f"Loaded {len(samples)} validation samples.")
    return samples

def build_samples_file(imagesets_dir, image_root_dir, wnid_to_class):
    img_dict = {}  # image_path -> set of class_ids
    img_info = {}  # image_path -> (basename, wnid)

    for i in range(1, 201):  # train_1.txt to train_200.txt
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
        samples.append((full_path, sorted(list(class_ids)), basename, class_names))
        
    print(f"Loaded {len(samples)} training samples.")
    return samples


class ImageNetDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, filename, class_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        return image, label
        

def create_imagenet_dataloaders(base_data_dir, batch_size=32, img_size=224, test_split=0.1):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_folder = 'train'
    val_folder = 'val'
    map_dir = os.path.join(base_data_dir, 'map_det.txt')
    wnid_to_class = load_wnid_map(map_dir)
    
    # Load map and all training samples
    train_dir = os.path.join(base_data_dir, 'Data/DET', train_folder)
    train_ann_dir = os.path.join(base_data_dir, 'Annotations/DET', train_folder)
    imagesets_dir = os.path.join(base_data_dir, 'ImageSets/DET')
    train_samples = build_samples_file(imagesets_dir, train_dir, wnid_to_class)

    # Split training samples into train and test (e.g., 90/10)
    total = len(train_samples)
    test_size = int(test_split * total)
    train_size = total - test_size
    train_dataset, test_dataset = random_split(train_samples, [train_size, test_size])

    # Build val dataset from val folder
    val_dir = os.path.join(base_data_dir, 'Data/DET', val_folder)
    val_ann_dir = os.path.join(base_data_dir, 'Annotations/DET', val_folder)
    val_samples = build_samples_xml(val_dir, val_ann_dir, wnid_to_class)

    # Wrap all in Dataset + DataLoader
    train_loader = DataLoader(ImageNetDataset(train_dataset, transform=transform), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ImageNetDataset(val_samples, transform=transform), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ImageNetDataset(test_dataset, transform=transform), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

