import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        
        # Main convolution block
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )

        # Shortcut connection
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        else:
            self.shortcut = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv_block(x) + self.shortcut(x))

class DeepCNN(nn.Module):
    """
    Configurable ResNet architecture
    Common configurations:
    - ResNet50:  layers=[3, 4, 6, 3]
    - ResNet101: layers=[3, 4, 23, 3]
    - ResNet152: layers=[3, 8, 36, 3]
    """
    def __init__(self, num_classes=10, layers=[3, 4, 6, 3]):
        super(DeepCNN, self).__init__()
        
        # Create initial layers
        initial_layers = [
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]
        
        # Create all feature layers
        self.features = nn.Sequential(
            *initial_layers,
            self._make_layer(64, 64, layers[0]),
            self._make_layer(256, 128, layers[1], stride=2),
            self._make_layer(512, 256, layers[2], stride=2),
            self._make_layer(1024, 512, layers[3], stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        # First block might need to handle stride and channel changes
        layers.append(Bottleneck(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels * 4, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def activate_fine_tune(self, num_classes):
        """
        Toggle fine tuning mode and adjust classifier for a new number of classes.
        Freezes all feature extraction layers and replaces the classifier.
        """
        # Freeze feature layers
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Replace the classifier with a new one adapted for the new number of classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes) 
        )