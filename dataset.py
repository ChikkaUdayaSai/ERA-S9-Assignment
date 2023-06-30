import torch
from torchvision import transforms, datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import is_cuda_available

class Cifar10Dataset(datasets.CIFAR10):
    def __init__(self, root="./data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

train_transform = A.Compose([
A.HorizontalFlip(p=0.5),
A.ShiftScaleRotate(p=0.5),
A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, 
                min_width=1, fill_value=(0.4914, 0.4822, 0.4465), mask_fill_value = None),
A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
ToTensorV2(),
])

test_transform = A.Compose([
A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
ToTensorV2(),
])

train_dataset = Cifar10Dataset(root='./data', train=True, download=True, transform=train_transform)
test_dataset = Cifar10Dataset(root='./data', train=False, download=True, transform=test_transform)

data_loader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if is_cuda_available() else dict(shuffle=True, batch_size=128)

train_loader = torch.utils.data.DataLoader(train_dataset, **data_loader_args)
test_loader = torch.utils.data.DataLoader(test_dataset, **data_loader_args)
