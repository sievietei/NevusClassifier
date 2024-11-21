import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import random
import imageio
import numpy as np
from nevus_coco_model import NevusCocoClassifier


class NevusCocoImageDataset(Dataset):
    def __init__(self, nevus_dir: Path, coco_dir: Path, transform=None):
        self.transform = transform
        labeled_nevus = [(item, 0) for item in self._get_images(images_dir=nevus_dir)]
        labeled_coco = [(item, 1) for item in self._get_images(images_dir=coco_dir)]

        self._images = labeled_nevus + labeled_coco
        random.shuffle(self._images)

    def _get_images(self, images_dir: Path) -> list[Path]:
        res: list[Path] = []

        for f in sorted(images_dir.rglob("*.jpg")):
            if f.is_file() and not f.name.startswith("._"):
                if len(res) < 5000:
                    res.append(str(f))

        return res

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image, label = self._images[idx]
        img = imageio.v2.imread(image)

        # Check if the image is grayscale (2D array)
        if len(img.shape) == 2:
            # Convert grayscale to RGB by stacking along the last axis
            img = np.stack((img,) * 3, axis=-1)

        transformed = self.transform(img)
        return transformed, label


def train_model(nevus_dir: Path, coco_dir: Path, max_epochs: int, trains: Path):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                    ),
                ],
                p=0.8,
            ),
            transforms.RandomAffine(
                degrees=180, translate=(0.1, 0.1), scale=(0.7, 1.4), shear=10
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((224, 224)),
        ]
    )

    

    dataset = NevusCocoImageDataset(
        nevus_dir=nevus_dir, coco_dir=coco_dir, transform=transform
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    num_workers = 6

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    model = NevusCocoClassifier()
    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(model, train_data_loader, val_data_loader)

    # Unfreeze the backbone
    for param in model.backbone.parameters():
        param.requires_grad = True

    model.configure_optimizers = lambda: optim.Adam(model.parameters(), lr=1e-4)

    checkpoint_callback = ModelCheckpoint(
        dirpath=trains,
        filename="nevus-coco-{epoch}-{val_loss:.4f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[checkpoint_callback])
    trainer.fit(model, train_data_loader, val_data_loader)


if __name__ == "__main__":
    random.seed(0)
    # Training the model
    train_model(
        nevus_dir=Path("C:\\Development\\git\\datasets\\nevus"),
        coco_dir=Path("C:\\Development\\git\\datasets\\coco"),
        trains=Path("C:\\Development\\git\\sclassifier\\trains"),
        max_epochs=4,
    )
