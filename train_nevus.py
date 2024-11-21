import polars.series
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import random
import imageio
import numpy as np
from nevus_model import NevusClassifier
import polars
import torch.nn.functional as F


class NevusImageDataset(Dataset):
    def __init__(self, nevus_csv: Path, transform=None):
        self.transform = transform
        data_frame = polars.read_csv(nevus_csv)
        self.classes = sorted(data_frame["dx"].unique())

        file_paths = []
        class_label = []

        for row in data_frame.iter_rows():
            dx_label = row[data_frame.get_column_index("dx")]
            image_id = row[data_frame.get_column_index("image_id")]
            class_label.append(list.index(self.classes, dx_label))
            found = False
            for sub_dir in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
                test_file = nevus_csv.parent / sub_dir / f"{image_id}.jpg"
                if test_file.exists():
                    file_paths.append(str(test_file))
                    found = True

            if not found:
                raise ValueError(f"Can't find file for: {image_id}")

        data_frame = data_frame.with_columns(
            polars.Series("class_label", class_label),
            polars.Series("file_path", file_paths),
        )

        self.data_frame = data_frame

        pass

    def __len__(self):
        return self.data_frame.shape[0]

    def __getitem__(self, idx):
        row = self.data_frame.row(idx)
        file_path = row[self.data_frame.get_column_index("file_path")]
        img = imageio.v2.imread(file_path)

        # Check if the image is grayscale (2D array)
        if len(img.shape) == 2:
            # Convert grayscale to RGB by stacking along the last axis
            img = np.stack((img,) * 3, axis=-1)

        transformed = self.transform(img)
        label = row[self.data_frame.get_column_index("class_label")]
        one_hot_vector = F.one_hot(torch.tensor(label), len(self.classes)).float()
        return transformed, one_hot_vector


def train_model(nevus_csv: Path, max_epochs: int, trains: Path):
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

    dataset = NevusImageDataset(nevus_csv=nevus_csv, transform=transform)

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

    fraction_of_data = 1.0

    model = NevusClassifier(num_classes=len(dataset.classes))
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        limit_train_batches=fraction_of_data,
        limit_val_batches=fraction_of_data,
    )
    trainer.fit(model, train_data_loader, val_data_loader)

    # Unfreeze the backbone
    for param in model.backbone.parameters():
        param.requires_grad = True

    model.configure_optimizers = lambda: optim.Adam(model.parameters(), lr=1e-4)

    checkpoint_callback = ModelCheckpoint(
        dirpath=trains,
        filename="nevus-{epoch}-{val_loss:.4f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        limit_train_batches=fraction_of_data,
        limit_val_batches=fraction_of_data,
    )
    trainer.fit(model, train_data_loader, val_data_loader)


if __name__ == "__main__":
    random.seed(0)
    # Training the model
    train_model(
        nevus_csv=Path("C:\\Development\\git\\datasets\\nevus\\HAM10000_metadata.csv"),
        trains=Path("C:\\Development\\git\\sclassifier\\trains"),
        max_epochs=40,
    )
