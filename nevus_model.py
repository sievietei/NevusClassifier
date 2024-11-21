import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import pytorch_lightning as pl
import imageio
from io import BytesIO
from torchvision import transforms


class NevusClassifier(pl.LightningModule):
    def __init__(self, num_classes: int = 7):
        super(NevusClassifier, self).__init__()
        self.backbone = models.mobilenet_v3_small(pretrained=True)

        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace the classifier
        num_ftrs = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = torch.sum(preds == labels.argmax(dim=1)).item() / (len(labels) * 1.0)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def infer_image(self, jpeg_image: bytes) -> list[float]:
        image = imageio.v2.imread(BytesIO(jpeg_image), format="jpeg")

        with torch.no_grad():
            inf_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                ]
            )
            transformed = inf_transform(image).float()
            res = self(transformed.unsqueeze(0).float())
            return res[0].numpy().tolist()
