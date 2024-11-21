import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import pytorch_lightning as pl
import imageio
from io import BytesIO
from torchvision import transforms


class NevusCocoClassifier(pl.LightningModule):
    def __init__(self):
        super(NevusCocoClassifier, self).__init__()
        self.backbone = models.mobilenet_v2(pretrained=True)

        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace the classifier
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(num_ftrs, 1)
        )

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.float().unsqueeze(
            1
        )  # Convert to float and add a channel dimension
        outputs = self(images)
        loss = nn.BCEWithLogitsLoss()(outputs, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch[0]
        labels = batch[1]
        labels = labels.float().unsqueeze(1)
        outputs = self(images)
        loss = nn.BCEWithLogitsLoss()(outputs, labels)
        preds = torch.sigmoid(outputs) > 0.5
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def infer_image(self, jpeg_image: bytes) -> float:
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
            return res[0].item()
