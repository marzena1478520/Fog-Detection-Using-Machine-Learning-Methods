from pathlib import Path
from PIL import Image
from typing import Any, Literal

import skimage as ski
import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchmetrics as tm
from torchvision.models import get_model
from torchvision.datasets import DatasetFolder


def load_image(path: str):
    img = ski.io.imread(path)
    if img.ndim == 2:  # Handle grayscale
        img = ski.color.gray2rgb(img)
    if img.shape[-1] == 4:  # Handle RGBA
        img = ski.color.rgba2rgb(img)
    img = ski.util.img_as_ubyte(img)
    img = img.squeeze()

    return Image.fromarray(img)


def get_dataloader(
    path: Path | str,
    transform: Any | None = None,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    return DataLoader(
        dataset=DatasetFolder(
            root=path,
            loader=load_image,
            extensions=[".jpg", ".png", ".jpeg"],
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )


def recursive_getattr(obj: object, name: str):
    names = name.split(".")
    for name in names:
        obj = getattr(obj, name)
    return obj


def recursive_setattr(obj: object, value: Any, name: str):
    names = name.split(".")
    for name in names[:-1]:
        obj = getattr(obj, name)
    setattr(obj, names[-1], value)


class CNNClassifier(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        loss: nn.Module | None = None,
        learning_rate: float = 1e-4,
        optimizer_name: Literal["adam", "adamw"] = "adam",
        weights: str | bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name

        self.model_name = model_name
        self.num_classes = num_classes
        self.weights = weights
        if weights == True:  # noqa: E712
            self.weights = f"{self.model_name}_Weights.DEFAULT"
        self.model = self._get_model()

        if loss is not None:
            self.criterion = loss
        else:
            self.criterion = nn.CrossEntropyLoss()

        task = "multiclass" if num_classes > 2 else "binary"

        self.train_metrics = tm.MetricCollection(
            {
                "accuracy": tm.classification.Accuracy(
                    task=task, num_classes=num_classes
                ),
                "f1": tm.classification.F1Score(task=task, num_classes=num_classes),
                "precision": tm.classification.Precision(
                    task=task, num_classes=num_classes
                ),
                "recall": tm.classification.Recall(task=task, num_classes=num_classes),
            },
            prefix="train_",
        )
        self.validation_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def _get_model(self):
        model = get_model(self.model_name, weights=self.weights)

        classifier_path = None
        classifier_in_features = None
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            classifier_path = "fc"
            classifier_in_features = model.fc.in_features
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Linear):
                classifier_path = "classifier"
                classifier_in_features = model.classifier.in_features
            elif isinstance(model.classifier, nn.Sequential):
                for name, module in reversed(list(model.classifier.named_children())):
                    if isinstance(module, nn.Linear):
                        classifier_path = f"classifier.{name}"
                        classifier_in_features = module.in_features
                        break
        elif (
            hasattr(model, "heads")
            and hasattr(model.heads, "head")
            and isinstance(model.heads.head, nn.Linear)
        ):
            classifier_path = "heads.head"
            classifier_in_features = model.heads.head.in_features
        elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
            classifier_path = "head"
            classifier_in_features = model.head.in_features
        else:
            raise NotImplementedError(f"Model {model} not implemented yet")

        recursive_setattr(
            model, nn.Linear(classifier_in_features, self.num_classes), classifier_path
        )

        return model

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds, labels = self._common_step(batch, batch_idx)

        self.log("train_loss", loss)
        self.log_dict(self.train_metrics(preds, labels))

        return loss

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds, labels = self._common_step(batch, batch_idx)
        self.validation_metrics.update(preds, labels)
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.validation_metrics.compute())
        self.validation_metrics.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds, labels = self._common_step(batch, batch_idx)
        self.test_metrics.update(preds, labels)
        return loss

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "adamw":
            optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}.")
        return optimizer

    def predict_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            batch = batch[0]
        if isinstance(batch, list):
            batch = batch[0]
        return self(batch)
