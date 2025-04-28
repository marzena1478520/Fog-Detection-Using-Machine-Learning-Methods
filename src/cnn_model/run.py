import os
import sys
from pathlib import Path
from time import time

from prettytable import PrettyTable
import lightning as L
from torchvision.models import get_weight


from src.cnn_model.utils import CNNClassifier, get_dataloader

DATASET_PATHS = {
    "fog-detection": Path("./datasets/fog-detection-dataset-prepared"),
    "fog-or-smog": Path("./datasets/fog-or-smog-detection-dataset-prepared"),
    "foggy-cityscapes": Path("./datasets/foggy-cityscapes-image-dataset-prepared"),
    "combined": Path("./datasets/fog-combined"),
}

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python run.py <run_name> <model_name> <dataset_name> [<Weights>]")
        sys.exit(1)
    RUN_NAME = sys.argv[1]
    MODEL_NAME = sys.argv[2]
    DATASET = sys.argv[3]
    WEIGHTS = sys.argv[4] if len(sys.argv) > 4 else False

    SAVE_DIR = Path(f"runs/classify/{RUN_NAME}")
    VERSION = 1

    model = CNNClassifier(
        model_name=MODEL_NAME,
        num_classes=2,
        weights=WEIGHTS,
    )
    _transform = get_weight(
        f"{MODEL_NAME}_Weights.DEFAULT" if not WEIGHTS else model.weights
    ).transforms()

    trainer = L.Trainer(
        max_epochs=20,
        logger=L.pytorch.loggers.TensorBoardLogger(
            save_dir=SAVE_DIR,
            name=f"{MODEL_NAME}-{DATASET}",
            version=VERSION,
        ),
        callbacks=[
            L.pytorch.callbacks.early_stopping.EarlyStopping(
                monitor="val_loss", mode="min", patience=5, verbose=False
            ),
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="val_f1",
                mode="max",
                dirpath=SAVE_DIR / f"{MODEL_NAME}-{DATASET}" / f"version_{VERSION}",
                filename="{epoch}-{val_loss:.2f}-{val_f1:.2f}",
            ),
        ],
        log_every_n_steps=1,
    )

    _start = time()
    trainer.fit(
        model,
        train_dataloaders=get_dataloader(
            path=DATASET_PATHS[DATASET] / "train",
            transform=_transform,
            batch_size=32,
            shuffle=True,
            num_workers=11,
        ),
        val_dataloaders=get_dataloader(
            path=DATASET_PATHS[DATASET] / "val",
            transform=_transform,
            batch_size=32,
            shuffle=False,
            num_workers=11,
        ),
    )
    _end = time()

    res = {
        dataset_name: trainer.test(
            model, get_dataloader(path=path / "test", transform=_transform)
        )[0]
        for dataset_name, path in DATASET_PATHS.items()
    }

    print(f"Training time: {(_end - _start):.2f}")
    table = PrettyTable()
    table.field_names = ["Dataset", *list(next(iter(res.values())).keys())]
    table.add_rows(
        [
            [dataset, *[round(m, 4) for m in metrics.values()]]
            for dataset, metrics in res.items()
        ]
    )
    print(table)
