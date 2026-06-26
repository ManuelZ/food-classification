# Usage:
#   python main.py fit --config config.yaml --model.optimizer=SGD
#   python main.py validate --config config.yaml --ckpt_path path/to/checkpoint.ckpt
#   python main.py test --config config.yaml --ckpt_path path/to/checkpoint.ckpt

# External imports
from lightning.pytorch.cli import LightningCLI

# Local imports
from datamodule import KenyaDataModule
from model import ImageClassifier


def cli_main():
    LightningCLI(ImageClassifier, KenyaDataModule, auto_configure_optimizers=False)


if __name__ == "__main__":
    cli_main()
