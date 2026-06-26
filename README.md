# Kenyan Food Classification (13 Classes)

Image classification of 13 traditional Kenyan food categories using transfer learning with ResNet-50, built with PyTorch Lightning.

**Result: 75.2% validation accuracy after ~200 epochs.**

---

## Dataset

8,174 images split into **6,536 training** and **1,638 validation** images (75/25 stratified split) across 13 classes:

> Bhaji, Chapati, Githeri, Kachumbari, Kukuchoma, Mandazi, Masalachips, Matoke, Mukimo, Nyamachoma, Pilau, Sukumawiki, Ugali

The dataset expects the following structure under `root_folder` (configured in `config.yaml`):

```
root_folder/
├── train.csv          # columns: id, class
├── test.csv           # column: id
└── images/images/     # all image files
```

## Method

Fine-tuning a pretrained **ResNet-50** backbone with a linear classification head.

**Augmentations** (training only, via Albumentations):
- Color jitter (brightness, contrast, saturation, hue)
- Random grayscale conversion
- Horizontal and vertical flips
- Random 90° rotation and affine shifts
- Elastic transform and grid distortion

**Training setup:**
- Loss: Cross-Entropy
- Optimizer: Adam (configurable via `config.yaml`)
- Scheduler: CosineAnnealingLR (configurable via `config.yaml`)
- Mixed precision: `16-mixed`
- Early stopping on `valid/loss` (patience: 10)
- Best checkpoint saved by `valid/acc`
- Logs: TensorBoard

## Project Structure

```
├── main.py          # Entry point — wires LightningCLI
├── model.py         # FineTuningWithResNet (LightningModule)
├── datamodule.py    # KenyaDataModule (LightningDataModule)
├── dataset.py       # KenyanFood13Dataset (torch Dataset)
└── config.yaml      # All hyperparameters
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Set `data.root_folder` in `config.yaml` to your local dataset path, then run:

```bash
# Train
python main.py fit --config config.yaml

# Validate a checkpoint
python main.py validate --config config.yaml --ckpt_path path/to/checkpoint.ckpt

# Generate predictions on the test set
python main.py predict --config config.yaml --ckpt_path path/to/checkpoint.ckpt
```

Any `config.yaml` value can be overridden from the command line:

```bash
python main.py fit --config config.yaml --model.optimizer.init_args.lr 0.001 --trainer.max_epochs 50
```

Run `python main.py fit --print_config` to see all available options.

## Visualizing Training

```bash
tensorboard --logdir logs/
```

---

**Note:** This project was originally developed as the second assignment of the OpenCV University course ["Deep Learning with PyTorch"](https://opencv.org/university/deep-learning-with-pytorch/).
