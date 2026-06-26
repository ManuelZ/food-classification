# Classification: 13 Kenyan food types

This is the second project of the Opencv University course ["Deep Learning with PyTorch"](https://opencv.org/university/deep-learning-with-pytorch/).
It focuses on classifying images of 13 different Kenyan food classes.


## Introduction

Classification in computer vision is a task where the objective is to determine the class that the image belongs to. 
This project focuses on classifying images from 13 classes.


## Data

The dataset, containing 8174 images of various sizes, was split into 6536 training images and 1638 validation images.
These images are categorized into the following classes: 

    Bhaji, Chapati, Githeri, Kachumbari, Kukuchoma, Mandazi, Masalachips, Matoke, Mukimo, Nyamachoma, Pilau, Sukumawiki, Ugali

## The method used

Fine-tuning of a ResNet-50 backbone with a linear classifier using PyTorch Lightning.

- Various augmentations techniques were used to try to improve generalization:
  - Color jitter
  - Conversion to gray
  - Horizontal and vertical flips
  - Random shifting and rotation
  - Elastic transformations
  - Grid distortion

- The loss function used was Cross-Entropy.

- An SGD optimizer with weight decay.

- A learning rate scheduler that implements the 1-cycle policy. It adjusts the learning rate from an initial rate to a 
maximum, then decreases it to a much lower minimum.


## Usage

Configure everything in `config.yaml`, then run a subcommand:

```bash
# Train from scratch
python main.py fit --config config.yaml

# Validate a checkpoint
python main.py validate --config config.yaml --ckpt_path path/to/checkpoint.ckpt

# Generate predictions
python main.py predict --config config.yaml --ckpt_path path/to/checkpoint.ckpt
```

Any value in `config.yaml` can be overridden directly on the command line:

```bash
python main.py fit --config config.yaml --model.optimizer.init_args.lr 0.001 --trainer.max_epochs 50
```


## Discussion

Training this model for ~200 epochs resulted in an accuracy of 75.2% on the test set.


See the [notebook](project-2-deep-learning-with-pytorch-2024.ipynb).