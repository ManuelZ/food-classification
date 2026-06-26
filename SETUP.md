# Setup & Usage

## Project Structure

```
├── main.py          # Entry point — wires LightningCLI
├── model.py         # ImageClassifier (LightningModule)
├── datamodule.py    # KenyaDataModule (LightningDataModule)
├── dataset.py       # KenyanFood13Dataset (torch Dataset)
└── config.yaml      # All hyperparameters
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Layout

The dataset expects the following structure under `root_folder` (configured in `config.yaml`):

```
root_folder/
├── train.csv          # columns: id, class
├── test.csv           # column: id
└── images/images/     # all image files
```

## Data Preparation (optional)

Pre-resize images to a fixed square size before training to avoid repeated on-the-fly resizing:

```bash
python resize_images.py \
  --images_dir /workspace/opencv-pytorch-classification-project-2/images/images \
  --output_dir /workspace/opencv-pytorch-classification-project-2-resize/images/images \
  --size 256
```

| Argument | Default | Description |
|---|---|---|
| `--images_dir` | *(required)* | Source directory containing images |
| `--output_dir` | same as `--images_dir` | Destination directory; omit to resize in-place |
| `--size` | `256` | Target width and height in pixels |
| `--quality` | `95` | JPEG output quality (1–100) |
| `--workers` | CPU count | Parallel worker processes |

## Training

Set `data.root_folder` in `config.yaml` to your local dataset path, then run:

```bash
# Train
python main.py fit --config config.yaml

# Validate a checkpoint on the validation set
python main.py validate --config config.yaml --ckpt_path path/to/checkpoint.ckpt

# Evaluate on the test set
python main.py test --config config.yaml --ckpt_path path/to/checkpoint.ckpt
```

Any `config.yaml` value can be overridden from the command line:

```bash
python main.py fit --config config.yaml --model.optimizer.init_args.lr 0.001 --trainer.max_epochs 50
```

Run `python main.py fit --print_config` to see all available options.

### Training Best Model

Train with the best hyperparameters found by Optuna (trial #21: convnext_tiny, val F1=0.7835):

```bash
python main.py fit --config config.yaml \
  --model.model_name convnext_tiny \
  --model.optimizer.init_args.lr 2.007821448001929e-04 \
  --model.optimizer.init_args.weight_decay 1.060491933490234e-06 \
  --data.batch_size 128 \
  --trainer.max_epochs 100 \
  --trainer.logger.init_args.name convnext_tiny \
  --trainer.callbacks[1].init_args.filename fine-tuning-convnext_tiny-epoch-{epoch:02d}
```

Best small model (trial #5: efficientnet_b0, val F1=0.7561):

```bash
python main.py fit --config config.yaml \
  --model.model_name efficientnet_b0 \
  --model.optimizer.init_args.lr 5.840867524690305e-04 \
  --model.optimizer.init_args.weight_decay 2.0797397710366865e-03 \
  --data.batch_size 128 \
  --trainer.max_epochs 100 \
  --trainer.logger.init_args.name efficientnet_b0 \
  --trainer.callbacks[1].init_args.filename fine-tuning-efficientnet_b0-epoch-{epoch:02d}
```

## Hyperparameter Search

Automated search over learning rate, weight decay, batch size, and backbone architecture using [Optuna](https://optuna.org/) with early pruning of unpromising trials.

```bash
python hparam_search.py                          # defaults from config.yaml
python hparam_search.py --n-trials 30 --max-epochs 10
```

The search space is defined in `config.yaml` under `hparam_search.search_space`. Each parameter needs a `type` (`float`, `int`, or `categorical`) and the corresponding bounds or choices:

```yaml
hparam_search:
  n_trials: 20
  max_epochs: 5
  search_space:
    lr:
      type: float
      low: 1.0e-6
      high: 1.0e-2
      log: true
    batch_size:
      type: categorical
      choices: [64, 128, 256]
    model_name:
      type: categorical
      choices: [resnet18, efficientnet_b0, convnext_tiny]
```

Each trial is logged to TensorBoard under `logs/optuna/trial_N/`. The study uses `MedianPruner` to stop unpromising trials after a warmup period, optimizing `valid/f1_macro`.

Results are persisted to `optuna.db` in the project root. To monitor trials live with the Optuna Dashboard (run in a separate terminal while the search is running):

```bash
optuna-dashboard sqlite:///optuna.db
```

## Visualizing Training

```bash
tensorboard --logdir logs/
```

---

## Cloud Training (Vast.ai)

**Prerequisites:** create two S3 buckets (one for datasets, one for artifacts), create an IAM user with appropriate S3 permissions, and generate access keys.

In the platform instance settings, add these variables to the secrets/environment panel before launching:

```bash
AWS_ACCESS_KEY_ID=<YOUR_ACCESS_KEY_ID>
AWS_SECRET_ACCESS_KEY=<YOUR_SECRET_ACCESS_KEY>
AWS_DEFAULT_REGION=<YOUR_REGION>
ARTIFACTS_BUCKET=<BUCKET_NAME>
DATASETS_BUCKET=<ANOTHER_BUCKET_NAME>
```

Both the AWS CLI and boto3 recognize these variable names natively. `DATASETS_BUCKET` is used to download the dataset; `ARTIFACTS_BUCKET` is used to upload trained checkpoints and logs after training.

Clone repo and install dependencies:
```bash
git clone https://github.com/ManuelZ/food-classification.git
cd food-classification
pip install -r requirements.txt
```

Download dataset:
```bash
aws s3 cp s3://$DATASETS_BUCKET/opencv-pytorch-classification-project-2.zip opencv-pytorch-classification-project-2.zip
```

Unzip downloaded dataset:
```bash
unzip opencv-pytorch-classification-project-2.zip -d /workspace/opencv-pytorch-classification-project-2
```

Pre-resize images (optional, avoids repeated on-the-fly resizing during training):
```bash
python resize_images.py \
  --images_dir /workspace/opencv-pytorch-classification-project-2/images/images \
  --output_dir /workspace/opencv-pytorch-classification-project-2-resize/images/images \
  --size 256
```

Copy CSV files to the new folder with the resized images:
```bash
cp /workspace/opencv-pytorch-classification-project-2/*.csv /workspace/opencv-pytorch-classification-project-2-resize/
```

Start training:
```bash
python main.py fit --config config.yaml \
  --trainer.max_epochs=10 \
  --data.num_workers=16 \
  --model.model_name=convnext_tiny \
  --trainer.logger.init_args.name=convnext_tiny \
  --trainer.callbacks[1].init_args.filename=fine-tuning-convnext_tiny-epoch-{epoch:02d}
```

Monitor training with TensorBoard (run on the remote instance):
```bash
tensorboard --logdir food-classification/logs/
```

Forward the TensorBoard port to your local machine (run locally):
```bash
ssh -p <remote_port> root@<remote_host> -L 16006:localhost:6006 -i ~/.ssh/id_ed25519_platform
```

Then open http://localhost:16006 in your local browser.

**Alternatively, connect with VS Code Remote SSH** (port forwarding included):

1. Install the **Remote - SSH** extension in VS Code
2. Press `Ctrl+Shift+P` → **Remote-SSH: Add New SSH Host...**
3. Paste the SSH command from the Vast.ai instance panel (e.g. `ssh root@<IP> -p <PORT>`)
4. VS Code will create an entry in `~/.ssh/config` — open that file and update the block to:

```
Host vastai
    HostName <IP>
    Port <PORT>
    User root
    LocalForward 16006 localhost:6006
    IdentityFile ~/.ssh/<PRIVATE_KEY_FILENAME>
```

5. Press `Ctrl+Shift+P` → **Remote-SSH: Connect to Host...** → select `vastai`

The `LocalForward` line automatically tunnels TensorBoard (no separate SSH command needed). Update `HostName` and `Port` each time you start a new Vast.ai instance.

Vast.ai drops you into an existing tmux session on login. To start a fresh one from within tmux:

```
Ctrl+b :new-session
```

Upload artifacts:
```bash
zip -r logs.zip logs/
aws s3 cp logs.zip s3://$ARTIFACTS_BUCKET/food_classification/logs.zip
```
