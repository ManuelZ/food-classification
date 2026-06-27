# Usage:
#   python hparam_search.py
#   python hparam_search.py --n-trials 30 --max-epochs 5
#   python hparam_search.py --storage sqlite:///optuna_20260626_143022.db  # resume a previous study

# Standard Library imports
import argparse
import functools
from datetime import datetime

# External imports
import lightning as pl
import optuna
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from optuna_integration import PyTorchLightningPruningCallback

# Local imports
from datamodule import KenyaDataModule
from model import ImageClassifier

with open("config.yaml") as f:
    _config = yaml.safe_load(f)

with open("hparam_config.yaml") as f:
    _HPARAM = yaml.safe_load(f)

_DATA = _config["data"]
_MODEL = _config["model"]


def _suggest(trial: optuna.Trial, search_space: dict) -> dict:
    params = {}
    for name, spec in search_space.items():
        t = spec["type"]
        if t == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif t == "int":
            params[name] = trial.suggest_int(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif t == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown search space type: {t!r}")
    return params


def objective(trial: optuna.Trial, max_epochs: int, run_name: str) -> float:
    params = _suggest(trial, _HPARAM["search_space"])
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    batch_size = params["batch_size"]
    model_name = params.get("model_name", _MODEL["model_name"])

    datamodule = KenyaDataModule(
        root_folder=_DATA["root_folder"],
        batch_size=batch_size,
        num_workers=_DATA["num_workers"],
        prefetch_factor=_DATA.get("prefetch_factor"),
        target_size=_DATA["target_size"],
        train_split=_DATA["train_split"],
    )

    optimizer_callable = functools.partial(
        torch.optim.Adam, lr=lr, weight_decay=weight_decay
    )
    scheduler_callable = functools.partial(
        torch.optim.lr_scheduler.CosineAnnealingLR, T_max=max_epochs
    )

    model = ImageClassifier(
        model_name=model_name,
        weights="DEFAULT",
        num_classes=_MODEL["num_classes"],
        label_names=_MODEL["label_names"],
        optimizer=optimizer_callable,
        scheduler=scheduler_callable,
    )

    logger = TensorBoardLogger(
        save_dir="logs", name=run_name, version=f"trial_{trial.number}"
    )
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="valid/f1_macro")
    early_stop_callback = EarlyStopping(
        monitor="valid/f1_macro",
        patience=_HPARAM["early_stopping_patience"],
        mode="max",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        precision=_config["trainer"]["precision"],
        callbacks=[pruning_callback, early_stop_callback],
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    trainer.fit(model, datamodule)

    return trainer.callback_metrics["valid/f1_macro"].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=_HPARAM["n_trials"])
    parser.add_argument("--max-epochs", type=int, default=_HPARAM["max_epochs"])
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="SQLite DB path to resume a previous study (e.g. sqlite:///optuna_20260626_143022.db). "
        "If omitted, a new timestamped DB is created.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    storage = args.storage or f"sqlite:///optuna_{timestamp}.db"
    load_if_exists = args.storage is not None

    pruner = optuna.pruners.MedianPruner(**_HPARAM["pruner"])
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        study_name="food-classification-hpo",
        storage=storage,
        load_if_exists=load_if_exists,
    )
    study.optimize(
        lambda trial: objective(
            trial, args.max_epochs, f"{_HPARAM['logger_name']}_{timestamp}"
        ),
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best valid/f1_macro:  {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
