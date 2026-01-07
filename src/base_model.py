import torch
from torchinfo import summary as torchinfo_summary
import copy
from abc import ABC, abstractmethod
from typing import List, TypeVar, Literal, Type, Dict, Callable
from pathlib import Path
from .configurations import TrainingParams, Task, TrainingHistory
from .early_stopping import EarlyStoppingHandler, StoppingCriteria
import numpy as np
import random
import os

T = TypeVar("T", bound="BaseTaskModel")


class BaseTaskModel(torch.nn.Module, ABC):
    def __init__(
        self,
        task: Task,
        device: torch.device = torch.device("cpu"),
        track_best_model: bool = True,
        stopping_criteria: List[StoppingCriteria] | None = None,
        random_state: int | None = None,
        **kwargs,  # Captures subclass-specific architecture params
    ):
        super().__init__()

        # 1. Immediate Seed Injection
        # We set this BEFORE any layers are initialized in subclasses
        self.random_state = (
            random_state if random_state is not None else np.random.randint(0, 10000)
        )
        self.seed_everything(self.random_state)

        # 2. Capture init_params for Export/Import reproducibility
        # This stores exactly what's needed to rebuild the object from disk
        self.init_params = {
            "task": task,
            "device": device,
            "track_best_model": track_best_model,
            "stopping_criteria": stopping_criteria,
            **kwargs,
        }

        self.task = task
        self.device = device
        self.network: torch.nn.Sequential | None = None

        self.history: List[TrainingHistory] = []
        self.track_best_model = track_best_model
        self.best_state_dict = None
        self.best_epoch = None
        self.best_metrics = None
        self.best_val_loss = float("inf")

        self.stopping_criteria = stopping_criteria
        self.to(self.device)

    @staticmethod
    def seed_everything(seed: int):
        """Standardizes randomness across all engines and hardware backends."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Absolute Determinism for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Support for newer PyTorch versions forcing deterministic algorithms
        # torch.use_deterministic_algorithms(True) # Uncomment if high-strictness is needed

    def _run_training_loop(self, x, y, *, optimizer, loss_fn, params: TrainingParams):
        # --- Reproducible Split Logic ---
        # We use a local generator tied to the model's random_state
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.random_state)

        indices = torch.randperm(x.size(0), generator=generator, device=self.device)
        split_point = int(x.size(0) * (1 - params.val_size))

        train_idx = indices[:split_point]
        val_idx = indices[split_point:]

        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        # --- Training Initialization ---
        history = TrainingHistory(params=params, phase=params.phase)
        history.initialize()
        self.history.append(history)

        es_handler = EarlyStoppingHandler(self.stopping_criteria or [])

        for epoch in range(1, params.epochs + 1):
            # 1. Training Phase
            self.train()
            _, train_logits = self._run_one_epoch(
                x_train,
                y_train,
                optimizer=optimizer,
                loss_fn=loss_fn,
                batch_size=params.batch_size,
                training_step=params.training_step,
                output_layer=params.output_layer,
            )
            train_metrics = self._compute_metrics(
                train_logits, y_train, loss_fn, params.metrics
            )

            # 2. Validation Phase
            self.eval()
            with torch.no_grad():
                val_outputs = self._run_evaluation_pass(
                    x_val, output_layer=params.output_layer
                )
                val_metrics = self._compute_metrics(
                    val_outputs, y_val, loss_fn, params.metrics
                )

            # 3. Logging & Checkpointing
            history.log_train(train_metrics)
            history.log_val(val_metrics)

            if self.track_best_model and val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_epoch = epoch
                self.best_metrics = val_metrics
                # Ensure we clone to CPU to avoid VRAM leaks during long experiments
                self.best_state_dict = {
                    k: v.cpu().clone() for k, v in self.state_dict().items()
                }

            # 4. Early Stopping & Console Output
            if es_handler.check(epoch, train_metrics, val_metrics):
                print(f"Early stopping triggered at epoch {epoch}")
                break

            if epoch % params.print_every == 0:
                self._print_epoch_log(epoch, params, train_metrics, val_metrics)

    # ... [forward, _run_one_epoch, _run_evaluation_pass, etc. remain as you had them] ...

    def recover_best_model(self) -> None:
        if self.best_state_dict is None:
            print("No best model stored.")
            return
        self.load_state_dict(self.best_state_dict)
        print(f"\n✔ Best model recovered from Epoch {self.best_epoch}")

    def copy(self, *, reset_history: bool = True, reset_best: bool = True):
        """
        Creates a deep copy.
        Crucial for Fine-Tuning experiments to keep pre-trained weights.
        """
        model_copy = copy.deepcopy(self)
        if reset_history:
            model_copy.history = []
        if reset_best:
            model_copy.best_state_dict = None
            model_copy.best_epoch = None
            model_copy.best_metrics = None
            model_copy.best_val_loss = float("inf")
        return model_copy

    @classmethod
    def import_(
        cls: Type[T], path: str | Path, device: torch.device | str = "cpu"
    ) -> T:
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Extract metadata
        seed = checkpoint.get("random_state", 42)
        init_params = checkpoint.get("init_params", {})

        # Reconstruct with the same random state and architecture params
        model = cls(random_state=seed, **init_params)
        model.load_state_dict(checkpoint["state_dict"])

        # Restore Training Metadata
        model.best_state_dict = checkpoint.get("best_state_dict")
        model.best_epoch = checkpoint.get("best_epoch")
        model.best_metrics = checkpoint.get("best_metrics")
        model.best_val_loss = checkpoint.get("best_val_loss")
        model.history = checkpoint.get("history", [])

        return model.to(device)

    def export(self, path: str | Path) -> None:
        path = Path(path)
        checkpoint = {
            "state_dict": self.state_dict(),
            "random_state": self.random_state,
            "init_params": self.init_params,
            "task": self.task.name if isinstance(self.task, Task) else str(self.task),
            "best_state_dict": self.best_state_dict,
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }
        torch.save(checkpoint, path)
