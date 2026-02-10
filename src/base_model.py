import time
import torch
from torchinfo import summary as torchinfo_summary
import copy
from abc import ABC, abstractmethod
from typing import List, TypeVar, Literal, Type, Dict, Callable
from pathlib import Path
from .configurations import (
    TrainingParams,
    Task,
    TrainingHistory,
    TrainingTermination,
    TerminationReason,
)
from .early_stopping import EarlyStoppingHandler
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

    def forward(
        self, x: torch.Tensor, *, output_layer: str | None = None
    ) -> torch.Tensor:
        if self.network is None:
            raise RuntimeError("self.network is not defined.")

        for name, layer in self.network.named_children():
            x = layer(x)
            if output_layer is not None and name == output_layer:
                return x

        if output_layer is not None:
            raise ValueError(f"Layer '{output_layer}' not found.")
        return x

    @abstractmethod
    def _compute_metrics(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable,
        metrics: List = None,
    ) -> Dict[str, float]:
        pass

    def summary(self, input_size, **kwargs):
        return torchinfo_summary(self, input_size, **kwargs)

    def _run_evaluation_pass(
        self, x: torch.Tensor, output_layer: str | None = None
    ) -> torch.Tensor:
        """
        Default evaluation pass hook.
        Subclasses override this for Siamese/Contrastive logic.
        """
        return self(x, output_layer=output_layer)

    def _run_one_epoch(
        self, x, y, *, optimizer, loss_fn, batch_size, training_step, output_layer
    ) -> tuple[float, torch.Tensor]:
        self.train()

        start_time = time.time()

        epoch_loss = 0.0
        if batch_size == "full":
            batch_size = x.size(0)
        num_batches = (x.size(0) + batch_size - 1) // batch_size

        all_outputs = []
        for i in range(num_batches):
            xb, yb = (
                x[i * batch_size : (i + 1) * batch_size],
                y[i * batch_size : (i + 1) * batch_size],
            )
            loss, logits = training_step(
                model=self,
                xb=xb,
                yb=yb,
                optimizer=optimizer,
                loss_fn=loss_fn,
                output_layer=output_layer,
            )
            epoch_loss += loss * xb.size(0)
            all_outputs.append(logits)

        duration = time.time() - start_time

        return epoch_loss / x.size(0), torch.cat(all_outputs, dim=0), duration

    def _run_training_loop(self, x, y, *, optimizer, loss_fn, params: TrainingParams):
        start_wall_time = time.time()

        # 1. Stopping Criteria & Epoch setup
        criteria = params.stopping_criteria or []
        if params.epochs is None:
            if not criteria:
                raise ValueError(
                    "Must provide stopping criteria for infinite-loop training."
                )
            max_epochs = 1_000_000
        else:
            max_epochs = params.epochs

        # 2. Reproducible Split Logic
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.random_state)
        indices = torch.randperm(x.size(0), generator=generator, device=self.device)
        split_point = int(x.size(0) * (1 - params.val_size))

        x_train, y_train = x[indices[:split_point]], y[indices[:split_point]]
        x_val, y_val = x[indices[split_point:]], y[indices[split_point:]]

        # 3. Initialization
        history = TrainingHistory(params=params, phase=params.phase)
        history.initialize()
        self.history.append(history)
        es_handler = EarlyStoppingHandler(criteria_list=criteria)

        # Termination tracking variables
        final_epoch = 0
        termination_reason = TerminationReason.MAX_EPOCHS
        stop_details = "Completed all requested epochs."
        latest_val_metrics = {}

        try:
            for epoch in range(1, max_epochs + 1):
                final_epoch = epoch

                # --- Training Step ---
                self.train()
                _, train_logits, duration = self._run_one_epoch(
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
                history.log_epoch_time(duration)

                # --- Validation Step ---
                self.eval()
                with torch.no_grad():
                    val_outputs = self._run_evaluation_pass(
                        x_val, output_layer=params.output_layer
                    )
                    val_metrics = self._compute_metrics(
                        val_outputs, y_val, loss_fn, params.metrics
                    )

                latest_val_metrics = val_metrics
                history.log_train(train_metrics)
                history.log_val(val_metrics)

                # --- Best Model Checkpointing ---
                if self.track_best_model and val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.best_epoch = epoch
                    self.best_metrics = val_metrics
                    self.best_state_dict = {
                        k: v.cpu().clone() for k, v in self.state_dict().items()
                    }

                # --- Early Stopping Check ---
                triggered = es_handler.check(epoch, train_metrics, val_metrics)
                if triggered:
                    termination_reason = TerminationReason.EARLY_STOPPING
                    stop_details = (
                        triggered.message or f"Criteria met for {triggered.metric_name}"
                    )
                    break

                if epoch % params.print_every == 0:
                    self._print_epoch_log(
                        epoch, params, train_metrics, val_metrics, duration
                    )

            else:  # If loop finishes without 'break'
                if params.epochs is None:
                    termination_reason = TerminationReason.HARD_SAFETY_LIMIT
                    stop_details = f"Reached safety limit of {max_epochs} epochs."

        except KeyboardInterrupt:
            termination_reason = TerminationReason.MANUAL_INTERRUPTION
            stop_details = "Training manually interrupted by user."
            raise  # Re-raise to let the caller handle the interrupt if needed

        finally:
            # --- Finalize Termination Data ---
            recovered = False
            if self.track_best_model and self.best_state_dict:
                self.load_state_dict(self.best_state_dict)
                recovered = True
                print(f"Restored best model from epoch {self.best_epoch}")

            total_duration = time.time() - start_wall_time
            history.set_total_time(total_duration)

            history.termination = TrainingTermination(
                epoch=final_epoch,
                reason=termination_reason,
                details=stop_details,
                final_val_metrics=latest_val_metrics,
                best_model_recovered=recovered,
            )

    def _print_epoch_log(self, epoch, params, train_m, val_m, duration=None):
        # 1. Format the metrics string
        m_str = (
            " | ".join(
                [
                    f"T-{m.name}: {train_m[m.name]:.4f} | V-{m.name}: {val_m[m.name]:.4f}"
                    for m in params.metrics
                ]
            )
            if params.metrics
            else ""
        )

        # 2. Handle "Infinite" epochs display
        total_epochs = params.epochs if params.epochs is not None else "NONE"

        # 3. Format the timing string
        # If duration is very small, we show ms; otherwise, seconds.
        time_str = ""
        if duration is not None:
            if duration < 0.001:
                time_str = f" | Epoch Elapsed: {duration * 1000:.3f} mSec"
            else:
                time_str = f" | Epoch Elapsed: {duration:.4f} Sec"

        # 4. Final print statement
        print(
            f"[{params.phase.upper()} | {epoch}/{total_epochs}] "
            f"Loss: T-{train_m['loss']:.4f} / V-{val_m['loss']:.4f} "
            f"{m_str}{time_str}"
        )

    def fit(self, x: torch.Tensor, y: torch.Tensor, training_params: TrainingParams):
        self.to(self.device)

        start_fit_time = time.time()

        optimizer = self._optimizer_creator(training_params)
        self._run_training_loop(
            x=x.to(self.device),
            y=y.to(self.device),
            optimizer=optimizer,
            loss_fn=training_params.loss_fn,
            params=training_params,
        )

        if self.history:
            self.history[-1].set_total_time(time.time() - start_fit_time)

    def set_layers_grad(
        self, layer_names: List[str] | Literal["all"], requires_grad: bool
    ):
        if layer_names == "all":
            for p in self.network.parameters():
                p.requires_grad = requires_grad
        else:
            for name in layer_names:
                found = False
                for p_name, param in self.named_parameters():
                    if p_name.startswith(f"network.{name}."):
                        param.requires_grad = requires_grad
                        found = True
                if not found:
                    raise ValueError(f"Layer {name} not found.")

    def freeze_layers(self, names="all"):
        self.set_layers_grad(names, False)

    def unfreeze_layers(self, names="all"):
        self.set_layers_grad(names, True)

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
        model_copy.seed_everything(model_copy.random_state)
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

    def _optimizer_creator(self, params: TrainingParams):
        return params.optimizer(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=params.lr,
            **(params.optimizer_params or {}),
        )
