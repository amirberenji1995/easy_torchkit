from enum import StrEnum
from pydantic import BaseModel
from typing import Literal, Optional


class EffectiveSet(StrEnum):
    TRAIN = "train"
    VAL = "val"


class StoppingCriteria(BaseModel):
    metric_name: str = "loss"
    effective_set: EffectiveSet = EffectiveSet.VAL
    mode: Literal["min", "max"] = "min"
    patience: int = 50
    min_epoch: int = 0
    target_value: Optional[float] = None


class EarlyStoppingHandler:
    def __init__(self, criteria_list: list[StoppingCriteria]):
        # Store criteria (could be an empty list)
        self.criteria = criteria_list or []

        # Initialize tracking state only for provided criteria
        self.state = []
        for c in self.criteria:
            init_val = float("inf") if c.mode == "min" else float("-inf")
            self.state.append({"best_val": init_val, "patience": 0})

    def check(self, epoch: int, train_metrics: dict, val_metrics: dict) -> bool:
        """
        Returns True if any stopping criteria is met.
        If criteria_list is empty, this always returns False (no early stopping).
        """
        if not self.criteria:
            return False

        for i, c in enumerate(self.criteria):
            # Select the correct set (Train or Val)
            metrics = (
                train_metrics if c.effective_set == EffectiveSet.TRAIN else val_metrics
            )
            current_val = metrics.get(c.metric_name)

            if current_val is None:
                continue

            # Strategy 1: Target Value reached
            if c.target_value is not None:
                reached = (
                    current_val >= c.target_value
                    if c.mode == "max"
                    else current_val <= c.target_value
                )
                if reached and epoch >= c.min_epoch:
                    print(f"🎯 Target {c.target_value} reached for {c.metric_name}")
                    return True

            # Strategy 2: Improvement/Patience
            improved = (
                current_val < self.state[i]["best_val"]
                if c.mode == "min"
                else current_val > self.state[i]["best_val"]
            )

            if improved:
                self.state[i]["best_val"] = current_val
                self.state[i]["patience"] = 0
            else:
                self.state[i]["patience"] += 1

            if self.state[i]["patience"] >= c.patience and epoch >= c.min_epoch:
                print(f"⏹ Patience exceeded for {c.metric_name} ({c.effective_set})")
                return True

        return False
