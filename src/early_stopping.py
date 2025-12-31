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
        self.criteria = criteria_list or []
        self.state = []
        for c in self.criteria:
            init_val = float("inf") if c.mode == "min" else float("-inf")
            self.state.append(
                {
                    "best_val": init_val,
                    "patience": 0,
                    "target_hold_count": 0,  # New: Tracks consecutive epochs at target
                }
            )

    def check(self, epoch: int, train_metrics: dict, val_metrics: dict) -> bool:
        if not self.criteria:
            return False

        for i, c in enumerate(self.criteria):
            metrics = (
                train_metrics if c.effective_set == EffectiveSet.TRAIN else val_metrics
            )
            current_val = metrics.get(c.metric_name)

            if current_val is None:
                continue

            # --- New Strategy: Target Value with Hold Duration ---
            if c.target_value is not None:
                at_target = (
                    current_val >= c.target_value
                    if c.mode == "max"
                    else current_val <= c.target_value
                )

                if at_target:
                    self.state[i]["target_hold_count"] += 1
                else:
                    self.state[i]["target_hold_count"] = (
                        0  # Reset if it drops below target
                    )

                # Stop only if target is held for 'patience' duration AND min_epoch is passed
                if (
                    self.state[i]["target_hold_count"] >= c.patience
                    and epoch >= c.min_epoch
                ):
                    print(
                        f"✅ Target {c.target_value} held for {c.patience} epochs.",
                        "\n",
                        f"Epoch: {epoch}",
                        "\n",
                        f"Val {c.metric_name}: {current_val}",
                    )
                    return True

                # If we are using target_value logic, we skip the standard patience check
                # for this specific metric to avoid double-triggering.
                continue

            # --- Standard Strategy: Improvement/Patience ---
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
                print(f"⏹ Patience exceeded for {c.metric_name}")
                return True

        return False
