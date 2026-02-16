import torch
import torch.nn as nn
from src.classification import ClassificationModel
from src.contracts.configurations import (
    TrainingPhaseType,
    EvaluationMetric,
)
from src.contracts.training_params import TrainingParams
from src.early_stopping import StoppingCriteria, EffectiveSet
from typing import OrderedDict
from sklearn.metrics import accuracy_score
from src.training_steps.supervised_training_step import SupervisedTrainingStep


class SimpleClassifier(ClassificationModel):
    def __init__(self, input_dim: int, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.network = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(input_dim, 32)),
                    ("relu1", nn.ReLU()),
                    ("linear2", nn.Linear(32, num_classes)),
                ]
            )
        )

        self.init_params = {
            "input_dim": input_dim,
            "num_classes": num_classes,
            **kwargs,
        }


torch.manual_seed(42)

N = 1000
input_dim = 10
num_classes = 3

X = torch.randn(N, input_dim)
y = torch.randint(0, num_classes, (N,))

accuracy_metric = EvaluationMetric(
    name="accuracy",
    function=accuracy_score,
)

loss_criteria = [
    StoppingCriteria(
        metric_name="loss",
        effective_set=EffectiveSet.VAL,
        mode="min",
        patience=5,
        message="EARLY STOPPING: Validation loss has not improved for 5 epochs.",
    )
]

lr = 0.001
total_epochs_estimate = 100

training_params = TrainingParams(
    epochs=None,
    lr=lr,
    batch_size="full",
    val_size=0.25,
    print_every=1,
    metrics=[accuracy_metric],
    loss_fn=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam,
    optimizer_params={"weight_decay": lr / total_epochs_estimate},
    training_step=SupervisedTrainingStep(),
    phase=TrainingPhaseType.training,
    stopping_criteria=loss_criteria,
)

model = SimpleClassifier(
    input_dim=input_dim,
    num_classes=num_classes,
    device=torch.device("cpu"),
)


model.fit(X, y, training_params)

print(model.history[-1].termination)

model.visualize_training_history(title="Training history", show_or_export="both")

model.recover_best_model()

results = model.evaluate(X, y, training_step=SupervisedTrainingStep())
print("Final evaluation:", results)

print(
    f"Total time: {model.history[-1].total_time} sec",
    "\n",
    f"Average time per epoch: {model.history[-1].average_epoch_time} sec",
)
