import torch
import torch.nn as nn
from src.classification import ClassificationModel
from src.configurations import TrainingPhaseType, TrainingParams, EvaluationMetric
from src.early_stopping import StoppingCriteria, EffectiveSet  # Added imports
from typing import OrderedDict
from sklearn.metrics import accuracy_score
from src.utils import supervised_step


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

# Define the criteria for 5 epochs of no improvement in validation loss
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
# We use a dummy epoch value for weight decay calculation since training is dynamic
total_epochs_estimate = 100

training_params = TrainingParams(
    epochs=None,  # Training will run until criteria is met
    lr=lr,
    batch_size="full",
    val_size=0.25,
    print_every=1,  # Set to 1 to see the early stopping more clearly
    metrics=[accuracy_metric],
    loss_fn=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam,
    optimizer_params={"weight_decay": lr / total_epochs_estimate},
    training_step=supervised_step,
    phase=TrainingPhaseType.training,
    stopping_criteria=loss_criteria,  # Pass criteria into TrainingParams
)

model = SimpleClassifier(
    input_dim=input_dim,
    num_classes=num_classes,
    device=torch.device("cpu"),
)

# This will now monitor validation loss and trigger the 5-epoch patience rule
model.fit(X, y, training_params)

print(model.history[-1].termination)

model.visualize_training_history(title="Training history", show_or_export="both")

model.recover_best_model()

results = model.evaluate(X, y)
print("Final evaluation:", results)

print(
    f"Total time: {model.history[-1].total_time} sec",
    "\n",
    f"Average time per epoch: {model.history[-1].average_epoch_time} sec",
)
