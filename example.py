import torch
import torch.nn as nn
from src.classification import ClassificationModel, TrainingPhaseType
from src.configurations import TrainingParams, EvaluationMetric
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

        # Required for export/import
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

lr = 0.001
epochs = 50

training_params = TrainingParams(
    epochs=epochs,
    lr=lr,
    batch_size="full",
    val_size=0.25,
    print_every=10,
    metrics=[accuracy_metric],
    loss_fn=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam,
    optimizer_params={"weight_decay": lr / epochs},
    training_step=supervised_step,
    phase=TrainingPhaseType.training,
)


model = SimpleClassifier(
    input_dim=input_dim,
    num_classes=num_classes,
    device=torch.device("cpu"),
)

model.fit(X, y, training_params)

model.visualize_training_history(title="Training history")

model.recover_best_model()


results = model.evaluate(X, y)
print("Final evaluation:", results)