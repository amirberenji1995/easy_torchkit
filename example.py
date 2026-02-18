import torch
import torch.nn as nn
from collections import OrderedDict
from sklearn.metrics import accuracy_score

from src.classification import ClassificationModel
from src.contracts.configurations import TrainingPhaseType, EvaluationMetric
from src.contracts.training_params import TrainingParams
from src.early_stopping import StoppingCriteria, EffectiveSet
from src.training_steps.supervised_training_step import SupervisedTrainingStep
from src.training_steps.siamese_training_step import (
    SiameseTrainingStep,
    ContrastiveLoss,
)


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


accuracy_metric = EvaluationMetric(name="accuracy", function=accuracy_score)

loss_criteria = [
    StoppingCriteria(
        metric_name="loss",
        effective_set=EffectiveSet.VAL,
        mode="min",
        patience=5,
        message="EARLY STOPPING: Validation loss has not improved for 5 epochs.",
    ),
    # StoppingCriteria(
    #     metric_name="total_time",
    #     target_value=0.25,
    #     effective_set=None,
    #     mode="max",
    #     patience=1,
    #     message="EARLY STOPPING: Total training time exceeded 0.25 seconds.",
    # ),
    # StoppingCriteria(
    #     metric_name="epoch_time",
    #     target_value=0.05,
    #     effective_set=None,
    #     mode="max",
    #     patience=1,
    #     message="EARLY STOPPING: Epoch training time exceeded 0.05 seconds.",
    # ),
]

lr = 0.001
total_epochs_estimate = 100

training_params = TrainingParams(
    epochs=10,
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

# Initialize model
model = SimpleClassifier(
    input_dim=input_dim, num_classes=num_classes, device=torch.device("cpu")
)

# Supervised training
print("=== Supervised training ===")
model.fit(X, y, training_params)
print("Termination info:", model.history[-1].termination)
model.visualize_training_history(
    title="Supervised Training History", show_or_export="both"
)
model.recover_best_model()

# Evaluate
results = model.evaluate(x=X, y=y, training_step=SupervisedTrainingStep())
print("Final supervised evaluation:", results)


pairs = torch.randn(N, 2, input_dim)
labels = torch.randint(0, 2, (N,))

print("\n=== Contrastive Fine-Tuning ===")

contrastive_params = TrainingParams(
    epochs=5,
    lr=0.001,
    batch_size="full",
    val_size=0.25,
    print_every=1,
    metrics=[],
    loss_fn=ContrastiveLoss(margin=1.0),
    optimizer=torch.optim.Adam,
    optimizer_params={"weight_decay": 1e-4},
    training_step=SiameseTrainingStep(),
    phase=TrainingPhaseType.fine_tuning,
    stopping_criteria=[],
)

# Fine-tune with Siamese contrastive loss
model.fit(pairs, labels, contrastive_params)

# Evaluate agian
print(
    "Contrastive embedding evaluation:",
    model.evaluate(
        X,
        y,
    ),
)
