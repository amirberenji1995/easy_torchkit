import torch
import torch.nn as nn
from src.classification import ClassificationModel
from src.configurations import TrainingParams, EvaluationMetric, TrainingPhaseType

class SimpleClassifier(ClassificationModel):
    def __init__(self, input_dim: int, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

        # Required for export/import
        self.init_params = {
            "input_dim": input_dim,
            "num_classes": num_classes,
            **kwargs,
        }

    def forward(self, x):
        return self.net(x)
    
torch.manual_seed(42)

N = 1000
input_dim = 10
num_classes = 3

X = torch.randn(N, input_dim)
y = torch.randint(0, num_classes, (N,))



accuracy_metric = EvaluationMetric(
    name="accuracy",
    function=lambda y_true, y_pred: (y_true == y_pred).mean(),
)

training_params = TrainingParams(
    epochs=50,
    lr=0.001,
    batch_size="full",
    val_size=0.25,
    print_every=10,
    optimizer=torch.optim.Adam,
    optimizer_params={"weight_decay": 0.001, "betas": (0.9, 0.999)}
)


model = SimpleClassifier(
    input_dim=input_dim,
    num_classes=num_classes,
    device=torch.device("cpu"),
)

model.loss_fn = torch.nn.CrossEntropyLoss()

model.fit(X, y, training_params)

model.recover_best_model()


results = model.evaluate(X, y)
print("Final evaluation:", results)

preds = model.predict(X[:5])
print("Sample predictions:", preds)


model.visualize_training_history()

model.evaluate(X, y)