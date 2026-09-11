import torch
from pathlib import Path

from dataset import get_dataloaders
from model import MNISTModel


# -----------------------
# Paths
# -----------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "mnist_model.pth"


# -----------------------
# Device
# -----------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)


# -----------------------
# Data
# -----------------------

_, _, test_loader = get_dataloaders(
    batch_size=64
)


# -----------------------
# Model
# -----------------------

model = MNISTModel().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# -----------------------
# Evaluation
# -----------------------

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()


accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")