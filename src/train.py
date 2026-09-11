import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from dataset import get_dataloaders
from model import MNISTModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "mnist_model.pth"
# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)


# Data
train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=64
)


# Model
model = MNISTModel().to(device)


# Loss
loss_function = nn.CrossEntropyLoss()


# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training
epochs = 5


train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(epochs):

    # -----------------------
    # Training
    # -----------------------

    model.train()

    total_train_loss = 0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_function(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        # Loss
        total_train_loss += loss.item()

        # Accuracy
        predictions = outputs.argmax(dim=1)

        total_train += labels.size(0)

        correct_train += (
            predictions == labels
        ).sum().item()


    # -----------------------
    # Training metrics
    # -----------------------

    train_loss = (
        total_train_loss / len(train_loader)
    )

    train_accuracy = (
        100 * correct_train / total_train
    )


    # -----------------------
    # Validation
    # -----------------------

    model.eval()

    total_val_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = loss_function(
                outputs,
                labels
            )

            # Loss
            total_val_loss += loss.item()

            # Accuracy
            predictions = outputs.argmax(dim=1)

            total_val += labels.size(0)

            correct_val += (
                predictions == labels
            ).sum().item()


    # -----------------------
    # Validation metrics
    # -----------------------

    val_loss = (
        total_val_loss / len(val_loader)
    )

    val_accuracy = (
        100 * correct_val / total_val
    )

    # Store metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)


    # -----------------------
    # Print metrics
    # -----------------------

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}%"
    )


# Save model
torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("Model saved!")

# -----------------------
# Plot metrics
# -----------------------

import matplotlib.pyplot as plt

epochs_range = range(1, epochs + 1)

# Loss plot
plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    train_losses,
    label="Training Loss"
)

plt.plot(
    epochs_range,
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()

plt.show()


# Accuracy plot
plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    train_accuracies,
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Epoch")
plt.legend()

plt.show()