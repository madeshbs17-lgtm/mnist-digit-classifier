# MNIST Digit Classifier

A deep learning project that classifies handwritten digits from the MNIST dataset using **PyTorch**. The project implements a complete training, validation, and testing pipeline using a custom fully connected neural network.

The model takes a `28 × 28` grayscale handwritten digit image as input and predicts one of the ten digit classes from `0` to `9`.

---

## 📌 Project Overview

Handwritten digit classification is a fundamental computer vision problem and is commonly used to understand the basics of neural networks and deep learning.

In this project, I built a complete MNIST classification pipeline using PyTorch.

The project covers:

- Dataset downloading and preprocessing
- Training and validation dataset splitting
- PyTorch Dataset and DataLoader usage
- Neural network architecture design
- Forward propagation
- Loss calculation
- Backpropagation
- Model optimization using Adam
- Training and validation evaluation
- Model saving
- Model loading
- Test set evaluation
- Training and validation performance visualization
- CPU/GPU device selection

---

## 🧠 How the Model Works

The MNIST dataset contains grayscale images of handwritten digits.

Each image has a resolution of:

text
28 × 28 pixels

Before being passed to the neural network, the image is converted into a tensor.

The 28 × 28 image is then flattened into:

28 × 28 = 784

input features.

The neural network processes these features through a hidden layer and produces predictions for the 10 possible digit classes.

Model Architecture
              MNIST Image
               28 × 28
                  │
                  ▼
              Flatten
                  │
                  ▼
             784 Features
                  │
                  ▼
          Fully Connected Layer
              784 → 128
                  │
                  ▼
                ReLU
                  │
                  ▼
          Fully Connected Layer
              128 → 10
                  │
                  ▼
          Digit Prediction
             0 - 9

The model is implemented using PyTorch's nn.Sequential.

📊 Dataset

This project uses the MNIST handwritten digit dataset.

The dataset contains grayscale images representing handwritten digits from 0 to 9.

Image Properties
Property	Value
Image Size	28 × 28
Image Type	Grayscale
Number of Classes	10
Classes	0–9
Input Features	784

The dataset is downloaded automatically using torchvision.

There is no need to manually download the MNIST dataset.

🔀 Dataset Splitting

The original MNIST training dataset is divided into:

MNIST Training Dataset
        │
        ├───────────────┐
        │               │
        ▼               ▼
     90% Train       10% Validation

A separate MNIST test dataset is used for the final evaluation.

Therefore, the project uses three datasets:

Training dataset
Validation dataset
Test dataset

The training and validation datasets are created using random_split.

⚙️ Data Loading

The project uses PyTorch DataLoader to load the images in batches.

The batch size used is:

64

The training DataLoader uses shuffling, while the validation and test DataLoaders do not shuffle the data.

The images are converted to tensors using:

transforms.ToTensor()
🏗️ Model Architecture

The neural network is defined in model.py.

The architecture consists of:

1. Flatten Layer

The 28 × 28 image is converted into a one-dimensional vector containing 784 values.

28 × 28 → 784
2. Fully Connected Layer

The first linear layer maps:

784 → 128

This layer learns useful representations from the input pixels.

3. ReLU Activation

A ReLU activation function is applied after the first linear layer.

ReLU(x) = max(0, x)
4. Output Layer

The final linear layer maps:

128 → 10

The ten outputs correspond to the ten possible digit classes.

🎯 Loss Function

The project uses:

nn.CrossEntropyLoss()

Cross Entropy Loss is used to measure the difference between the model's predicted class scores and the actual digit labels.

The loss is calculated during both training and validation.

🚀 Optimizer

The model is trained using the Adam optimizer.

Configuration:

Optimizer: Adam
Learning Rate: 0.001

Adam updates the model parameters during training based on the gradients calculated through backpropagation.

💻 Device Selection

The project automatically checks whether CUDA is available.

torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

If a CUDA-compatible GPU is available, the model runs on the GPU.

Otherwise, the project automatically runs on the CPU.

Example:

Using: cpu
🔄 Training Process

The training process runs for:

5 Epochs

For every epoch, the project performs the following steps:

Load Batch
    ↓
Move Data to Device
    ↓
Forward Pass
    ↓
Calculate Loss
    ↓
Backpropagation
    ↓
Optimizer Step
    ↓
Calculate Training Metrics
    ↓
Validation
    ↓
Calculate Validation Metrics

During training, the project tracks:

Training Loss
Training Accuracy
Validation Loss
Validation Accuracy

📈 Training Results

The model was trained for 5 epochs.

Epoch	Train Loss	Train Accuracy	Validation Loss	Validation Accuracy
1	0.3630	90.16%	0.2112	94.17%
2	0.1643	95.24%	0.1620	95.25%
3	0.1146	96.72%	0.1272	96.38%
4	0.0882	97.37%	0.1063	96.95%
5	0.0698	97.94%	0.1010	97.13%

The training and validation metrics show that the model's performance improves across the five epochs.

🧪 Test Performance

After training, the saved model was evaluated on the separate MNIST test dataset.

Test Accuracy
97.38%

The evaluation script produced:

Using: cpu
Test Accuracy: 97.38%

This represents the final performance of the trained model on the MNIST test dataset.

📉 Training Visualization

The training script also tracks the loss and accuracy values across epochs.

Two plots are generated during training:

Loss vs Epoch

The plot compares:

Training Loss
Validation Loss
Accuracy vs Epoch

The plot compares:

Training Accuracy
Validation Accuracy

These visualizations make it possible to observe how the model learns throughout training.

💾 Model Saving

After training, the model parameters are saved using PyTorch:

torch.save(
    model.state_dict(),
    MODEL_PATH
)

The saved model is stored at:

models/mnist_model.pth

The model can then be loaded later for evaluation without retraining it.

🔍 Model Evaluation

The evaluate.py script loads the saved model:

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

The model is then switched to evaluation mode:

model.eval()

Predictions are generated using:

predictions = outputs.argmax(dim=1)

The predicted labels are compared with the actual labels to calculate the final test accuracy.

📁 Project Structure
mnist-digit-classifier/
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   └── mnist_model.pth
│
├── data/
│
├── requirements.txt
└── README.md
src/dataset.py

Responsible for:

Downloading the MNIST dataset
Applying ToTensor() transformation
Splitting the training dataset
Creating DataLoaders
Returning training, validation, and test DataLoaders
src/model.py

Contains the neural network architecture.

784 → 128 → 10

with a ReLU activation between the linear layers.

src/train.py

Responsible for:

Selecting CPU/GPU
Loading the datasets
Creating the model
Defining the loss function
Defining the optimizer
Training the model
Calculating training metrics
Calculating validation metrics
Displaying training plots
Saving the trained model
src/evaluate.py

Responsible for:

Loading the saved model
Loading the test dataset
Running inference
Calculating test accuracy
🛠️ Technologies Used
Technology	Purpose
Python	Programming language
PyTorch	Deep learning framework
Torchvision	MNIST dataset and transformations
NumPy	Numerical operations
Matplotlib	Training visualization
📦 Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/mnist-digit-classifier.git

Navigate into the project directory:

cd mnist-digit-classifier

Install the required dependencies:

pip install -r requirements.txt
▶️ Train the Model

Run the training script:

python src/train.py

The script will:

Download the MNIST dataset if it is not already available.
Create the training, validation, and test DataLoaders.
Initialize the neural network.
Train the model for 5 epochs.
Calculate training and validation metrics.
Display loss and accuracy plots.
Save the trained model.

The trained model will be saved to:

models/mnist_model.pth
🧪 Evaluate the Model

After training, run:

python src/evaluate.py

Example output:

Using: cpu
Test Accuracy: 97.38%
📋 Requirements

The project dependencies are listed in requirements.txt.

The main libraries used are:

torch
torchvision
numpy
matplotlib
🎓 What I Learned

Through this project, I gained practical experience with:

Building neural networks using PyTorch
Working with the MNIST dataset
Creating PyTorch DataLoaders
Splitting datasets into training and validation sets
Implementing training loops
Implementing validation loops
Forward propagation
Backpropagation
Cross Entropy Loss
Adam optimization
Tracking model performance
Saving and loading PyTorch models
Evaluating models on unseen test data
Using CPU/GPU device selection
Visualizing training and validation performance
📌 Results Summary
Model: Fully Connected Neural Network
Framework: PyTorch
Training Epochs: 5
Batch Size: 64
Optimizer: Adam
Learning Rate: 0.001

Final Training Accuracy:     97.94%
Final Validation Accuracy:   97.13%
Test Accuracy:               97.38%
👨‍💻 Author

Madesh

GitHub: madeshbs17-lgtm



