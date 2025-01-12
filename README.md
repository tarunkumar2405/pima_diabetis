 Pima Diabetes Prediction Model

This repository contains a simple deep learning project to predict diabetes outcomes based on the Pima Indian Diabetes dataset. The project demonstrates the creation, training, and testing of a neural network model using Keras. Below is a brief description of the files included in the repository:

### Files in the Repository
1. **`train.py`**:
   - This script is used to train a neural network model to predict diabetes outcomes.
   - Key Steps:
     - Loads the Pima Diabetes dataset (`pima_diabetes.csv`).
     - Defines a Sequential neural network with three layers.
     - Compiles and trains the model using binary cross-entropy loss and the Adam optimizer.
     - Saves the trained model architecture to `model.json` and weights to `weights.weights.h5`.

2. **`test.py`**:
   - This script tests the trained model using the saved architecture and weights.
   - Key Steps:
     - Loads the Pima Diabetes dataset (`pima_diabetes.csv`).
     - Loads the saved model architecture (`model.json`) and weights (`weights.weights.h5`).
     - Makes predictions on the dataset and compares the predicted values with the actual outcomes.

3. **`weights.weights.h5`**:
   - Contains the saved weights of the trained neural network model.

4. **`model.json`**:
   - Contains the saved architecture of the trained neural network model in JSON format.

5. **`pima_diabetes.csv`**:
   - The dataset used for training and testing the model.
   - This dataset consists of 768 rows and 9 columns. The first 8 columns represent input features (e.g., glucose level, BMI, etc.), and the last column represents the binary outcome (1 for diabetic, 0 for non-diabetic).

### Key Features of the Model
- Input Features: 8 (e.g., glucose level, age, BMI, etc.).
- Output: Binary (1 for diabetic, 0 for non-diabetic).
- Architecture:
  - Layer 1: 12 neurons with ReLU activation.
  - Layer 2: 8 neurons with ReLU activation.
  - Layer 3: 1 neuron with Sigmoid activation.
- Optimizer: Adam.
- Loss Function: Binary Crossentropy.

### How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pima-diabetes-prediction.git
   ```
2. Ensure the required libraries (`numpy`, `keras`, etc.) are installed:
   ```bash
   pip install numpy keras
   ```
3. Train the model:
   ```bash
   python train.py
   ```
4. Test the model:
   ```bash
   python test.py
   ```
