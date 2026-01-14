# =============================================================================
# BLOCK 0: IMPORT LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# Set a random seed for reproducible results every time the code is run
np.random.seed(42)


# =============================================================================
# BLOCK 1: LOAD YOUR WEKA-PREPARED DATASET
# =============================================================================
print("--- BLOCK 1: Loading Your Weka-Prepared Dataset ---")

# Load your pre-processed CSV from Weka
df = pd.read_csv('ObesityProccessed.csv')  # Changed to your actual filename

print(f"-> Successfully loaded dataset.")
print(f"-> Dataset shape: {df.shape}")
print(f"-> Columns: {list(df.columns)}")
print(f"Target column: {df.columns[-1]}")

# Add this after loading your dataset
print("\n--- Data Analysis ---")
print(f"Dataset balance:")
print(df.iloc[:, -1].value_counts())
print(f"Imbalance ratio: {df.iloc[:, -1].value_counts().max() / df.iloc[:, -1].value_counts().min():.2f}")



\


# =============================================================================
# BLOCK 2: DATA PREPARATION
# =============================================================================
print("\n--- BLOCK 2: Preparing Data ---")

# --- Step 2.1: Prepare Data ---
print("\nStep 2.1: Preparing data for the model...")
# Assuming last column is the target/label
X = df.iloc[:, :-1].values  # All columns except last
y = df.iloc[:, -1].values  # target values 0-7

print(f"-> Features shape: {X.shape}")
print(f"-> Target shape: {y.shape}")

# --- Step 2.2: Split the data ---
print("\nStep 2.2: Splitting data into training and testing sets...")
num_samples = X.shape[0]
indices = np.arange(num_samples)
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

split_point = int(0.8 * num_samples)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

print(f"-> Training data shape: {X_train.shape}")
print(f"-> Testing data shape: {X_test.shape}")

# --- Step 2.3: Scale the data ---
print("\nStep 2.3: Scaling the data...")
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
train_std[train_std == 0] = 1
X_train_scaled = (X_train - train_mean) / train_std
X_test_scaled = (X_test - train_mean) / train_std
print("-> Data has been successfully split and scaled.")


# =============================================================================
# BLOCK 3: NEURAL NETWORK IMPLEMENTATION
# =============================================================================
class NeuralNetwork:
    """A from-scratch implementation of a two-layer neural network with ReLU."""
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.biases1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.biases2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        """Activation function for the output layer (for binary classification)."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid for backpropagation."""
        return x * (1 - x)

    def relu(self, x):
        """Activation function for the hidden layer."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU for backpropagation."""
        return (x > 0).astype(float)

    def forward(self, X):
        """Perform the forward pass."""
        self.z1 = np.dot(X, self.weights1) + self.biases1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.biases2
        return self.sigmoid(self.z2)

    def backward(self, X, y, output, learning_rate):
        """Perform backpropagation and update weights."""
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        hidden_error = np.dot(output_delta, self.weights2.T)
        hidden_delta = hidden_error * self.relu_derivative(self.a1)

        d_weights2 = np.dot(self.a1.T, output_delta)
        d_biases2 = np.sum(output_delta, axis=0, keepdims=True)
        d_weights1 = np.dot(X.T, hidden_delta)
        d_biases1 = np.sum(hidden_delta, axis=0, keepdims=True)

        self.weights1 += learning_rate * d_weights1
        self.biases1 += learning_rate * d_biases1
        self.weights2 += learning_rate * d_weights2
        self.biases2 += learning_rate * d_biases2

    def predict(self, X):
        """Make predictions on new data."""
        output = self.forward(X)
        return (output > 0.5).astype(int)


# =============================================================================
# BLOCK 4: NETWORK TRAINING
# =============================================================================
print("\n--- BLOCK 4: Starting Model Training ---")

# --- HYPERPARAMETERS ---
input_size = X_train_scaled.shape[1]
hidden_size = 128
output_size = 1
epochs = 5000
learning_rate = 0.001
max_error = 0.001

print(f"-> Input features: {input_size}")
print(f"-> Hidden neurons: {hidden_size}")
print(f"-> Training epochs: {epochs}")
print(f"-> Learning rate: {learning_rate}")

# --- Create and Train the Network ---
nn = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

# Lists to store metrics for plotting
history = {
    'train_loss': [], 'train_accuracy': [],
    'val_loss': [], 'val_accuracy': []
}

# Training loop
print(f"\n-> Starting training...")
for epoch in range(epochs):
    current_lr = learning_rate * (0.1 ** (epoch // 1000))
    train_output = nn.forward(X_train_scaled)  # FIXED: was X.shape, now X_train_scaled
    nn.backward(X_train_scaled, y_train, train_output, current_lr)  # FIXED: was X.shape, now X_train_scaled

    if (epoch + 1) % 100 == 0 or epoch == 0:
        train_loss = np.mean((y_train - train_output) ** 2)
        train_preds = (train_output > 0.5).astype(int)
        train_accuracy = np.mean(train_preds == y_train)
        val_output = nn.forward(X_test_scaled)
        val_loss = np.mean((y_test - val_output) ** 2)
        val_preds = (val_output > 0.5).astype(int)
        val_accuracy = np.mean(val_preds == y_test)
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

    if (epoch + 1) % 500 == 0:
        print(f"-> Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f} | Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}")

    if train_loss <= max_error:
        print(f"\n--- Stopping training early at Epoch {epoch+1} ---")
        print(f"-> Target maximum error of {max_error} was reached.")
        break

print("--- Training Finished ---")


# =============================================================================
# BLOCK 5: MODEL EVALUATION
# =============================================================================
print("\n--- BLOCK 5: Evaluating Model Performance on Test Data ---")
predictions = nn.predict(X_test_scaled)
accuracy = np.mean(predictions == y_test) * 100
print(f"-> Final Overall Test Accuracy: {accuracy:.2f}%")

predictions = nn.predict(X_test_scaled)





print("\n--- End of Report ---")



#head
#df.head()

#tail
#df.tail()

#shape
#df.shape

#info
#df.info()

#find missing values
#df.isnull().sum()

#find duplicated
#df.duplicated().sum()

#identify useless value
#for i in df.select_dtypes(include="object").columns:
 #   print(df[i].value_counts())
  #  print("***"*10)

#remove missing values
#df.isnull().sum()
#df.dropna(inplace=True)

#def wisker(col):
 ###   q1,q3=np.percentile(col,[30,70])
    #iqr=q3-q1
    #lw=q1-1.5*iqr
    #uw=q3+1.5*iqr
    #return lw,uw

#wisker(df['GDP'])
#df.columns