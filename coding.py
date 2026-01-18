# =============================================================================
# BLOCK 0: IMPORT LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# Set a random seed for reproducible results
np.random.seed(42)

# =============================================================================
# BLOCK 1: LOAD AND PREPROCESS DATA
# =============================================================================
print("--- BLOCK 1: Loading and Preprocessing ---")
df = pd.read_csv('obesityProccessed.csv')

# Describe the Table
print("\n Descriptibe Statistics :")
print(df.describe().T)

# Detect and handle missing values
#print("\nMissing Values per Column\n")
#print(df.isnull().sum())

missing_count = df.isnull().sum().sum()
print(f"-> Missing values detected: {missing_count}")
if missing_count > 0:
    df.dropna(inplace=True)
    print("-> Rows with missing values have been dropped.")

# Map target 'NObeyesdad' to 0-6 range (since values are 0,1,3,4,5,6,7)
unique_targets = sorted(df['NObeyesdad'].unique())
target_mapping = {val: i for i, val in enumerate(unique_targets)}
inv_mapping = {i: val for val, i in target_mapping.items()}
df['NObeyesdad_Mapped'] = df['NObeyesdad'].map(target_mapping)

print(f"-> Target mapping: {target_mapping}")
print(f"-> Dataset shape: {df.shape}")

# =============================================================================
# BLOCK 2: DATA PREPARATION
# =============================================================================
X = df.drop(['NObeyesdad', 'NObeyesdad_Mapped'], axis=1).values
y = df['NObeyesdad_Mapped'].values

# Split the data (80% Train, 20% Test)
num_samples = X.shape[0]
indices = np.arange(num_samples)
np.random.shuffle(indices)
X, y = X[indices], y[indices]

split_point = int(0.8 * num_samples)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

# Scale the data
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
train_std[train_std == 0] = 1 # Prevent division by zero
X_train_scaled = (X_train - train_mean) / train_std
X_test_scaled = (X_test - train_mean) / train_std

# One-hot encode labels for the Neural Network
def to_one_hot(labels, num_classes):
    return np.eye(num_classes)[labels]

y_train_oh = to_one_hot(y_train, 7)

# =============================================================================
# BLOCK 3: NEURAL NETWORK IMPLEMENTATION (Multi-Class)
# =============================================================================
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # He initialization for better convergence
        self.weights1 = np.random.randn(input_size, hidden_size) * np.sqrt(2./input_size)
        self.biases1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * np.sqrt(2./hidden_size)
        self.biases2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.biases1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.biases2
        self.probs = self.softmax(self.z2)
        return self.probs

    def backward(self, X, y_oh, learning_rate):
        m = X.shape[0]
        # Backprop for Softmax + Cross-Entropy is simply (Probs - Labels)
        dz2 = self.probs - y_oh
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = np.dot(dz2, self.weights2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.weights1 -= learning_rate * dw1
        self.biases1 -= learning_rate * db1
        self.weights2 -= learning_rate * dw2
        self.biases2 -= learning_rate * db2

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

# =============================================================================
# BLOCK 4: TRAINING
# =============================================================================
nn = NeuralNetwork(input_size=X_train_scaled.shape[1], hidden_size=128, output_size=7)
epochs = 2000
learning_rate = 0.1

print("\n--- Starting Model Training ---")
for epoch in range(epochs):
    probs = nn.forward(X_train_scaled)
    nn.backward(X_train_scaled, y_train_oh, learning_rate)
    
    if epoch % 500 == 0:
        loss = -np.mean(np.sum(y_train_oh * np.log(probs + 1e-15), axis=1))
        acc = np.mean(np.argmax(probs, axis=1) == y_train)
        print(f"Epoch {epoch}: Loss {loss:.4f}, Accuracy {acc*100:.2f}%")

# =============================================================================
# BLOCK 5: EVALUATION
# =============================================================================
test_preds = nn.predict(X_test_scaled)
print(f"\nFinal Test Accuracy: {np.mean(test_preds == y_test) * 100:.2f}%")

# Convert back to original labels for the report
y_test_orig = [inv_mapping[i] for i in y_test]
test_preds_orig = [inv_mapping[i] for i in test_preds]

print("\nDetailed Classification Report:")
print(classification_report(y_test_orig, test_preds_orig))

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