# =============================================================================
# BLOCK 0: IMPORT LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# =============================================================================
# BLOCK 1: LOAD DATA
# =============================================================================
np.random.seed(42)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
df = pd.read_csv('ShillBidding.csv') 

print("Summary Statistics:")
print(df.iloc[:, :7].describe())
print("\n")
print(df.iloc[:, 7:].describe())


# Features & target
X = df.drop('Class', axis=1)
y = df['Class']  

# Check for missing values
missingX = X.isnull().sum()
print("\nMissing values per feature:")
print(missingX)

missingY = y.isnull().sum()
print("\nMissing values in target:")
print(missingY)



# Convert y to numpy only after checking
y = y.values.reshape(-1,1)


# Unique values for each feature
unique_table = pd.DataFrame({
    "Feature": X.columns,
    "Total Unique Values": [X[col].nunique() for col in X.columns]
})

print("\n=== UNIQUE VALUES TABLE ===")
print(unique_table)

# Feature with highest unique values
highest_unique_feature = unique_table.sort_values(
    by="Total Unique Values", ascending=False
).iloc[0]

print("\n=== FEATURE WITH HIGHEST UNIQUE VALUES ===")
print(highest_unique_feature)

# Unique values in target class
target_unique_count = pd.Series(y.flatten()).nunique()

print("\n=== TOTAL UNIQUE VALUES IN TARGET CLASS ===")
print(f"Target Class -> Total Unique Values: {target_unique_count}")




# Convert categorical columns to numeric
categorical_cols = X.select_dtypes(include='object').columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

X = X.values  # convert to numpy array

print("\nPreprocessed successfully.")



# Split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
print(f"\nTraining samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Scale numeric features
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
train_std[train_std==0] = 1
X_train_scaled = (X_train - train_mean)/train_std
X_test_scaled = (X_test - train_mean)/train_std




# =============================================================================
# BLOCK 2: NEURAL NETWORK (Binary) BPNN
# =============================================================================

# i) Relu Hidden + Sigmoid Output (first method to test)
class BPNN:
    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate, epochs):
        # STEP 0: Initialize weights (small random values)
        self.lr = learning_rate
        self.epochs = epochs
        self.w1 = np.random.randn(n_inputs, n_hidden) * np.sqrt(2 / n_inputs)
        self.b1 = np.zeros((1, n_hidden))
        self.w2 = np.random.randn(n_hidden, n_outputs) * np.sqrt(1 / n_hidden)
        self.b2 = np.zeros((1, n_outputs))
        
        self.encoder = OneHotEncoder(sparse_output=False)

    # Activation functions
    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, a):
        return 1 - a**2  
    
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Training method
    def train(self, X, y):
        ys = self.encoder.fit_transform(y.reshape(-1,1))
        m = X.shape[0]

        for epoch in range(self.epochs):

            # STEP 1: Calculate hidden layer activations
            z1 = np.dot(X, self.w1) + self.b1
            a1 = self.relu(z1) 

            # STEP 2: Calculate output layer activations
            z2 = np.dot(a1, self.w2) + self.b2
            a2 = self.sigmoid(z2) 

            # STEP 3: Calculate error
            error = ys - a2
            mse = np.mean(error ** 2)
            
            # STEP 4: Calculate output layer error
            d2 = error * self.sigmoid_derivative(a2)
            
            # STEP 5: Calculate hidden layer error
            d1 = np.dot(d2, self.w2.T) * self.relu_derivative(a1)

            # STEP 6: Update output layer weights
            self.w2 += np.dot(a1.T, d2) * self.lr / m
            self.b2 += np.sum(d2, axis=0, keepdims=True) * self.lr / m

            # STEP 7: Update hidden layer weights
            self.w1 += np.dot(X.T, d1) * self.lr / m
            self.b1 += np.sum(d1, axis=0, keepdims=True) * self.lr / m


            if epoch % 200 == 0:
                preds = np.argmax(a2, axis=1)
                actuals = np.argmax(ys, axis=1)
                acc = np.mean(preds == actuals) * 100
                print(f"Epoch {epoch}, Mean Squared Error: {mse:.4f}, Accuracy: {acc:.2f}%")

    # Predict method
    def predictBPNN(self, X):
        a1 = self.relu(np.dot(X, self.w1) + self.b1)
        a2 = self.sigmoid(np.dot(a1, self.w2) + self.b2)
        return np.argmax(a2, axis=1)

    # Accuracy
    def accuracyBPNN(self, X, y):
        return np.mean(self.predictBPNN(X) == y.flatten()) * 100
    
    def saveFile(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def loadFile(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)


# =============================================================================
# BLOCK 3: TRAINING
# =============================================================================
input_size = X_train_scaled.shape[1]
hidden_size = 256
output_size = np.unique(y).shape[0]
epochs = 2000
lr = 0.2


print("Neural Network Structure:")
print(f"Input Size: {input_size}")
print(f"Hidden Size: {hidden_size}")
print(f"Output Size: {output_size}\n")

print("Training Settings :")
print(f"Hidden Size:", hidden_size)
print(f"Epochs:", epochs)
print(f"Learning Rate:", lr, "\n")


#This will go back to BLOCK 2 (BPNN & Epochs)
nn = BPNN(n_inputs=input_size, n_hidden=hidden_size, n_outputs=output_size,
          learning_rate=lr, epochs=epochs)


nn.train(X_train_scaled, y_train)


# =============================================================================
# BLOCK 4: EVALUATION
# =============================================================================
# Evaluate on training data


# Evaluate
train_acc = nn.accuracyBPNN(X_train_scaled, y_train)
test_preds = nn.predictBPNN(X_test_scaled)
test_acc = np.mean(test_preds == y_test.flatten()) * 100
print("\n=== FINAL RESULTS ===")
print(f"Training Accuracy: {train_acc:.2f}%") 
print(f"Test Accuracy: {test_acc:.2f}%")

# Save the trained neural network
nn.saveFile("nn_shillbidding.pkl")
print("Model saved successfully!")

# =============================================================================
# BLOCK 5: VISUALIZATION (Confusion Matrix)
# =============================================================================

y_pred = nn.predictBPNN(X_test_scaled)

cm = confusion_matrix(y_test.flatten(), y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Shill Bidding'], 
            yticklabels=['Normal', 'Shill Bidding'])

plt.title('Confusion Matrix - Shill Bidding Detection')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test.flatten(), y_pred))



