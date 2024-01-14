import numpy as np
import tensorflow as tf
from sklearn.tree import DecisionTreeRegressor

# Generate some synthetic data
#X = np.random.rand(1000, 1)  # 1000 samples, 1 feature
#y = 4 * X.squeeze() + 3 + np.random.randn(1000)  # Linear relation with some noise

# Initialize and fit a basic decision tree
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y
# Here we're simplifying by assuming that each sample in X ends up in one leaf.
leaf_indices = tree.apply(X)

# Create and train neural networks for each leaf
leaf_to_nn = {}
for leaf in np.unique(leaf_indices):
    # Extract samples that end up in this leaf
    X_leaf = X[leaf_indices == leaf]
    y_leaf = y[leaf_indices == leaf]
    # Define a simple neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.fit(X_leaf, y_leaf, epochs=10, verbose=0)
    # Store the trained model
    leaf_to_nn[leaf] = model

# Function to make predictions
def hybrid_predict(X):
    leaf_indices = tree.apply(X)
    predictions = np.zeros(X.shape[0]) 
    for leaf in np.unique(leaf_indices):
        X_leaf = X[leaf_indices == leaf]
        nn = leaf_to_nn[leaf]
        predictions[leaf_indices == leaf] = nn.predict(X_leaf).squeeze()
    return predictions

# Test the hybrid model
X_test = np.random.rand(200, 1)
y_test = 4 * X_test.squeeze() + 3 + np.random.randn(200)
y_pred = hybrid_predict(X_test)

# Compute MSE for evaluation
mse = np.mean((y_test - y_pred)**2)
print(f"Mean Squared Error on test data: {mse}")
