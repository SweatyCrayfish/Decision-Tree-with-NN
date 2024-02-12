import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

# Generate a more complex dummy dataset
num_samples = 10000
num_features = 10
X = np.random.rand(num_samples, num_features)
weights = np.random.rand(num_features, 1)
y = X.dot(weights).squeeze() + np.random.randn(num_samples) * 0.5  # Adding some noise
data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(num_features)])
data['target'] = y

# Save the DataFrame to a CSV file
data.to_csv('generated_complex_dataset.csv', index=False)
start_time_whole_alg = time.time()
# Split the data into training, validation, and test sets
X = data.drop('target', axis=1).values
y = data['target'].values
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train XGBoost model
params = {'max_depth': 3, 'objective': 'reg:squarederror'}
dtrain = xgb.DMatrix(X_train, label=y_train)

start_time = time.time()  # Start timing
bst = xgb.train(params, dtrain, num_boost_round=5)
end_time = time.time()  # End timing
print(f"XGBoost Training Time: {end_time - start_time:.2f} seconds")

# Extract leaf indices
leaf_indices = bst.predict(dtrain, pred_leaf=True)

# Initialize storage for neural networks and performance metrics
leaf_to_nn = {}
training_maes, validation_maes, validation_mses, training_mses = [], [], [], []

for i, tree_leaf_indices in enumerate(leaf_indices.T):
    for leaf in np.unique(tree_leaf_indices):
        X_leaf_train = X_train[tree_leaf_indices == leaf]
        y_leaf_train = y_train[tree_leaf_indices == leaf]
        
        dval = xgb.DMatrix(X_val, label=y_val)
        val_leaf_indices = bst.predict(dval, pred_leaf=True)
        X_leaf_val = X_val[val_leaf_indices[:, i] == leaf]
        y_leaf_val = y_val[val_leaf_indices[:, i] == leaf]
        
        if len(X_leaf_train) > 0 and len(X_leaf_val) > 0:
            # Enhanced neural network model
            model = tf.keras.Sequential([
                Dense(128, activation='gelu', input_shape=(num_features,), kernel_regularizer=l2(0.001)),
                BatchNormalization(),
                Dropout(0.2),
                Dense(64, activation='gelu', kernel_regularizer=l2(0.001)),
                BatchNormalization(),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
            history = model.fit(X_leaf_train, y_leaf_train, epochs=50, verbose=0, validation_data=(X_leaf_val, y_leaf_val))
            
            leaf_to_nn[(i, leaf)] = model
            training_maes.append(history.history['mae'][-1])
            training_mses.append(history.history['loss'][-1])
            validation_maes.append(history.history['val_mae'][-1])
            validation_mses.append(history.history['val_loss'][-1])

def hybrid_predict(X):
    dtest = xgb.DMatrix(X)
    leaf_indices = bst.predict(dtest, pred_leaf=True)
    predictions = np.zeros(X.shape[0])
    for i, tree_leaf_indices in enumerate(leaf_indices.T):
        for leaf in np.unique(tree_leaf_indices):
            X_leaf = X[tree_leaf_indices == leaf]
            nn = leaf_to_nn.get((i, leaf))
            if nn:
                predictions[tree_leaf_indices == leaf] += nn.predict(X_leaf).squeeze()
    return predictions / leaf_indices.shape[1]
end_time_whole_alg = time.time()
print(f"Hybrid Model Full Training Time: {end_time_whole_alg - start_time_whole_alg:.2f} seconds")
start_time = time.time()
y_pred = hybrid_predict(X_test)
end_time = time.time()
print(f"Hybrid Model Prediction Time: {end_time - start_time:.2f} seconds")

test_mae = mean_absolute_error(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)

print(f"Training Mean Absolute Error (MAE): {np.mean(training_maes)}")
print(f"Test Mean Absolute Error (MAE): {test_mae}")
print(f"Validation Mean Absolute Error (MAE): {np.mean(validation_maes)}")
print(f"Training Mean Squared Error (MSE): {np.mean(training_mses)}") 
print(f"Test Mean Squared Error (MSE): {test_mse}")
print(f"Validation Mean Squared Error (MSE): {np.mean(validation_mses)}")
