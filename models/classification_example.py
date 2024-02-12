import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa
import time

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# One-hot encode the target labels
y_train_encoded = to_categorical(y_train, num_classes=3)
y_val_encoded = to_categorical(y_val, num_classes=3)
y_test_encoded = to_categorical(y_test, num_classes=3)
# Train XGBoost model
num_features = X_train.shape[1]
params = {'max_depth': 3}
dtrain = xgb.DMatrix(X_train, label=y_train)
start_time = time.time()
bst = xgb.train(params, dtrain, num_boost_round=5)
end_time = time.time() 
print(f"XGBoost Training Time: {end_time - start_time:.2f} seconds")
# Extract leaf indices
leaf_indices = bst.predict(dtrain, pred_leaf=True)

# Initialize storage for neural networks and performance metrics
leaf_to_nn = {}
training_maes, validation_maes, validation_mses, training_mses = [], [], [], []

for i, tree_leaf_indices in enumerate(leaf_indices.T):
    for leaf in np.unique(tree_leaf_indices):
        # Prepare data for each leaf
        X_leaf_train = X_train[tree_leaf_indices == leaf]
        y_leaf_train = y_train_encoded[tree_leaf_indices == leaf]
        # Extract validation data for the leaf
        dval = xgb.DMatrix(X_val, label=y_val)
        val_leaf_indices = bst.predict(dval, pred_leaf=True)
        X_leaf_val = X_val[val_leaf_indices[:, i] == leaf]
        y_leaf_val = y_val_encoded[val_leaf_indices[:, i] == leaf]        
        if len(X_leaf_train) > 0 and len(X_leaf_val) > 0:
            # Train neural network for each leaf
            model = tf.keras.Sequential([
                Dense(128, activation='gelu', input_shape=(num_features,), kernel_regularizer=l2(0.001)),
                BatchNormalization(),
                Dropout(0.2),
                Dense(64, activation='gelu', kernel_regularizer=l2(0.001)),
                BatchNormalization(),
                Dropout(0.2),
                Dense(3, activation='softmax')
            ])
            model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.001),
                          loss='categorical_crossentropy', 
                          metrics=['accuracy'])
            history = model.fit(X_leaf_train, y_leaf_train, epochs=50, verbose=0, validation_data=(X_leaf_val, y_leaf_val))            
            # Store trained parameters
            leaf_to_nn[(i, leaf)] = model
            training_maes.append(history.history['accuracy'][-1])  
            training_mses.append(history.history['loss'][-1])
            validation_maes.append(history.history['val_accuracy'][-1])  
            validation_mses.append(history.history['val_loss'][-1])

def hybrid_predict(X):
    dtest = xgb.DMatrix(X)
    leaf_indices = bst.predict(dtest, pred_leaf=True)
    predictions = np.zeros((X.shape[0], 3)) 
    for i, tree_leaf_indices in enumerate(leaf_indices.T):
        for leaf in np.unique(tree_leaf_indices):
            X_leaf = X[tree_leaf_indices == leaf]  
            nn = leaf_to_nn.get((i, leaf))
            if nn:
                class_predictions = nn.predict(X_leaf) 
                # Ensure accumulation only for samples in the current leaf
                predictions[np.where(tree_leaf_indices == leaf), :] += class_predictions
    return np.argmax(predictions, axis=1)

# Evaluate hybrid model
start_time = time.time()
y_pred = hybrid_predict(X_test) 
end_time = time.time()
print(f"Hybrid Model Prediction Time: {end_time - start_time:.2f} seconds")

# Calculate and print performance metrics
test_mae = mean_absolute_error(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)

# Get predicted labels
y_train_pred = hybrid_predict(X_train)
y_test_pred = hybrid_predict(X_test)

# Calculate classification metrics
train_report = classification_report(y_train, y_train_pred, target_names=iris.target_names, output_dict=True)
test_report = classification_report(y_test, y_test_pred, target_names=iris.target_names, output_dict=True)

# Extract metrics
train_accuracy = train_report['accuracy']
train_recall = train_report['macro avg']['recall']
train_precision = train_report['macro avg']['precision']
train_f1 = train_report['macro avg']['f1-score']

test_accuracy = test_report['accuracy']
test_recall = test_report['macro avg']['recall']
test_precision = test_report['macro avg']['precision']
test_f1 = test_report['macro avg']['f1-score']

print(f"Test Mean Absolute Error (MAE): {test_mae}")
print(f"Test Mean Squared Error (MSE): {test_mse}")
print(f"Training Mean Absolute Error (MAE): {np.mean(training_maes)}")
print(f"Test Mean Absolute Error (MAE): {test_mae}")
print(f"Validation Mean Absolute Error (MAE): {np.mean(validation_maes)}")
print(f"Training Mean Squared Error (MSE): {np.mean(training_mses)}")
print(f"Test Mean Squared Error (MSE): {test_mse}")
print(f"Validation Mean Squared Error (MSE): {np.mean(validation_mses)}")

# Print classification metrics
print("Training Metrics:")
print(f"Accuracy: {train_accuracy}")
print(f"Recall: {train_recall}")
print(f"Precision: {train_precision}")
print(f"F1 Score: {train_f1}")
print("\nTesting Metrics:")
print(f"Accuracy: {test_accuracy}")
print(f"Recall: {test_recall}")
print(f"Precision: {test_precision}")
print(f"F1 Score: {test_f1}")
