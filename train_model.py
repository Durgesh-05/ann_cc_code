import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("./credit_score_data/large_demographic_data.csv")

# 1. Preprocessing
# Encode categorical features
categorical_cols = ["Gender", "Education", "Marital Status", "Home Ownership"]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le
    # Save each encoder
    joblib.dump(le, f"{col.replace(' ', '_')}_encoder.pkl")

# Encode target variable
label_encoder = LabelEncoder()
data["Credit Score"] = label_encoder.fit_transform(data["Credit Score"])
joblib.dump(label_encoder, "label_encoder.pkl")

# Split features and target
X = data.drop("Credit Score", axis=1)
y = data["Credit Score"]

# Scale features - IMPORTANT: Fit then transform to preserve feature names
scaler = StandardScaler()
scaler.fit(X)  # Fit first to store feature names
X_scaled = scaler.transform(X)
joblib.dump(scaler, "scaler.pkl")

# Save feature order for reference
pd.Series(scaler.feature_names_in_).to_csv("feature_order.csv", index=False)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Convert y to categorical for neural network
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# 2. Neural Network Model
ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

ann_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = ann_model.fit(
    X_train, y_train_cat,
    epochs=50,
    batch_size=8,
    validation_data=(X_test, y_test_cat),
    verbose=1
)

# Save ANN model
ann_model.save("credit_model.h5")

# 3. Other Models
logistic_model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
logistic_model.fit(X_train, y_train)
joblib.dump(logistic_model, "logistic_model.pkl")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, "rf_model.pkl")

# 4. Evaluation
print("Training complete! All models saved.")
print("Feature order:", scaler.feature_names_in_)