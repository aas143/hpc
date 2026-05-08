#%pip install pandas numpy tensorflow scikit-learn matplotlib

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('boston_house_prices.csv')
print(df.head())

# Separate features and target
X = df.drop('PRICE', axis=1)
y = df['PRICE']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1) # Output layer for regression (no activation function)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()



history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

mse, mae = model.evaluate(X_test, y_test)
print(f"\nTest Mean Squared Error: {mse:.2f}")
print(f"Test Mean Absolute Error: {mae:.2f}")

# Predict
y_pred = model.predict(X_test)

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.show()


