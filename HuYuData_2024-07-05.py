import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

# Example data: Generate a simple sine wave
def generate_data(seq_length, num_samples):
    X = []
    y = []
    for _ in range(num_samples):
        start = np.random.rand() * 2 * np.pi
        x = np.sin(np.linspace(start, start + seq_length * 0.1, seq_length))
        X.append(x)
        y.append(np.sin(start + seq_length * 0.1))
    return np.array(X), np.array(y)

# Generate data
seq_length = 50
num_samples = 1000
X, y = generate_data(seq_length, num_samples)

# Reshape data for GRU [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data into training and testing sets
split = int(0.8 * num_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define the GRU model
model = Sequential()
model.add(GRU(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make a prediction
test_input = X_test[0].reshape((1, seq_length, 1))
predicted = model.predict(test_input, verbose=0)
print(f'Actual: {y_test[0]}, Predicted: {predicted[0][0]}')