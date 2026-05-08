#%pip install tensorflow numpy


import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

# 1. Load the pre-tokenized dataset
print("Loading data...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# 2. Pad sequences to ensure uniform input size
X_train = pad_sequences(X_train, maxlen=200)
X_test = pad_sequences(X_test, maxlen=200)

# 3. Build the DNN Model (Clean version)
model = Sequential([
    tf.keras.Input(shape=(200,)),
    Embedding(10000, 32),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 4. Compile the model
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.summary()

# 5. Train the model
model.fit(X_train, y_train, epochs=3, batch_size=32)

# 6. Evaluate and print final accuracy
loss, acc = model.evaluate(X_test, y_test)
print("\nFinal Accuracy:", acc)