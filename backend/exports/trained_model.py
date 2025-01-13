
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

# Define the model architecture
model = Sequential([
    Dense(112, activation='relu', input_shape=(4,)),
Dense(3, activation='softmax'),
])

model.compile(optimizer='sgd', loss='categorical cross-entropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=12, batch_size=32, validation_split=0.2)
model.save('trained_model.keras')
    