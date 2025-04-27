# Purpose: Main script to execute the entire process of training, evaluation, and prediction.

from utils import load_data, build_model
from callbacks import get_callbacks
from tensorflow.keras.models import load_model
import os

# Configuration
from config import TRAIN_PATH, TEST_PATH, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, EPOCHS

# Load data
train_data, test_data = load_data(TRAIN_PATH, TEST_PATH, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)

# Build and compile the model
model = build_model(IMG_HEIGHT, IMG_WIDTH)

# Get callbacks
callbacks = get_callbacks()

# Train the model
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save the model
model.save('final_model.h5')

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot the results
from utils import plot_results
plot_results(history)
