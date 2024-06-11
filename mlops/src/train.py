import tensorflow as tf # type: ignore
from model import build_model
from loss import yolo_loss
from data import load_data
from constants import *
import os

# Hyperparameters
learning_rate = 0.001
batch_size = 32
epochs = 100
print('loading')
# Data loading and preprocessing
(train_images, train_labels), (test_images, test_labels) = load_data()
print('converting')
# Convert data to TensorFlow tensors
train_images = tf.convert_to_tensor(train_images, dtype=tf.float32)
train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)
test_images = tf.convert_to_tensor(test_images, dtype=tf.float32)
test_labels = tf.convert_to_tensor(test_labels, dtype=tf.float32)
print('datasets')
# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

print('building')
# Build the YOLOv1 model
model = build_model()
print('done building')
# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Define the loss function
def yolo_loss_wrapper(y_true, y_pred):
    return yolo_loss(y_pred, y_true)

# Compile the model
model.compile(optimizer=optimizer, loss=yolo_loss_wrapper, run_eagerly=True)

# Define a checkpoint callback to save the model during training
checkpoint_path = "cp.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model
model.fit(train_dataset, epochs=epochs, callbacks=[cp_callback])

# Save the trained model
model.save('yolov1_model.h5')
