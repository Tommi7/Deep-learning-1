import tensorflow as tf # type: ignore
from tensorflow.keras import layers, models # type: ignore
from constants import BOUNDING_BOXES

def build_model(input_shape=(270, 480, 3), num_bboxes=BOUNDING_BOXES):
    """
    Build the YOLOv1 model.
    
    Args:
        input_shape (tuple): Shape of the input images. Default is (270, 480, 3).
        num_classes (int): Number of classes for object detection. Default is 20.
        num_bboxes (int): Number of bounding boxes per cell. Default is 2.
    
    Returns:
        tf.keras.Model: YOLOv1 model.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Feature extraction layers
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    x = layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    x = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # for _ in range(4):
    #     x = layers.Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    #     x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    
    # x = layers.Conv2D(512, (1, 1), padding='same', activation='relu')(x)
    # x = layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(x)
    # x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # for _ in range(2):
    #     x = layers.Conv2D(512, (1, 1), padding='same', activation='relu')(x)
    #     x = layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(x)
    
    x = layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    
    x = layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(x)
    
    # Flatten the feature maps
    x = layers.Flatten()(x)
    
    # Fully connected layers
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(9 * 16 * (num_bboxes * 5), activation='linear')(x)
    x = layers.Reshape((9, 16, num_bboxes, 5))(x)
    
    # Create model
    model = models.Model(inputs, x)
    
    return model


if __name__ == '__main__':
    # Example usage:    
    model = build_model()
    model.summary()
