import cv2
import os
import pandas as pd # type: ignore
import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
from constants import *

def preprocess_images(image_names, target_size=NEW_IMAGE_SIZES, normalize=True):
  """
  Preprocesses an image for YOLO input.

  Args:
    image_path: Path to the image file.

  Returns:
    A preprocessed image as a numpy array.
  """
  images = []
  image_paths = [IMAGES_FOLDER + image for image in image_names]
  for image_path in image_paths:
    # Load the image
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, target_size)
    # Normalize the pixel values
    if normalize:
      normalized_image = resized_image / 255.0
      # Return the preprocessed image
      images.append(normalized_image)
    else:
      images.append(resized_image)
  return images


def preprocess_csv(csv_path=CSV_PATH, image_size=ORIGINAL_IMAGE_SIZE,  grid_width=16, grid_height=9, bboxes=BOUNDING_BOXES):
  """
  Preprocesses a CSV file containing bounding box annotations to prepare
  the data for YOLOv1 model input with a 16x9 grid size.

  Args:
      csv_path (str): Path to the CSV file containing bounding box annotations.
      image_size (tuple): Size of the original images (height, width).

  Returns:
      tuple: Tuple of numpy arrays (X, Y). X represents the grids and Y represents the bounding boxes.
  """
  images = []
  df = pd.read_csv(csv_path)
  
  df['y'] = df['y'].apply(lambda x: 1279 if x >= 1280 else x)
  df['x'] = df['x'].apply(lambda x: 719 if x >= 720 else x)

  # Normalize bounding box coordinates
  df['x'] = df['x'] / image_size[1]
  df['y'] = df['y'] / image_size[0]
  df['width'] = df['width'] / image_size[1]
  df['height'] = df['height'] / image_size[0]

  grouped = df.groupby('image').agg(list).reset_index()

  # Initialize output arrays
  labels = np.zeros((len(grouped), grid_height, grid_width, bboxes, 5), dtype=np.float32)

  # Populate labels array
  for idx, row in grouped.iterrows():
      images.append(row['image'])
      x_centers = row['x']
      y_centers = row['y']
      widths = row['width']
      heights = row['height']

      for i in range(len(x_centers)):
          x_center = x_centers[i] + widths[i] / 2
          y_center = y_centers[i] + heights[i] / 2

          grid_x = int(x_center // (1/grid_width))
          grid_y = int(y_center // (1/grid_height))

          box = [x_center, y_center, widths[i], heights[i], 1]  # 1 indicates object presence
          for j in range(bboxes):
            labels[idx, grid_y, grid_x, j] = box 
          
  return (images, labels)



def load_data(test_size=0.2, random_state=42):
    """
    Loads and preprocesses images and CSV data, then splits into train and test sets.

    Args:
        image_paths (list): Paths to the image files.
        csv_path (str): Path to the CSV file.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Tuples containing train and test sets for images and corresponding CSV DataFrames.
    """
    # Preprocess CSV data
    image_names, labels = preprocess_csv()[:64]
    
    # Split image names into train and test sets
    train_image_names, test_image_names, train_labels, test_labels = train_test_split(image_names, labels, test_size=test_size, random_state=random_state)

    train_images = preprocess_images(train_image_names)
    test_images = preprocess_images(test_image_names)
    
    return (train_images, train_labels), (test_images, test_labels)

if __name__ == '__main__':
  print(preprocess_images()[1])
