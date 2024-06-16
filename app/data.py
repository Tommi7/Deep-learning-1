import os
import pandas as pd # type: ignore
import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
import random
import shutil
import yaml

def csv_to_labels():
  """
  Preprocesses a CSV file containing bounding box annotations to prepare
  the data for YOLOv1 model input with a 16x9 grid size.

  Args:
      csv_path (str): Path to the CSV file containing bounding box annotations.
      image_size (tuple): Size of the original images (height, width).

  Returns:
      tuple: Tuple of numpy arrays (X, Y). X represents the grids and Y represents the bounding boxes.
  """
  image_w, image_h = (1280, 720)
  df = pd.read_csv('new_data/coords.csv')
  
  df['x'] = (df['x1']+df['x2']) /2
  df['y'] = (df['y1']+df['y2']) /2
  df['w'] = df['x2'] - df['x1']
  df['h'] = df['y2'] - df['y1']
  
  df = df.drop(columns=['x1', 'y1', 'x2', 'y2'])
  
  df['y'] = df['y'].apply(lambda x: 1279 if x >= 1280 else x)
  df['x'] = df['x'].apply(lambda x: 719 if x >= 720 else x)

  # Normalize bounding box coordinates
  df['x'] = df['x'] / image_w
  df['y'] = df['y'] / image_h
  df['w'] = df['w'] / image_w
  df['h'] = df['h'] / image_h
    
  for _, row in df.iterrows():
    path = 'new_data/labels/' + row['image']
    path = path.rstrip('.jpg')
    with open(path + '.txt', 'a') as file:
      x = row['x']
      y = row['y']
      w = row['w']
      h = row['h']
      file.write(f'0 {x} {y} {w} {h}')
      file.write('\n')

def create_training_run():
  face_frames_folder = 'new_data/face_frames/'
  resized_videos_folder = 'new_data/resized_videos/'
  
  for face_frame in os.listdir(face_frames_folder):
    file_path = os.path.join(face_frames_folder, face_frame)
    os.unlink(file_path)
    
  for video in os.listdir(resized_videos_folder):
    file_path = os.path.join(resized_videos_folder, video)
    os.unlink(file_path)
  
  try:
    os.unlink('new_data/coords.csv')
  except:
    pass
    
  training_folder = 'Training_runs'
  
  image_folder = 'new_data/frames'
  label_folder = 'new_data/labels'
  
  counter = 1
  new_training_path = f"{training_folder}/training_run_{counter}"
  
  
  
  while os.path.exists(new_training_path):
      counter += 1
      new_training_path = f"{training_folder}/training_run_{counter}"
  os.makedirs(new_training_path)
  
  train_image_folder = f'{new_training_path}/train/images'
  val_image_folder = f'{new_training_path}/val/images'
  train_label_folder = f'{new_training_path}/train/labels'
  val_label_folder = f'{new_training_path}/val/labels'
  
  os.makedirs(train_image_folder)
  os.makedirs(val_image_folder)
  os.makedirs(train_label_folder)
  os.makedirs(val_label_folder)
  
  images = [f for f in os.listdir(image_folder)]
  
  # Shuffle images for random splitting
  random.shuffle(images)
  
  # Determine the split index
  split_index = int(len(images) * (1 - 0.2))

  # Split into training and validation sets
  train_images = images[:split_index]
  val_images = images[split_index:]

  # Move files to their respective directories
  for img in train_images:
      shutil.move(os.path.join(image_folder, img), os.path.join(train_image_folder, img))
      shutil.move(os.path.join(label_folder, os.path.splitext(img)[0] + '.txt'), os.path.join(train_label_folder, os.path.splitext(img)[0] + '.txt'))
  
  for img in val_images:
      shutil.move(os.path.join(image_folder, img), os.path.join(val_image_folder, img))
      shutil.move(os.path.join(label_folder, os.path.splitext(img)[0] + '.txt'), os.path.join(val_label_folder, os.path.splitext(img)[0] + '.txt'))
    
  create_yaml()

def get_highest_training_run():
    training_run_folders = [f for f in os.listdir('Training_runs/')]
    training_run_numbers = [int(f.split('_')[-1]) for f in training_run_folders]
    highest_number = max(training_run_numbers)
    return f"training_run_{highest_number}"

def create_yaml():
  highest_training_run = get_highest_training_run()
  yaml_content = {
      'path': 'training_runs',
      'train': f"{highest_training_run}/train",
      'val': f"{highest_training_run}/val",
      'names': {
          0: 'face'
      }
  }
  with open(os.path.join(f'Training_runs/{highest_training_run}', f'{highest_training_run}.yaml'), 'w') as yaml_file:
      yaml.dump(yaml_content, yaml_file, default_flow_style=False)

if __name__ == '__main__':
  create_training_run()
