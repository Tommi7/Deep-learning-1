import os

IMAGES_FOLDER = 'Data/images/'
IMAGE_PATHS = [IMAGES_FOLDER + image for image in os.listdir(IMAGES_FOLDER)]
CSV_PATH = 'Data/Coords.csv'
ORIGINAL_IMAGE_SIZE = (720, 1280)
NEW_IMAGE_SIZES = (480, 270)

BOUNDING_BOXES = 3