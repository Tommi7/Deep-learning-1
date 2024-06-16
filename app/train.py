from ultralytics import YOLO
import os

def get_highest_training_run():
    training_run_folders = [f for f in os.listdir('Training_runs/')]
    training_run_numbers = [int(f.split('_')[-1]) for f in training_run_folders]
    highest_number = max(training_run_numbers)
    return f"training_run_{highest_number}"


def train_model():
    os.environ['WANDB_DISABLED'] = 'true'
    highest_number_training_run = get_highest_training_run()
    
    yaml_path = f'Training_runs/{highest_number_training_run}/{highest_number_training_run}.yaml'
    # Load and train the YOLO model
    model = YOLO('Data/training/weights/best.pt')
    
    # Train the model using the provided YAML configuration
    try:
        model.train(data=yaml_path, epochs=50, single_cls=True, rect=True, plots=False, imgsz=[720, 1280], project=f'Training_runs', name=f'{highest_number_training_run}_training', device=0)
    except:
        model.train(data=yaml_path, epochs=50, single_cls=True, rect=True, plots=False, imgsz=[720, 1280], project=f'Training_runs', name=f'{highest_number_training_run}_training', device='cpu')
if __name__ == "__main__":
    train_model()