from ultralytics import YOLO
import cv2
import cvzone
import math
import os

def get_highest_training_run():
    training_run_folders = [f for f in os.listdir('Training_runs/')]
    training_run_numbers = [int(f.split('_')[-1]) for f in training_run_folders if f.split('_')[-1].isnumeric()]
    try:
        highest_number = max(training_run_numbers)
    except:
        highest_number = 0
    return f"training_run_{highest_number}"

def show_detection():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    highest_training_run = get_highest_training_run()
    
    if not os.listdir('Training_runs'):
        model = YOLO('Data/training/weights/best.pt')
    else:
        model = YOLO(f'Training_runs/{highest_training_run}_training/weights/best.pt')
    # model = YOLO('yolo-Weights/yolov8n.pt')


    classNames = ["face"]

    while True:
        success, img = cap.read()
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2-x1, y2-y1
                cvzone.cornerRect(img, (x1, y1, w, h))

                conf = math.ceil((box.conf[0]*100))/100

                cls = box.cls[0]
                name = classNames[int(cls)]

                cvzone.putTextRect(
                    img, f'{name} 'f'{conf}', (max(0, x1), max(35, y1)), scale=2, thickness=2, 
                    colorT=(255,255,255), colorR=(54,250,74))

        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    show_detection()