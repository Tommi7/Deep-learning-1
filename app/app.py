import tkinter as tk
from tkinter import messagebox, filedialog
from labeler import resize_video, imagePainter
from detection import show_detection
from data import csv_to_labels, create_training_run
from train import train_model

def button1_action():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.mp4")]
    )
    new_path = resize_video(file_path)
    imagePainter(new_path, 0)
    csv_to_labels()
    
def button2_action():
    create_training_run()
    train_model()

def button3_action():
    show_detection()

# Create the main window
root = tk.Tk()
root.title("Object detection")
root.geometry('400x60')


# Create buttons
button1 = tk.Button(root, text="Label new video", command=button1_action)
button1.grid(column=0, row=1, pady=15, padx=10)

button2 = tk.Button(root, text="Retrain model", command=button2_action)
button2.grid(column=1, row=1, pady=15, padx=10)

button3 = tk.Button(root, text="Detect faces from livestream", command=button3_action)
button3.grid(column=2, row=1, pady=15, padx=10)

# Run the application
root.mainloop()
