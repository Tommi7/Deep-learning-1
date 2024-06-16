import cv2
import os
import moviepy.editor as mp
import pandas as pd
import pathlib

def resize_video(path):
    clip = mp.VideoFileClip(path)
    path = os.path.basename(path)
    new_path = f'new_data/resized_videos/' + path
    clip_resized = clip.resize((1280, 720))
    clip_resized.write_videofile(new_path, codec='libx264')
    return new_path
    
class imagePainter:
    def __init__(self, video_path, frame_start, title='ImagePainter!'):
        self.running = True
        
        self.title = title
        self.frame_start = frame_start
        
        self.video_name = os.path.basename(video_path).rstrip('.mp4')
        self.video_path = video_path.rstrip(self.video_name)
        self.coords_path = f'new_data/coords.csv'
        
        if 'coords.csv' not in os.listdir('new_data/'):
            self.rectangle_coords = pd.DataFrame({'image': [], 'x1': [], 'y1': [], 'x2': [], 'y2': []})
            self.rectangle_coords.to_csv(self.coords_path)
        else:
            self.rectangle_coords = pd.read_csv(self.coords_path)
        
        self.draw_on_frames_from_video(video_path)
        
    def create_folders(self):
        data_folder = 'data'
        frame_folder = f'{data_folder}/{self.video_name}_frames'
        face_folder =  f'{frame_folder}_face_boxes'

        self.check_and_create_folder(data_folder)
        self.check_and_create_folder(frame_folder)
        self.check_and_create_folder(face_folder)
        
        return (frame_folder, face_folder, data_folder)
    
    def check_and_create_folder(self, folder):
        try:
            os.mkdir(folder)
            print(f'Creating {folder}')
        except FileExistsError:
            print(f'{folder} folder already exists! skipping creation...')
            
    def get_data_paths(self, count):
        frame_path = f'new_data/frames/{self.video_name}_frame_{count}.jpg'
        face_path = f'new_data/face_frames/{self.video_name}_face_frame_{count}.jpg'
        
        return (frame_path, face_path)
             
    def draw_on_frames_from_video(self, video_path):
        self.show_previous_frame = False
        vid = cv2.VideoCapture(video_path)
        vid.set(cv2.CAP_PROP_POS_FRAMES, self.frame_start)
        frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        count = self.frame_start
        
        while self.running and vid.isOpened():
            self.title = f'ImagePainter - {self.video_name} - Image{count}/{frame_count}'  
            if self.show_previous_frame:
                count -= 1
                vid.set(cv2.CAP_PROP_POS_FRAMES, current_frame-1)
                self.running, self.img = vid.read()
                self.show_previous_frame = False
            else:
                self.running, self.img = vid.read()
                
                
            next_frame = vid.get(cv2.CAP_PROP_POS_FRAMES)
            current_frame = next_frame - 1
            
            try:
                self.original_img = self.img.copy()
            except AttributeError:
                cv2.destroyAllWindows() 
                self.rectangle_coords.to_csv(self.coords_path, index=False)
                break
            self.drawing_img = self.img.copy()

            self.image_painter(count)
            
            count += 1
            
        self.rectangle_coords.to_csv(self.coords_path, index=False)
        
    def image_painter(self, count):
        self.image_name = f'{self.video_name}_frame_{count}.jpg'
        frame_path, face_path = self.get_data_paths(count)
        
        self.ix = -1
        self.iy = -1
        self.drawing = False
        
        while self.running:
            cv2.namedWindow(winname = self.title) 
            cv2.setMouseCallback(self.title,  self.draw_rectangle_with_drag) 
            cv2.imshow(self.title, self.img)
                
            key = cv2.waitKey(0)
            if key == 27:
                self.running = False
                cv2.destroyAllWindows() 
                self.rectangle_coords.to_csv(self.coords_path, index=False)
            else:
                cv2.imwrite(frame_path, self.original_img)
                cv2.imwrite(face_path, self.drawing_img)
                cv2.destroyAllWindows()  
                break
    
    def draw_rectangle_with_drag(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN: 
            self.drawing = True
            self.ix = x 
            self.iy = y
            
        elif event == cv2.EVENT_MOUSEMOVE: 
            if self.drawing == True: 
                self.drawing_img = self.img.copy()
                cv2.rectangle(self.drawing_img, 
                              pt1=(self.ix, self.iy), pt2=(x, y), 
                              color=(0, 0, 255), 
                              thickness=1, 
                              lineType=cv2.LINE_AA)
                cv2.imshow(self.title, self.drawing_img)
        
        elif event == cv2.EVENT_LBUTTONUP: 
            self.img = self.drawing_img
            coords = pd.DataFrame({'image': [self.image_name], 'x1': [self.ix], 'y1': [self.iy], 'x2': [x], 'y2': [y]})
            self.rectangle_coords = pd.concat([self.rectangle_coords, coords], axis=0)
            self.drawing = False
            