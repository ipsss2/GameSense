# import os
# import threading
# import cv2
# import time
# from PIL import Image
# import mss

# class ScreenshotManager:

#     def __init__(self, save_dir="video_picture_save/action", frame_rate=60, region=(1120, 630, 1600, 900)):

#         self._save_dir = save_dir
#         self.region = region
#         self.frame_rate = frame_rate 
#         self._ensure_dir_exists(self._save_dir)


#         self.recording = False  
#         self.stop_recording_flag = threading.Event() 
#         self.sct = mss.mss() 


#         self.capture_thread = None

#     @property
#     def save_dir(self):

#         return self._save_dir

#     @save_dir.setter
#     def save_dir(self, new_dir):

#         self._save_dir = new_dir
#         self._ensure_dir_exists(self._save_dir)

#     def _ensure_dir_exists(self, directory):

#         if not os.path.exists(directory):
#             os.makedirs(directory)

#     def _capture_screen(self):
 
#         monitor = {
#             "left": self.region[0],
#             "top": self.region[1],
#             "width": self.region[2],
#             "height": self.region[3],
#         }
#         screenshot = self.sct.grab(monitor)
#         img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
#         return img

#     def _capture_loop(self):

#         while not self.stop_recording_flag.is_set(): 
#             timestamp = time.time()
#             img = self._capture_screen() 

 
#             file_path = os.path.join(self._save_dir, f"screenshot_{timestamp:.2f}.png")
#             img.save(file_path)
#             print(f"Screenshot saved: {file_path}")

#             time.sleep(1 / self.frame_rate)  # 等待下一帧

#         print("Stopped recording screenshots.")

#     def start_recording(self):
      
#         if not self.recording: 
#             self.recording = True
#             self.stop_recording_flag.clear()
#             self.capture_thread = threading.Thread(target=self._capture_loop)
#             self.capture_thread.start()
#             print("Started recording screenshots.")

#     def stop_recording(self):

#         if self.recording:
#             self.stop_recording_flag.set() 
#             self.capture_thread.join() 
#             self.recording = False
#             print("Recording stopped.")


# frame_recorder.py

import os
import threading
import time
from PIL import Image
import mss

class ScreenshotManager:


    def __init__(self, save_dir="video_picture_save/action", frame_rate=60, region=(1120, 630, 1600, 900), use_coordinates=True):

        self._save_dir = save_dir
        self.frame_rate = frame_rate
        self.recording = False
        self.stop_recording_flag = threading.Event()
        self.sct = mss.mss()
        self.capture_thread = None
        if use_coordinates:

            self.monitor = {
                "left": region[0],
                "top": region[1],
                "width": region[2] - region[0],
                "height": region[3] - region[1],
            }
            if self.monitor["width"] <= 0 or self.monitor["height"] <= 0:
                raise ValueError("Region coordinates are invalid: right must be greater than left, and bottom must be greater than top.")
        else:
            self.monitor = {
                "left": region[0],
                "top": region[1],
                "width": region[2],
                "height": region[3],
            }
        
        self._ensure_dir_exists(self._save_dir)

    @property
    def save_dir(self):

        return self._save_dir

    @save_dir.setter
    def save_dir(self, new_dir):

        self._save_dir = new_dir
        self._ensure_dir_exists(self._save_dir)

    def _ensure_dir_exists(self, directory):

        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory created: {directory}")

    def _capture_screen(self):

        screenshot = self.sct.grab(self.monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        return img

    def _capture_loop(self):
        interval = 1.0 / self.frame_rate
        while not self.stop_recording_flag.is_set():
            start_time = time.perf_counter()
            
            img = self._capture_screen()

            timestamp = time.time()
            file_path = os.path.join(self._save_dir, f"screenshot_{timestamp:.6f}.png")
            img.save(file_path)
            
            elapsed_time = time.perf_counter() - start_time
            sleep_time = interval - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("Stopped recording screenshots.")

    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.stop_recording_flag.clear()
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.start()
            print(f"Started recording screenshots. Saving to: {self._save_dir}")

    def stop_recording(self):
        if self.recording:
            self.stop_recording_flag.set()
            if self.capture_thread:
                self.capture_thread.join()
            self.recording = False
            print("Recording stopped and thread joined.")