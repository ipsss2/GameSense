import os
import pyautogui
import cv2
import numpy as np
import time
import base64

class ScreenCapture:
    def __init__(self, region=None, output_folder="video_picture_save/picture"):
        self.region = region  # (x, y, width, height)
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def set_region(self, region):
        """Set the capture region."""
        self.region = region

    def capture_image(self, convert_to_base64=False, quality=60, resize_factor=1.0):
        """
        Capture an image, compress and save it. Optionally convert to Base64.

        Args:
            convert_to_base64 (bool): Whether to convert to base64
            quality (int): JPEG quality 1-100 (default 60)
            resize_factor (float): Resize factor 0-1.0 (default 0.8)
        """
        screenshot = pyautogui.screenshot(region=self.region)
        img_array = np.array(screenshot)


        if resize_factor < 1.0:
            new_size = (int(img_array.shape[1] * resize_factor),
                        int(img_array.shape[0] * resize_factor))
            img_array = cv2.resize(img_array, new_size, interpolation=cv2.INTER_AREA)


        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


        filename = f"capture_{int(time.time())}.jpg"  
        path = os.path.join(self.output_folder, filename)
        cv2.imwrite(path, img_array, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

        if convert_to_base64:
            success, buffer = cv2.imencode('.jpg', img_array, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not success:
                raise Exception("Failed to encode image")
            return base64.b64encode(buffer).decode('utf-8')

        return path

    def capture_video(self, duration, convert_to_base64=False):
        """Capture a video of specified duration. Optionally convert to Base64."""
        end_time = time.time() + duration
        timestamp = int(time.time())
        video_path = os.path.join(self.output_folder, f"capture_{timestamp}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, 8.0, (self.region[2], self.region[3]))

        while time.time() < end_time:
            img = pyautogui.screenshot(region=self.region)
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            out.write(frame)

        out.release()

        if convert_to_base64:
            with open(video_path, "rb") as video_file:
                return base64.b64encode(video_file.read()).decode('utf-8')
        else:
            return video_path

