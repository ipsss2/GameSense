import cv2
import numpy as np
import mss
import time
import pyautogui


class BloodBarDetector:
    def __init__(self, x_start=30, y_start=950, width=400, height=50):

        self.x_start = x_start
        self.y_start = y_start
        self.width = width
        self.height = height
        self._max_blood_length = None  
        self.kernel = np.ones((3, 3), np.uint8) 

    def crop_blood_bar_region(self, img):

        return img[self.y_start:self.y_start + self.height, self.x_start:self.x_start + self.width]

    def detect_white_blood(self, blood_hsv_img):

        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])

        mask = cv2.inRange(blood_hsv_img, lower_white, upper_white)
        return mask

    def process_mask(self, mask):


        dilated_mask = cv2.dilate(mask, self.kernel, iterations=1)  
        eroded_mask = cv2.erode(dilated_mask, self.kernel, iterations=1)
        return eroded_mask

    def calculate_blood_length(self, mask):

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return w
        else:
            return 0

    def detect_blood_bar(self, img):

        try:

            blood_region = self.crop_blood_bar_region(img)

            hsv_img = cv2.cvtColor(blood_region, cv2.COLOR_BGR2HSV)

            mask = self.detect_white_blood(hsv_img)

            processed_mask = self.process_mask(mask)

            current_blood_length = self.calculate_blood_length(processed_mask)

            if self._max_blood_length is None or current_blood_length > self._max_blood_length:
                self._max_blood_length = current_blood_length
                print(f" {self._max_blood_length}")

            if self._max_blood_length > 0:
                blood_percentage = (current_blood_length / self._max_blood_length) * 100
            else:
                blood_percentage = 0


            self.display_debug_info(blood_region, processed_mask, current_blood_length, blood_percentage)

            return blood_percentage

        except Exception as e:
            print(f"error: {e}")
            return 0

    def display_debug_info(self, blood_region, processed_mask, current_blood_length, blood_percentage):

        display_text = f"Length: {current_blood_length} px, {blood_percentage:.2f}%"

        cv2.putText(blood_region, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Blood Bar Region", blood_region)

        cv2.imshow("Processed Mask", processed_mask)

        print(f" {current_blood_length} px, {blood_percentage:.2f}%")

    def capture_screen(self, x_start, y_start, width, height):

        with mss.mss() as sct:
            monitor = {"top": y_start, "left": x_start, "width": width, "height": height}
            img = np.array(sct.grab(monitor))
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) 

    def run_test(self, screen_x, screen_y, screen_width, screen_height, capture_interval, capture_duration):
 
        I=1
        start_time = time.time()
        while time.time() - start_time < capture_duration:
            print(I)
            I+=1
            screenshot = self.capture_screen(screen_x, screen_y, screen_width, screen_height)
            self.detect_blood_bar(screenshot)
            time.sleep(capture_interval)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
