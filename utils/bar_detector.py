
#
import cv2
import numpy as np
import mss
import time
import pyautogui

class StatusBarDetector:
    def __init__(self, blood_bar, mana_bar, stamina_bar, potion_bar,boss_bar,left, top, capture_width, capture_height):

        self.blood_bar = blood_bar
        self.mana_bar = mana_bar
        self.stamina_bar = stamina_bar
        self.potion_bar = potion_bar
        self.boss_bar = boss_bar

        # self._max_blood_length = 121  # 存储最大血条长度
        self._max_mana_length = 115  # 存储最大法力条长度
        self._max_stamina_length = 155  # 存储最大体力条长度
        # self._max_boss_length = 330
        self._max_potion_length = 49  # 存储最大药剂条长度（竖向）
        self.kernel = np.ones((3, 3), np.uint8)  # 形态学操作的卷积核
        self.left=left
        self.top=top
        self.capture_width = capture_width
        self.capture_height = capture_height




    def crop_region(self, img, region):

        x_start, y_start, width, height = region
        return img[y_start:y_start + height, x_start:x_start + width]

    def detect_colored_bar(self, hsv_img, color_range):

        lower_bound, upper_bound = color_range
        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
        return mask



    def boss_blood_percentage(self,cropped_img):

        hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 200])  
        upper_white = np.array([180, 30, 255])  
        lower_gray = np.array([0, 0, 150])  
        upper_gray = np.array([180, 50, 255])  

  
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        combined_mask = cv2.bitwise_or(white_mask, gray_mask)


        blood_pixels = cv2.countNonZero(combined_mask)

   
        total_pixels = 330*5  # 高度 * 宽度

        blood_percentage = blood_pixels / total_pixels

        return blood_percentage

    def self_blood_count_hsv(self, cropped_img):


        hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)


        lower_white = np.array([0, 0, 200])  
        upper_white = np.array([180, 40, 255])  

        lower_light_red = np.array([0, 15, 180])  
        upper_light_red = np.array([15, 120, 255])  

        lower_real_red = np.array([1, 50, 69])
        upper_real_red = np.array([6, 80, 85])


        lower_light_green = np.array([35, 20, 150])  
        upper_bright_green = np.array([90, 255, 255])  


        lower_dark_red = np.array([2, 51, 23])  
        upper_dark_red = np.array([9, 59, 40]) 


        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        light_red_mask = cv2.inRange(hsv, lower_light_red, upper_light_red)
        cyan_green_mask = cv2.inRange(hsv, lower_light_green, upper_bright_green)
        real_red_mask = cv2.inRange(hsv, lower_real_red, upper_real_red)


        dark_red_mask = cv2.inRange(hsv, lower_dark_red, upper_dark_red)

  
        combined_mask = cv2.bitwise_or(white_mask, light_red_mask)
        combined_mask = cv2.bitwise_or(combined_mask, cyan_green_mask)
        combined_mask = cv2.bitwise_or(combined_mask, real_red_mask)

        final_mask = cv2.bitwise_and(combined_mask, cv2.bitwise_not(dark_red_mask))

        blood_pixels = cv2.countNonZero(final_mask)
        total_pixels = 145*8

        blood_percentage = blood_pixels / total_pixels

        return blood_percentage

    def process_mask(self, mask):

        dilated_mask = cv2.dilate(mask, self.kernel, iterations=1) 
        eroded_mask = cv2.erode(dilated_mask, self.kernel, iterations=1)
        return eroded_mask

    def calculate_bar_length(self, mask, vertical=False):

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return h if vertical else w
        else:
            return 0

    def self_blood_count_edge(self, obs_gray):
       
        height = obs_gray.shape[0]
        center_y = height // 2
        strip_height = 4  
        blood_strip = obs_gray[center_y - strip_height // 2:center_y + strip_height // 2, :]

        
        blurred = cv2.GaussianBlur(blood_strip, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 150)


        right_edge = edges.shape[1] - 1
        while right_edge > 0:
            if np.any(edges[:, right_edge] > 0):
                break
            right_edge -= 1


        blood_ratio = right_edge / edges.shape[1]

        return blood_ratio

    def detect_status_bars(self, img):

        try:
            
            white_range = (np.array([0, 0, 180]), np.array([360, 30, 220])) 
            blue_range = (np.array([100, 150, 0]), np.array([140, 255, 255]))  
            brown_range = (np.array([10, 50, 100]), np.array([30, 255, 200])) 
            gray_white_range = (np.array([0, 0, 100]), np.array([180, 50, 255]))  

            

            current_blood_length = self.self_blood_count_hsv(blood_region)


            blood_percentage = current_blood_length
            
            boss_blood_region = self.crop_region(img, self.boss_bar)
            
            current_boss_blood_length = self.boss_blood_percentage(boss_blood_region)
            
            boss_percentage = current_boss_blood_length
            
            mana_region = self.crop_region(img, self.mana_bar)
            hsv_mana = cv2.cvtColor(mana_region, cv2.COLOR_BGR2HSV)
            mana_mask = self.detect_colored_bar(hsv_mana, blue_range)
            processed_mana_mask = self.process_mask(mana_mask)
            current_mana_length = self.calculate_bar_length(processed_mana_mask)


            mana_percentage = (current_mana_length / self._max_mana_length) if self._max_mana_length > 0 else 0

            stamina_region = self.crop_region(img, self.stamina_bar)
            hsv_stamina = cv2.cvtColor(stamina_region, cv2.COLOR_BGR2HSV)
            stamina_mask = self.detect_colored_bar(hsv_stamina, brown_range)
            processed_stamina_mask = self.process_mask(stamina_mask)
            current_stamina_length = self.calculate_bar_length(processed_stamina_mask)

            if self._max_stamina_length is None or current_stamina_length > self._max_stamina_length:
                self._max_stamina_length = current_stamina_length

            stamina_percentage = (current_stamina_length / self._max_stamina_length) if self._max_stamina_length > 0 else 0

            potion_region = self.crop_region(img, self.potion_bar)
            hsv_potion = cv2.cvtColor(potion_region, cv2.COLOR_BGR2HSV)
            potion_mask = self.detect_colored_bar(hsv_potion, gray_white_range)
            processed_potion_mask = self.process_mask(potion_mask)
            current_potion_length = self.calculate_bar_length(processed_potion_mask, vertical=True)


            potion_percentage = (current_potion_length / self._max_potion_length) if self._max_potion_length > 0 else 0

          
            return {
                "blood_percentage": blood_percentage,
                "boss_percentage": boss_percentage,
                "mana_percentage": mana_percentage,
                "stamina_percentage": stamina_percentage,
                "potion_percentage": potion_percentage,
               
            },img

        except Exception as e:
            print(f"error: {e}")
            return {"blood_percentage": 0, "mana_percentage": 0, "stamina_percentage": 0, "potion_percentage": 0}

    def display_debug_info(self, region, processed_mask, current_length, percentage, bar_name, vertical=False):

        display_text = f"{bar_name} Length: {current_length} px, {percentage:.2f}%"

        cv2.putText(region, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(f"{bar_name} Region", region)


        orientation = "Vertical" if vertical else "Horizontal"
        print(f"{bar_name} ({orientation}) lenth: {current_length} px, percentage: {percentage:.2f}%")

    def capture_screen(self, x_start, y_start, width, height):

        with mss.mss() as sct:
            monitor = {"top": y_start, "left": x_start, "width": width, "height": height}
            img = np.array(sct.grab(monitor))
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  
    def run_test(self, screen_x, screen_y, screen_width, screen_height, capture_interval, capture_duration):

        start_time = time.time()
        while time.time() - start_time < capture_duration:

            screenshot = self.capture_screen(screen_x, screen_y, screen_width, screen_height)


            self.detect_status_bars(screenshot)


            time.sleep(capture_interval)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def get_status(self):
        screenshot = self.capture_screen(self.left, self.top, self.capture_width, self.capture_height)


        result=self.detect_status_bars(screenshot)

        return result


    def capture_state(self,screen_x, screen_y, screen_width, screen_height):
        screenshot = self.capture_screen(screen_x, screen_y, screen_width, screen_height)

        state=self.detect_status_bars(screenshot)

        return state

    def get_frame(self):
        screenshot = self.capture_screen(self.left, self.top, self.capture_width, self.capture_height)


        return screenshot




blood_bar = (170, 820, 145, 8)  
mana_bar = (172, 833, 120, 10)  
stamina_bar = (172, 846, 155, 10)  
potion_bar = (103, 800, 8, 51)  
boss_bar = (636,763,330,5)

screen_width, screen_height = pyautogui.size()


capture_width = 1600
capture_height = 900


left = (screen_width - capture_width) // 2
top = (screen_height - capture_height) // 2



detector = StatusBarDetector(blood_bar, mana_bar, stamina_bar, potion_bar, boss_bar,left, top, capture_width, capture_height)
