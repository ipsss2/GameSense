import pyautogui
import cv2
import time
import matplotlib.pyplot as plt
from action_manager.base_action2keybord import SimpleGameController
from utils.bar_detector import detector
import pyautogui
import time

c=SimpleGameController()

class boss_env(object):
    def __init__(self,check_interval=0.2, check_attempts=5):
        super().__init__()

        self.action_manager = c

        self.detector = detector

        self.check_interval = check_interval 
        self.check_attempts = check_attempts  

    def _is_state_stable(self, state):

        return (
                state['blood_percentage'] != 0 or
                state['mana_percentage'] != 0 or
                state['stamina_percentage'] != 0
        )

    def check_done(self,state):

        if state['blood_percentage'] ==0.0:
            for _ in range(self.check_attempts):
                state, _ = self.get_state() 
                if not state['blood_percentage'] ==0.0:
                    return False, False
                else:
                    time.sleep(self.check_interval)
            return True, False  # done = True, success = False

        if state['boss_percentage'] < 0.05 and state['blood_percentage'] != 0:
            time.sleep(self.check_interval * 2)
            for _ in range(self.check_attempts*10):
                state, _ = self.get_state()  
                if state['boss_percentage'] == 0 and state['blood_percentage'] != 0 and self._is_state_stable(state):
                    time.sleep(self.check_interval)
                else:
                    return False, False
            return True, True  

        return False, False

    def take_action(self,action):
        self.action_manager.execute_action(action_names=action)

    def get_state(self):
        state, obs_img = self.detector.get_status()

        return state, obs_img

    def step(self, action):
        self.take_action(action)
        next_state,obs_img=self.get_state()
        i = 0
        while (next_state['boss_percentage'] < 0.05 or next_state['blood_percentage'] == 0):
            i += 1
            state, obs_img = self.detector.get_status()
            time.sleep(0.1)
            if i == 3:
                break
        done,success = self.check_done(next_state)



        return next_state,obs_img,done,success

    def restart(self):
        time.sleep(25)
 
        pyautogui.press('l')
        pyautogui.click(button='middle')

        state, obs_img = self.get_state()
        return state,obs_img

