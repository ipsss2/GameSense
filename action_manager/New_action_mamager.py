from pynput.keyboard import Key, Controller as KeyboardController
import time
from typing import List, Tuple
import subprocess
import os
import ctypes
from ctypes import wintypes

from sympy.physics.units import current


class Controller:
    def __init__(self, train_script_path: str = 'fight_with_boss.py',fight_path: str = 'fight_mobs.py'):
        self.keyboard = KeyboardController()
        self.train_script_path = train_script_path
        self.fight_path = fight_path

        self.special_keys = {
            'SPACE': Key.space,
            'ESC': Key.esc,
            'STOP':Key.esc,
            'ENTER': Key.enter,
            'UP': Key.up,  
            'DOWN': Key.down,  
            'LEFT': Key.left,  
            'RIGHT': Key.right,  
        }

        self.mouse_buttons = {
            'MOUSE1': (0x0002, 0x0004),  # (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP)
            'MOUSE2': (0x0008, 0x0010),  # (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP)
            'MOUSE3': (0x0020, 0x0040)   # (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP)
        }

        self.view_movements = {
            'VIEW_LEFT': (-500, 0),
            'VIEW_RIGHT': (500, 0),
            'VIEW_UP': (0, -500),
            'VIEW_DOWN': (0, 500)
        }

    def execute_training(self):
        try:
            if not self.train_script_path:
                raise ValueError("Training script path not specified")

            if not os.path.exists(self.train_script_path):
                raise FileNotFoundError(f"Training script not found at {self.train_script_path}")

            # Activate instant mobile cheat to enable the training script to run normally (returning to the front of the boss at the beginning of each training session)
            self.keyboard.press('k')
            time.sleep(0.05)
            self.keyboard.release('k')
            time.sleep(1)
            self.keyboard.press('l')

            file_extension = os.path.splitext(self.train_script_path)[1].lower()

            if file_extension == '.py':
                subprocess.Popen(['python', self.train_script_path])
            elif file_extension == '.exe':
                subprocess.Popen([self.train_script_path])
            else:
                raise ValueError(f"Unsupported script type: {file_extension}")

            print(f"Training script launched: {self.train_script_path}")

        except Exception as e:
            print(f"Error executing training script: {str(e)}")

    def execute_fight(self):
        try:
            if not self.fight_path:
                raise ValueError("fight script path not specified")

            if not os.path.exists(self.fight_path):
                raise FileNotFoundError(f"fight script not found at {self.fight_path}")

            file_extension = os.path.splitext(self.fight_path)[1].lower()

            if file_extension == '.py':
                subprocess.Popen(['python', self.fight_path])
            elif file_extension == '.exe':
                subprocess.Popen([self.fight_path])
            else:
                raise ValueError(f"Unsupported script type: {file_extension}")

            print(f"fight script launched: {self.fight_path}")

        except Exception as e:
            print(f"Error executing fight script: {str(e)}")

    def send_mouse_click(self, down_flag, up_flag):
        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))
            ]

        class INPUT(ctypes.Structure):
            _fields_ = [
                ("type", wintypes.DWORD),
                ("mi", MOUSEINPUT)
            ]

        extra = ctypes.pointer(ctypes.c_ulong(0))
        mouse_down = MOUSEINPUT(0, 0, 0, down_flag, 0, extra)
        input_down = INPUT(0, mouse_down)
        ctypes.windll.user32.SendInput(1, ctypes.byref(input_down), ctypes.sizeof(INPUT))
        time.sleep(0.05) 
        mouse_up = MOUSEINPUT(0, 0, 0, up_flag, 0, extra)
        input_up = INPUT(0, mouse_up)
        ctypes.windll.user32.SendInput(1, ctypes.byref(input_up), ctypes.sizeof(INPUT))

    def mouse_click(self, button_key: str):

        try:
            if button_key in self.mouse_buttons:
                down_flag, up_flag = self.mouse_buttons[button_key]
                self.send_mouse_click(down_flag, up_flag)
        except Exception as e:
            print(f"Error executing mouse click {button_key}: {str(e)}")

    def send_mouse_input(self, dx: int, dy: int):

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))
            ]

        class INPUT(ctypes.Structure):
            _fields_ = [
                ("type", wintypes.DWORD),
                ("mi", MOUSEINPUT)
            ]

        extra = ctypes.pointer(ctypes.c_ulong(0))
        mouse_input = MOUSEINPUT(dx, dy, 0, 0x0001, 0, extra)  
        input_event = INPUT(0, mouse_input) 

        ctypes.windll.user32.SendInput(1, ctypes.byref(input_event), ctypes.sizeof(INPUT))

    def move_view(self, direction: str):

        try:
            if direction in self.view_movements:
                dx, dy = self.view_movements[direction]
                self.send_mouse_input(dx, dy)
                #self.mouse.position = (current_x, current_y)
        except Exception as e:
            print(f"Error moving view {direction}: {str(e)}")

    def press_key(self, action: Tuple[str, float]):

        try:
            key, duration = action


            if key == 'TRAIN':
                self.execute_training()
                return

            if key == 'FIGHT':
                self.execute_fight()
                return

            if key in self.view_movements:
                self.move_view(key)
                time.sleep(duration)
                return

            if key in self.mouse_buttons:
                self.mouse_click(key)
                time.sleep(duration)
                return


            if key in self.special_keys:
                k = self.special_keys[key]
            else:
                k = key.lower()
                #print('we act in ', k)

            self.keyboard.press(k)
            #print('we success act in ', k)
            time.sleep(duration)
            self.keyboard.release(k)
            time.sleep(0.05)

        except Exception as e:
            print(f"Error executing key {key}: {str(e)}")

    def execute_action_sequence(self, action_code: List[Tuple[str, float]], delay_between_actions: float = 0.1):

        try:
            for action in action_code:

                self.press_key(action)

                time.sleep(delay_between_actions)

        except Exception as e:
            print(f"Error executing action sequence: {str(e)}")
