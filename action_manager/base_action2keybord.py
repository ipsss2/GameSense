from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController
import time
from enum import Enum
import threading





class Sub_GameAction(Enum):

    MOVE_FORWARD = 'w'
    MOVE_BACKWARD = 's'
    MOVE_LEFT = 'a'
    MOVE_RIGHT = 'd'


    DODGE = Key.space

    DRINK_RESTORE_BLOOD_VOLUME = 'r'


    BODY_FIXING = '1'

    LIGHT_ATTACK = Button.left

  
    HEAVY_ATTACK = Button.right


    STOP=Key.esc


class Controller:
    pass

class SimpleGameController:
    def __init__(self):
        self.keyboard = KeyboardController()
        self.mouse = MouseController()
        self._active_actions = set()
        self.valid_actions = set(action.name for action in Sub_GameAction)

    def is_valid_action(self, action_name: str) -> bool:
        return action_name.upper() in self.valid_actions


    def execute_action(self, action_names: list, duration: float = 0.1):
        actions = [Sub_GameAction[action_name.upper()] for action_name in action_names]

        for action in actions:
            if isinstance(action.value, Button):
                self.mouse.press(action.value)
            else:
                self.keyboard.press(action.value)

        time.sleep(duration)

        for action in actions:
            if isinstance(action.value, Button):
                self.mouse.release(action.value)
            else:
                self.keyboard.release(action.value)

    def execute_sequence(self, action_sequence: list, duration_per_action: float = 0.5):

        for action_names in action_sequence:
            if isinstance(action_names, list):
                self.execute_action(action_names, duration_per_action)


