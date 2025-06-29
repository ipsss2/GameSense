from utils.ask_model import MultiTurnChatService
from utils.picture2chat import make_pic_content
from utils.video_capture import ScreenCapture


from agent.prompt_bank.prompts_bmw import system_prompt,user_prompt
from action_manager.base_action2keybord import SimpleGameController
import time
import pyautogui
import json

def log_response_to_file(response, filename="responses.txt"):
    """
    Append a response to a text file.

    :param response: str, the response to log
    :param filename: str, the name of the file to append to
    """
    try:
        with open(filename, "a") as file:
            file.write(response + "\n")
    except Exception as e:
        print(f"Error writing to file: {e}")

game_controller = SimpleGameController()
bmw_agent = MultiTurnChatService(system_prompt=system_prompt)

attack_5=[('LIGHT_ATTACK',0.05),('LIGHT_ATTACK',0.05),('LIGHT_ATTACK',0.05),('LIGHT_ATTACK',0.05),('LIGHT_ATTACK',0.05)]



screen_width, screen_height = pyautogui.size()

capture_width = 1600
capture_height = 900

left = (screen_width - capture_width) // 2
top = (screen_height - capture_height) // 2
capture = ScreenCapture(region=(left, top, capture_width,capture_height), output_folder="video_picture_save/picture")


def run(memory_length=50,max_step=100):
    image_base64_org = capture.capture_image(convert_to_base64=True)


    for i in range(max_step):
        bmw_agent.trim_chat_history(memory_length)
        print("Token usage:", bmw_agent.get_token_usage())

        content = make_pic_content(text=user_prompt, base64_encode=image_base64_org)

        is_correct = False
        while not is_correct:
            #print(content)
            response = bmw_agent.chat({"role": "user", "content": content})
            log_response_to_file(response)
            print(response)
            is_correct, message = check_output_format(response)

            if not is_correct:
                content= 'your output format is wrong' + message +'please give me the correct format without any explain'
                print("Invalid format, trying again:", message)

            else:
                print("Response is correct:", message)


        response_dict = json.loads(response)

        action_list = response_dict["action_list"]
        for action_item in action_list:
            action, duration = action_item
            game_controller.execute_action(action, duration)

        image_base64_after = capture.capture_image(convert_to_base64=True)
        #content_re = make_pic_content(text=user_prompt, base64_encode=image_base64_after)


    time.sleep(2)

run()










