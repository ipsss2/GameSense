# import time
# from action_manager.New_action_mamager import Controller
# from utils.bar_detector import detector
# import inspect

# fight_controller = Controller()

# def func_to_str(func):
#     return inspect.getsource(func)


# def str_to_func(func_str):
#     try:

#         func_str = func_str.strip()
#         if not func_str.startswith('def '):
#             raise ValueError("Function string must start with 'def'")

#         namespace = {}

#         print("Executing function string...")
#         exec(func_str, namespace)
#         print("Namespace keys:", list(namespace.keys()))


#         func_name = func_str[4:func_str.index('(')].strip()
#         print(f"Extracted function name: {func_name}")

#         if func_name not in namespace:
#             raise ValueError(f"Function {func_name} not found in namespace")

#         return namespace[func_name]

#     except Exception as e:
#         print(f"Debug - Function string starting characters: {func_str[:50]}...")
#         raise ValueError(f"Cannot convert string to function: {str(e)}")
    

# def execute_fight():

#     fight_controller.press_key(('MOUSE3', 0.05))
#     state, _ = detector.get_status()
#     init_blood = state['blood_percentage']
#     print('Initial blood percentage:', init_blood)

#     def check_blood_change():
#         state1, _ = detector.get_status()
#         current_blood = state1['blood_percentage']
#         for _ in range(20):
#             time.sleep(1)
#             state2, _ = detector.get_status()
#             new_blood = state2['blood_percentage']
#             if new_blood < current_blood-0.05:
#                 return True
#         fight_controller.press_key(('MOUSE3', 0.05))
#         return False

#     fight_num = 0
#     while True:
#         for _ in range(6):
#             fight_controller.press_key(('MOUSE1', 0.05))
#             time.sleep(0.1)

#         fight_controller.press_key(('SPACE', 0.1))

#         fight_num += 1
#         print('fight', fight_num)
#         if fight_num % 4 == 0:
#             if check_blood_change():
#                 continue
#             else:
#                 return


# fight_mobs.py

import os
import json
import inspect
import time
import traceback
import multiprocessing

# 导入你的自定义模块
from action_manager.New_action_mamager import Controller
from utils.bar_detector import detector
from utils.frame_recorder import ScreenshotManager
from agent.fast_module_trainner_agent.fight import opt_action 

JSON_FILE = "fight_logic.json"

LOCK_FILE = "fight_logic.json.lock" 
MAX_VERSIONS = 0
SAVE_DIR_BASE = "video_picture_save" 


INITIAL_FIGHT_LOGIC = """
def execute_fight(fight_controller, detector, time):
    \"\"\"
    Version 1: Initial fighting logic.
    - Clicks MOUSE3 to lock on.
    - Attacks 6 times, then dodges.
    - Repeats this loop 4 times.
    - After 4 loops, checks if the enemy is still attacking.
    \"\"\"
    print("Executing fight logic: Version 1")

    fight_controller.press_key(('MOUSE3', 0.05))
    state, _ = detector.get_status()
    init_blood = state['blood_percentage']
    print('Initial blood percentage:', init_blood)

    def check_blood_change():
        state1, _ = detector.get_status()
        current_blood = state1['blood_percentage']
        for _ in range(20):
            time.sleep(1)
            state2, _ = detector.get_status()
            new_blood = state2['blood_percentage']
            if new_blood < current_blood-0.05:
                return True
        fight_controller.press_key(('MOUSE3', 0.05)) # Lock-on target.
        return False
    fight_num = 0
    while True:
        for _ in range(4):
            fight_controller.press_key(('MOUSE1', 0.05)) #Light attack
            time.sleep(0.1)
        fight_controller.press_key(('MOUSE2', 0.05)) #Heavy attack
        fight_controller.press_key(('SPACE', 0.1))

        fight_num += 1
        print('fight', fight_num)
        if fight_num % 4 == 0:
            if check_blood_change():
                continue
            else:
                return
"""


def func_to_str(func):
    return inspect.getsource(func)

def str_to_func(func_str, context):
    try:
        func_str = func_str.strip()
        if not func_str.startswith('def '):
            raise ValueError("Function string must start with 'def'")

        func_name_start = 4
        func_name_end = func_str.find('(')
        if func_name_end == -1:
            raise ValueError("Cannot find function signature '(...'")
        func_name = func_str[func_name_start:func_name_end].strip()


        local_namespace = {}
        exec(func_str, context, local_namespace)

        if func_name not in local_namespace:
            raise ValueError(f"Function '{func_name}' not found after exec.")

        return local_namespace[func_name]

    except Exception as e:
        print(f"Error converting string to function: {e}")
        print(f"Problematic code string:\n---\n{func_str}\n---")
        raise


def run_llm_optimization_process(json_filepath, lock_filepath, last_func_str, image_folder_path):

    print(f"\n--- [Optimizer-Process {os.getpid()}] Started ---")
    
    new_function_str, suggestions = opt_action(
        fight_logic_code=last_func_str,
        screenshot_dir=image_folder_path
    )
    if new_function_str is None:
        print("[Optimizer-Process] Code optimization failed. The process will now exit without updating the logic.")
        return

    print(f"[Optimizer-Process] Waiting to acquire lock for '{json_filepath}'...")
    while os.path.exists(lock_filepath):
        time.sleep(1)
    try:
        with open(lock_filepath, 'w') as f:
            f.write(str(os.getpid()))
        print("[Optimizer-Process] Lock acquired.")

        if not os.path.exists(json_filepath):
            functions_list = [INITIAL_FIGHT_LOGIC]
        else:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                functions_list = json.load(f)
        
        functions_list.append(new_function_str)
        
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(functions_list, f, indent=4, ensure_ascii=False)
        
        print(f"[Optimizer-Process] New logic (Version {len(functions_list)}) saved to '{json_filepath}'.")

    except Exception as e:
        print(f"[Optimizer-Process] CRITICAL ERROR during file write: {e}")
    finally:
        if os.path.exists(lock_filepath):
            os.remove(lock_filepath)
            print("[Optimizer-Process] Lock released.")
    
    print(f"--- [Optimizer-Process {os.getpid()}] Finished ---")


# --- 主执行逻辑 ---
def main():
    """
    脚本主入口点。
    """
    print("="*50)
    print(f"Fight Mobs Management Script Initialized (PID: {os.getpid()})")
    print("="*50)

    if os.path.exists(LOCK_FILE):
        print("Lock file exists. Another optimization process may be running. Waiting...")
    
        # time.sleep(5) 
        # if os.path.exists(LOCK_FILE):
        #     print("Lock file still exists. Exiting to prevent conflict.")
        #     return
        pass

    try:
        execution_context = {
            "fight_controller": Controller(),
            "detector": detector,
            "time": time
        }
    except Exception as e:
        print(f"Error initializing context (Controller, detector): {e}")
        return

    if not os.path.exists(JSON_FILE):
        print(f"JSON file not found. Using initial logic for the first run.")
        functions_list = [INITIAL_FIGHT_LOGIC]
    else:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            functions_list = json.load(f)

    current_version_count = len(functions_list)
    print(f"Found {current_version_count} versions of fight logic. Max versions set to {MAX_VERSIONS}.")

    if not functions_list:
        print("CRITICAL: JSON file is empty. Re-initializing with base logic.")
        functions_list = [INITIAL_FIGHT_LOGIC]
        current_version_count = 1

    if current_version_count >= MAX_VERSIONS:
        print("\n[Mode: Stable] Optimization limit reached. Executing the final version.")
        final_func_str = functions_list[-1]
        try:
            fight_func = str_to_func(final_func_str, execution_context)
            fight_func(**execution_context)
            print("Final function executed successfully.")
        except Exception as e:
            print(f"CRITICAL: Final function failed to execute: {e}")
            traceback.print_exc()
    else:
        print(f"\n[Mode: Optimization] Running version {current_version_count}.")
        
        run_save_dir = os.path.join(SAVE_DIR_BASE, f"run_v{current_version_count}")
        recorder = ScreenshotManager(save_dir=run_save_dir, frame_rate=10, region=(1120, 630, 1600, 900), use_coordinates=True)
        recorder.start_recording()

        latest_func_str = functions_list[-1]
        execution_successful = False
        try:
            print(f"\nAttempting to execute logic (Version {current_version_count})...")
            fight_func = str_to_func(latest_func_str, execution_context)
            fight_func(**execution_context)
            print(f"Version {current_version_count} executed successfully.")
            execution_successful = True

        except Exception as e:
            print(f"\nERROR: Execution of Version {current_version_count} failed: {e}")
            traceback.print_exc()
            
            print("Rolling back to the previous version.")
            functions_list.pop()
            while os.path.exists(LOCK_FILE): time.sleep(0.5)
            with open(LOCK_FILE, 'w') as f: f.write(str(os.getpid()))
            with open(JSON_FILE, 'w', encoding='utf-8') as f: json.dump(functions_list, f, indent=4)
            os.remove(LOCK_FILE)
            print("Failed version removed from JSON.")

        finally:
            recorder.stop_recording()
            
            if execution_successful:
                print("\nSpawning background process for LLM optimization...")
                p = multiprocessing.Process(
                    target=run_llm_optimization_process,
                    args=(JSON_FILE, LOCK_FILE, latest_func_str, run_save_dir)
                )
                p.start() 
                print("Optimization process launched in the background.")

    print("\nMain fight script finished its task and is now exiting.")
    print("="*50)


if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main()