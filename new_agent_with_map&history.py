from collections import deque
import asyncio

from executing.executing import attr_names_match

from utils.bar_detector import detector
from utils.picture2chat import process_image_to_content,make_pic_content
from game_info.BMW_info import BMW_info
from agent.player_agent.task_planner import new_task_planner
from agent.player_agent.Env_gather import get_env_info
from agent.player_agent.action_build import action_builder
from agent.player_agent.Self_Reflection import new_action_reflection,new_task_reflection
from agent.player_agent.history_evaluate_summary import history_evaluate_summary
from agent.player_agent.Map_evaluator import single_map_eval,overall_eval
from action_manager.New_action_mamager import Controller
import logging
import time
import os
import json
from datetime import datetime
from memory.RAG import RAGDatabase

print('init program')


class Logger:
    def __init__(self, log_dir="logs_step"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"agent_log_{timestamp}.txt")

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_step(self, step_data):
        try:
            formatted_data = json.dumps(step_data, ensure_ascii=False, indent=2)
            self.logger.info(f"\n=== Step {step_data.get('step', 'N/A')} ===\n{formatted_data}")
        except Exception as e:
            self.logger.error(f"Error logging step data: {str(e)}")

    def log_error(self, error_msg):
        self.logger.error(error_msg)


def split_control(control,cut=4.0):
    key, duration = control

    if key not in ['W', 'A', 'S', 'D'] or duration <= cut:
        return [(key, duration)]


    full_parts = int(duration // cut)  
    remainder = duration % cut 

    result = []

    for _ in range(full_parts):
        result.append((key, cut))

    if remainder > 0:
        result.append((key, remainder))

    print('移动指令拆分为了：', result)
    return result

class RPGAgent:
    def __init__(self):

        self.frame_capturer = detector  
        self.game_info = BMW_info
  
        self.state_analyzer = get_env_info  

        #2 map reflection, map summary, task reflection, task summary, action reflection
        # if ana_map is true, we need analysis the information of maps.
        self.single_map_summary = single_map_eval
        self.overall_map_summary = overall_eval
        #
        self.self_reflection_task = new_task_reflection  
        self.history_evaluate_summary = history_evaluate_summary
        self.self_reflection_action = new_action_reflection

        self.task_inference = new_task_planner
        self.action_builder = action_builder

        self.controller = Controller()
        #self.memory_task = RAGDatabase(fields_to_embed=['env_info'],storage_dir='memory/task')  
        self.memory_action = RAGDatabase(fields_to_embed=['action_name_description'],storage_dir='memory/action')
        #self.memory_difficulty = RAGDatabase(fields_to_embed=['difficulty_ana'],storage_dir='/memory/difficulty')
        self.memory_top_k = 3
        self.start_rag=5


        self.task_history = deque(maxlen=8)
        self.action_history = deque(maxlen=8)
        self.map_history = deque(maxlen=7)
        self.task_action_frame=[]


        self.ana_map = False 

        self.logger = Logger()

    def queue_to_str(self,task_history):
        if task_history is None or len(task_history) == 0:
            return ""
        return "\n".join(str(d) for d in task_history)

    def get_map(self):
        i = 0
        while True:
            i += 1
            self.controller.press_key(('M', 0.05))
            # time.sleep(0.2)
            current_state, current_frame = self.frame_capturer.get_status()
            print('try to enter map in time ', i)
            if current_state['mana_percentage'] < 0.1:
                time.sleep(1.5)
                cs, current_frame = self.frame_capturer.get_status()
                if cs['mana_percentage'] < 0.1:
                    print('we enter in the map')
                    self.controller.press_key(('ESC', 0.1))
                    return current_frame

    async def game_loop(self):
        token_usage=0
        step=0

        while True:
            print('******************* begin game play **********************')
            step += 1
            step_log = {
                'step': step,
                'token_usage': token_usage,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            }


            current_step_memory_task_history = {}

            current_step_memory_action_history = {}
            current_step_memory_task_history['step'] = step
            current_step_memory_action_history['step'] = step

 
            current_state,current_frame = self.frame_capturer.get_status()
            current_frame=process_image_to_content(current_frame)
            print('We capture a game frame, the size of it is: %.2f kb. Now in step: %d'%(len(current_frame)/1024,step))
            #step_first_map=self.map_history[-1]
            #ENV_frames=[current_frame,step_first_map]

    
            start_time=time.time()
            env_analysis_result = self.state_analyzer(current_frame,self.game_info)
            env_analysis = env_analysis_result['response']
            end_time = time.time()
            print('the time we use to get reflections and summaries is %.2f.'%((end_time-start_time)))



            if 'Gameplay Screen' in env_analysis:
                print('we are playing the game, needing to use the information of maps, change ana_map to True, step num is:',step)
                self.ana_map = True
                current_map=process_image_to_content(self.get_map(),jpeg_quality=65)
                self.map_history.append(current_map)
                print('we get latest map and add it in history map deque. The size of current map is %.2f kb'%(len(current_map)/1024))
            else:
                print('we are in the Ui screen, ignore the map info, and change ana_map to False')
                self.ana_map = False
    
            token_usage += env_analysis_result['token_use']
            del current_state['boss_percentage']
            current_env_state_info = env_analysis + '\n player state is: \n' +str(current_state)
            print(current_env_state_info)

            step_log['env_analysis'] = current_env_state_info
            step_log['env_token'] =  env_analysis_result['token_use']

            current_step_memory_task_history['env_info'] = current_env_state_info

            if step > 1:
                if self.ana_map:
                    latest_map=self.map_history[-1]
                    current_info_single_map=[current_frame,latest_map]
                    maps_history_eval=list(self.map_history)
                    print('we need to summary the change of maps history, their length is %d'%(len(maps_history_eval)))


                    pass_step_memory_task_history = self.task_history[-1]
                    print('need to reflect on the pass step task: \n %s' % (str(pass_step_memory_task_history)))
                    pass_step_memory_action_history = self.action_history[-1]
                    print('need to reflect on the pass step action design: \n %s' % (str(pass_step_memory_action_history)))

           
                    pass_task_frames=self.task_action_frame # frame
                    maps_task_eval = [self.map_history[-2], latest_map] # map
                    pass_task_info = pass_step_memory_task_history['reason_and_task'] # task
                    pass_env_info = pass_step_memory_task_history['env_info'] # pass env info
                    current_env_info = current_step_memory_task_history['env_info'] # current env info
                    #pass_action_list = pass_step_memory_action_history['action_list']
                    pass_action_list_code = pass_step_memory_action_history['action_list_code']

                    history_str = self.queue_to_str(self.task_history)


                    pass_action_frames=pass_task_frames[:-1]

                    start_time=time.time()
                    # 并行反思
                    (map_single_ref,token_m_s), (map_overall_summary,token_m_a), (pass_task_reflection, token_t_r), (pass_task_history_summary,token_t_h), (pass_action_reflection,token_a_r)  =await asyncio.gather(
                        asyncio.to_thread(self.single_map_summary, current_info_single_map,self.game_info),
                        asyncio.to_thread(self.overall_map_summary, maps_history_eval,self.game_info),
                        asyncio.to_thread(self.self_reflection_task, pass_task_frames, maps_task_eval, pass_task_info, pass_env_info,current_env_info,pass_action_list_code),
                        asyncio.to_thread(self.history_evaluate_summary, history_str),
                        asyncio.to_thread(self.self_reflection_action, pass_action_frames, pass_task_info, pass_action_list_code,4)
                    )


                    end_time=time.time()
                    token_usage_re = token_m_s + token_m_a + token_t_r + token_t_h + token_a_r
                    token_usage += token_usage_re
                    print('the time we use to get reflections and summaries is %.2f and we use token number is %d.'%((end_time-start_time),token_usage_re))

                    #写进log
                    step_log['reflection_token']=token_usage_re
                    step_log['map_reflection_summary']=map_single_ref+'\n'+map_overall_summary
                    step_log['task_history_summary']=pass_task_history_summary
                    step_log['pass_task_reflection']=pass_task_reflection
                    step_log['pass_action_reflection']=str(pass_action_reflection)

                    print(step_log['pass_action_reflection'])


                    self.task_action_frame=[]
                    print('we clear the task and action frames, its length now is:',len(self.task_action_frame))

   
                    self.task_history[-1]['task_reflection'] = pass_task_reflection
                    self.action_history[-1]['action_reflection'] = str(pass_action_reflection)

                    self.memory_action.add_documents(pass_action_reflection)
                    print('we save the reflection data, for task reflection we save it in self.task_history, for action reflection we save it in self.action_history and RAG database')

                else:

                    latest_map = None

                    pass_step_memory_task_history = self.task_history[-1]
                    print('need to reflect on the pass step task: \n %s' % (str(pass_step_memory_task_history)))
                    pass_step_memory_action_history = self.action_history[-1]
                    print('need to reflect on the pass step action design: \n %s' % (str(pass_step_memory_action_history)))

                    pass_task_frames = self.task_action_frame  # frame
                    maps_task_eval = None
                    pass_task_info = pass_step_memory_task_history['reason_and_task']  # task
                    pass_env_info = pass_step_memory_task_history['env_info']  # pass env info
                    current_env_info = current_step_memory_task_history['env_info']  # current env info
                    #pass_action_list = pass_step_memory_action_history['action_list']
                    pass_action_list_code = pass_step_memory_action_history['action_list_code']

                    history_str = self.queue_to_str(self.task_history)

                    pass_action_frames = pass_task_frames[:-1]

                    start_time = time.time()

                    (pass_task_reflection, token_t_r), (pass_task_history_summary, token_t_h), (pass_action_reflection, token_a_r) = await asyncio.gather(
                        asyncio.to_thread(self.self_reflection_task, pass_task_frames, maps_task_eval, pass_task_info,
                                          pass_env_info, current_env_info, pass_action_list_code),
                        asyncio.to_thread(self.history_evaluate_summary, history_str),
                        asyncio.to_thread(self.self_reflection_action, pass_action_frames, pass_task_info, pass_action_list_code,4)
                    )


                    map_single_ref, map_overall_summary=None,None
                    end_time = time.time()
                    token_usage_re = token_t_r + token_t_h + token_a_r
                    token_usage += token_usage_re
                    print('the time we use to get reflections and summaries is %.2f and we use token number is %d.' % ((end_time - start_time), token_usage_re))

                    step_log['reflection_token'] = token_usage_re
                    step_log['task_history_summary'] = pass_task_history_summary
                    step_log['pass_task_reflection'] = pass_task_reflection
                    step_log['pass_action_reflection'] = str(pass_action_reflection)


                    print(step_log['pass_action_reflection'])

                    self.task_action_frame = []
                    print('we clear the task and action frames, its length now is:', len(self.task_action_frame))

                    self.task_history[-1]['task_reflection'] = pass_task_reflection
                    self.action_history[-1]['action_reflection'] = str(pass_action_reflection)

                    self.memory_action.add_document(pass_action_reflection)
                    print('we save the reflection data, for task reflection we save it in self.task_history, for action reflection we save it in self.action_history and RAG database')

            else:
                pass_task_history_summary, pass_task_reflection,map_overall_summary=None,None,None
                if self.ana_map:
                    print('we need analysis map info')
                    start_time=time.time()
                    latest_map = self.map_history[-1]
                    current_info_single_map = [current_frame, latest_map]
                    map_single_ref, token_map_s=self.single_map_summary(current_info_single_map,self.game_info)
                    token_usage+=token_map_s
                    end_time = time.time()
                    print('we need to summary the map, time use is %.2f, their length is %d'%((end_time-start_time),token_map_s))
                    step_log['reflection_token'] = token_map_s
                    step_log['map_reflection_summary'] = map_single_ref
                    print(step_log['map_reflection_summary'])
                else:
                    latest_map = None
                    map_single_ref = None



            print('we use the following task history: \n task history summary: \n %s; \n task reflection: \n %s;'%(pass_task_history_summary,pass_task_reflection))
            print('the map_single_ref, map_overall_summary are:',map_single_ref, map_overall_summary)

            start_time=time.time()
            response=self.task_inference(latest_map,current_frame,pass_task_history_summary,pass_task_reflection,map_single_ref, map_overall_summary,current_step_memory_task_history['env_info'],self.game_info,step)
            end_time = time.time()
            reason_and_task=response['reason_task']
            token_usage += response['token_use']
            action_plan = response['action_plan']
            print('Task and Reason:\n',reason_and_task)
            print('Action plan:\n',action_plan)
            print('Task inference time: %.2f seconds'%(end_time-start_time))

            step_log['reason_and_task'] = reason_and_task
            step_log['action_list'] = action_plan
            step_log['task_token'] = response['token_use']


            current_step_memory_task_history['reason_and_task'] = reason_and_task
            current_step_memory_action_history['task_info'] = reason_and_task
            #current_step_memory_action_history['action_list'] = action_plan
            print('current step memory (both task and action) has saved: \n reason_and_task(current_step_memory_task_history)\ntask_info(current_step_memory_action_history)')

            start_time = time.time()
            RAG4actions=None
            if step > self.start_rag:
                search_key = action_plan
                RAG4actions = self.memory_action.search(query=search_key, k=self.memory_top_k)
                print('through rag we find follow actions\n',RAG4actions)

            action_list_code, token_usage4action = self.action_builder(RAG4actions, current_frame, reason_and_task, self.game_info,action_plan)
            end_time = time.time()
            print('Action plan time: %.2f seconds'%(end_time-start_time))
            token_usage += token_usage4action

            step_log['action_list_code'] = str(action_list_code)
            step_log['action_token'] = token_usage4action
            #print('action list LLM design is:\n',action_list)
            print('action code LLM design is:\n',str(action_list_code))

            #current_step_memory_action_history['action_list'] = action_list
            current_step_memory_action_history['action_list_code'] = action_list_code
            print('current step memory (action) has saved: action_list_code')


            action_code=[]
            for item in action_list_code:
                action_code_s=item["action_code"]
                for code in action_code_s:
                    action_code.append(code)
            print('we plan action number is %d, code number is %d'%(len(action_list_code),len(action_code)))
            current_frame = process_image_to_content(self.frame_capturer.get_frame(),jpeg_quality=30)
            self.task_action_frame=[]
            self.task_action_frame.append(current_frame)
            pic_len=0
            for act in action_code:
                split_actions = split_control(act)
                for split_action in split_actions:
                    pic_len+=1
                    self.controller.press_key(split_action)
                    print('we are act in ', split_action)
                    current_frame = process_image_to_content(self.frame_capturer.get_frame(),jpeg_quality=30)
                    if pic_len>5:
                        print('picture num can not be more than 5, now is: ',pic_len)
                        continue
                    else:
                        self.task_action_frame.append(current_frame)

            for i in range(1):
                current_frame = process_image_to_content(self.frame_capturer.get_frame(),jpeg_quality=25)
                self.task_action_frame.append(current_frame)
                time.sleep(0.5)
            length = 0
            for frame in self.task_action_frame:
                length += len(frame)/1024
            print('the size of task and action frames are:',length/1024)
            print('task and action frame len is: ',len(self.task_action_frame))

            self.task_history.append(current_step_memory_task_history)
            self.action_history.append(current_step_memory_action_history)
            self.logger.log_step(step_log)
            print('we have save one step memory without reflection')


    def start(self):
        """启动游戏循环"""
        asyncio.run(self.game_loop())


if __name__ == "__main__":
    agent = RPGAgent()
    agent.start()