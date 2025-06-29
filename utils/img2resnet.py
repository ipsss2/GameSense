import cv2
import numpy as np
import torch

def process_img_to_obs(img, obs_window=(391,51,1178,677), observation_w=160, observation_h=160):

    x_min, y_min, x_max, y_max = obs_window
    cropped_img = img[y_min:y_max, x_min:x_max]  

    resized_img = cv2.resize(cropped_img, (observation_w, observation_h))

    if resized_img.shape[-1] == 4:  
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGRA2RGB)  
    elif resized_img.shape[-1] == 3:  
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)  

    resized_img = np.array(resized_img, dtype=np.float32) / 255.0

    resized_img = np.transpose(resized_img, (2, 0, 1))  
    obs_tensor = torch.from_numpy(resized_img)  

    obs_tensor = obs_tensor.unsqueeze(0)  

    return obs_tensor
