import os
import pandas as pd
from mediapipe.python.solutions.holistic import Holistic
from model_utils import get_keypoints, insert_keypoints_sequence
from utils.constants import DATA_PATH, FRAME_ACTIONS_PATH, ROOT_PATH

def create_keypoints(frame_actions,save_path):
  
    data=pd.DataFrame([]) 

    with Holistic() as model_holistic: # fr_act == [sample1,sample2,.....]
        for n_samples, samples_name in enumerate(os.listdir(frame_actions),1): 
            sample_path = os.path.join(frame_actions,samples_name)  #dir\samples
            keypoints_sequence = get_keypoints(model_holistic, sample_path)
            data = insert_keypoints_sequence(data, n_samples, keypoints_sequence) 

