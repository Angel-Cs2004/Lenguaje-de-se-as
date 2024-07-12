import os
import pandas as pd
from mediapipe.python.solutions.holistic import Holistic
from utils.model_utils import get_keypoints, insert_keypoints_sequence, extract_keypoints 
from utils.constants import DATA_PATH, FRAME_ACTIONS_PATH, ROOT_PATH,PROCESSED_DATA_PATH

def create_new_keypoints(path):
    data = pd.DataFrame([])
    name_word = os.path.basename(path)
    save_path = os.path.join(PROCESSED_DATA_PATH, f'{name_word}.h5')
    if f"{name_word}.h5" in os.listdir(PROCESSED_DATA_PATH):
        return False
    else:
        with Holistic() as model_holistic:
            for n_sample, sample_name in enumerate(os.listdir(path), 1):
                sample_path = os.path.join(path, sample_name)  
                kp_sq = get_keypoints(model_holistic, sample_path) 
                data = insert_keypoints_sequence(data, n_sample, kp_sq)  
        
        data.to_hdf(save_path, key='data', mode='w')    
        return True

path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, "perro")  # Ruta a la carpeta con imágenes
print(create_new_keypoints(path))

















# def create_keypoints(frames_path,save_path): # me bota hasta fA
#     data=pd.DataFrame([]) # estructura bidimencional en forma de filas,columnas

#     "Aqui asignamos el nombre al modelo 'holistic()' como 'model_holistic'"
#     "'.listdir' devuelve una lista de los nombres de lso archivos en fram_actions"
#     "'enumerate()', devuelve 2 valores tanto el indice como la caperta "

#     with Holistic() as model_holistic: # fr_act == [sample1,sample2,.....]
#         for n_samples, samples_name in enumerate(os.listdir(frames_path),1): 
#             sample_path = os.path.join(frames_path,samples_name)  #dir\samples  sample_path == 'camino de la muestra'
#             keypoints_sequence = get_keypoints(model_holistic, sample_path) #(holistic(), fotos==> samples)
#             "me vota una lista o array con todos los kp de cada frame"
#             data = insert_keypoints_sequence(data, n_samples, keypoints_sequence) 
#     data.to_hdf(save_path, key='data', mode='w')
#     return True