import os
import pandas as pd
from mediapipe.python.solutions.holistic import Holistic
from utils.model_utils import get_keypoints, insert_keypoints_sequence
from utils.constants import DATA_PATH, FRAME_ACTIONS_PATH, ROOT_PATH,PROCESSED_DATA_PATH

def create_keypoints(frames_path,save_path): #(folder_samples,guardar[processed]) de aqui se toma las varibables de (line 30)
  
    data=pd.DataFrame([]) # estructura bidimencional en forma de filas,columnas

    "Aqui asignamos el nombre al modelo 'holistic()' como 'model_holistic'"
    "'.listdir' devuelve una lista de los nombres de lso archivos en fram_actions"
    "'enumerate()', devuelve 2 valores tanto el indice como la caperta "
    with Holistic() as model_holistic: # fr_act == [sample1,sample2,.....]
        for n_samples, samples_name in enumerate(os.listdir(frames_path),1): 
            sample_path = os.path.join(frames_path,samples_name)  #dir\samples  sample_path == 'camino de la muestra'
            keypoints_sequence = get_keypoints(model_holistic, sample_path) #(holistic(), fotos==> samples)
            "me vota una lsita o array con toodos los kp de cada frame"
            data = insert_keypoints_sequence(data, n_samples, keypoints_sequence) 

    data.to_hdf(save_path, key='data', model='w')

#INICIALIZACION DE LA ESTRACCION DE KEYPOINTS 

if __name__=="__main__": #********************************************

    # hallamos la ruta en donde esta la palabra creada 
    #tambien se dalcar la variable words_path
    words_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH)

    # utilizamos .listdir para crear una lista de los datos de words_path ==> [pal1, pal2,....]
    for word_name in os.listdir(words_path):   
        word_path= os.path.join(words_path, word_name)  #aqui se une la un archivo de la lista creada de words_path
        hdf_path= os.path.join(PROCESSED_DATA_PATH, f'{word_name}.h5') #ruteamos en donde se va guardar, f'{word_name}.h5 == perro.h5 
        print(f'En proceso de creacion de Keypoints de "{word_name}".h5') #perro.h5 en proceso....m, esto se imprimira en pantalla
        create_keypoints(word_path,hdf_path)
        print(f"Keypoints creados!")