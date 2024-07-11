import os
import numpy as np
import cv2
import pandas as pd
from mediapipe.python.solutions.drawing_utils import DrawingSpec, draw_landmarks # se importa una funcion(draw...) y una clase(Drawing..), mas a fondo en (46,54,62)
from mediapipe.python.solutions.holistic import (
    FACEMESH_CONTOURS,
    HAND_CONNECTIONS,
    POSE_CONNECTIONS,
)
from typing_extensions import NamedTuple

def dir_exists(path):
    # Verify if exist the directory
    return True if os.path.exists(path) else False

def create_dir(path):
    # Create the directory if not exists
    if not os.path.exists(path):
        os.mkdir(path)


def mediapipe_detection(image, model): # (frame,holistic_model) , se ve en capture_samples
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #image = "imagen ya covertida"
    image.flags.writeable = False # tacho a la image de no ser modificable
    results = model.process(image) #aqui "model" es llamado como "holistic()" en capture_samples(49 line)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results


def there_hand(results: NamedTuple) -> bool:
    return results.left_hand_landmarks or results.right_hand_landmarks # v o f // f o v


def save_frames(frames, output_dir):
    for num_frame, frame in enumerate(frames):
        frame_path = os.path.join(output_dir, f"{num_frame+1}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA))


def draw_keypoints(image, results):
    """
    Dibuja los keypoints en la imagen
    """
    draw_landmarks( # esta funcion "draw_landmarks" se utiliza para indicar donde se dibuja las marcas en el frame
        image, #arg
        results.face_landmarks, #arg los puntos detectados en cara
        FACEMESH_CONTOURS, #arg las conexiones entre la cara
        DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1), #arg color del las marcas de la cara
        DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1), #arg color de las conexiones 
    )
    # Draw pose connections
    draw_landmarks(
        image,
        results.pose_landmarks,
        POSE_CONNECTIONS, 
        DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
    )
    # Draw left hand connections
    draw_landmarks(
        image,
        results.left_hand_landmarks,
        HAND_CONNECTIONS,# deteccion de mano izquierda 
        DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
    )
    # Draw right hand connections
    draw_landmarks(
        image,
        results.right_hand_landmarks, 
        HAND_CONNECTIONS, #deteccion de mano derecha
        DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )

# FUNCIONES NECESARIAS PARA "CREATE_KEYPOINT.py"

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def get_keypoints(model,path): #(holistic,samples path==> fotos en carpeta)
    kp_sequence = np.array([]) #estoy creando una lista de elementos comunes, para samples_path
    for name_img in os.listdir(path): #name_img , recorrera todas la img de sample_path
        path_img = os.path.join(path,name_img) # ruteamos samples_path/ name_img
        frame = cv2.imread(path_img) #leemos o process la ruta de la img directa
        results = mediapipe_detection(frame, model)# aquise procesa la img y es trasformada a matriz  (img, process)
       #ya se extrajo los kp , tenemos todos
        kp_frame = extract_keypoints(results) 
        # el ".concatenate"
        kp_sequence= np.concatenate([kp_sequence, [kp_frame]] if kp_sequence.size>0 else [[kp_frame]])
    return kp_sequence #me vota una array con todos los kp 

def insert_keypoints_sequence(df, n_sample: int, kp_seq):
    #                        (data,n_sample=int, kp_seq== kp de una mig de un models)
    '''
    ### INSERTA LOS KEYPOINTS DE LA MUESTRA AL DATAFRAME
    Retorna el mismo DataFrame pero con los keypoints de la muestra agregados
    '''
    for frame, keypoints in enumerate(kp_seq):
        data = {'sample': n_sample, 'frame': frame + 1,'keypoints': [keypoints]} #sample: 1, frame: 2, kepoypoints: 01.25
        df_keypoints = pd.DataFrame(data) #lo ponemos en cuadro
        df = pd.concat([df, df_keypoints])#lo agregamos a data(cre_kp) para que este en la array
    return df