import os
import numpy as np
import cv2
import pandas as pd
from mediapipe.python.solutions.drawing_utils import DrawingSpec, draw_landmarks
from mediapipe.python.solutions.holistic import (
    FACEMESH_CONTOURS,
    HAND_CONNECTIONS,
    POSE_CONNECTIONS,
)
from typing_extensions import NamedTuple


def create_dir(path):
    # Create the directory if not exists
    if not os.path.exists(path):
        os.mkdir(path)


def dir_exists(path):
    # Verify if exist the directory
    return True if os.path.exists(path) else False


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def there_hand(results: NamedTuple) -> bool:
    return results.left_hand_landmarks or results.right_hand_landmarks


def save_frames(frames, output_dir):
    for num_frame, frame in enumerate(frames):
        frame_path = os.path.join(output_dir, f"{num_frame+1}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA))


def draw_keypoints(image, results):
    """
    Dibuja los keypoints en la imagen
    """
    draw_landmarks(
        image,
        results.face_landmarks,
        FACEMESH_CONTOURS,
        DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
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
        HAND_CONNECTIONS,
        DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
    )
    # Draw right hand connections
    draw_landmarks(
        image,
        results.right_hand_landmarks,
        HAND_CONNECTIONS,
        DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )

# FUNCIONES NECESARIAS PARA "CREATE_KEYPOINT.py"

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

def get_keypoints(model, path):
    kp_sequence = np.array([]) 
    for name_img in os.listdir(path): 
        print(name_img)
        path_img = os.path.join(path, name_img) 
        frame = cv2.imread(path_img) 
        _, results = mediapipe_detection(frame, model)  # Corregido para obtener también los resultados
        kp_frame = extract_keypoints(results) 
        kp_sequence = np.concatenate([kp_sequence, [kp_frame]]) if kp_sequence.size > 0 else np.array([kp_frame])  # Corregido para crear correctamente el array de keypoints
    return kp_sequence 


def insert_keypoints_sequence(df, n_sample: int, kp_seq):
    
    for frame, keypoints in enumerate(kp_seq):
        data = {'sample': n_sample, 'frame': frame + 1, 'keypoints': [keypoints]}  # Crear diccionario de datos
        df_keypoints = pd.DataFrame(data)  # Convertir datos en DataFrame
        df = pd.concat([df, df_keypoints], ignore_index=True)  # Concatenar al DataFrame principal
    return df


