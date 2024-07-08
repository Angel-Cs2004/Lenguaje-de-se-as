import os

import cv2
import numpy as np

# import pygetwindow as pw
from mediapipe.python.solutions.holistic import Holistic

from utils.constants import BLUE_COLOR, FONT, FONT_POS, FONT_SIZE, RED_COLOR # de
from utils.model_utils import (
    create_dir,
    draw_keypoints,
    mediapipe_detection,
    save_frames,
    there_hand,
)


def handle_action(action, *args, **kwargs):
    try:
        action(*args, **kwargs)
    except Exception as e:
        print(f"{type(e).__name__}: {str(e)}")
        return None


"""
def bring_window_to_front(window_name):
    try:
        window = pw.getWindowsWithTitle(window_name)[0]
        window.activate()
    except IndexError:
        print(f"No se encontró la ventana con el nombre: {window_name}")
    except pw.PyGetWindowException:
        pass
"""


def model_capture_sign(path, margin_frame=2, min_cant_frames=5, video_device=0):
    create_dir(path)
  #VARIABLES 
    cant_samples_exists = len(os.listdir(path))
    quantity_sample = cant_samples_exists
    new_sample_requested = False
    count_frame = 0
    capturing = True #usado en (64)
    frames = [] #lista donde se almacenan los frames

    with Holistic() as holistic_model:  # de aqui optenemos la variable 
        video = cv2.VideoCapture(video_device)

        if not video.isOpened():
            raise RuntimeError("No fue posible abrir el dispositivo de video")

        window_name = f'Toma de muestras para la palabra "{os.path.basename(path)}"'

        while video.isOpened():
            # "ret" alamacena un bool y "frame" alamcena el una matriz en representacion de img (line 59)
            ret, frame = video.read() # la funcion .read() me ayuda a leer los frames de un video, me da una variable bool
            if not ret: #ret == False
                break

            image, results = mediapipe_detection(frame, holistic_model) #(58,49) se hace llamado a la fucion medi_....
            if capturing and there_hand(results): #capturing(46), se hace llamado a there_hand
                count_frame += 1
                if count_frame > margin_frame: # 3>2
                    cv2.putText(
                        image,
                        f"Capturando la seña de {os.path.basename(path)}",
                        FONT_POS,
                        FONT,
                        FONT_SIZE,
                        BLUE_COLOR,
                    )
                    frames.append(np.asarray(frame))
            else:
                if len(frames) > min_cant_frames + margin_frame:
                    quantity_sample += 1
                    frames = frames[:-margin_frame]
                    output_dir = os.path.join(path, f"sample_{quantity_sample}")
                    create_dir(output_dir)
                    save_frames(frames, output_dir)

                frames = []
                count_frame = 1
                cv2.putText(
                    image,
                    "Listo para capturar...",
                    FONT_POS,
                    FONT,
                    FONT_SIZE,
                    RED_COLOR,
                    2,
                )
                cv2.putText(
                    image,
                    f"Muestra numero: {quantity_sample}",
                    (11, 100),
                    FONT,
                    FONT_SIZE,
                    RED_COLOR,
                    2,
                )

#DESPUES DE HABER PASADO POR EL IF MAS DE MAYOR NIVEL , CONTINUA CON "DRAW_KEYPOINTS"

            draw_keypoints(image, results)
            cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}"', image) #(,la img en matriz)
            # bring_window_to_front(window_name)

            key = cv2.waitKey(10) & 0xFF
            if key == 28:
                break
            elif key == ord("q"):
                video.release()
                cv2.destroyWindow("video")
            elif key == ord(" "):
                capturing = not capturing
            elif key == 14:
                new_sample_requested = True
                break

    return new_sample_requested
