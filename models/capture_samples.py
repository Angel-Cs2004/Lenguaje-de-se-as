import os

import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic

from utils.constants import BLUE_COLOR, FONT, FONT_POS, FONT_SIZE, RED_COLOR
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


def model_capture_sign(path, margin_frame=2, min_cant_frames=5, video_device=0):
    """
    Captura de muestras para una palabra.

    Parameters:
    - path (str): Ruta de la carpeta de la palabra.
    - margin_frame (int): Cantidad de frames que se ignoran al comienzo y al final.
    - min_cant_frames (int): Cantidad de frames mínimos para cada muestra.
    - video_device (int): Dispositivo de video a usar (default es 0 para webcam).

    Controles:
    - 'Esc' para salir y eliminar la muestra actual.
    - 'Enter' para terminar el registro de la muestra actual.
    - ' ' (espacio) para pausar la captura, pulsar de nuevo espacio para reanudad con la captura
    - 'q' para guardar la muestra capturada, salir y cerrar el programa.
    """
    create_dir(path)

    cant_samples_exists = len(os.listdir(path))
    quantity_sample = cant_samples_exists
    new_sample_deleted = False
    count_frame = 0
    capturing = True
    frames = []

    with Holistic() as holistic_model:
        video = cv2.VideoCapture(video_device)

        if not video.isOpened():
            raise RuntimeError("No fue posible abrir el dispositivo de video")

        window_name = f'Toma de muestras para la palabra "{os.path.basename(path)}"'

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic_model)

            if capturing:
                if there_hand(results):
                    count_frame += 1
                    if count_frame > margin_frame:
                        cv2.putText(
                            image,
                            f"Capturando la seña de {os.path.basename(path)}",
                            FONT_POS,
                            FONT,
                            FONT_SIZE,
                            BLUE_COLOR,
                            2,
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
                    count_frame = 0
                    cv2.putText(
                        image,
                        "Listo para capturar..",
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
            else:
                cv2.putText(
                    image,
                    "Se pauso la captura de samples",
                    FONT_POS,
                    FONT,
                    FONT_SIZE,
                    RED_COLOR,
                    2,
                )

            draw_keypoints(image, results)
            cv2.imshow(window_name, image)

            key = cv2.waitKey(10) & 0xFF
            if key == 13:  # Pressed Enter key
                break
            elif key == ord("q"):  # Pressed q key
                video.release()
                cv2.destroyAllWindows()
            elif key == ord(" "):  # Pressed space key
                capturing = not capturing
            elif key == 27:  # Pressed Esc key
                new_sample_deleted = True
                break

        video.release()
        cv2.destroyWindow(window_name)

    return new_sample_deleted
