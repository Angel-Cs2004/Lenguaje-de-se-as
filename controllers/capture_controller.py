import os
import shutil
from tkinter import simpledialog

from models.capture_samples import handle_action, model_capture_sign
from utils.constants import FRAME_ACTIONS_PATH, ROOT_PATH
from utils.model_utils import extract_keypoints 


def capture_new_sign(input):
    while True:
        dir_word_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, input)
        new_sample_requested = model_capture_sign(dir_word_path)
        if new_sample_requested:
            kp = extract_keypoints(new_sample_requested)
            continue
        delete_sample = simpledialog.askstring(
            "Eliminar muestra",
            f"¿Desea eliminar la muestra de la palabra {input} y comenzar de nuevo? (y/n)",
        )
        if delete_sample == "y":
            handle_action(shutil.rmtree, dir_word_path)
        else:
            return True, dir_word_path


def detect_and_talk():
    return "te mueves y yo hablo"
