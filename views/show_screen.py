import tkinter as tk
from tkinter import messagebox, simpledialog
from tkinter.font import BOLD

import utils.generic as utl
from controllers.capture_controller import capture_new_sign, detect_and_talk


class App:
    def __init__(self) -> None:
        self.window = tk.Tk()
        self.window.title("Detector de lenguaje de señas")
        self.window.geometry("1000x625")
        self.window.config(bg="#fcfcfc")
        self.window.resizable(width=0, height=0)
        utl.center_window(self.window, 1000, 625)

        logo = utl.read_image_and_resize("./assets/logo.png", (250, 250))

        # frame-logo
        frame_logo = tk.Frame(
            self.window,
            bd=0,
            width=375,
            relief=tk.SOLID,
            padx=0,
            pady=0,
            bg="#121b26",
        )
        frame_logo.pack(side="left", expand=tk.NO, fill=tk.BOTH)
        label = tk.Label(frame_logo, image=logo, bg="#121b26")
        label.place(x=0, y=0, relwidth=1, relheight=1)

        # frame-right
        frame_right = tk.Frame(self.window, bd=0, relief=tk.SOLID, bg="#fcfcfc")
        frame_right.pack(side="right", expand=tk.YES, fill=tk.BOTH)

        # frame-title
        frame_right_title = tk.Frame(
            frame_right, height=62, bd=0, relief=tk.SOLID, bg="black"
        )
        frame_right_title.pack(side="top", fill=tk.X)
        title = tk.Label(
            frame_right_title,
            text="La palabra del mudo",
            font=("courrier", 30),
            fg="#666a88",
            pady=50,
        )
        title.pack(expand=tk.YES, fill=tk.BOTH)

        # frame-buttons
        frame_buttons = tk.Frame(
            frame_right, height=50, bd=0, relief=tk.SOLID, bg="#fff"
        )
        frame_buttons.pack(side="bottom", expand=tk.YES, fill=tk.BOTH)

        # Create buttons
        capture_btn = utl.read_image("./assets/capture_btn.png")
        capture_btn_hover = utl.read_image("./assets/capture_btn_active.png")
        self.create_btn(
            200,
            100,
            capture_btn,
            capture_btn_hover,
            frame_buttons,
            self.capture_new_sign,
        )

        detect_btn = utl.read_image("./assets/detect_btn.png")
        detect_btn_hover = utl.read_image("./assets/detect_btn_active.png")
        self.create_btn(
            217, 200, detect_btn, detect_btn_hover, frame_buttons, self.detect_and_talk
        )

        quit_btn = utl.read_image("./assets/quit_btn.png")
        quit_btn_hover = utl.read_image("./assets/quit_btn_active.png")
        self.create_btn(
            273, 300, quit_btn, quit_btn_hover, frame_buttons, self.close_window
        )

        self.window.mainloop()

    def create_btn(self, x, y, img, img_hov, frame, cmd):
        new_btn = tk.Button(
            frame,
            image=img,
            border=0,
            cursor="hand2",
            bg="#fff",
            relief=tk.SUNKEN,
            activebackground="#fff",
            highlightthickness=0,
            command=cmd,
        )

        def on_enter(e):
            new_btn["image"] = img_hov

        def on_leave(e):
            new_btn["image"] = img

        new_btn.bind("<Enter>", on_enter)
        new_btn.bind("<Leave>", on_leave)

        new_btn.place(x=x, y=y)

    def capture_new_sign(self):
        word = simpledialog.askstring(
            "Capturar nueva seña", "Indique el significado de la nueva palabra"
        )
        if word:
            res, name_path = capture_new_sign(word)
            if res == "exists":
                messagebox.showerror(
                    "Resultado de la captura",
                    f"La muestra de la palabra {word} ha sido capturada anteriormente",
                )
            elif res == "create":
                messagebox.showinfo(
                    "Resultado de la captura",
                    f"La muestra de la palabra {word} se encuentra en la carpeta {name_path}, ademas se extrajo correctmente los kp",
                )
            else:
                messagebox.showwarning(
                    "Resultado de la captura",
                    f'Se elimino correctamente la muestra de la palabra "{word}"',
                )

    def detect_and_talk(self):
        res = detect_and_talk()
        print(res)

    def close_window(self):
        self.window.destroy()
