import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

def process_frame():
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 90, 50])
    upper = np.array([60, 220, 180])
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    update_label(frame, label_frame)
    update_label(mask, label_mask)
    update_label(res, label_result)
    
    ventana.after(20, process_frame)

def update_label(image, label):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_image(image, 300)
    image = Image.fromarray(image)
    image_tk = ImageTk.PhotoImage(image=image)
    
    label.config(image=image_tk)
    label.image = image_tk

def resize_image(image, max_width):
    original_height, original_width, _ = image.shape
    ratio = max_width / original_width
    height = int(original_height * ratio)
    return cv2.resize(image, (max_width, height))

ventana = tk.Tk()
ventana.title("Real-time Image Processing with Tkinter")

cap = cv2.VideoCapture(0)

label_frame = tk.Label(ventana)
label_frame.grid(row=0, column=0, padx=10, pady=10)

label_mask = tk.Label(ventana)
label_mask.grid(row=1, column=0, padx=10, pady=10)

label_result = tk.Label(ventana)
label_result.grid(row=2, column=0, padx=10, pady=10)

# Iniciar el procesamiento de cuadros
process_frame()

ventana.mainloop()
