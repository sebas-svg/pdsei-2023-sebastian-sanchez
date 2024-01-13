import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

def process_frame():
    #LECTURA RGB ORIGINAL
    _, frame = cap.read()
    #LECTURA YUV
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    #LECTURA BALANCE DE BLANCOS 
    u_mean = np.mean(yuv[:, :, 1])
    v_mean = np.mean(yuv[:, :, 2])
    yuv[:, :, 1] = yuv[:, :, 1] + (128 - u_mean)
    yuv[:, :, 2] = yuv[:, :, 2] + (128 - v_mean)
    print(yuv)
    imagen_balanceada = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    update_label(frame, label_frame)
    update_label(yuv, label_mask)
    update_label(imagen_balanceada, label_result)
    #Agregar media final
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
