import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
# Imagen original sin segmentar (usar sliders)
h = 0
s = 0
v   = 0
 
def process_frame():
    _, frame = cap.read() #iamgen 1

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # IMAGEN 2
    
    lower = np.array([h,s,v]) # Límites inferiores (variables)
    upper = np.array([180, 255, 255]) # Límites sup (max)
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=mask) #IMAGEN 3

    update_label(frame, label_frame)
    update_label(hsv, label_mask)
    update_label(res, label_result)

    ventana.after(20, process_frame)
def update_label(image, label):
    # Para actualizar segmentación:
    # Verificación y conversión del espacio de color de la imagen
    if len(image.shape) == 3:  
        # Si la imagen es a color (tres canales), convertir de BGR a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    else:
        # Si la imagen es en escala de grises, convertir a formato de tres canales (RGB)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) 


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

def obtener_valores_segmentacion(inicial):
    return sliderh.get(), sliders.get(), sliderv.get()

def actualizar_parametros_segmentacion():
    global h, s, v
    inicial = 0
    h, s, v = obtener_valores_segmentacion(inicial)
    
ventana = tk.Tk()
ventana.title("Real-time Image Processing with Tkinter")

cap = cv2.VideoCapture(0)

label_frame = tk.Label(ventana)
label_frame.grid(row=0, column=0, padx=10, pady=10)

label_mask = tk.Label(ventana)
label_mask.grid(row=0, column=1, padx=10, pady=10)

label_result = tk.Label(ventana)
label_result.grid(row=0, column=2, padx=10, pady=10)

# SLIDER 1
sliderh = tk.Scale(ventana, from_=0, to=180, resolution=1, orient=tk.HORIZONTAL,
                    label="Valor de H", command=obtener_valores_segmentacion)
sliderh.set(0)
sliderh.grid(row=1, column=0,columnspan=1, pady=10)

# SLIDER 2
sliders = tk.Scale(ventana, from_=0, to=255, resolution=1, orient=tk.HORIZONTAL,
                    label="Valor de S", command=obtener_valores_segmentacion)
sliders.set(0)
sliders.grid(row=2, column=0,columnspan=1, pady=10)

# SLIDER 3
sliderv = tk.Scale(ventana, from_=0, to=255, resolution=1, orient=tk.HORIZONTAL,
                    label="Valor de V", command=obtener_valores_segmentacion)
sliderv.set(0)
sliderv.grid(row=3, column=0,columnspan=1, pady=10)

# BUTTON
boton = tk.Button(ventana,text="Procesar", command=actualizar_parametros_segmentacion)
boton.grid(row=2, column=1, columnspan=1, pady=10)

process_frame()
ventana.mainloop()