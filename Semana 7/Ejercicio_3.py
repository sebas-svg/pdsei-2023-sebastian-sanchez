import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2

def escalar_imagen(image,max_width):
    original_height,original_width=image.shape
    ratio=max_width/original_width
    height=int(original_height*ratio)
    return cv2.resize(image,(max_width,height))

def mostrar_imagen(image, label, max_width):
    imagen_resized = escalar_imagen(image, max_width)
    imagen_tk = ImageTk.PhotoImage(image=Image.fromarray(imagen_resized))
    label.config(image=imagen_tk)
    label.image = imagen_tk

def actualizar_filtro(valor):
    #n = int(valor)
    n = 2
    D =  np.sqrt( (f-fil)**2 + (c-col)**2 )
    D0 = (0.2/int(valor)*fil)
    H_pa_butter = 1 - 1/(1 + (D/D0)**(2*n))

    img_fft = np.fft.fft2(img_padded)
    img_fft_shift = np.fft.fftshift(img_fft)

    G_fft = img_fft_shift * H_pa_butter

    G_fft_ishift = np.fft.ifftshift(G_fft)
    g = np.fft.ifft2(G_fft_ishift)
    g_real = np.real(g)[0:fil, 0:col]
    filtro = np.log(1 + np.abs(H_pa_butter))
    espectro = np.log(1 + np.abs(G_fft))
    
    mostrar_imagen(255-255*filtro, label_filtro, 300)
    mostrar_imagen(10*espectro, label_matriz, 300)
    mostrar_imagen(g_real, label_g_real, 300)
    
img= cv2.imread("huerequeque.jpg", 0)
fil, col = img.shape
img_padded = np.pad(img,((0,fil),(0,col)),'constant',constant_values=((0, 0),(0,0)))
f, c = np.ogrid[0:2*fil,0:2*col]
fs, cs = np.ogrid[0:fil,0:col]



ventana = tk.Tk()
ventana.title("Mostrar Imagen y Matriz con Tkinter")

#Títulos:
titulo_label = tk.Label(ventana, text="Imagen original")
titulo_label.grid(row=0, column=0, padx=10)

titulo_label = tk.Label(ventana, text="Filtro aplicado")
titulo_label.grid(row=0, column=1, padx=10)

titulo_label = tk.Label(ventana, text="Espectro de la imagen modificada")
titulo_label.grid(row=3, column=0, padx=10)

titulo_label = tk.Label(ventana, text="Imagen modificada")
titulo_label.grid(row=3, column=1, padx=10)


label_imagen = tk.Label(ventana)
mostrar_imagen(img, label_imagen, 300)
label_imagen.grid(row=1, column=0,padx=10,pady=10)

label_filtro = tk.Label(ventana)
label_filtro.grid(row=1, column=1,padx=10,pady=10)

label_matriz = tk.Label(ventana)
label_matriz.grid(row=4, column=0,padx=10,pady=10)

label_g_real = tk.Label(ventana)
label_g_real.grid(row=4, column=1,padx=10,pady=10)

slider_n=tk.Scale(ventana, from_=1, to=20, orient=tk.HORIZONTAL, label="Valor de n", command=actualizar_filtro)

slider_n.set(2)
slider_n.config(length=500, width=20)
slider_n.grid(row=5, column=0,columnspan=3, pady=10)

ventana.mainloop()
