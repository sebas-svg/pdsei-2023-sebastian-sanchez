import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("huerequeque.jpg", 0)
fil, col = img.shape

img_padded = np.pad(img,((0,fil),(0,col)),'constant',constant_values=((0, 0),(0,0)))

f, c = np.ogrid[0:2*fil,0:2*col]

n = 2
D =  np.sqrt( (f-fil)**2 + (c-col)**2 )
D0 = (0.12*fil)
H_pb_butter = 1/(1 + (D/D0)**(2*n))

img_fft = np.fft.fft2(img_padded)
img_fft_shift = np.fft.fftshift(img_fft)

G_fft = img_fft_shift * H_pb_butter

G_fft_ishift = np.fft.ifftshift(G_fft)
g = np.fft.ifft2(G_fft_ishift)
g_real = np.real(g)[0:fil, 0:col]

espectro = np.log(1 + np.abs(img_fft))

plt.figure(figsize=(8,8))
plt.imshow(espectro, cmap='gray')
#plt.imshow(g_real, cmap='gray')
plt.show()