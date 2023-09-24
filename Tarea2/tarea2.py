import numpy as np
import soundfile
import matplotlib.pyplot as plt
from IPython import display
import pylab as pl
plt.style.use(["dark_background"])

audio_signal, fm = soundfile.read("ardillas-cantando.wav")

t= np.arange(0, 9.06447, 1/fm) # variable independiente discreta

dimension_t=len(t)

audio_izquierdo=audio_signal[:, 0]

type(audio_signal)

print("Tasa de muestreo: {} muestras/segundo" .format(fm))
print("Tamaño de la señal: {} muestras" .format(audio_izquierdo.shape[0]))
print("Duración: {:.3f} segundos" .format(audio_izquierdo.shape[0]/fm))

h_n=0.2*np.exp(-np.linspace(-2,2,31)**2)
print(-np.linspace(-2,2,31)**2)
print(h_n)
len_x=len(audio_izquierdo)
len_h=len(h_n)


plt.figure(figsize=(15,5))
plt.subplot(211)
plt.plot(t,audio_izquierdo, "b")
#plt.xlim([0,len_x])
plt.grid(True)
plt.subplot(212)
plt.plot(h_n,'r')
#plt.xlim([0,len_x])
plt.grid(True)

y_n=np.convolve(audio_izquierdo,h_n)
y_n_2=np.correlate(audio_izquierdo,h_n)

plt.figure(figsize=(10,3))

plt.subplot(311)
plt.plot(t,audio_izquierdo, "b")
plt.grid(True)
plt.subplot(312)
plt.plot(y_n,"r", linewidth=3)
plt.xlim([0,len_x])
plt.grid(True)
#plt.plot(x_n)
plt.subplot(313)
plt.plot(y_n_2,"g", linewidth=3)
plt.xlim([0,len_x])
plt.grid(True)
plt.show()
soundfile.write("filtradoG.wav", y_n,fm)
soundfile.write("filtrado_correlate.wav",y_n_2,fm)
