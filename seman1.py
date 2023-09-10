import soundfile
import matplotlib.pyplot as plt
import numpy as np

audio_signal, fm = soundfile.read("ardillas-cantando.wav")

t= np.arange(0, 9.06447, 1/fm) # variable independiente discreta

dimension_t=len(t)

audio_izquierdo=audio_signal[:, 0]

type(audio_signal)

print("Tasa de muestreo: {} muestras/segundo" .format(fm))
print("Tamaño de la señal: {} muestras" .format(audio_izquierdo.shape[0]))
print("Duración: {:.3f} segundos" .format(audio_izquierdo.shape[0]/fm))
plt.plot(t,audio_izquierdo)

k=101
signal_filtrada=np.zeros_like(audio_izquierdo)

for i in range(k,dimension_t-k+1):
  signal_filtrada[i]=np.mean(audio_izquierdo[i-k:i+k])

size_ventana=1000*(2*k+1/fm)
print(size_ventana)
plt.plot(t,signal_filtrada)
plt.title("Señal Filtrada (Media Movil)")
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud")
plt.legend(["Senal Original","Senal Filtrada"])
plt.xlim(2.9,2.930)

soundfile.write("filtrado.wav", signal_filtrada,fm)

FWHM=25
k=80

gauss_t=1000*np.arange(-k,k)/fm

filtro_gaussiano=np.exp(-4*(np.log(2)*gauss_t**2)/FWHM**2)
filtro_gaussiano_normalizado=filtro_gaussiano/np.sum(filtro_gaussiano)

signal_filtradag=np.zeros_like(audio_izquierdo)

for i in range(k,dimension_t-k+1):
  signal_filtradag[i]=np.sum(audio_izquierdo[i-k:i+k]*filtro_gaussiano_normalizado)

size_ventana=1000*(2*k+1/fm)
print(size_ventana)
plt.figure()
plt.plot(t,audio_izquierdo)
plt.plot(t,signal_filtradag)
plt.title("Señal Filtrada (Gaussiano)")
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud")
plt.legend(["Senal Original","Senal Filtrada"])
plt.xlim(2.9,2.930)
plt.show()

soundfile.write("filtradog.wav", signal_filtradag,fm)