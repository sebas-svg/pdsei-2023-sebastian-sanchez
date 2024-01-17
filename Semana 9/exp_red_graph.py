import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow import keras as tf_keras
import matplotlib.pyplot as plt
import numpy as np

# Establecer la semilla para reproducibilidad
np.random.seed(42)
tf_keras.backend.clear_session()

# Cargar el conjunto de datos Iris
data = load_iris()

# Separar características (features) y etiquetas (labels)
features = data["data"]
labels = data["target"]

# División en conjunto de entrenamiento y prueba
input_train, input_test, output_train, output_test = train_test_split(
    features, labels, test_size=0.2, random_state=42)

# Preprocesamiento de la data
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalización de las características
mean = input_train.mean(axis=0)
input_train -= mean
std = input_train.std(axis=0)
input_train /= std

input_test -= mean
input_test /= std

# Codificación one-hot para las etiquetas
output_train = tf_keras.utils.to_categorical(output_train, 3)
output_test = tf_keras.utils.to_categorical(output_test, 3)

# Lista para almacenar resultados de experimentos
experiment_results = []

# Experimento 1
experiment_1 = {
    "epochs": 60,
    "num_layers": 2,
    "neurons_per_layer": 128,
    "activation_function": "relu",
    "optimizer": "Adam",
    "learning_rate": 0.002
}

# Experimento 2
experiment_2 = {
    "epochs": 50,
    "num_layers": 3,
    "neurons_per_layer": 256,
    "activation_function": "relu",
    "optimizer": "SGD",
    "learning_rate": 0.001
}

# Experimento 3
experiment_3 = {
    "epochs": 80,
    "num_layers": 2,
    "neurons_per_layer": 64,
    "activation_function": "tanh",
    "optimizer": "Adam",
    "learning_rate": 0.005
}

# Experimento 4
experiment_4 = {
    "epochs": 70,
    "num_layers": 3,
    "neurons_per_layer": 128,
    "activation_function": "relu",
    "optimizer": "Adam",
    "learning_rate": 0.002
}

experiment_5 = {
    "epochs": 60,
    "num_layers": 2,
    "neurons_per_layer": 64,
    "activation_function": "relu",
    "optimizer": "Adam",
    "learning_rate": 0.001
}

experiment_6 = {
    "epochs": 50,
    "num_layers": 3,
    "neurons_per_layer": 128,
    "activation_function": "relu",
    "optimizer": "SGD",
    "learning_rate": 0.005
}

experiment_7 = {
    "epochs": 70,
    "num_layers": 2,
    "neurons_per_layer": 128,
    "activation_function": "tanh",
    "optimizer": "Adam",
    "learning_rate": 0.002
}

experiment_8 = {
    "epochs": 80,
    "num_layers": 3,
    "neurons_per_layer": 64,
    "activation_function": "relu",
    "optimizer": "SGD",
    "learning_rate": 0.001
}

experiment_9 = {
    "epochs": 60,
    "num_layers": 2,
    "neurons_per_layer": 256,
    "activation_function": "tanh",
    "optimizer": "Adam",
    "learning_rate": 0.002
}

experiment_10 = {
    "epochs": 70,
    "num_layers": 3,
    "neurons_per_layer": 64,
    "activation_function": "relu",
    "optimizer": "SGD",
    "learning_rate": 0.005
}


# Agregar resultados de experimentos a la lista
experiment_results.extend([experiment_1, experiment_2, experiment_3, experiment_4, experiment_5, experiment_6, experiment_7, experiment_8, experiment_9, experiment_10])

# Gráficos de pérdida y exactitud en subplots específicos
plt.figure(figsize=(19, 12))

for idx, experiment in enumerate(experiment_results):
    model = Sequential([
        Dense(experiment["neurons_per_layer"], activation=experiment["activation_function"], input_shape=(4,),
              kernel_initializer='glorot_uniform'),
        Dense(experiment["neurons_per_layer"], activation=experiment["activation_function"],
              kernel_initializer='glorot_uniform'),
        Dense(3, activation="softmax")
    ])

    optimizer = getattr(tf_keras.optimizers, experiment["optimizer"])(learning_rate=experiment["learning_rate"])

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    history = model.fit(input_train, output_train,
                        epochs=experiment["epochs"],
                        validation_split=0.2,
                        verbose=0)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(loss))

    plt.subplot(3, 4, idx + 1)
    plt.title(f"Experimento {idx + 1}")
    plt.xlabel('Épocas')

    plt.plot(epochs, loss, "-b", label="Pérdidas de entrenamiento")
    plt.plot(epochs, val_loss, "-r", label="Pérdidas de validación")
    plt.plot(epochs, acc, "-g", label="Exactitud de entrenamiento")
    plt.plot(epochs, val_acc, "-m", label="Exactitud de validación")

    plt.legend(fontsize='small')  # Ajustar el tamaño de la leyenda

    # Imprimir resultados
    print(f"\nExperimento {idx + 1}:\n", experiment)
    print("Última pérdida en conjunto de entrenamiento:", loss[-1])
    print("Última pérdida en conjunto de validación:", val_loss[-1])
    print("Última exactitud en conjunto de entrenamiento:", acc[-1])
    print("Última exactitud en conjunto de validación:", val_acc[-1])

# Ajustes finales y mostrar gráficas
plt.tight_layout()
plt.show()

# Imprimir resultados
for idx, experiment in enumerate(experiment_results):
    print(f"\nExperimento {idx + 1}:\n", experiment)
