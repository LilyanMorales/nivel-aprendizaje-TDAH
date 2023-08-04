# Importar las bibliotecas necesarias
import numpy as np                  # Para manipulación numérica
import tensorflow as tf             # Para construir y entrenar el modelo de aprendizaje automático
from sklearn.model_selection import train_test_split  # Para dividir los datos en conjuntos de entrenamiento y prueba

# Establecer una semilla para la generación de números aleatorios para reproducibilidad
np.random.seed(0)

# Generar 1000 números aleatorios entre 0 y 100 para niveles de atención
niveles_atencion = np.random.rand(1000) * 100

# Generar 1000 números aleatorios entre 0 y 5 para actividad física
actividad_fisica = np.random.rand(1000) * 5

# Generar 1000 números aleatorios entre 0 y 10 para horas de sueño
horas_sueno = np.random.rand(1000) * 10

# Definir el máximo posible para el rendimiento escolar
max_rendimiento = 100

# Generar datos de rendimiento escolar en base a los datos de atención, actividad física y sueño
rendimiento_escolar = (niveles_atencion / 100) * 0.3 * max_rendimiento + \
                      (actividad_fisica / 5) * 0.2 * max_rendimiento + \
                      (horas_sueno / 10) * 0.5 * max_rendimiento

# Agrupar los datos de entrada en un solo array
X = np.column_stack((niveles_atencion, actividad_fisica, horas_sueno))

# El rendimiento escolar es nuestro objetivo, es lo que queremos predecir
y = rendimiento_escolar

# Dividir los datos en un conjunto de entrenamiento (80% de los datos) y un conjunto de prueba (20% de los datos)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Función de pérdida personalizada para ajustar el error de predicción al rango 0-100
def mean_squared_error_0_to_100(y_true, y_pred):
    # Asegurarse de que las predicciones estén en el rango 0-100
    y_pred_clipped = tf.clip_by_value(y_pred, 0, 100)
    
    # Calcular el error cuadrático medio entre los valores verdaderos y las predicciones
    return tf.keras.losses.mean_squared_error(y_true, y_pred_clipped)

# Crear la arquitectura de la red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(3,)),  # Primera capa oculta con 128 neuronas y función de activación ReLU
    tf.keras.layers.Dense(64, activation='relu'),  # Segunda capa oculta con 64 neuronas y función de activación ReLU
    tf.keras.layers.Dense(32, activation='relu'),  # Tercera capa oculta con 32 neuronas y función de activación ReLU
    tf.keras.layers.Dense(1, activation='linear')  # Capa de salida con 1 neurona y función de activación lineal
])

# Compilar el modelo con la función de pérdida personalizada y el optimizador Adam
model.compile(loss=mean_squared_error_0_to_100, optimizer='adam')

# Entrenar el modelo con los datos de entrenamiento, usando un tamaño de lote de 10 y validando con los datos de prueba
model.fit(X_train, y_train, epochs=200, batch_size=10, validation_data=(X_test, y_test))

# Función para realizar la predicción del desempeño escolar
def predecir_desempeno_alumno():
    # Solicitar los datos de entrada del alumno
    niveles_atencion_alumno = float(input("Ingrese los niveles de atención del alumno (0-100): "))
    actividad_fisica_alumno = float(input("Ingrese la actividad física del alumno (0-5): "))
    horas_sueno_alumno = float(input("Ingrese las horas de sueño del alumno (0-10): "))
    
    # Organizar los datos de entrada en un formato que el modelo pueda usar
    input_data_alumno = np.array([[niveles_atencion_alumno, actividad_fisica_alumno, horas_sueno_alumno]])
    
    # Hacer la predicción
    prediccion_desempeno = model.predict(input_data_alumno)
    
    # Asegurarse de que la predicción esté en el rango 0-100
    prediccion_desempeno_clipped = np.clip(prediccion_desempeno, 0, 100)
    
    # Mostrar la predicción
    print("Predicción del desempeño escolar:", prediccion_desempeno_clipped[0][0])

# Bucle para continuar ingresando datos de nuevos alumnos
while True:
    predecir_desempeno_alumno()
    # Preguntar al usuario si quiere hacer otra predicción
    continuar = input("¿Desea ingresar datos de otro alumno? (y/n): ")
    # Si el usuario no quiere hacer otra predicción, romper el bucle
    if continuar.lower() != 'y':
        break
