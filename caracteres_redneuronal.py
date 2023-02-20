#Developer: Angel Rodriguez Zuñiga

import tensorflow as tf
import numpy as np 
from tensorflow import keras

ent =[]
cat = []

#ASCII de Mayusculas
for i in range(65,91):
  ent.append([i])
  cat.append(0)

#ASCII de Minusculas
for i in range(97,123):
  ent.append([i])
  cat.append(1)

#ASCII de Digitos
for i in range(48,58):
  ent.append([i])
  cat.append(2)
    
#ASCII de Caracteres Especiales
for i in range(58,65):
  ent.append([i])
  cat.append(3)

for i in range(91,97):
  ent.append([i])
  cat.append(3)

for i in range(0,48):
  ent.append([i])
  cat.append(3)
    
for i in range(123,256):
  ent.append([i])
  cat.append(3)

#Convertir el valor a un tipo de dato compatible con la funcion de entrenamiento fit()
ent = np.array(ent, dtype=np.float32)
cat = np.array(cat, dtype=np.int32)

#Crear modelo y las capas con el numero de neuronas correspondientes, 32 en este caso
modelo=keras.Sequential()
modelo.add(keras.layers.Dense(32, input_shape=(1,), activation='relu'))
modelo.add(keras.layers.Dense(4, activation='softmax'))

#Compilar el modelo con el potimizador Adam y una tasa de aprendizaje del 2%
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.02),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#Ejecutar el entrenamiento
print("Ejecutando Entrenamiento...")
hist = modelo.fit(ent, cat, epochs=3000, verbose=False)
print("---¡Entrenamiento Terminado!---")
precision= modelo.evaluate(ent, cat, verbose=0)
print(">La precision es: %.2f" % precision[1] + "<")

#Realizar prediccion pidiendo dato al usuario (Codigo ASCII)
print("\n¡Prediccion!")
v = input("Ingresa el codigo ASCII(Int): \n")
v = int(v)
#Realizamos la prediccion
resultados = modelo.predict([v])
print("Salidas: " + str(resultados))
#Comparamos los valores devueltos para ver cual indice presenta y asi asignarlo a su categoria
for valor in resultados:
    indice_categorias = np.argmax(resultados)
    if indice_categorias == 0:
      print("Es Mayuscula")
    elif indice_categorias ==1:
      print("Es minuscula")
    elif indice_categorias==2:
      print("Es digito")
    elif indice_categorias==3:
      print("Invalido, CARACTER ESPECIAL")
    else:
      print("El valor es invalido o fuera del rango")


