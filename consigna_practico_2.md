# Práctico 2

En este práctico se aplicarán los conocimientos adquiridos de redes avanzadas.
Basándose en lo aprendido con el práctico 1, esperamos que apliquen la misma
metodología en la creación de un pipeline de trabajo para el entrenamiento y
evaluación de modelos de redes neuronales avanzados en tareas que les
conciernan.

Se les darán como base dos posibles tareas a trabajar: clasificación de
imágenes del **CIFAR-10** o generación de textos en lenguaje natural. Sin embargo
son bienvenidos de trabajar en la tarea que vean pertinente mientras utilicen
algún modelo neuronal avanzado.

## Propuesta: Clasificación de imágenes del CIFAR-10 con redes convolucionales

El CIFAR-10 es un conjunto de datos utilizado comúnmente para el entrenamiento
de tareas de visión por computadoras. Similar al conjunto estándar MNIST, este
conjunto de datos tiene información de 60 mil imágenes y 10 clases. La
diferencia es que en este caso las imágenes son un poco más grandes (32x32
píxeles) y son a color.

Esto hace que el conjunto de datos sea un poco más complejo a trabajar que el
MNIST, por lo que un perceptrón multicapa es muy probable que no sea suficiente
para generar un modelo del problema. El entrenamiento de un modelo de
clasificación del CIFAR-10 es una tarea clásica en la que se pueden aplicar
redes convolucionales.

Keras viene con el conjunto de datos necesario disponible para descargar desde
su módulo `keras.datasets`.

## Propuesta: Generación de texto mediante redes recurrentes

Basándonos en una idea de un [artículo de Andrej
Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) esperamos
que entrenen, partiendo del conjunto de textos que les interese, un modelo
de lenguaje utilizando redes recurrentes.

La idea es que el modelo sea entrenado con la secuencia de caracteres (en lugar
de palabras) y que a partir de su entrenamiento puedan generar texto
automáticamente.

El conjunto de datos, para este caso, queda a criterio del alumno.  Sin
embargo, una recomendación que les hacemos, es tomar todos los textos
disponibles de algún autor específico y ver si efectivamente el texto generado
luego de entrenada la red evoca los textos de dicho autor.
