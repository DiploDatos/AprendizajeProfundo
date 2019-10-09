# Práctico 1

En este práctico trabajaremos con el conjuto de datos de petfinder que utilizaron en la materia *Aprendizaje Supervisado*. La tarea es predecir la velocidad de adopción de un conjunto de mascotas. Para ello, también utilizaremos [esta competencia de Kaggle](https://www.kaggle.com/t/8842af91604944a9974bd6d5a3e097c5).

Durante esta etapa implementaremos modelos MLP básicos y no tan básicos, y veremos los diferentes hiperparámetros y arquitecturas que podemos elegir. Compararemos además dos tipos de representaciones comunes para datos categóricos: *one-hot-encodings* y *embeddings*. El primer ejercicio consiste en implementar y entrenar un modelo básico, y el segundo consiste en explorar las distintas combinaciones de características e hiperparámetros.

Para resolver los ejercicios, les proveemos un esqueleto que pueden completar en el archivo `practico_1_train_petfinder.py`. Este esqueleto ya contiene muchas de las funciones para combinar las representaciones de las distintas columnas que vimos en la notebook 2, aunque pueden agregar más columnas y las columnas con valores numéricos.

## Ejercicio 1

1. Construir un pipeline de clasificación con un modelo Keras MLP. Pueden comenzar con una versión simplicada que sólo tenga una capa de Input donde pasen los valores de las columnas de *one-hot-encodings*.

2. Entrenar uno o varios modelos (con dos o tres es suficiente, veremos más de esto en el práctico 2). Evaluar los modelos en el conjunto de dev y test.


## Ejercicio 2

1. Utilizar el mismo modelo anterior y explorar cómo cambian los resultados a medida que agregamos o quitamos columnas.

2. Volver a ejecutar una exploración de hyperparámetros teniendo en cuenta la información que agregan las nuevas columnas.

4. Subir los resultados a la competencia de Kaggle.


Finalmente, tienen que reportar los hyperparámetros y resultados de todos los modelos entrenados. Para esto, pueden utilizar los resultados que recolectan con *mlflow* y procesarlos con una notebook. Tiene que presentar esa notebook o un archivo (pdf|md) con las conclusiones que puedan sacar. Dentro de este reporte tiene que describir:
  * Hyperparámetros con los que procesaron cada columna del dataset. ¿Cuáles son las columnas que más afectan al desempeño? ¿Por qué?
  * Las decisiones tomadas al construir cada modelo: regularización, batch normalization, dropout, número y tamaño de las capas, optimizador.
  * Proceso de entrenamiento: división del train/dev, tamaño del batch, número de épocas, métricas de evaluación. Seleccione los mejores hiperparámetros en función de su rendimiento. El proceso de entrenamiento debería ser el mismo para todos los modelos.
  * Analizar si el clasificador está haciendo overfitting. Esto se puede determinar a partir del resultado del método fit.


