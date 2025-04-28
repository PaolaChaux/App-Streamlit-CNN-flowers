# Proyecto: Clasificación de flores con CNN + Streamlit

Estructura Final del Proyecto:
De datos se uso un dataset de flores, el dataset se descargo de kaggle: https://www.kaggle.com/datasets/imsparsh/flowers-dataset?resource=download
Para el modelo se realizo un modelo CNN basico para la clasificacion de estas en donde se desarrollo y probo en GoogleColab: https://colab.research.google.com/drive/1xlOT2T1lpqWGn3ux1O_OX2qIlGJFoS5P?usp=sharing
En el streamlit se uso el script: app.py, el cual tiene toda la estructura basica para la app funcional que carga una imagen de una flor, la clasifica, predice y nos da un porcentaje de confiabilidad, Código principal de Streamlit
Requeriments. txt tiene toda las librerias basicas necesasrias que necesitamos para desarrollar nuestra app


Este proyecto permite clasificar imágenes de flores usando una red neuronal convolucional (CNN) construida en PyTorch, integrada en una aplicación web hecha con Streamlit.

## 🚀 ¿Qué incluye?

- Red neuronal convolucional (CNN) simple.
- Clasificación de 5 clases de flores: daisy, dandelion, rose, sunflower, tulip.
- Subida de imágenes en Streamlit.
- Predicción en tiempo real con el modelo preentrenado.
- Visualización de la probabilidad de predicción.

## Instalación

1. Clona el repositorio:
    ```bash
    git clone https://github.com/tu_usuario/tu_repositorio.git
    cd tu_repositorio
    ```
Debes descomprimir el archivo.zip que se descarga en el repositorio Kaggle

2. Crea y activa un entorno virtual en tu computadora:
    ```bash
    python -m venv venv
    # Activar en Windows:
    .\venv\Scripts\activate
    ```

3. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
Nota: dees tener en cuenta que debes asegurarte de tener Streamlit, Pytorch, Python y descargar el modelo,pth del google colab o tener una ya previamente entrenado

4. Ejecuta la app:
    ```bash
    streamlit run app.py
    ```
----------------------------------------------
# En cueanto a lo que se realizo en el script de Google Colab:

1. Librerías Importadas:
Se importaron las siguientes librerías necesarias para procesamiento de imágenes, construcción de redes neuronales y visualización:

- torch, torch.nn, torch.optim: para crear y entrenar la CNN.
- pytorch_lightning: para estructurar de manera más ordenada el flujo de datos.
- matplotlib.pyplot: para visualizar imágenes.
- numpy, PIL.Image, os, pandas: para manejo de datos y archivos.
- torchvision.transforms: para preprocesamiento de imágenes.

2. Creación de un Dataset personalizado
Se definió una clase Dataset personalizada:

class Dataset(torch.utils.data.Dataset) que recibe una lista de rutas de imágenes y etiquetas, se carga cada imagen usando PIL.Image, se aplica transformaciones como redimensionamiento y conversión a tensor y 
devuelve cada imagen y su respectiva etiqueta como un torch.Tensor con el objetivo de preparar los datos para ser usados por PyTorch de forma compatible.

3.  Preprocesamiento y manejo de datos - FlowerDataModule
Se definió la clase:
class FlowerDataModule(pl.LightningDataModule) para la carga imágenes desde carpetas train/ organizadas por clases (por ejemplo: train/daisy, train/rose...), se asigna un índice a cada clase automáticamente.
Separa los datos en:
80% para entrenamiento - 20% para validación

Se aplica transformaciones de:
- Redimensionar todas las imágenes a 128x128 píxeles.
- Convertirlas en tensores normalizados.
- Proporciona dataloaders para el entrenamiento (train_dataloader()) y la validación (val_dataloader()).

4. Visualización de imágenes
Se visualizaron 64 imágenes del dataset de entrenamiento, se organizaron en un grid de 8 filas x 8 columnas, cada imagen se mostró junto con su etiqueta predicha,
se transformaron de formato [C, H, W] a [H, W, C] para que puedan ser interpretadas por matplotlib para verificar que el preprocesamiento de datos es correcto antes de entrenar.

5.  Definición del modelo SimpleCNN
Se definió la red neuronal convolucional SimpleCNN:

class SimpleCNN(nn.Module):
De la cual su arquitectura es:
- Entrada: imágenes RGB (3 canales).
- Capas:
Convolución 1: 32 filtros, kernel 3x3, padding 1 → seguida de ReLU y MaxPooling.
Convolución 2: 64 filtros, kernel 3x3, padding 1 → seguida de ReLU y MaxPooling.
Aplanado (Flatten).
Capa densa: 64x32x32 → 128 neuronas → ReLU.
Capa final de clasificación: 128 → 5 neuronas (una por cada clase de flor).

S contruyo para que sea simple, eficiente y suficiente para un problema de clasificación básico con imágenes pequeñas.

6.  Entrenamiento del modelo
Se definió una función train:

def train(model, datamodule, epochs=5, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):

Esta carga los datos de entrenamiento y validación. 

Usa:
- Función de pérdida: CrossEntropyLoss (para clasificación multiclase).
- Optimizador: Adam con learning rate de 0.001.

Realiza 5 épocas de entrenamiento:
- Propagación hacia adelante (forward).
- Cálculo de la pérdida.
- Propagación hacia atrás (backward).
- Actualización de pesos (optimizer.step).

Evalúa desempeño tanto en entrenamiento como en validación, imprimiendo:
- Pérdida promedio (avg_loss).
- Precisión (train_acc y val_acc).

Resultados:
Cada época muestra el progreso del entrenamiento y la calidad de la validación.
Epoch 1/5 - Train Loss: 1.3674, Train Acc: 0.4076, Val Acc: 0.0000
Epoch 2/5 - Train Loss: 0.9900, Train Acc: 0.6011, Val Acc: 0.0000
Epoch 3/5 - Train Loss: 0.8309, Train Acc: 0.6726, Val Acc: 0.0036
Epoch 4/5 - Train Loss: 0.7204, Train Acc: 0.7272, Val Acc: 0.0018
Epoch 5/5 - Train Loss: 0.5991, Train Acc: 0.7755, Val Acc: 0.0636

7.  Guardado y carga del modelo
Después del entrenamiento, el modelo se guardó usando:

def save_model(model, path='model_cnn.pth'): El cual solo se guardaron solo los pesos (state_dict) en el archivo model_cnn.pth para cargarlo más rápido y ocupar menos espacio.

Para cargar el modelo después se usó:

def load_model(model_class, path='model_cnn.pth', device='cuda' if torch.cuda.is_available() else 'cpu'): Que carga correctamente los pesos en la misma arquitectura SimpleCNN.

8. Preparación del dataset de prueba: TestDataset
Se definió la clase -> class TestDataset(Dataset):
La cual carga imágenes sueltas de la carpeta test/ (sin subcarpetas).
Usa un CSV (Testing_set_flower.csv) que contiene el nombre de cada archivo.
Solo devuelve las imágenes y sus nombres, sin etiquetas reales (no se usa ground truth).

Transformación aplicada: 
- Redimensiona a 128x128 píxeles.
- Conversión a tensor.

9.  Inferencia en el conjunto de prueba
Se definió una -> def mostrar_predicciones_solo_pred(model, test_dataset, idx_to_class, n=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
Para que seleccione aleatoriamente 5 imágenes del dataset de prueba, pasa cada imagen por el modelo, predice la clase (índice máximo del vector de salida) y traduce el índice al nombre de la flor
usando el diccionario idx_to_class.

Muestra:
- Imagen
- Nombre de archivo
- Clase predicha

Visualización:
- Imágenes en una fila horizontal (o varias si se necesita).
- Etiquetas predichas como título de cada imagen.

Resultado final
Se entrenó una CNN básica capaz de clasificar 5 tipos de flores, se guardó el modelo para inferencia futura, se construyó un pipeline completo de: Carga de datos, Preprocesamiento, Entrenamiento, Validación y 
Predicción en imágenes nuevas.

Todo el sistema es reutilizable y puede integrarse fácilmente como se realizo en Streamlit.

Conclusión
El proyecto completo de Google Colab posee:

- Preparación y carga de imágenes.
- Diseño e implementación de una CNN sencilla.
- Entrenamiento supervisado en un conjunto de datos propio.
- Guardado y carga de modelos de PyTorch.
- Realización de inferencia sobre datos nuevos.
- Visualización de resultados de forma clara.

  Gracias por leer!

