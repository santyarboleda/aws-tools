# Adaptando un contenedor personalizado para Sagemaker Proccesses

El proceso que se describe en este repositorio consiste en la creación de entorno personalizado para el entrenamiento e inferencias con Scikit-learn 1.5 en Amazon Sagemaker.

¿Cuándo debería crear un contenedor personalizado?

Cuando los frameworks provistos por AWS no contienen una funcionalidad o una versión de librería que necesito para poder hacer un entrenamiento.

## Pasos para la creación del Contenedor Personalizado

## 1. Creación de un dockerfile

Es necesario crear un contenedor con las librerías que se requieren, que no estan incluidas en los frameworks provistos por AWS. Es importante considerar la instalación de las librerias `sagemaker-training` y `sagemaker-containers`, que son librerías necesarias para que el contenedor trabaje para entrenamiento e inferencia con los parámetros de SageMaker. Al momento de la creación de este repositorio estas librerias y sus dependencias son compatibles con al versión 3.9 de Python.

Este es un ejemplo de archivo requirements.txt

```txt 
scikit-learn==1.5.0
pandas==2.2.2
numpy==1.26.4
joblib==1.4.2
boto3==1.34.138
sagemaker-training
sagemaker-containers>=2.0.0
protobuf==3.20.3
Jinja2==3.0.3
itsdangerous==2.0.1 
Werkzeug==2.0.3
gunicorn==20.1.0
Flask==2.0.2 
``` 

A partir de la librería jinja, las siguientes librerías son necesarias para levantar un servicio web que de respuesta a las inferencias a través de endpoints, esto si es necesario que el estimador que se entrena de respuestas en tiempo real, y si este es el caso, es necesario crear un archivo serve para la descarga del archivo wsgi que realiza el lanzamiento del web server. El archivo wsgi debe ser almacenado en S3 para que se haga la descarga y se haga el lanzamiento del servidor. En este ejemplo ademas, el archivo `app.py`, que contiene la información para entrenamiento e inferencias tambien es contenido en esa ruta, sin embargo, para que el contenedor quede en funcionamiento para modelos diferentes y no tengamos un contenedor por modelo, la descarga de ese archivo debe realizarse desde el `wsgi.py`.

Este es un ejemplo de archivo serve:

```bash
#!/bin/bash

echo "--- Verificando contenido de /opt/ml/code ---"
ls -lR /opt/ml/code
echo "--- Fin de la verificación ---"
echo "--- Verificando contenido de /opt/ml/input ---"
ls -lR /opt/ml/input
echo "--- Fin de la verificación ---"

# Ruta del bucket y carpeta donde se encuenta el archivo wsgi y app (en este caso)
S3_CODE_PATH="s3://machine-learning-serviciosnutresa-modelos-lab/titanic/artifacts/code/"
LOCAL_CODE_PATH="/opt/ml/code"

echo "Creando directorio de código en ${LOCAL_CODE_PATH}"
mkdir -p ${LOCAL_CODE_PATH}

echo "Descargando scripts de ${S3_CODE_PATH} a ${LOCAL_CODE_PATH}"
aws s3 sync ${S3_CODE_PATH} ${LOCAL_CODE_PATH}

echo "Contenido del directorio después de la descarga:"
ls -lR ${LOCAL_CODE_PATH}

# --- LÓGICA DEL SERVIDOR (igual que antes) ---
export PYTHONPATH=${LOCAL_CODE_PATH}:$PYTHONPATH
cd ${LOCAL_CODE_PATH}
exec gunicorn --timeout 60 --bind 0.0.0.0:8080 --workers 1 "wsgi:app"
``` 

Este es un ejemplo de archivo wsgi.py:

```python 
import flask
import os
import importlib

# usamos la ruta estándar y fija donde SageMaker SIEMPRE descomprime el modelo.
model_dir = "/opt/ml/model"

module_name = "app"

print(f"Intentando importar el módulo de usuario: {module_name}")
user_module = importlib.import_module(module_name)
print("Módulo de usuario importado exitosamente.")

print(f"Cargando modelo desde la ruta fija: {model_dir}")
model = user_module.model_fn(model_dir)
print("Modelo cargado exitosamente.")

# Crea la aplicación Flask
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    return flask.Response(response='\n', status=200, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    input_data = user_module.input_fn(flask.request.data, flask.request.content_type)
    prediction = user_module.predict_fn(input_data, model)
    output, mimetype = user_module.output_fn(prediction, flask.request.accept_mimetypes)
    return flask.Response(response=output, status=200, mimetype=mimetype) 
```

Este es un ejemplo de archivo `app.py`, el cual contiene los métodos para entranamiento e inferencia. Para este ejemplo esta almacenado en S3 en la misma ruta del wsgi y se descarga desde el archivo serve, sin embargo, para hacer un contenedor genérico este archivo debería ser descargado desde el `wsgi.py`.

```python 
#!/usr/bin/env python

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from io import StringIO
import json

# =============================================================================
# FUNCIONES DE INFERENCIA
# =============================================================================
def model_fn(model_dir):
    print("Inferencing: Loading model from disk.")
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def input_fn(request_body, request_content_type):
    print(f"Inferencing: Received request with Content-Type: {request_content_type}")
    if request_content_type == 'text/csv':
        decoded_body = request_body.decode('utf-8')
        # Le decimos a pandas que no hay encabezado
        df = pd.read_csv(StringIO(decoded_body), header=None)

        # Asignamos los nombres de las columnas en el orden correcto
        try:
            df.columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        except Exception as e:
            # Esto ayuda a depurar si el número de columnas no coincide
            print(f"Error al asignar columnas: {e}")
            print(f"Número de columnas esperadas: 6, número de columnas recibidas: {len(df.columns)}")
            raise e

        df['Sex_male'] = df['Sex'].map({'male': 1, 'female': 0})
        df['Age'] = df['Age'].fillna(30.0)
        
        features_for_model = ['Pclass', 'Sex_male', 'Age', 'SibSp', 'Parch', 'Fare']
        return df[features_for_model]
    else:
        raise ValueError(f"Content-Type no soportado: {request_content_type}")

def predict_fn(input_data, model):
    print("Inferencing: Performing prediction.")
    return model.predict(input_data)

def output_fn(prediction, accept):
    print(f"Inferencing: Serializing prediction for Accept type: {accept}")
    
    # Comprueba si 'application/json' está en la lista de tipos aceptados
    if "application/json" in accept:
        response_data = {'predictions': prediction.tolist()}
        return json.dumps(response_data), "application/json"

     if "text/csv" in accept:
        response_data = {'predictions': prediction.tolist()}
        return json.dumps(response_data), "application/json"
        
        
    raise ValueError(f"Tipo 'Accept' no soportado: {accept}")

# =============================================================================
# LÓGICA DE ENTRENAMIENTO (VERSIÓN ROBUSTA)
# =============================================================================
if __name__ == '__main__':
    print("--- Starting Training Script (Robust Version) ---")

    # --- Leer parámetros directamente de las variables de entorno de SageMaker ---
    model_dir = os.environ['SM_MODEL_DIR']
    training_dir = os.environ['SM_CHANNEL_TRAINING']
    n_estimators = int(os.environ.get('SM_HP_N_ESTIMATORS', 100))
    random_state = int(os.environ.get('SM_HP_RANDOM_STATE', 42))

    print(f"Hyperparameters: n-estimators={n_estimators}, random-state={random_state}")
    print(f"Model directory: {model_dir}")
    print(f"Training data directory: {training_dir}")

    # Cargar datos
    training_data_path = os.path.join(training_dir, 'train.csv')
    df = pd.read_csv(training_data_path)

    # Preprocesamiento y entrenamiento
    print("Preprocessing data...")
    df['Sex_male'] = df['Sex'].map({'male': 1, 'female': 0})
    df['Age'] = df['Age'].fillna(df['Age'].median())
    features = ['Pclass', 'Sex_male', 'Age', 'SibSp', 'Parch', 'Fare']
    target = 'Survived'
    X_train = df[features]
    y_train = df[target]

    print("Training the RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # Guardar el modelo en la ruta de salida correcta
    model_path = os.path.join(model_dir, "model.joblib")
    print(f"Saving final model to: {model_path}")
    joblib.dump(model, model_path)
    
    print("--- Training Script finished successfully ---")
``` 

Este es un ejemplo de cómo debe quedar el dockerfile:

```Dockerfile 
FROM python:3.9-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=off

# Instala las dependencias del sistema operativo, incluyendo awscli
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential nginx awscli && \
    rm -rf /var/lib/apt/lists/*

# Actualiza pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copia e instala las dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Crear directorios necesarios
WORKDIR /opt/ml/code

# Copia nuestro script 'serve' personalizado al PATH del sistema
COPY serve /usr/local/bin/serve

# Le da permisos de ejecución
RUN chmod +x /usr/local/bin/serve
``` 

# 2. Lanzamiento del contenedor

Debe realizarse vía SDK o CLI la publicación del contenedor en ECR. Una vez el contenedor quede disponible en ese servicio podrá ser utilizado para entrenar o inferir.

El archivo train-inference.ipynb contiene un ejemplo de cómo realizar un entrenamiento e inferencia basados en el contenedor personalizado creado.

# 3. Estructura repositorio

**Carpeta Container:** contiene los archivos necesarios para lanzar el contenedor, incluyendo el Dockerfile

**Carpeta src:** contiene los archivos wsgi.py y app.py, que a su vez contiene el codigo de lanzamiento del web server, y el codigo de entrenamiento e inferencias respectivamente.

**Carpeta data:** contiene los dataset usados para el entrenamiento e inferencias, el cual corresponde al caso popular de supervivencia el evento del titanic.