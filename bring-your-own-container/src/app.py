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