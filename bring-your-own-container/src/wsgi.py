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