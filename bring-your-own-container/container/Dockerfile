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