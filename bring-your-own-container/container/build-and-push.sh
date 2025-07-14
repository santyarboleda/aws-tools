#!/bin/bash
# ------------------------------------------------------------------------------------
image_name="byco2-sklearn-1-5"
# ------------------------------------------------------------------------------------

# Obtener la cuenta de AWS y la región desde la configuración de la AWS CLI
account=$(aws sts get-caller-identity --query Account --output text)
region=$(aws configure get region)

# Verificar que se obtuvieron la cuenta y la región
if [ -z "$account" ] || [ -z "$region" ]; then
    echo "Error: No se pudo obtener la cuenta de AWS o la región. Asegúrate de que la AWS CLI esté configurada (aws configure)."
    exit 1
fi

# Nombre completo de la imagen en ECR
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image_name}:latest"

echo "Construyendo y publicando la imagen en: ${fullname}"

# Iniciar sesión en el registro de ECR
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com

# Crear el repositorio en ECR si no existe
aws ecr describe-repositories --repository-names "${image_name}" > /dev/null 2>&1 || aws ecr create-repository --repository-name "${image_name}" > /dev/null

# Construir la imagen de Docker localmente
docker build --no-cache -t ${image_name} .

# Etiquetar la imagen para poder publicarla en ECR
docker tag ${image_name} ${fullname}

# Publicar la imagen en ECR
docker push ${fullname}

echo "---"
echo "¡Éxito! Imagen publicada correctamente."
echo "URI de la imagen: ${fullname}"