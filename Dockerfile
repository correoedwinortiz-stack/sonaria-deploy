FROM ubuntu:22.04

# Evitar prompts interactivos durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-pyaudio \
    ffmpeg \
    portaudio19-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias
COPY requirements_fixed.txt .

# Instalar dependencias de Python
RUN pip3 install --no-cache-dir -r requirements_fixed.txt

# Copiar el código de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p jingles downloads ambientes videos

# Exponer el puerto
EXPOSE 8080

# Comando por defecto
CMD ["python", "sonaria.py"]

