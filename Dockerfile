# Usar una imagen oficial de Python 3.12. Esto nos da Python y un sistema Debian base.
FROM python:3.12-slim

# Establecer una variable de entorno para evitar que las instalaciones pidan input.
ENV DEBIAN_FRONTEND=noninteractive

# Instalar TODAS las dependencias del sistema en un solo paso.
# Incluye python-dev, build-essential y portaudio para compilar, y ffmpeg para el audio.
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo dentro del contenedor.
WORKDIR /app

# Copiar el archivo de requerimientos de Python.
COPY requirements.txt .

# Instalar las dependencias de Python.
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el resto del código de tu proyecto al contenedor.
COPY . .

# Crear los directorios que tu aplicación necesita para funcionar.
RUN mkdir -p jingles downloads ambientes videos

# Exponer el puerto 8080 para que el servidor web sea accesible.
EXPOSE 8080

# El comando final para ejecutar la aplicación usando Gunicorn.
# --preload es clave para que el hilo del bot se inicie correctamente.
CMD ["gunicorn", "--worker-class", "geventwebsocket.gunicorn.workers.GeventWebSocketWorker", "-w", "1", "--bind", "0.0.0.0:8080", "sonaria:flask_app"]
