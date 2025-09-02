# =================================================================
# Fase 1: Build - Instalar dependencias y compilar si es necesario
# =================================================================
# Usamos una imagen completa de Python 3.12 para tener las herramientas de compilación
FROM python:3.12-slim as builder

# Instalar dependencias del sistema necesarias para algunas librerías de Python
# ¡AÑADIMOS portaudio19-dev AQUÍ!
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar las dependencias de Python en un entorno virtual
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copiar solo el archivo de requerimientos primero para aprovechar el cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# =================================================================
# Fase 2: Final - Crear la imagen final ligera
# =================================================================
# Usamos una imagen "slim" que es mucho más pequeña
FROM python:3.12-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el entorno virtual con las dependencias ya instaladas desde la fase 'builder'
COPY --from=builder /opt/venv /opt/venv

# Copiar las dependencias del sistema (ffmpeg) desde la fase 'builder'
COPY --from=builder /usr/bin/ffmpeg /usr/bin/ffmpeg

# Activar el entorno virtual para los comandos siguientes
ENV PATH="/opt/venv/bin:$PATH"

# Copiar el resto del código de la aplicación
COPY . .

# Crear los directorios que tu aplicación necesita
RUN mkdir -p jingles downloads ambientes videos

# Exponer el puerto en el que corre tu aplicación Flask/SocketIO
EXPOSE 8080

# Comando para ejecutar la aplicación
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:8080", "sonaria:flask_app"]
