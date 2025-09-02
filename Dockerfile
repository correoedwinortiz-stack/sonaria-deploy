# =================================================================
# Fase 1: Build - Instalar dependencias y compilar si es necesario
# =================================================================
FROM python:3.12-slim as builder

# Instalar dependencias del sistema necesarias
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
FROM python:3.12-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el entorno virtual con las dependencias ya instaladas desde la fase 'builder'
COPY --from=builder /opt/venv /opt/venv

# --- ¡CORRECCIÓN IMPORTANTE AQUÍ! ---
# Copiar el ejecutable de ffmpeg Y sus librerías compartidas necesarias.
COPY --from=builder /usr/bin/ffmpeg /usr/bin/ffmpeg
COPY --from=builder /usr/lib/x86_64-linux-gnu/libavdevice.so.* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libavfilter.so.* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libavformat.so.* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libavcodec.so.* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libswresample.so.* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libswscale.so.* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libavutil.so.* /usr/lib/x86_64-linux-gnu/
# Es posible que necesites más, pero estas son las principales.

# Activar el entorno virtual para los comandos siguientes
ENV PATH="/opt/venv/bin:$PATH"

# Copiar el resto del código de la aplicación
COPY . .

# Crear los directorios que tu aplicación necesita
RUN mkdir -p jingles downloads ambientes videos

# Exponer el puerto
EXPOSE 8080

# Comando para ejecutar la aplicación con precarga
CMD ["gunicorn", "--preload", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:8080", "sonaria:flask_app"]
