#!/bin/bash

# Script de inicio para SONARIA Bot en producción
# Este script maneja el inicio correcto del bot y el servidor web

echo "🚀 Iniciando SONARIA Bot en modo producción..."

# Verificar que existe el archivo .env
if [ ! -f ".env" ]; then
    echo "❌ Error: No se encontró el archivo .env"
    echo "   Copia .env.example a .env y configura tus credenciales"
    exit 1
fi

# Verificar que existen los directorios necesarios
mkdir -p jingles downloads ambientes videos frontend

# Verificar variables de entorno críticas
if [ -z "$DISCORD_TOKEN" ]; then
    echo "⚠️  Advertencia: DISCORD_TOKEN no está configurado"
    echo "   El bot no podrá conectarse a Discord"
fi

# Iniciar con Gunicorn usando la configuración específica
echo "🌐 Iniciando servidor web con Gunicorn..."
exec gunicorn --config gunicorn_config.py sonaria:flask_app

