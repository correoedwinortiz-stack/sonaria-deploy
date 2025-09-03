# ==============================================================================
# ||                                                                          ||
# ||                 SONARÍA - BOT DE DISCORD (VERSIÓN 2.0.0)                ||
# ||               -- VERSIÓN WEB + OAUTH + WEBSOCKETS --                    ||
# ==============================================================================
import os
import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv
import re
import unicodedata
from collections import deque
import asyncio
import random
import yt_dlp
from gtts import gTTS
import requests
import lyricsgenius
import time
import json
import jwt
from datetime import datetime, timedelta
import numpy as np  # Importar numpy para cálculos de audio
from flask_socketio import SocketIO, emit, join_room, leave_room

from flask import (
    Flask,
    request,
    jsonify,
    render_template_string,
    redirect,
    session,
    send_from_directory,
)
from flask_cors import CORS
from threading import Thread
import logging
import numpy as np
import aiohttp
import numpy as np
import functools
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import threading


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Carga de Variables de Entorno ---
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CLIENT_ID = os.getenv("DISCORD_CLIENT_ID")  # NUEVO
DISCORD_CLIENT_SECRET = os.getenv("DISCORD_CLIENT_SECRET")  # NUEVO
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "tu-clave-secreta-muy-segura")  # NUEVO
DISCORD_GUILD_ID = os.getenv("DISCORD_GUILD_ID")
peer_connections = {}

if not DISCORD_TOKEN:
    print("❌ FATAL: No se encontró DISCORD_TOKEN en el archivo .env.")
    exit()

if not DISCORD_CLIENT_ID or not DISCORD_CLIENT_SECRET:
    print("⚠️ ADVERTENCIA: No se encontraron credenciales OAuth de Discord.")
    print("   Las funciones de login web estarán limitadas.")

# --- Configuración existente (sin cambios) ---
ELEVENLABS_VOICE_IDS_ESPANOL = ["94zOad0g7T7K4oa7zhDq", "ajOR9IDAaubDK5qtLUqQ"]
GENIUS_API_TOKEN = os.getenv("GENIUS_API_TOKEN")

if GENIUS_API_TOKEN:
    genius = lyricsgenius.Genius(
        GENIUS_API_TOKEN, verbose=False, remove_section_headers=True
    )
    print("✅ Cliente de Genius API cargado.")
else:
    genius = None
    print(
        "⚠️ No se encontró GENIUS_API_TOKEN. La búsqueda por letra estará desactivada."
    )


def cargar_claves_elevenlabs():
    claves = []
    i = 1
    while True:
        clave = os.getenv(f"ELEVENLABS_API_KEY_{i}")
        if clave:
            claves.append(clave.strip())
            i += 1
        else:
            break
    if not claves:
        print("⚠️ No se encontraron claves de ElevenLabs. Usando gTTS por defecto.")
    return claves


# --- Configuración del Cliente de Spotify ---
sp = None
try:
    # Usamos las credenciales del archivo .env
    client_credentials_manager = SpotifyClientCredentials(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    )
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    logger.info("✅ Cliente de Spotify API cargado y autenticado.")
except Exception as e:
    logger.error(
        f"❌ No se pudo inicializar el cliente de Spotify. Verifica las credenciales. Error: {e}"
    )


ELEVENLABS_API_KEYS = cargar_claves_elevenlabs()
clave_index = 0

# --- Rutas y estructuras existentes (sin cambios) ---
JINGLES_PATH = os.path.join(os.getcwd(), "jingles")
DOWNLOAD_PATH = os.path.join(os.getcwd(), "downloads")
MUSICAFONDO_PATH = os.path.join(os.getcwd(), "ambientes")
VIDEO_PATH = os.path.join(os.getcwd(), "videos")  # Nueva ruta para videos
os.makedirs(JINGLES_PATH, exist_ok=True)
os.makedirs(DOWNLOAD_PATH, exist_ok=True)
os.makedirs(MUSICAFONDO_PATH, exist_ok=True)
os.makedirs(VIDEO_PATH, exist_ok=True)  # Crear directorio de videos

# Variables globales existentes
cola_canciones = deque()
cola_reproduccion = deque()
radio_activa = False
playlist_en_curso = False
ultimo_jingle_relleno = 0
INTERVALO_JINGLE_RELLENO = 150
silence_duration = 0.0
SILENCE_THRESHOLD = 50.0  # Umbral de volumen para considerar "silencio"
SILENCE_TIMEOUT = 2.0  # Tiempo en segundos para considerar que está "en silencio"
banco_recomendaciones = []
recomendaciones_actuales = []
VOICE_CHANNEL_ID = os.getenv("CANAL_DE_VOZ_ID")


# sonaria.py

# --- FRASES PREDEFINIDAS PARA EL BOT DEL CHAT ---
FRASES_PETICION_RECIBIDA = [
    "¡Petición recibida! Procesando '{cancion}'. ¡Pronto estará en la cola!",
    "¡Anotado! '{cancion}' está en camino. ¡Qué buen gusto tienes!",
    "¡Claro que sí! Preparando '{cancion}' para ti. ¡Gracias por participar!",
]

FRASES_DEDICATORIA_RECIBIDA = [
    "¡Qué detallazo! La dedicatoria para {para} de tu parte con la canción '{cancion}' ha sido recibida. ❤️",
    "¡Precioso! Preparando '{cancion}' para {para}. ¡Seguro que le encanta!",
    "¡Entendido! Una dedicatoria especial para {para} en camino. ¡La radio se llena de amor!",
]

FRASES_DESCARGA_COMPLETADA = [
    "¡Buenas noticias! '{cancion}' ya está descargada y lista para sonar.",
    "¡Todo listo! '{cancion}' acaba de entrar en la cola de reproducción.",
    "¡Atención! La canción '{cancion}' está preparada. ¡Sube el volumen!",
]


# --- NUEVAS VARIABLES PARA LA LIMPIEZA DE ARCHIVOS ---
ARCHIVOS_A_CONSERVAR = [
    "background_music.mp3",
    ".gitkeep",
]  # Archivos que nunca se deben borrar
TIEMPO_EXPIRACION_SEGUNDOS = 3600  # 1 horas (24 * 60 * 60)
INTERVALO_LIMPIEZA_SEGUNDOS = 3600  # 1 hora (60 * 60)


# --- NUEVAS VARIABLES GLOBALES ---
cancion_actual = {"titulo": None, "artista": None, "usuario": None}
usuarios_conectados = set()
canal_radio_id = None  # ID del canal de voz de la radio

# Variable global para el nivel de audio
audio_level = 0.0
radio_activa = False

# Cargar banco de jingles existente
todos_los_jingles = os.listdir(JINGLES_PATH)
BANCO_JINGLES = {
    "APERTURA": [f for f in todos_los_jingles if f.startswith("apertura_")],
    "ENTRE_CANCIONES": [f for f in todos_los_jingles if f.startswith("frase_")],
    "PUENTE_DEDICATORIA": [
        f for f in todos_los_jingles if f.startswith("dedicatoria_")
    ],
    "SIN_DEDICATORIA": [
        f for f in todos_los_jingles if f.startswith("sin_dedicatoria_")
    ],
    "PRINCIPAL": [f for f in todos_los_jingles if f.startswith("jingle_principal")],
}
jingles_de_relleno = []
jingles_de_relleno.extend(BANCO_JINGLES["ENTRE_CANCIONES"])
jingles_de_relleno.extend(BANCO_JINGLES["PRINCIPAL"])
BANCO_JINGLES["RELLENO"] = list(set(jingles_de_relleno))

print("✅ Banco de jingles cargado.")


def emit_with_context(event, data, **kwargs):
    """Emite un evento de SocketIO dentro del contexto de la aplicación Flask"""
    with flask_app.app_context():
        socketio.emit(event, data, **kwargs)


def obtener_recomendaciones_deezer(nombre_artista):
    try:
        # 1. Buscar el artista en Deezer
        search_url = f"https://api.deezer.com/search/artist?q={nombre_artista}"
        resp = requests.get(search_url).json()
        if not resp.get("data"):
            logger.info(f"No se encontró el artista '{nombre_artista}' en Deezer.")
            return []

        artist_id = resp["data"][0]["id"]

        # 2. Obtener artistas relacionados
        related_url = f"https://api.deezer.com/artist/{artist_id}/related"
        related = requests.get(related_url).json()

        if not related.get("data"):
            logger.warning(
                f"No se encontraron artistas relacionados para '{nombre_artista}'."
            )
            return []

        lista_formateada = []
        for artist in related["data"][:5]:
            # 3. Obtener su canción más popular
            top_url = f"https://api.deezer.com/artist/{artist['id']}/top?limit=1"
            top_track = requests.get(top_url).json()
            if top_track.get("data"):
                lista_formateada.append(
                    {
                        "titulo": top_track["data"][0]["title"],
                        "artista": artist["name"],
                        "mensaje_corto": f"Porque te gusta {nombre_artista}, quizás disfrutes este artista.",
                    }
                )

        logger.info(
            f"✅ Se generaron {len(lista_formateada)} recomendaciones con Deezer."
        )
        return lista_formateada

    except Exception as e:
        logger.error(f"❌ Error en Deezer API: {e}", exc_info=True)
        return []


def cargar_recomendaciones():
    global banco_recomendaciones
    try:
        with open("recomendaciones.json", "r", encoding="utf-8") as f:
            banco_recomendaciones = json.load(f)
        logger.info(
            f"✅ Banco de recomendaciones cargado con {len(banco_recomendaciones)} canciones."
        )
    except FileNotFoundError:
        logger.error(
            "❌ No se encontró el archivo 'recomendaciones.json'. El sistema no podrá dar sugerencias."
        )
    except json.JSONDecodeError:
        logger.error(
            "❌ Error al leer 'recomendaciones.json'. Asegúrate de que el formato sea correcto."
        )


# Llama a esta función al final del bloque de configuración inicial, antes de definir los comandos.
cargar_recomendaciones()


def buscar_cancion_por_letra(frase):
    if not genius:
        return None
    try:
        song = genius.search_song(frase)
        return f"{song.title} - {song.artist}" if song else None
    except Exception:
        return None


def normalizar_texto(texto):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", texto.lower())
        if unicodedata.category(c) != "Mn"
    )


def sanitizar_nombre_archivo(nombre):
    nombre_limpio = normalizar_texto(nombre)
    nombre_limpio = re.sub(r"[\s/\\:]+", "_", nombre_limpio)
    return f"{nombre_limpio}.mp3"


def normalizar_nombre_cancion(nombre: str) -> str:
    if not nombre:
        return "cancion_desconocida"
    nombre_norm = "".join(
        c
        for c in unicodedata.normalize("NFD", nombre)
        if unicodedata.category(c) != "Mn"
    )
    nombre_norm = re.sub(r"[\/\\\:\|\*\?\"<>\\]", " ", nombre_norm)
    nombre_norm = re.sub(r"\s+", " ", nombre_norm)
    return nombre_norm.strip().title()


def parsear_dedicatoria(texto):
    try:
        t_norm = normalizar_texto(texto)
        pos_c = t_norm.rfind("cancion")
        if pos_c == -1:
            return None

        cancion = texto[pos_c + len("cancion") :].strip(" :.,-–—")
        if not cancion:
            return None

        pre = texto[:pos_c].strip()
        pre_norm = normalizar_texto(pre)
        p_para = pre_norm.rfind("para ")
        p_de = pre_norm.rfind("de ")

        def limpiar(s):
            return s.strip(" ,.:;—–-")

        if p_para == -1 and p_de == -1:
            return None

        destinatario, remitente, mensaje = "", "", ""

        if p_para != -1 and p_de != -1:
            if p_para < p_de:
                destinatario = limpiar(pre[p_para + 5 : p_de])
                resto = limpiar(pre[p_de + 3 :])
                partes = re.split(r"[:\-—,]\s*", resto, maxsplit=1)
                remitente = limpiar(partes[0]) if partes else limpiar(resto)
                mensaje = limpiar(partes[1]) if len(partes) > 1 else ""
            else:
                remitente = limpiar(pre[p_de + 3 : p_para])
                resto = limpiar(pre[p_para + 5 :])
                partes = re.split(r"[:\-—,]\s*", resto, maxsplit=1)
                destinatario = limpiar(partes[0]) if partes else limpiar(resto)
                mensaje = limpiar(partes[1]) if len(partes) > 1 else ""
        elif p_para != -1:
            resto = limpiar(pre[p_para + 5 :])
            partes = re.split(r"[:\-—,]\s*", resto, maxsplit=1)
            destinatario = limpiar(partes[0]) if partes else limpiar(resto)
            mensaje = limpiar(partes[1]) if len(partes) > 1 else ""
        elif p_de != -1:
            resto = limpiar(pre[p_de + 3 :])
            partes = re.split(r"[:\-—,]\s*", resto, maxsplit=1)
            remitente = limpiar(partes[0]) if partes else limpiar(resto)
            mensaje = limpiar(partes[1]) if len(partes) > 1 else ""

        return (
            destinatario.title() if destinatario else "Todos",
            remitente.title() if remitente else "Alguien",
            mensaje.strip(" .,:;—–-"),
            cancion,
        )
    except Exception:
        return None


# ==============================================================================
# ||                 TAREA DE LIMPIEZA DE ARCHIVOS ANTIGUOS                   ||
# ==============================================================================
@tasks.loop(seconds=INTERVALO_LIMPIEZA_SEGUNDOS)
async def limpiar_archivos_antiguos():
    logger.info("🧹 Ejecutando tarea de limpieza de archivos antiguos en /downloads...")
    ahora = time.time()
    archivos_borrados = 0

    try:
        for nombre_archivo in os.listdir(DOWNLOAD_PATH):
            # Ignorar archivos que queremos conservar
            if nombre_archivo in ARCHIVOS_A_CONSERVAR:
                continue

            ruta_completa = os.path.join(DOWNLOAD_PATH, nombre_archivo)

            # Solo procesar archivos, no directorios
            if os.path.isfile(ruta_completa):
                try:
                    # Obtener la fecha de última modificación del archivo
                    fecha_modificacion = os.path.getmtime(ruta_completa)

                    # Comprobar si el archivo es más antiguo que el tiempo de expiración
                    if (ahora - fecha_modificacion) > TIEMPO_EXPIRACION_SEGUNDOS:
                        os.remove(ruta_completa)
                        logger.info(f"🗑️ Archivo antiguo borrado: {nombre_archivo}")
                        archivos_borrados += 1
                except Exception as e:
                    logger.error(
                        f"❌ No se pudo borrar el archivo {nombre_archivo}: {e}"
                    )

    except Exception as e:
        logger.error(f"❌ Error general durante la limpieza de archivos: {e}")

    if archivos_borrados > 0:
        logger.info(
            f"🧹 Limpieza completada. Se borraron {archivos_borrados} archivos."
        )
    else:
        logger.info("🧹 Limpieza completada. No se encontraron archivos para borrar.")


def descargar_audio_youtube(busqueda, archivo_salida):
    ruta_absoluta = os.path.join(DOWNLOAD_PATH, archivo_salida)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": ruta_absoluta.replace(".mp3", ""),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "noplaylist": True,
        "quiet": True,
        "default_search": "ytsearch",
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([busqueda])
        return ruta_absoluta
    except Exception:
        return None


def texto_a_voz_gTTS(texto, nombre_archivo):
    try:
        tts = gTTS(text=texto, lang="es", slow=False)
        ruta_completa = os.path.join(DOWNLOAD_PATH, nombre_archivo)
        tts.save(ruta_completa)
        return ruta_completa
    except Exception:
        return None


def texto_a_voz_elevenlabs(texto, nombre_archivo):
    global clave_index
    if not ELEVENLABS_API_KEYS:
        return None
    voice_id = random.choice(ELEVENLABS_VOICE_IDS_ESPANOL)
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEYS[clave_index],
        "Content-Type": "application/json",
    }
    data = {
        "text": texto,
        "voice_settings": {"stability": 0.7, "similarity_boost": 0.8},
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            ruta = os.path.join(DOWNLOAD_PATH, nombre_archivo)
            with open(ruta, "wb") as f:
                f.write(response.content)
            clave_index = (clave_index + 1) % len(ELEVENLABS_API_KEYS)
            return ruta
        else:
            clave_index = (clave_index + 1) % len(ELEVENLABS_API_KEYS)
            return (
                None
                if clave_index == 0
                else texto_a_voz_elevenlabs(texto, nombre_archivo)
            )
    except Exception:
        return None


def generar_audio_dedicatoria(texto, nombre_archivo):
    ruta_audio = texto_a_voz_elevenlabs(texto, nombre_archivo)
    return ruta_audio or texto_a_voz_gTTS(texto, nombre_archivo)


# --- NUEVAS FUNCIONES PARA OAUTH Y JWT ---
def generar_jwt_token(user_data):
    payload = {
        "user_id": user_data["id"],
        "username": user_data["username"],
        "exp": datetime.utcnow() + timedelta(days=7),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")


def verificar_jwt_token(token):
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


async def obtener_usuario_discord(access_token):
    """Obtiene información del usuario desde la API de Discord"""
    async with aiohttp.ClientSession() as session:
        headers = {"Authorization": f"Bearer {access_token}"}
        async with session.get(
            "https://discord.com/api/users/@me", headers=headers
        ) as resp:
            if resp.status == 200:
                return await resp.json()
    return None


async def unir_usuario_servidor(access_token, user_id):
    """Une al usuario al servidor de Discord de la radio"""
    if not DISCORD_CLIENT_SECRET or not canal_radio_id:
        return False

    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bot {DISCORD_TOKEN}",
            "Content-Type": "application/json",
        }
        data = {"access_token": access_token}

        # Aquí necesitarás el ID del servidor
        guild_id = os.getenv("DISCORD_GUILD_ID")
        if not guild_id:
            return False

        url = f"https://discord.com/api/guilds/{guild_id}/members/{user_id}"
        async with session.put(url, headers=headers, json=data) as resp:
            return resp.status in [200, 201, 204]


# --- Configuración del Bot (sin cambios) ---
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)


# --- Funciones de reproducción existentes (MODIFICADAS) ---


class AudioLevelSource(discord.PCMVolumeTransformer):
    def __init__(self, original):
        super().__init__(original)
        self.volume = 1.0

    def read(self):
        data = super().read()
        if not data:
            return data

        # La lógica de análisis de audio para Socket.IO se queda.
        # Ya no necesita enviar audio a ningún listener de WebRTC.
        try:
            audio_array = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(np.square(audio_array.astype(np.float64))))
            level = float(np.clip((20 * np.log10(rms + 1e-10) + 60) / 60, 0, 1))
            emit_with_context("audio_level", {"level": level})

            if rms >= SILENCE_THRESHOLD:
                fft_data = np.abs(np.fft.rfft(audio_array))
                freqs = np.fft.rfftfreq(len(audio_array), 1 / 48000)
                bass = np.mean(fft_data[(freqs >= 20) & (freqs < 250)])
                mid = np.mean(fft_data[(freqs >= 250) & (freqs < 4000)])
                high = np.mean(fft_data[(freqs >= 4000)])
                total = np.max(fft_data) if np.max(fft_data) > 0 else 1

                emit_with_context(
                    "audio_bands",
                    {
                        "bass": float(bass / total),
                        "mid": float(mid / total),
                        "high": float(high / total),
                    },
                )

        except Exception as e:
            logger.error(f"Error en análisis de audio: {e}", exc_info=False)

        return data


async def reproducir_archivo(voice_client, ruta_archivo):
    if not os.path.exists(ruta_archivo):
        logger.error(f"❌ No se pudo encontrar el archivo a reproducir: {ruta_archivo}")
        return
    try:
        # Volvemos a una fuente simple, sin opciones de tiempo
        source = discord.FFmpegPCMAudio(ruta_archivo)
        audio_source_with_level = AudioLevelSource(source)
        voice_client.play(audio_source_with_level)

        # Esperamos a que termine de sonar
        while voice_client.is_playing():
            await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"Error al reproducir {ruta_archivo}: {e}")


# REEMPLAZA tu radio_manager con esta versión final y simplificada


# REEMPLAZA tu radio_manager con esta versión que SÍ reproduce los jingles


@tasks.loop(seconds=2.0)
async def radio_manager(ctx):
    global ultimo_jingle_relleno, cancion_actual

    voice_client = discord.utils.get(bot.voice_clients, guild=ctx.guild)

    if not radio_activa or not voice_client or not voice_client.is_connected():
        return

    if playlist_en_curso:
        return

    # --- PRIORIDAD 1: Peticiones de la cola ---
    if cola_reproduccion:
        if voice_client.is_playing():
            voice_client.stop()
            logger.info("⏹️ Deteniendo audio actual para dar paso a una petición.")
            await asyncio.sleep(0.5)

        logger.info("▶️ Iniciando reproducción de la playlist de peticiones.")
        await reproducir_playlist(ctx)
        return

    # --- PRIORIDAD 2: Jingles de relleno ---
    # Comprobamos si es hora de un jingle ANTES de comprobar si algo está sonando.
    if (time.time() - ultimo_jingle_relleno) > INTERVALO_JINGLE_RELLENO:
        if jingles_relleno := BANCO_JINGLES.get("RELLENO"):
            # Si es hora de un jingle, detenemos lo que esté sonando (la música de fondo).
            if voice_client.is_playing():
                voice_client.stop()
                logger.info(
                    "⏹️ Deteniendo música de fondo para dar paso a un jingle de relleno."
                )
                await asyncio.sleep(0.5)

            logger.info("⏳ Reproduciendo jingle de relleno...")
            jingle_a_reproducir = os.path.join(
                JINGLES_PATH, random.choice(jingles_relleno)
            )
            await reproducir_archivo(voice_client, jingle_a_reproducir)
            ultimo_jingle_relleno = time.time()
            return

    # Si ya hay algo sonando (y no es una petición ni un jingle), lo dejamos estar.
    if voice_client.is_playing():
        return

    # Si llegamos aquí, el reproductor está LIBRE y no toca ni petición ni jingle.
    # --- PRIORIDAD 3: Música de fondo (acción por defecto) ---
    try:
        pistas_de_fondo = [
            f
            for f in os.listdir(MUSICAFONDO_PATH)
            if f.endswith((".mp3", ".wav", ".ogg"))
        ]
        if not pistas_de_fondo:
            return

        pista_elegida = random.choice(pistas_de_fondo)
        ruta_pista = os.path.join(MUSICAFONDO_PATH, pista_elegida)

        logger.info(f"📻 Poniendo música de fondo: {pista_elegida}")

        cancion_actual = {
            "titulo": "Música de Ambiente",
            "artista": "SONARÍA Radio",
            "usuario": None,
        }
        emit_with_context("now_playing", cancion_actual)

        await reproducir_archivo(voice_client, ruta_pista)

    except Exception as e:
        logger.error(f"❌ Error al intentar poner música de fondo: {e}")


# python
async def reproducir_playlist(ctx):
    global playlist_en_curso, ultimo_jingle_relleno, cancion_actual

    if not cola_reproduccion:
        return

    playlist_en_curso = True
    logger.info("▶️ Iniciando playlist. Flag 'playlist_en_curso' establecido a True.")

    try:
        voice_client = discord.utils.get(bot.voice_clients, guild=ctx.guild)
        if not voice_client or not voice_client.is_connected():
            logger.error(
                "❌ No se puede reproducir la playlist, el bot no está en un canal de voz."
            )
            playlist_en_curso = False
            return

        cancion_actual_obj = cola_reproduccion.popleft()

        # Actualizar 'Ahora Sonando' en la web
        cancion_actual = {
            "titulo": cancion_actual_obj["titulo"].split(" - ")[0],
            "artista": (
                cancion_actual_obj["titulo"].split(" - ")[1]
                if " - " in cancion_actual_obj["titulo"]
                else "Artista Desconocido"
            ),
            "usuario": cancion_actual_obj["autor"],  # Ahora es directamente el string
            "cover_url": cancion_actual_obj.get("cover_url"),
        }
        emit_with_context("now_playing", cancion_actual)

        # --- LÓGICA DE CONSTRUCCIÓN DE PLAYLIST CORREGIDA ---
        playlist_de_rutas = []
        dedicatoria_info = cancion_actual_obj.get("dedicatoria")

        if dedicatoria_info:
            # 1. Añadir jingle de puente para dedicatorias
            if jingles_puente := BANCO_JINGLES.get("PUENTE_DEDICATORIA"):
                playlist_de_rutas.append(
                    os.path.join(JINGLES_PATH, random.choice(jingles_puente))
                )

            # 2. ¡¡PASO CLAVE!! Generar el audio de la dedicatoria
            texto_dedicatoria = f"Esta canción es para {dedicatoria_info['para']}, de parte de {dedicatoria_info['de']}"
            if dedicatoria_info.get("mensaje"):
                texto_dedicatoria += (
                    f" con el siguiente mensaje: {dedicatoria_info['mensaje']}"
                )

            logger.info(f"🎙️ Generando audio para la dedicatoria: '{texto_dedicatoria}'")
            nombre_archivo_dedicatoria = f"dedicatoria_{int(time.time())}.mp3"

            loop = asyncio.get_running_loop()
            ruta_audio_dedicatoria = await loop.run_in_executor(
                None,
                functools.partial(
                    generar_audio_dedicatoria,
                    texto_dedicatoria,
                    nombre_archivo_dedicatoria,
                ),
            )

            if ruta_audio_dedicatoria:
                playlist_de_rutas.append(ruta_audio_dedicatoria)
                logger.info("✅ Audio de dedicatoria generado y añadido a la playlist.")
            else:
                logger.error("❌ Falló la generación del audio de la dedicatoria.")

        else:
            # Si no hay dedicatoria, usar el jingle correspondiente
            if jingles_sin_dedicatoria := BANCO_JINGLES.get("SIN_DEDICATORIA"):
                playlist_de_rutas.append(
                    os.path.join(JINGLES_PATH, random.choice(jingles_sin_dedicatoria))
                )

        # 3. Añadir la canción principal
        playlist_de_rutas.append(cancion_actual_obj["ruta_archivo"])

        # 4. Añadir jingle de cierre
        if jingles_cierre := BANCO_JINGLES.get("ENTRE_CANCIONES"):
            playlist_de_rutas.append(
                os.path.join(JINGLES_PATH, random.choice(jingles_cierre))
            )

        # --- FIN DE LA LÓGICA CORREGIDA ---

        # Reproducir la secuencia completa
        for item_ruta in playlist_de_rutas:
            if voice_client.is_connected():
                await reproducir_archivo(voice_client, item_ruta)
            else:
                logger.warning(
                    "⚠️ La reproducción se detuvo porque el bot fue desconectado."
                )
                break

        ultimo_jingle_relleno = time.time()

    except Exception as e:
        logger.error(f"❌ Error en reproducir_playlist: {e}", exc_info=True)
    finally:
        playlist_en_curso = False
        logger.info(
            "⏹️ Finalizando playlist. Flag 'playlist_en_curso' establecido a False."
        )
        # Notificar al frontend que la cola ha cambiado (se ha quitado un elemento)
        emit_with_context("queue_updated", {})


# ==============================================================================
# ||      TAREA procesar_cola_canciones CON DESCARGA NO BLOQUEANTE           ||
# ==============================================================================


@tasks.loop(seconds=10.0)
async def procesar_cola_canciones():
    # LÍNEA DE DEBUG CRÍTICA - mantén esto temporalmente
    logger.info(
        f"🔍 DEBUG: Revisando colas. cola_canciones: {len(cola_canciones)}, cola_reproduccion: {len(cola_reproduccion)}"
    )

    if not cola_canciones:
        return

    # No procesar más si la cola de reproducción ya tiene canciones esperando
    if len(cola_reproduccion) > 1:
        logger.info(
            "🔥 Pausando descargas, la cola de reproducción tiene items esperando."
        )
        return

    logger.info("🔥 Iniciando procesamiento de la siguiente petición en cola.")

    try:
        cancion_o_letra, autor, dedicatoria_info = cola_canciones.popleft()

        # DEBUG: Mostrar exactamente qué estamos procesando
        logger.info(
            f"🎵 Procesando: '{cancion_o_letra}' de {autor} (tipo: {type(autor)})"
        )

        titulo_final = cancion_o_letra
        nombre_archivo = sanitizar_nombre_archivo(titulo_final)
        ruta_archivo = os.path.join(DOWNLOAD_PATH, nombre_archivo)

        # 1. LÓGICA DE DESCARGA
        if not os.path.exists(ruta_archivo):
            logger.info(f"⬇️ Descargando '{titulo_final}'... (Esto puede tardar)")
            loop = asyncio.get_running_loop()
            ruta_descargada = await loop.run_in_executor(
                None,
                functools.partial(
                    descargar_audio_youtube, titulo_final, nombre_archivo
                ),
            )
            if not ruta_descargada:
                logger.error(f"❌ La descarga de '{titulo_final}' falló.")
                # Enviar notificación de error al chat
                emit_with_context(
                    "mensaje_a_cliente",
                    {
                        "texto": f"❌ No se pudo descargar '{titulo_final}'. Intenta con otro nombre.",
                        "usuario": "Bot SONARIA",
                        "esBot": True,
                    },
                )
                return
        else:
            logger.info(f"📁 Archivo '{titulo_final}' ya existe, usando versión local.")

        logger.info(f"✅ '{titulo_final}' listo para reproducir.")

        # 2. LÓGICA DE METADATOS (carátula, recomendaciones, etc.)
        titulo_base = titulo_final.split(" - ")[0]
        artista_base = (
            titulo_final.split(" - ")[1]
            if " - " in titulo_final
            else "Artista Desconocido"
        )
        cover_url = None

        # Obtener carátula de Spotify
        if sp:
            try:
                query = f"{titulo_base} {artista_base}"
                results = await bot.loop.run_in_executor(
                    None, lambda: sp.search(q=query, type="track", limit=1)
                )
                if results["tracks"]["items"]:
                    cover_url = results["tracks"]["items"][0]["album"]["images"][0][
                        "url"
                    ]
                    logger.info(f"🎨 Carátula encontrada para '{titulo_final}'")
            except Exception as e:
                logger.warning(f"No se pudo obtener la carátula de Spotify: {e}")

        # Obtener recomendaciones
        try:
            loop = asyncio.get_running_loop()
            nuevas_sugerencias = await loop.run_in_executor(
                None, obtener_recomendaciones_deezer, artista_base
            )
            if nuevas_sugerencias:
                global recomendaciones_actuales
                recomendaciones_actuales = nuevas_sugerencias
                emit_with_context("recommendations_updated", recomendaciones_actuales)
                logger.info(
                    f"🎯 Nuevas recomendaciones generadas basadas en {artista_base}"
                )
        except Exception as e:
            logger.warning(f"Error obteniendo recomendaciones: {e}")

        # 3. AÑADIR A LA COLA DE REPRODUCCIÓN (¡LA PARTE MÁS CRÍTICA!)
        cola_reproduccion.append(
            {
                "ruta_archivo": ruta_archivo,
                "titulo": titulo_final,
                "autor": str(autor),  # Asegurar que sea string
                "dedicatoria": dedicatoria_info,
                "cover_url": cover_url,
            }
        )

        logger.info(
            f"🎶 '{titulo_final}' añadido a la cola de reproducción. Cola actual: {len(cola_reproduccion)} elementos"
        )

        # Notificar a la UI que la cola ha cambiado
        emit_with_context("queue_updated", {})

        # 4. ENVIAR NOTIFICACIÓN DE ÉXITO
        frase_descarga = random.choice(FRASES_DESCARGA_COMPLETADA).format(
            cancion=titulo_final
        )
        emit_with_context(
            "mensaje_a_cliente",
            {"texto": frase_descarga, "usuario": "Bot SONARIA", "esBot": True},
        )
        logger.info(
            f"✔️ Notificación de descarga completada enviada al chat para '{titulo_final}'."
        )

        # 5. INTERRUMPIR MÚSICA DE FONDO (OPCIONAL PERO RECOMENDADO)
        if bot.voice_clients:
            voice_client = bot.voice_clients[0]
            if voice_client.is_playing() and not playlist_en_curso:
                logger.info(
                    "⚡️ Canción lista. Interrumpiendo audio de fondo para dar paso a la petición."
                )
                voice_client.stop()

    except Exception as e:
        logger.error(
            f"❌ Error catastrófico en procesar_cola_canciones: {e}", exc_info=True
        )
        # Enviar notificación de error
        emit_with_context(
            "mensaje_a_cliente",
            {
                "texto": "❌ Hubo un error procesando la última petición. Inténtalo de nuevo.",
                "usuario": "Bot SONARIA",
                "esBot": True,
            },
        )


# --- NUEVA VARIABLE GLOBAL PARA COMANDOS DE LA WEB ---
web_command_queue = asyncio.Queue()


async def process_web_commands():
    """Procesa comandos que llegan desde la interfaz web."""
    while True:
        # Espera un nuevo comando de la cola (esto es no-bloqueante)
        command_data = await web_command_queue.get()
        logger.info(f"Comando de la web recibido: {command_data}")

        command = command_data.get("command")
        user_data = command_data.get("user_data")

        # Aquí manejas los diferentes comandos
        if command == "join":
            channel_id = int(user_data.get("channel_id"))

            # Obtiene el objeto del canal de voz
            voice_channel = bot.get_channel(channel_id)
            if voice_channel and isinstance(voice_channel, discord.VoiceChannel):
                # Conecta el bot al canal de voz
                logger.info(
                    f"Intentando conectar al canal de voz: {voice_channel.name}"
                )
                await voice_channel.connect()
                emit_with_context(
                    "bot_status",
                    {
                        "message": f"✅ Conectado al canal: {voice_channel.name}",
                        "status": "connected",
                    },
                )
            else:
                logger.warning(
                    f"ID de canal no válido o no es un canal de voz: {channel_id}"
                )
                emit_with_context(
                    "bot_status",
                    {
                        "message": "❌ ID de canal no válido. Asegúrate de que es un canal de voz.",
                        "status": "error",
                    },
                )

        # Si quieres añadir más comandos, puedes hacerlo aquí
        elif command == "leave":
            # Lógica para que el bot salga del canal
            if bot.voice_clients:
                await bot.voice_clients[0].disconnect()
                emit_with_context(
                    "bot_status",
                    {"message": "👋 Bot desconectado.", "status": "disconnected"},
                )


@bot.command(name="empezar", aliases=["start"])
@commands.has_role("DJ")
async def empezar(ctx):
    global radio_activa, ultimo_jingle_relleno, canal_radio_id
    if radio_activa:
        return await ctx.send("¡La radio ya está en marcha!")
    voice_client = ctx.guild.voice_client
    if not voice_client:
        if not ctx.author.voice:
            return await ctx.send(
                f"{ctx.author.mention}, ¡debes estar en un canal de voz!"
            )
        try:
            voice_client = await ctx.author.voice.channel.connect()
        except Exception as e:
            return await ctx.send(f"❌ No pude unirme al canal de voz: {e}")

    radio_activa = True
    canal_radio_id = voice_client.channel.id
    emit_with_context("bot_status_update", {"is_ready": True})
    logger.info("✅ Radio activada. Señal 'is_ready: True' enviada a los clientes.")

    await ctx.send("📻 **¡Iniciando SONARIA Radio!**")
    if BANCO_JINGLES.get("APERTURA"):
        jingle_apertura = os.path.join(
            JINGLES_PATH, random.choice(BANCO_JINGLES["APERTURA"])
        )
        await reproducir_archivo(voice_client, jingle_apertura)

    ultimo_jingle_relleno = time.time()
    if not radio_manager.is_running():
        radio_manager.start(ctx)
    await ctx.send("📻 **¡SONARIA Radio está AL AIRE!**")


@bot.event
async def on_ready():
    # El nombre de usuario puede tener un # y números, lo limpiamos para el log
    bot_username = str(bot.user).split("#")[0]
    print(f"✅ ¡Conectado como {bot_username}!")
    logger.info(
        f"✅ ¡Conectado como {bot_username}!"
    )  # Añadimos un logger para consistencia

    # 🚀 Inicia la tarea que procesa los comandos de la web

    bot.loop.create_task(process_web_commands())

    if not procesar_cola_canciones.is_running():
        logger.info("🚀 Iniciando la tarea 'procesar_cola_canciones'.")
        procesar_cola_canciones.start()

    if not limpiar_archivos_antiguos.is_running():
        logger.info("🚀 Iniciando la tarea 'limpiar_archivos_antiguos'.")
        limpiar_archivos_antiguos.start()


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.channel.name == "peticiones" and "cancion" in normalizar_texto(
        message.content
    ):
        texto_original = message.content
        dedicatoria_data = parsear_dedicatoria(texto_original)

        if dedicatoria_data:
            nombre, remitente, frase, cancion = dedicatoria_data
            cancion_normalizada = normalizar_nombre_cancion(cancion)
            info_dedicatoria = {
                "para": nombre,
                "de": remitente,
                "mensaje": frase,
            }
            cola_canciones.append(
                (cancion_normalizada, message.author.display_name, info_dedicatoria)
            )
            print(f"✅ Pedido con dedicatoria AÑADIDO: {cancion_normalizada}")
            await message.add_reaction("💌")
        else:
            texto_norm_simple = normalizar_texto(texto_original)
            match_simple = re.search(r"cancion[:.,\s]*(.*)", texto_norm_simple)
            if match_simple:
                cancion = match_simple.group(1).strip()
                if cancion:
                    cancion_normalizada = normalizar_nombre_cancion(cancion)
                    cola_canciones.append(
                        (cancion_normalizada, message.author.display_name, None)
                    )
                    print(f"✅ Pedido simple AÑADIDO: {cancion_normalizada}")
                    await message.add_reaction("👍")

    await bot.process_commands(message)


# --- RESTO DE COMANDOS EXISTENTES ---


# Pausar audio actual
@bot.command()
async def pause(ctx):
    if ctx.voice_client and ctx.voice_client.is_playing():
        ctx.voice_client.pause()
        await ctx.send("⏸️ Reproducción pausada.")
    else:
        await ctx.send("⚠️ No hay nada reproduciéndose.")


# Reanudar audio pausado
@bot.command()
async def resume(ctx):
    if ctx.voice_client and ctx.voice_client.is_paused():
        ctx.voice_client.resume()
        await ctx.send("▶️ Reproducción reanudada.")
    else:
        await ctx.send("⚠️ No hay nada pausado.")


@bot.command()
async def test(ctx):
    if ctx.voice_client:
        ruta = "musica_fondo.mp3"  # pon el nombre exacto de un mp3 que tengas en la carpeta
        print("Probando:", ruta)
        ctx.voice_client.play(AudioLevelSource(discord.FFmpegPCMAudio(ruta)))
        await ctx.send("▶️ Probando reproducción local.")
    else:
        await ctx.send("⚠️ No estoy en un canal de voz.")


@bot.command(name="unirse", aliases=["join"])
async def unirse(ctx):
    global canal_radio_id
    if not ctx.author.voice:
        return await ctx.send(
            f"{ctx.author.mention}, ¡primero debes conectarte a un canal de voz!"
        )
    if not ctx.guild.voice_client:
        await ctx.author.voice.channel.connect()
        canal_radio_id = ctx.author.voice.channel.id
        await ctx.send(f"¡Hola! Me he unido a **{ctx.author.voice.channel.name}**.")
    else:
        await ctx.send("Ya estoy en un canal de voz.")


@bot.command(name="salir", aliases=["leave"])
@commands.has_role("DJ")
async def salir(ctx):
    global is_bot_ready_for_webrtc
    if ctx.guild.voice_client and ctx.guild.voice_client.is_connected():
        await parar(ctx)
        await ctx.guild.voice_client.disconnect()
        print("❌ Bot desconectado. WebRTC no disponible.")
        is_bot_ready_for_webrtc = False
        # Avisamos a TODOS los clientes web que ya NO estamos listos
        emit_with_context("bot_status_update", {"is_ready": False})
    else:
        await ctx.send("No estoy conectado a ningún canal de voz.")


@bot.command(name="parar", aliases=["stop"])
@commands.has_role("DJ")
async def parar(ctx):
    global radio_activa, canal_radio_id
    if not radio_activa:
        return await ctx.send("La radio no está en marcha.")

    radio_activa = False
    canal_radio_id = None
    emit_with_context("bot_status_update", {"is_ready": False})
    logger.info("❌ Radio detenida. Señal 'is_ready: False' enviada a los clientes.")

    if radio_manager.is_running():
        radio_manager.cancel()
    if ctx.guild.voice_client and ctx.guild.voice_client.is_playing():
        ctx.guild.voice_client.stop()
    cola_canciones.clear()
    cola_reproduccion.clear()
    await ctx.send("📻 La radio se ha detenido.")


@bot.command(name="vercola")
async def ver_cola(ctx):
    if not cola_canciones and not cola_reproduccion:
        return await ctx.send("🤔 La cola está vacía.")
    msg = "--- 🎶 Cola de Peticiones 🎶 ---\n"
    msg += (
        "\n".join(
            f"{i+1}. **{c[0].title()}** (de {c[1].display_name})"
            for i, c in enumerate(cola_canciones)
        )
        or "No hay nuevas peticiones."
    )
    msg += "\n\n--- 💿 En Espera para Reproducir 💿 ---\n"
    msg += (
        "\n".join(
            f"{i+1}. **{item['titulo'].title()}**"
            for i, item in enumerate(cola_reproduccion)
        )
        or "Nada listo para sonar."
    )
    await ctx.send(msg)


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.MissingRole):
        await ctx.send(
            f"Lo siento {ctx.author.mention}, no tienes el rol de 'DJ' para usar este comando. 🚫"
        )
    else:
        print(f"Ocurrió un error no manejado: {error}")


# --- CONFIGURACIÓN DE FLASK CON WEBSOCKETS ---
flask_app = Flask(__name__, static_folder="frontend", static_url_path="")
flask_app.secret_key = JWT_SECRET_KEY
CORS(flask_app)
socketio = SocketIO(flask_app, cors_allowed_origins="*", async_mode="threading")

# Añade esta nueva ruta a tu sección de API de Flask


@flask_app.route("/api/recommendations", methods=["GET"])
def obtener_recomendaciones():
    # Si ya hemos generado recomendaciones dinámicas, las servimos.
    if recomendaciones_actuales:
        return jsonify(recomendaciones_actuales)

    # Si no, servimos las iniciales desde el archivo JSON.
    if banco_recomendaciones:
        return jsonify(
            random.sample(banco_recomendaciones, min(5, len(banco_recomendaciones)))
        )

    return jsonify([])


# NUEVO: Ruta para servir videos estáticos
@flask_app.route("/videos/<path:filename>")
def serve_video(filename):
    # Asumimos que la carpeta 'videos' está DENTRO de la carpeta 'frontend'
    video_path = os.path.join(flask_app.static_folder, "videos")
    return send_from_directory(video_path, filename)


# --- RUTAS DE LA API WEB ---


@socketio.on("web_command")
def handle_web_command(data):
    """Maneja los comandos que llegan desde el frontend y los encola para el bot."""
    command = data.get("command")
    # Añade el comando a la cola. Esto es no-bloqueante
    asyncio.run_coroutine_threadsafe(
        web_command_queue.put({"command": command, "user_data": data}), bot.loop
    )
    logger.info(f"Comando '{command}' encolado desde la web.")


@socketio.on("nuevo_mensaje_chat")
def handle_chat_message(data):
    """
    Recibe un mensaje de un cliente y lo retransmite a todos los demás.
    """
    texto = data.get("texto")
    usuario = data.get("usuario", "Anónimo")

    if not texto:
        return

    logger.info(f"💬 Mensaje de chat recibido de {usuario}: '{texto}'")

    # Retransmitir el mensaje a TODOS los clientes conectados, incluyéndolo a él mismo
    # El frontend ya sabe cómo diferenciar si el mensaje es del propio usuario o de otro.
    emit_with_context(
        "mensaje_a_cliente",
        {
            "texto": texto,
            "usuario": usuario,
            "esBot": False,  # Marcamos que no es un mensaje del bot
        },
    )


@flask_app.route("/")
def index():
    # Usamos send_from_directory para servir el archivo HTML principal.
    # Es más seguro que otras opciones. [1, 3, 5]
    return send_from_directory(flask_app.static_folder, "index.html")


@socketio.on("peticion_desde_cliente")
def handle_song_request_from_client(data):
    usuario_nombre = data.get("usuario", "Usuario Web")
    cancion = data.get("cancion")
    dedicatoria = data.get("dedicatoria")
    mensaje = data.get("mensaje")

    if not cancion:
        return

    logger.info(f"✅ Petición por Socket.IO recibida: '{cancion}' de {usuario_nombre}")

    info_dedicatoria = None
    if dedicatoria:
        info_dedicatoria = {
            "para": dedicatoria,
            "de": usuario_nombre,
            "mensaje": mensaje,
        }

    cancion_normalizada = normalizar_nombre_cancion(cancion)

    # ASEGURAR que autor es un string, no un objeto de Discord
    cola_canciones.append((cancion_normalizada, str(usuario_nombre), info_dedicatoria))

    # DEBUG LOG - mantén esto temporalmente
    logger.info(
        f"🔍 Cola después de añadir: cola_canciones={len(cola_canciones)}, elementos={list(cola_canciones)}"
    )

    # Notificar a todos que la cola de peticiones visual se actualizó
    emit_with_context("queue_updated")

    # Enviar mensaje de CONFIRMACIÓN del Bot al chat
    try:
        if info_dedicatoria:
            frase = random.choice(FRASES_DEDICATORIA_RECIBIDA).format(
                para=info_dedicatoria["para"], cancion=cancion_normalizada
            )
        else:
            frase = random.choice(FRASES_PETICION_RECIBIDA).format(
                cancion=cancion_normalizada
            )

        emit_with_context(
            "mensaje_a_cliente",
            {"texto": frase, "usuario": "Bot SONARIA", "esBot": True},
        )
        logger.info(
            f"✔️ Notificación de petición recibida enviada al chat para '{cancion_normalizada}'."
        )

    except Exception as e:
        logger.error(
            f"❌ Error al enviar notificación de chat en handle_song_request: {e}"
        )


@flask_app.route("/api/current-song", methods=["GET"])
def obtener_cancion_actual():
    return jsonify(
        {
            "song": cancion_actual.get("titulo"),
            "artist": cancion_actual.get("artista"),
            "requested_by": cancion_actual.get("usuario"),
        }
    )


@flask_app.route("/api/queue", methods=["GET"])
def obtener_cola():
    cola_data = []
    for item in list(cola_canciones):
        cola_data.append(
            {
                "title": item[0],
                "user": (
                    item[1].display_name
                    if hasattr(item[1], "display_name")
                    else str(item[1])
                ),
                "dedication": item[2] if item[2] else None,
            }
        )

    reproduccion_data = []
    for item in list(cola_reproduccion):
        reproduccion_data.append(
            {
                "title": item["titulo"],
                "user": (
                    item["autor"].display_name
                    if hasattr(item["autor"], "display_name")
                    else str(item["autor"])
                ),
            }
        )

    return jsonify({"pending": cola_data, "ready": reproduccion_data})


# --- RUTAS OAUTH (básico) ---
@flask_app.route("/auth/discord")
def auth_discord():
    """
    Redirige al usuario a la página de autorización de Discord.
    """
    if not DISCORD_CLIENT_ID:
        return jsonify({"error": "OAuth no está configurado en el servidor."}), 500

    # El scope 'guilds.join' es crucial para poder añadir al usuario al servidor.
    # El scope 'identify' nos permite ver su información básica.
    scopes = "identify guilds.join"
    redirect_uri = request.url_root + "auth/callback"
    discord_auth_url = (
        f"https://discord.com/api/oauth2/authorize?client_id={DISCORD_CLIENT_ID}"
        f"&redirect_uri={redirect_uri}&response_type=code&scope={scopes}"
    )
    return redirect(discord_auth_url)


@flask_app.route("/auth/callback")
async def auth_callback():
    code = request.args.get("code")
    if not code:
        return "Error: No se recibió el código de autorización.", 400

    # --- 1. Intercambiar código por token de acceso ---
    token_url = "https://discord.com/api/oauth2/token"
    # ¡CORRECCIÓN IMPORTANTE! Asegúrate de que la redirect_uri aquí sea idéntica
    # a la que tienes en el portal de Discord.
    redirect_uri = request.url_root.rstrip("/") + "/auth/callback"
    data = {
        "client_id": DISCORD_CLIENT_ID,
        "client_secret": DISCORD_CLIENT_SECRET,
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    async with aiohttp.ClientSession() as session_http:
        async with session_http.post(token_url, data=data, headers=headers) as resp:
            if resp.status != 200:
                token_error = await resp.text()
                logger.error(f"Error al obtener token de Discord: {token_error}")
                return "Error al obtener token de Discord.", 500
            token_data = await resp.json()
            access_token = token_data.get("access_token")

    # --- 2. Obtener info del usuario y unir al servidor ---
    user_info = await obtener_usuario_discord(access_token)
    if not user_info:
        return "Error: No se pudo obtener la información del usuario.", 500

    # Unir al usuario al servidor (esto ya debería funcionar)
    await unir_usuario_servidor(access_token, user_info["id"])

    # 🚀 --- NUEVA LÓGICA CLAVE: ENVIAR COMANDO AL BOT PARA UNIRSE AL CANAL DE LA EMISORA ---
    if VOICE_CHANNEL_ID:
        # Poner el comando en la cola de forma no-bloqueante
        web_command_queue.put_nowait(
            {"command": "join", "user_data": {"channel_id": VOICE_CHANNEL_ID}}
        )
        logger.info(f"Comando 'join' para el canal de la radio encolado.")

    # --- 3. Crear JWT ---
    jwt_token = generar_jwt_token(user_info)

    # --- 4. Redirigir con un script que guarda el token ---
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Autenticando...</title>
        <script>
            // 1. Guarda el token recibido en el localStorage del navegador.
            localStorage.setItem('sonaria_jwt_token', '{jwt_token}');
            
            // 2. Redirige al usuario a la página principal.
            window.location.href = "/";
        </script>
    </head>
    <body>
        <p>Redirigiendo...</p>
    </body>
    </html>
    """


# REEMPLAZA tu ruta /api/auth/verify con esta


@flask_app.route("/api/auth/verify", methods=["GET"])
def verificar_auth():
    # El token ahora viene en la cabecera 'Authorization' desde el frontend
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"valid": False}), 401

    token = auth_header.split(" ")[1]

    user_data = verificar_jwt_token(token)
    if user_data:
        return jsonify(
            {
                "valid": True,
                "user": {"id": user_data["user_id"], "username": user_data["username"]},
            }
        )
    else:
        # Si el token es inválido, el frontend debería encargarse de borrarlo.
        return jsonify({"valid": False}), 401


@flask_app.route("/api/join-voice", methods=["POST"])
def unir_canal_voz():
    """Endpoint para unir usuario al canal de voz desde la web"""
    datos = request.json
    user_id = datos.get("user_id")

    if not user_id or not canal_radio_id:
        return jsonify({"success": False, "message": "Usuario o canal no válido"}), 400

    # En una implementación completa, aquí moverías al usuario
    # Por ahora, solo confirmamos que se puede hacer
    return jsonify({"success": True, "message": f"Conectándote al canal de radio..."})


@flask_app.route("/api/radio-info", methods=["GET"])
def get_radio_info():
    if radio_activa and canal_radio_id:
        # Construimos el enlace directo al canal de voz
        voice_channel_link = (
            f"https://discord.com/channels/{DISCORD_GUILD_ID}/{canal_radio_id}"
        )
        return jsonify({"is_ready": True, "voice_channel_link": voice_channel_link})
    else:
        return jsonify({"is_ready": False})


# --- WEBSOCKETS ---
@socketio.on("connect")
def handle_connect():
    sid = request.sid
    logger.info(f"Cliente web conectado: {sid}")
    emit("bot_status_update", {"is_ready": radio_activa}, room=sid)
    emit("now_playing", cancion_actual, room=sid)


# VERSIÓN A PRUEBA DE BALAS
@socketio.on("disconnect")
def handle_disconnect():
    logger.info(f"🔌 Cliente web desconectado: {request.sid}")
    usuarios_conectados.discard(request.sid)


def run_flask():
    try:
        socketio.run(
            flask_app, host="0.0.0.0", port=8080, debug=False, use_reloader=False
        )
    except Exception as e:
        print(f"❌ Error en el servidor Flask: {e}")


# --- INICIO DEL SISTEMA ---


def run_server():
    """Función para iniciar el servidor web."""
    # Usamos el puerto 8080 como lo tenías
    socketio.run(flask_app, host="0.0.0.0", port=8080)


bot_thread = None
is_bot_running = False


# --- INICIO DEL SISTEMA (VERSIÓN FINAL Y CORRECTA PARA PRODUCCIÓN) ---


def run_bot_in_thread():
    """
    Esta función se ejecutará en un hilo separado.
    Crea un nuevo event loop de asyncio exclusivo para el bot de Discord,
    evitando conflictos con el servidor web.
    """
    logger.info("🤖 Iniciando hilo para el bot de Discord...")
    try:
        # 1. Crear un nuevo loop de asyncio para este hilo
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # 2. Iniciar el bot de Discord. Esta llamada es bloqueante y mantendrá el hilo vivo.
        logger.info("🔌 Conectando el bot a Discord...")
        loop.run_until_complete(bot.start(DISCORD_TOKEN))

    except Exception as e:
        # Captura cualquier error catastrófico que pueda ocurrir en el bot
        logger.error(
            f"❌ Error fatal en el hilo del bot de Discord: {e}", exc_info=True
        )
    finally:
        logger.info("🛑 El hilo del bot de Discord ha finalizado.")


# ==============================================================================
# || ESTA ES LA PARTE CLAVE QUE RESTAURA LA LÓGICA DE INICIO INMEDIATO         ||
# ==============================================================================
# Cuando Gunicorn carga este archivo para encontrar "flask_app", este código
# en el nivel superior se ejecuta automáticamente.
logger.info(
    "🚀 Módulo sonaria.py cargado. Iniciando el bot de Discord en segundo plano..."
)

# Creamos e iniciamos el hilo del bot.
# `daemon=True` asegura que el hilo se cierre si el programa principal (Gunicorn) termina.
bot_thread = threading.Thread(target=run_bot_in_thread, daemon=True)
bot_thread.start()

# El código anterior inicia el bot. A partir de aquí, Gunicorn continuará
# y servirá la aplicación Flask (`flask_app`) en el hilo principal.
