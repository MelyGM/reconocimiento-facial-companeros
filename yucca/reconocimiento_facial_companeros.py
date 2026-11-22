#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================================================
# 1 - Paquetes
# =====================================================================
import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
import h5py
import cv2                       # OpenCV: detectar y recortar el rostro
import matplotlib
matplotlib.use('Agg')            # sin pantalla: las figuras se guardan como PNG
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

sys.stdout.reconfigure(line_buffering=True)   # los mensajes aparecen al momento en el .out de Slurm

print('TensorFlow:', tf.__version__, '| Keras:', keras.__version__, '| OpenCV:', cv2.__version__)
print('GPUs visibles:', tf.config.list_physical_devices('GPU') or 'ninguna (se usa CPU)')

# =====================================================================
# 1.1 - Rutas y parámetros
# =====================================================================
BASE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description='Reconocimiento facial de compañeros con FaceNet')
parser.add_argument('--carpeta', default=str(BASE), help='carpeta con el modelo y las imágenes')
parser.add_argument('--umbral', type=float, default=0.7, help='umbral de distancia L2')
parser.add_argument('--calibrar', action='store_true', help='usar el umbral sugerido por la calibración')
parser.add_argument('--probar', nargs='*', default=[], help='fotos nuevas para reconocer')
args = parser.parse_args()

CARPETA_PROYECTO = Path(args.carpeta).resolve()

# Modelo: facenet_recuperado.keras o model.json + model.h5
ARCHIVO_KERAS = CARPETA_PROYECTO / 'facenet_recuperado.keras'
CARPETA = CARPETA_PROYECTO / 'keras-facenet-h5' if (CARPETA_PROYECTO / 'keras-facenet-h5').is_dir() else CARPETA_PROYECTO
ARCHIVO_JSON = CARPETA / 'model.json'
ARCHIVO_H5 = CARPETA / 'model.h5'

# Imágenes: carpeta base_de_datos_imagenes o zip Fotos_para_FaceRecognition*.zip
CARPETA_IMAGENES = CARPETA_PROYECTO / 'base_de_datos_imagenes'
zips = sorted(CARPETA_PROYECTO.glob('Fotos_para_FaceRecognition*.zip'))
RUTA_ZIP = zips[0] if zips else None
CARPETA_FOTOS = CARPETA_PROYECTO / 'fotos_companeros'          # aquí se descomprime el zip
CARPETA_SIMULADAS = CARPETA_PROYECTO / 'fotos_simuladas'

# Resultados
CARPETA_SALIDA = CARPETA_PROYECTO / 'salida_reconocimiento'
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

# Umbral de distancia L2: distancia < UMBRAL  =>  misma persona
UMBRAL = args.umbral

assert ARCHIVO_KERAS.is_file() or (ARCHIVO_JSON.is_file() and ARCHIVO_H5.is_file()), \
    f'Falta el modelo: pon {ARCHIVO_KERAS.name} o model.json + model.h5 en {CARPETA_PROYECTO}'
assert CARPETA_IMAGENES.is_dir() or RUTA_ZIP, \
    f'Faltan las fotos: pon la carpeta base_de_datos_imagenes o Fotos_para_FaceRecognition.zip en {CARPETA_PROYECTO}'
print('Carpeta del proyecto:', CARPETA_PROYECTO)
print('Fotos desde:', CARPETA_IMAGENES if CARPETA_IMAGENES.is_dir() else RUTA_ZIP)


def guardar_figura(nombre):
    """Guarda la figura actual como PNG en la carpeta de resultados."""
    ruta = CARPETA_SALIDA / nombre
    plt.savefig(ruta, dpi=110, bbox_inches='tight')
    plt.close()
    print('Figura guardada:', ruta)


# =====================================================================
# 2 - Verificación ingenua (píxel a píxel)
# =====================================================================
def distancia_pixeles(ruta_a, ruta_b, tam=160):
    """Distancia L2 entre dos imágenes crudas (normalizadas a [0,1] y del mismo tamaño)."""
    a = np.asarray(Image.open(ruta_a).convert('RGB').resize((tam, tam)), dtype=np.float32) / 255.0
    b = np.asarray(Image.open(ruta_b).convert('RGB').resize((tam, tam)), dtype=np.float32) / 255.0
    return float(np.linalg.norm(a - b))


# =====================================================================
# 3 - Codificar rostros en un vector de 128 dimensiones: Triplet Loss
# =====================================================================
def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Triplet loss según la fórmula (3) de FaceNet.

    y_true -- etiquetas (Keras las exige en una pérdida; aquí no se usan)
    y_pred -- lista con tres tensores de forma (None, 128):
              anchor (ancla), positive (misma persona), negative (otra persona)
    alpha  -- margen mínimo entre la distancia positiva y la negativa
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Paso 1: distancia (al cuadrado) entre ancla y positivo -> queremos que sea PEQUEÑA
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Paso 2: distancia (al cuadrado) entre ancla y negativo -> queremos que sea GRANDE
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Paso 3: diferencia de distancias más el margen alpha
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Paso 4: max(basic_loss, 0) y suma sobre todos los ejemplos
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss


tf.random.set_seed(1)
y_true = (None, None, None)
y_pred = (tf.random.normal([3, 128], mean=6, stddev=0.1, seed=1),
          tf.random.normal([3, 128], mean=1, stddev=1, seed=1),
          tf.random.normal([3, 128], mean=3, stddev=4, seed=1))
print('Triplet loss:', float(triplet_loss(y_true, y_pred)))

# Casos simples con resultado conocido: (ancla, positivo, negativo, alpha, esperado)
for a, p, n, alpha, esperado in [
    ([[1., 1.]], [[1., 1.]], [[1., 1.]], 5, 5),
    ([[1., 1.]], [[1., 1.]], [[0., 0.]], 3, 1),
    ([[1., 1.]], [[0., 0.]], [[1., 1.]], 0, 2),
    ([[0., 0.]], [[0., 0.]], [[0., 0.]], -2, 0),
]:
    real = triplet_loss(y_true, (tf.constant(a), tf.constant(p), tf.constant(n)), alpha)
    assert np.isclose(float(real), esperado), (real, esperado)
print('Pruebas básicas de Triplet Loss superadas.')


# =====================================================================
# 4 - Cargar el modelo preentrenado
# =====================================================================
@keras.saving.register_keras_serializable(package='FaceNet')
class ResidualScaleSum(keras.layers.Layer):
    """Suma residual escalada de FaceNet: salida = a + scale * b."""
    def __init__(self, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = float(scale)

    def call(self, inputs):
        a, b = inputs
        return a + self.scale * b

    def get_config(self):
        return {**super().get_config(), 'scale': self.scale}


def reconstruir_facenet(ruta_json):
    """Recorre las capas del JSON en orden y las vuelve a crear con la API funcional de Keras 3."""
    with open(ruta_json, encoding='utf-8') as archivo:
        especificacion = json.load(archivo)
    capas = especificacion['config']['layers']
    tensores = {}   # nombre de capa -> tensor de salida

    for dato in capas:
        tipo, nombre = dato['class_name'], dato['name']
        config = dict(dato['config'])

        # Capa de entrada: (160, 160, 3)
        if tipo == 'InputLayer':
            forma = config.get('batch_input_shape', config.get('batch_shape'))
            tensores[nombre] = keras.Input(shape=tuple(forma[1:]), dtype=config.get('dtype', 'float32'), name=nombre)
            continue

        if tipo == 'Lambda':
            # No se ejecuta código Python embebido en el JSON; solo se admiten las sumas residuales.
            if not nombre.endswith('_ScaleSum'):
                raise ValueError(f'Lambda desconocida: {nombre}')
            capa = ResidualScaleSum(name=nombre, scale=config['arguments']['scale'])
        else:
            # El BatchNormalization antiguo guarda axis como lista [3]; Keras 3 espera un entero.
            if tipo == 'BatchNormalization' and isinstance(config.get('axis'), list):
                config['axis'] = config['axis'][0]
            # Campos antiguos que ya no son parámetros del constructor.
            config.pop('dtype', None)
            config.pop('trainable', None)
            capa = getattr(keras.layers, tipo)(**config)

        # Conectar la capa con sus entradas (todas las conexiones son del nodo 0).
        entradas = [tensores[conexion[0]] for conexion in dato['inbound_nodes'][0]]
        tensores[nombre] = capa(entradas if len(entradas) > 1 else entradas[0])

    entrada = especificacion['config']['input_layers'][0][0]
    salida = especificacion['config']['output_layers'][0][0]
    return keras.Model(tensores[entrada], tensores[salida], name='inception_resnet_v1')


def cargar_pesos_originales(model, ruta_h5):
    """Copia los pesos del H5 a cada capa, verificando nombres y formas."""
    total = 0
    with h5py.File(ruta_h5, 'r') as archivo:
        for capa in model.layers:
            esperados = capa.get_weights()
            grupo = archivo.get(capa.name)
            if grupo is None:
                raise KeyError(f'Capa ausente en H5: {capa.name}')
            nombres = [n.decode('utf-8') if isinstance(n, bytes) else n
                       for n in grupo.attrs.get('weight_names', [])]
            valores = [np.asarray(grupo[n]) for n in nombres]
            for v, e in zip(valores, esperados):
                assert v.shape == e.shape, f'{capa.name}: {v.shape} != {e.shape}'
            if valores:
                capa.set_weights(valores)
                total += sum(v.size for v in valores)
    return total


if ARCHIVO_KERAS.is_file():
    # Cargar el modelo ya reconstruido
    model = keras.models.load_model(ARCHIVO_KERAS)
    print('Modelo cargado desde', ARCHIVO_KERAS.name)
else:
    # Reconstruir desde el JSON antiguo y copiar los pesos del H5
    model = reconstruir_facenet(ARCHIVO_JSON)
    cargar_pesos_originales(model, ARCHIVO_H5)
    print('Modelo reconstruido desde model.json + model.h5')
print('Capas:', len(model.layers), '| Parámetros:', f'{model.count_params():,}')
print('Entrada:', model.input_shape, '| Salida:', model.output_shape)
assert model.input_shape[1:] == (160, 160, 3) and model.output_shape[-1] == 128

# Prueba técnica: una imagen negra debe producir 128 valores finitos.
y = np.asarray(model(np.zeros((1, 160, 160, 3), np.float32), training=False))
assert y.shape == (1, 128) and np.isfinite(y).all()
FRmodel = model
print('Modelo FaceNet listo.')


# =====================================================================
# 5 - Aplicar el modelo a las fotos
# =====================================================================
EXTENSIONES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Usar la carpeta base_de_datos_imagenes si está; si no, descomprimir el zip.
if CARPETA_IMAGENES.is_dir():
    CARPETA_FOTOS = CARPETA_IMAGENES
else:
    CARPETA_FOTOS.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RUTA_ZIP) as z:                 # se ignoran carpetas basura como __MACOSX
        z.extractall(CARPETA_FOTOS)


def es_imagen(p):
    return (p.suffix.lower() in EXTENSIONES and '__MACOSX' not in p.parts
            and not p.name.startswith('.'))


def nombre_desde_archivo(ruta):
    """'Melina_Gonzalez (2).jpg' -> 'melina_gonzalez'  |  'Edelmira3.png' -> 'edelmira'"""
    base = re.sub(r'[\s_\-\.]*\(?\d+\)?$', '', ruta.stem)
    return base.strip(' _-.').lower().replace(' ', '_')


def cargar_fotos(carpeta):
    """Devuelve un diccionario {persona: [rutas de las fotos]}."""
    raiz = carpeta
    # Bajar mientras solo haya una carpeta (por ejemplo 'Fotos_para_FaceRecognition/').
    while True:
        hijos = [h for h in raiz.iterdir() if h.name != '__MACOSX' and not h.name.startswith('.')]
        if len(hijos) == 1 and hijos[0].is_dir():
            raiz = hijos[0]
        else:
            break

    fotos = {}
    # El nombre de la persona sale del archivo: alejandra_1.jpeg -> alejandra
    for p in sorted(raiz.iterdir()):
        if es_imagen(p):
            fotos.setdefault(nombre_desde_archivo(p), []).append(p)
    modo = 'nombre de la persona en el archivo'
    return fotos, modo


fotos_por_persona, modo = cargar_fotos(CARPETA_FOTOS)
print('Organización detectada:', modo)
for persona, rutas in fotos_por_persona.items():
    print(f'  {persona:20s} {len(rutas)} foto(s)')
print('Total de personas:', len(fotos_por_persona))


# ---------------------------------------------------------------------
# De imagen a codificación (img_to_encoding)
# Se recorta solo el rostro con el detector Haar de OpenCV.
# Si no encuentra rostro, se usa la imagen completa.
# ---------------------------------------------------------------------
DETECTOR = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if DETECTOR.empty():
    print('AVISO: no se pudo cargar el detector de rostros; se usarán las imágenes completas.')


def abrir_rgb(ruta):
    """Abre la imagen en RGB respetando la orientación EXIF (fotos de celular)."""
    with Image.open(ruta) as im:
        return ImageOps.exif_transpose(im).convert('RGB')


def recortar_rostro(img, margen=0.20):
    """
    Detecta el rostro MÁS GRANDE de la imagen y devuelve un recorte cuadrado con un margen.
    Devuelve (imagen_recortada, se_detecto_rostro).
    """
    if DETECTOR.empty():
        return img, False
    arr = np.asarray(img)
    gris = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    lado_min = max(40, min(gris.shape) // 10)          # ignora detecciones diminutas
    caras = DETECTOR.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5,
                                      minSize=(lado_min, lado_min))
    if len(caras) == 0:
        return img, False
    x, y, w, h = max(caras, key=lambda c: c[2] * c[3])  # la cara más grande
    cx, cy = x + w / 2, y + h / 2
    lado = max(w, h) * (1 + 2 * margen)
    izq, arr_ = int(max(0, cx - lado / 2)), int(max(0, cy - lado / 2))
    der, aba = int(min(arr.shape[1], cx + lado / 2)), int(min(arr.shape[0], cy + lado / 2))
    return img.crop((izq, arr_, der, aba)), True


def preparar(ruta, recortar=True):
    """Imagen lista para FaceNet: rostro recortado, 160x160, valores en [0, 1]."""
    img = abrir_rgb(ruta)
    detectado = False
    if recortar:
        img, detectado = recortar_rostro(img)
    img = img.resize((160, 160), Image.Resampling.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    return x, detectado


def img_to_encoding(image_path, model, recortar=True):
    """Codificación de 128 dimensiones normalizada (norma L2 = 1)."""
    x, _ = preparar(image_path, recortar)
    embedding = np.asarray(model(np.expand_dims(x, 0), training=False))
    norma = np.linalg.norm(embedding)
    if not np.isfinite(norma) or norma == 0:
        raise ValueError(f'Embedding no válido para {image_path}')
    return embedding / norma    # forma (1, 128)


# Revisión visual: rostro recortado de la primera foto de cada persona
personas = list(fotos_por_persona)
cols = min(6, len(personas))
filas = int(np.ceil(len(personas) / cols))
plt.figure(figsize=(2.2 * cols, 2.5 * filas))
sin_rostro = []
for i, persona in enumerate(personas):
    x, ok = preparar(fotos_por_persona[persona][0])
    plt.subplot(filas, cols, i + 1)
    plt.imshow(x); plt.axis('off')
    plt.title(persona + ('' if ok else '\n(sin rostro)'), fontsize=9)
plt.tight_layout()
guardar_figura('recortes.png')

# Lista de todas las fotos donde no se detectó rostro (se usarán completas).
for persona, rutas in fotos_por_persona.items():
    for r in rutas:
        if not preparar(r)[1]:
            sin_rostro.append(r.name)
print('Fotos sin rostro detectado:', sin_rostro if sin_rostro else 'ninguna')


# ---------------------------------------------------------------------
# Construir la base de datos
# Se guardan todas las fotos de registro de cada persona y se compara contra la más parecida.
# Para evaluar se necesita una foto de prueba que no esté en la base de datos:
# si solo hay una foto, se genera una simulada (espejo, otra luz, rotación, menor calidad).
# ---------------------------------------------------------------------
def simular_foto_camara(ruta, destino, semilla=0):
    """
    Crea una foto de cámara artificial a partir de una foto registrada:
    espejo, otra iluminación y contraste, ligera rotación, encuadre distinto y compresión JPEG.
    """
    rng = np.random.default_rng(semilla)
    img = ImageOps.mirror(abrir_rgb(ruta))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
    img = img.rotate(rng.uniform(-8, 8), resample=Image.Resampling.BILINEAR, fillcolor=(128, 128, 128))
    w, h = img.size                                      # recorte de bordes = encuadre distinto
    dx, dy = int(w * rng.uniform(0, 0.06)), int(h * rng.uniform(0, 0.06))
    img = img.crop((dx, dy, w - int(w * rng.uniform(0, 0.06)), h - int(h * rng.uniform(0, 0.06))))
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.2)))
    destino.parent.mkdir(parents=True, exist_ok=True)
    img.save(destino, quality=60)
    return destino


fotos_registro, fotos_prueba, simuladas = {}, {}, []
for i, (persona, rutas) in enumerate(fotos_por_persona.items()):
    if len(rutas) >= 2:
        fotos_registro[persona] = rutas[:-1]
        fotos_prueba[persona] = rutas[-1]
    else:
        fotos_registro[persona] = rutas
        # se prueba con una versión simulada de esa misma foto
        fotos_prueba[persona] = simular_foto_camara(rutas[0], CARPETA_SIMULADAS / f'{persona}_camara.jpg', semilla=i)
        simuladas.append(persona)

# database: {persona: matriz (k, 128)} con una fila por foto registrada
database = {
    persona: np.vstack([img_to_encoding(r, FRmodel) for r in rutas])
    for persona, rutas in fotos_registro.items()
}
print('Personas registradas:', len(database))
print('Fotos de prueba reales:', len(fotos_prueba) - len(simuladas), '| simuladas:', len(simuladas))

# Ver algunas fotos de prueba (las simuladas aparecen marcadas).
muestra = list(fotos_prueba)[:6]
plt.figure(figsize=(2.6 * len(muestra), 5.4))
for i, persona in enumerate(muestra):
    plt.subplot(2, len(muestra), i + 1)
    plt.imshow(preparar(fotos_registro[persona][0])[0]); plt.axis('off'); plt.title(f'{persona}\nregistro', fontsize=9)
    plt.subplot(2, len(muestra), len(muestra) + i + 1)
    plt.imshow(preparar(fotos_prueba[persona])[0]); plt.axis('off')
    plt.title('prueba' + (' (simulada)' if persona in simuladas else ''), fontsize=9)
plt.tight_layout()
guardar_figura('fotos_de_prueba.png')

# Comparación: distancia en píxeles vs. distancia entre codificaciones.
con_prueba = list(fotos_prueba)
if len(con_prueba) >= 2:
    p1, p2 = con_prueba[0], con_prueba[1]
    a, a2, b = fotos_registro[p1][0], fotos_prueba[p1], fotos_prueba[p2]
    print(f'Píxeles   -> misma persona ({p1}): {distancia_pixeles(a, a2):7.2f} | distinta ({p1} vs {p2}): {distancia_pixeles(a, b):7.2f}')
    ea, ea2, eb = (img_to_encoding(r, FRmodel) for r in (a, a2, b))
    print(f'FaceNet   -> misma persona ({p1}): {np.linalg.norm(ea - ea2):7.3f} | distinta ({p1} vs {p2}): {np.linalg.norm(ea - eb):7.3f}')


# ---------------------------------------------------------------------
# Verificación facial (verify 1:1)
# Alguien dice ser 'identity': se compara su foto solo con las fotos registradas
# de esa persona; si la distancia L2 es menor al umbral, se abre la puerta.
# ---------------------------------------------------------------------
def distancia_a_persona(encoding, db_enc):
    """Distancia L2 mínima entre una codificación (1,128) y las k fotos (k,128) de una persona."""
    return float(np.min(np.linalg.norm(db_enc - encoding, axis=1)))


def verify(image_path, identity, database, model, umbral=None):
    """
    Verifica si la persona de image_path es 'identity'.
    Devuelve (dist, door_open): distancia y True si se le deja pasar.
    """
    umbral = UMBRAL if umbral is None else umbral
    # Paso 1: codificación de la imagen
    encoding = img_to_encoding(image_path, model)
    # Paso 2: distancia con las fotos registradas de 'identity'
    dist = distancia_a_persona(encoding, database[identity])
    # Paso 3: abrir la puerta solo si la distancia es menor al umbral
    if dist < umbral:
        print(f"Es {identity}, ¡bienvenido!  (distancia {dist:.3f})")
        door_open = True
    else:
        print(f"No es {identity}, acceso denegado.  (distancia {dist:.3f})")
        door_open = False
    return dist, door_open


# Prueba 1: la foto de prueba de una persona contra su propio registro -> debería abrir.
# Prueba 2: la misma foto contra el registro de otra persona -> no debería abrir.
if len(con_prueba) >= 2:
    persona, otra = con_prueba[0], con_prueba[1]
    print(verify(fotos_prueba[persona], persona, database, FRmodel))
    print(verify(fotos_prueba[persona], otra, database, FRmodel))


# ---------------------------------------------------------------------
# Reconocimiento facial (who_is_it 1:K)
# Se busca en toda la base la codificación más cercana. Si la distancia mínima
# supera el umbral, el rostro no está registrado y se responde "desconocido".
# ---------------------------------------------------------------------
def who_is_it(image_path, database, model, umbral=None, verbose=True):
    """
    Identifica a la persona de image_path.
    Devuelve (min_dist, identity): identity es 'desconocido' si min_dist >= umbral.
    """
    umbral = UMBRAL if umbral is None else umbral
    # Codificación de la imagen
    encoding = img_to_encoding(image_path, model)

    # Buscar la persona más cercana
    min_dist, identity = 100, None                 # 100 = valor inicial "muy grande"
    for (name, db_enc) in database.items():
        dist = distancia_a_persona(encoding, db_enc)
        if dist < min_dist:
            min_dist, identity = dist, name

    # Manejar rostros no reconocidos
    if min_dist >= umbral:
        if verbose:
            print(f"No está en la base de datos (más parecido: {identity}, distancia {min_dist:.3f})")
        identity = 'desconocido'
    elif verbose:
        print(f"Es {identity}, la distancia es {min_dist:.3f}")
    return min_dist, identity


# Reconocer cada foto de prueba (ninguna de ellas está en la base de datos).
for persona, ruta in fotos_prueba.items():
    print(f'[{persona}] ', end='')
    who_is_it(ruta, database, FRmodel)


# =====================================================================
# 6 - Evaluación y calibración del umbral
# =====================================================================
# 6.1 Codificar todas las fotos (registro + prueba, incluidas las simuladas) una sola vez
todas = ([(p, r) for p, rutas in fotos_registro.items() for r in rutas]
         + [(p, r) for p, r in fotos_prueba.items()])
etiquetas = np.array([p for p, _ in todas])
E = np.vstack([img_to_encoding(r, FRmodel) for _, r in todas])

# Matriz de distancias entre todas las parejas
D = np.linalg.norm(E[:, None, :] - E[None, :, :], axis=-1)
iu = np.triu_indices(len(todas), k=1)
misma = (etiquetas[:, None] == etiquetas[None, :])[iu]
genuinas, impostoras = D[iu][misma], D[iu][~misma]

# Umbral que maximiza la exactitud de "misma persona / distinta"
valores = np.sort(np.unique(D[iu]))
candidatos = np.concatenate([[valores[0] - 1e-6], (valores[:-1] + valores[1:]) / 2, [valores[-1] + 1e-6]])
exactitud = [((genuinas < t).sum() + (impostoras >= t).sum()) / len(D[iu]) for t in candidatos]
umbral_sugerido = float(candidatos[int(np.argmax(exactitud))])

plt.figure(figsize=(8, 4))
plt.hist(genuinas, bins=30, alpha=.6, label=f'Misma persona ({len(genuinas)})')
plt.hist(impostoras, bins=30, alpha=.6, label=f'Personas distintas ({len(impostoras)})')
plt.axvline(UMBRAL, color='k', ls='--', label=f'Umbral práctica = {UMBRAL}')
plt.axvline(umbral_sugerido, color='r', ls=':', label=f'Umbral sugerido = {umbral_sugerido:.2f}')
plt.xlabel('Distancia L2 entre codificaciones'); plt.ylabel('Número de parejas'); plt.legend()
guardar_figura('distancias_umbral.png')

print(f'Genuinas:   media {genuinas.mean():.3f}  máx {genuinas.max():.3f}' if len(genuinas) else 'No hay parejas genuinas (cada persona tiene 1 foto).')
print(f'Impostoras: media {impostoras.mean():.3f}  mín {impostoras.min():.3f}')
print(f'Umbral sugerido: {umbral_sugerido:.2f} (exactitud {max(exactitud):.1%})')

# 6.2 Elegir el umbral final (con pocas fotos el umbral calibrado puede ser optimista).
if args.calibrar:
    UMBRAL = round(umbral_sugerido, 2)
print('Umbral en uso:', UMBRAL)

# 6.3 Aciertos sobre las fotos de prueba (personas que SÍ están registradas)
aciertos = 0
for persona, ruta in fotos_prueba.items():
    _, pred = who_is_it(ruta, database, FRmodel, verbose=False)
    aciertos += (pred == persona)
    if pred != persona:
        print(f'  Error: {persona} -> {pred}')
if fotos_prueba:
    print(f'Reconocimiento de registrados: {aciertos}/{len(fotos_prueba)} = {aciertos/len(fotos_prueba):.1%}')

# 6.4 Rechazo de desconocidos: quitar a cada persona y ver si se la rechaza
rechazos = 0
for persona, ruta in fotos_prueba.items():
    db_sin_persona = {k: v for k, v in database.items() if k != persona}
    _, pred = who_is_it(ruta, db_sin_persona, FRmodel, verbose=False)
    rechazos += (pred == 'desconocido')
    if pred != 'desconocido':
        print(f'  {persona} (no registrado) fue confundido con {pred}')
if fotos_prueba:
    print(f'Desconocidos rechazados: {rechazos}/{len(fotos_prueba)} = {rechazos/len(fotos_prueba):.1%}')


# =====================================================================
# 7 - Probar con fotos nuevas (--probar foto1.jpg foto2.jpg ...)
# =====================================================================
for i, nombre in enumerate(args.probar, start=1):
    x, ok = preparar(nombre)
    min_dist, identidad = who_is_it(nombre, database, FRmodel)
    plt.figure(figsize=(3, 3)); plt.imshow(x); plt.axis('off')
    plt.title(f'{identidad} ({min_dist:.2f})' + ('' if ok else ' - sin rostro detectado'))
    guardar_figura(f'prueba_{i}.png')


# =====================================================================
# 8 - Exportar para la aplicación
# base_de_datos.json: nombres, codificaciones de 128 números y el umbral (la app compara contra esto).
# facenet.keras: el modelo reconstruido.
# =====================================================================
# Base de datos completa: para la app se registran TODAS las fotos (incluidas las de prueba).
database_final = {p: np.vstack([img_to_encoding(r, FRmodel) for r in rutas])
                  for p, rutas in fotos_por_persona.items()}
exportar = {
    'modelo': 'FaceNet Inception-ResNet v1 (entrada 160x160x3, valores 0-1, salida 128 normalizada L2)',
    'umbral': UMBRAL,
    'personas': {p: enc.round(6).tolist() for p, enc in database_final.items()},
}
with open(CARPETA_SALIDA / 'base_de_datos.json', 'w', encoding='utf-8') as f:
    json.dump(exportar, f, ensure_ascii=False)

FRmodel.save(CARPETA_SALIDA / 'facenet.keras')
print('Guardado en', CARPETA_SALIDA)
print(sorted(p.name for p in CARPETA_SALIDA.iterdir()))
