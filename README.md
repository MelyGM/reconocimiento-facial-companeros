# Reconocimiento facial de compañeros con FaceNet

El sistema identifica a los compañeros registrados y responde **"desconocido"** cuando un rostro no está en la base de datos. Usa FaceNet (Inception-ResNet v1), que convierte cada rostro en un vector de 128 números, y compara esos vectores con la distancia L2. El umbral es 0.7.

## Contenido de la entrega

```
entrega_reconocimiento_facial/
├── requirements.txt              1. Fichero de requisitos
├── base_de_datos_imagenes/       2. Base de datos de imágenes (las fotos se entregan aparte)
│   ├── etiquetas.csv             archivo -> persona
│   └── LEEME.md
├── app_web/                      3. Archivo HTML (aplicación de reconocimiento facial)
│   ├── index.html
│   ├── nucleo.js
│   ├── modelo/                   FaceNet para el navegador (TensorFlow.js, ~45 MB)
│   ├── base_de_datos.json        codificaciones de las 14 personas (128 números cada una, sin fotos)
│   └── haarcascade_frontalface_default.xml
├── notebook/
│   └── Reconocimiento_Facial_Companeros.ipynb    modelo (Google Colab)
└── yucca/                        el mismo modelo como script para el clúster Yucca (Slurm)
    ├── reconocimiento_facial_companeros.py
    └── script.slrm
```

El modelo `facenet_recuperado.keras` (88 MB) no viene en esta carpeta por su tamaño. Lo usa la libreta y debe estar en la misma carpeta de Drive que las imágenes.

---

## 1. Fichero de requisitos (`requirements.txt`)

Lista las bibliotecas con las versiones con las que se probó el proyecto: TensorFlow 2.21, Keras 3.15, h5py, NumPy, Pillow, OpenCV 4, Matplotlib y Jupyter. Necesita **Python 3.10 a 3.13**.

### Crear un entorno nuevo e instalar

**Con venv (Python normal):**
```bash
python -m venv entorno_face
# Windows:      entorno_face\Scripts\activate
# Linux / Mac:  source entorno_face/bin/activate
pip install -r requirements.txt
```

**Con conda:**
```bash
conda create -n face python=3.11 -y
conda activate face
pip install -r requirements.txt
```

**Comprobar la instalación:**
```bash
python -c "import tensorflow as tf, keras, cv2; print(tf.__version__, keras.__version__, cv2.__version__)"
```

**GPU AMD (clúster con ROCm 6.4):** primero instala `pip install tensorflow-rocm==2.18.1 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.3/` y después `pip install -r requirements.txt --no-deps`.

**En Google Colab** casi todo viene instalado. La primera celda de código de la libreta instala lo que falta.

---

## 2. Base de datos de imágenes (`base_de_datos_imagenes/`)

> **Privacidad:** las fotos de los compañeros **no se publican en este repositorio** (las excluye el `.gitignore`). Se entregan aparte al profesor. Aquí solo están la lista de archivos y etiquetas (`etiquetas.csv`) y la estructura que debe tener la carpeta.

- **14 personas, una foto cada una.** Las fotos son las originales, sin modificar (no incluidas en el repositorio).
- **Etiquetado por nombre de archivo:** `persona_número.extensión`, por ejemplo `alejandra_1.jpeg` o `edelmira_1.png`. La libreta quita el número final para obtener la etiqueta: `alejandra_1.jpeg` → `alejandra`.
- **Lista completa en `etiquetas.csv`**, con las columnas `archivo`, `persona` y `archivo_original`.
- **Agregar una foto:** usa el mismo formato, por ejemplo `daniel_2.jpg`. Con 2 o más fotos por persona, la libreta usa la última como foto de prueba real en lugar de una simulada.

| Persona | Archivo |
|---|---|
| alejandra | alejandra_1.jpeg |
| carlos | carlos_1.jpeg |
| daniel | daniel_1.jpeg |
| edelmira | edelmira_1.png |
| ernesto | ernesto_1.jpg |
| karol | karol_1.jpeg |
| kevin | kevin_1.jpg |
| luis | luis_1.jpg |
| matti | matti_1.jpg |
| melina | melina_1.jpg |
| raul | raul_1.jpeg |
| ricardo | ricardo_1.jpg |
| rogelio | rogelio_1.jpeg |
| sebastian | sebastian_1.jpeg |

---

## 3. Archivo HTML (`app_web/index.html`)

Es una aplicación web que corre FaceNet **dentro del navegador** con TensorFlow.js. Su base de datos solo guarda las codificaciones (no las fotos), así que en la lista de personas registradas se ve la inicial de cada nombre. Usa el mismo detector de rostros, el mismo recorte y la misma base de datos que la libreta, así que da las mismas distancias.

**Qué se puede hacer:**
- **Tomar foto / Subir foto:** detecta los rostros y los marca en verde (reconocido) o en rojo (desconocido). Muestra el nombre, la distancia, la cara más parecida y las 5 personas más cercanas.
- **Cámara en tiempo real:** sigue los rostros cuadro por cuadro y les pone nombre. Con **Usar cámara trasera / frontal** se cambia de cámara.
- **Registrar a alguien:** agrega una persona nueva con el último rostro detectado.
- **Ajustes:** cambia el umbral de distancia.

**Cómo abrirlo.** Por seguridad, el navegador solo permite la cámara en `https` o en `localhost`, así que no basta con dar doble clic al archivo:
```bash
cd app_web
python -m http.server 8000
```
Después abre **http://localhost:8000** en Chrome o Edge. Se necesita internet para cargar TensorFlow.js y OpenCV.js.

**En línea (opcional):** GitHub Pages solo publica desde la raíz del repositorio o desde una carpeta `docs`. Para publicar la app, cambia el nombre de `app_web` a `docs` y activa Pages en *Settings → Pages* (rama `main`, carpeta `/docs`).

---

## Cómo correr la libreta (Google Colab)

1. Sube a una carpeta de tu Google Drive `facenet_recuperado.keras` y la carpeta `base_de_datos_imagenes`. En lugar de la carpeta también sirve `Fotos_para_FaceRecognition.zip`.
2. Abre `notebook/Reconocimiento_Facial_Companeros.ipynb` en Colab y elige **Entorno de ejecución → Ejecutar todas**.
3. La primera celda monta Drive y **busca sola** la carpeta donde está el modelo. No hay que escribir rutas.
4. Al final se guardan `base_de_datos.json` y `facenet.keras` en `salida_reconocimiento/`.

## Cómo correrlo en el clúster Yucca

`yucca/reconocimiento_facial_companeros.py` hace los mismos pasos que la libreta, sin Google Colab. Las figuras se guardan como PNG en `salida_reconocimiento/`.

1. En la Yucca, copia a `~/reconocimiento_facial/` estos archivos: los dos de la carpeta `yucca/`, `facenet_recuperado.keras` y las fotos (la carpeta `base_de_datos_imagenes/` o `Fotos_para_FaceRecognition.zip`).
2. Prepara el ambiente con GPU AMD:
   ```bash
   conda create -n face -c conda-forge python=3.10 -y
   conda activate face
   pip install tensorflow-rocm==2.18.1 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.3/
   pip install pillow matplotlib "opencv-python>=4,<5"
   ```
3. Lanza el trabajo y revisa la salida:
   ```bash
   cd ~/reconocimiento_facial
   sbatch script.slrm
   tail -f slurm-*.out
   ```
4. Para probar fotos nuevas: `python reconocimiento_facial_companeros.py --probar foto_nueva.jpg`.

## Resultados

Como hay una sola foto por persona, la prueba usa una **foto simulada** de cada una: espejo, otra iluminación, rotación y menor calidad.

- **Misma persona:** distancias de 0.17 a 0.50 (media 0.30). Con el umbral 0.7 se reconoció a **14 de 14**.
- **Personas distintas:** distancia mínima de **0.86** (media 1.27), así que nadie se confunde con otra persona.
- **Desconocidos rechazados:** **14/14**. Para medirlo se quita a cada persona de la base y se busca su foto.
- **Umbral sugerido por la calibración:** 0.68. Es muy parecido al de la práctica, así que se mantuvo 0.7.

Las fotos simuladas cambian un poco según el orden de los archivos. En una versión anterior, con 11 personas, la foto simulada de *daniel* quedó en 0.77, arriba del umbral. Por eso 0.7 es un límite ajustado para algunas fotos.

Para una evaluación más realista, conviene agregar una segunda foto real de cada persona, tomada otro día o con otra luz.
