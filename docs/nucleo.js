/* =====================================================================
 * nucleo.js — Reconocimiento facial con FaceNet en el navegador
 * Reproduce EXACTAMENTE el pipeline de la libreta "Reconocimiento Facial
 * Compañeros":
 *   1. Escala de grises y detector Haar de OpenCV (mismos parámetros)
 *   2. Recorte cuadrado del rostro más grande con 20 % de margen
 *   3. Redimensionado a 160x160 igual que PIL (BILINEAR con antialias)
 *   4. Píxeles / 255 -> FaceNet -> vector de 128 números normalizado (L2)
 *   5. Distancia L2 mínima contra la base de datos y umbral
 * Funciona en el navegador (window.Nucleo) y en Node (module.exports).
 * ===================================================================== */
(function (raiz) {
  'use strict';

  // ------------------------------------------------------------------
  // Capa propia de FaceNet: suma residual escalada  salida = a + scale*b
  // (en Keras eran capas Lambda; TF.js necesita conocer la clase)
  // ------------------------------------------------------------------
  function registrarCapaResidual(tf) {
    class ResidualScaleSum extends tf.layers.Layer {
      constructor(config) {
        super(config);
        this.scale = config.scale;
      }
      computeOutputShape(formas) { return formas[0]; }
      call(entradas) {
        return tf.tidy(() => {
          const [a, b] = Array.isArray(entradas) ? entradas : [entradas[0], entradas[1]];
          return tf.add(a, tf.mul(b, this.scale));
        });
      }
      getConfig() { return Object.assign({}, super.getConfig(), { scale: this.scale }); }
      static get className() { return 'ResidualScaleSum'; }
    }
    tf.serialization.registerClass(ResidualScaleSum);
  }

  // ------------------------------------------------------------------
  // Redimensionado idéntico a PIL Image.resize(..., BILINEAR)
  // PIL usa un filtro triangular cuyo soporte crece al reducir la imagen
  // (antialias) y aritmética entera de 22 bits. Lo replicamos para que las
  // codificaciones del navegador coincidan con las de la libreta.
  // ------------------------------------------------------------------
  const BITS = 22;                          // PRECISION_BITS de PIL para imágenes de 8 bits
  const ESCALA_ENTERA = 1 << BITS;

  function coeficientes(tamEntrada, tamSalida) {
    const escala = tamEntrada / tamSalida;
    const escalaFiltro = Math.max(escala, 1);
    const soporte = 1.0 * escalaFiltro;     // el filtro bilineal tiene soporte 1
    const inicios = new Int32Array(tamSalida);
    const cuantos = new Int32Array(tamSalida);
    const lista = [];
    for (let i = 0; i < tamSalida; i++) {
      const centro = (i + 0.5) * escala;
      let xmin = Math.trunc(centro - soporte + 0.5); if (xmin < 0) xmin = 0;
      let xmax = Math.trunc(centro + soporte + 0.5); if (xmax > tamEntrada) xmax = tamEntrada;
      xmax -= xmin;
      const pesos = new Float64Array(xmax);
      let suma = 0;
      for (let x = 0; x < xmax; x++) {
        const t = Math.abs((x + xmin - centro + 0.5) / escalaFiltro);
        const w = t < 1 ? 1 - t : 0;
        pesos[x] = w; suma += w;
      }
      const enteros = new Int32Array(xmax);
      for (let x = 0; x < xmax; x++) {
        const k = suma !== 0 ? pesos[x] / suma : 0;
        enteros[x] = k < 0 ? Math.trunc(k * ESCALA_ENTERA - 0.5) : Math.trunc(k * ESCALA_ENTERA + 0.5);
      }
      inicios[i] = xmin; cuantos[i] = xmax; lista.push(enteros);
    }
    return { inicios, cuantos, lista };
  }

  function recortar8(v) {               // equivalente a clip8 de PIL
    const r = Math.floor(v / ESCALA_ENTERA);
    return r < 0 ? 0 : r > 255 ? 255 : r;
  }

  /** rgb: Uint8Array (ancho*alto*3). Devuelve Uint8Array (anchoS*altoS*3). */
  function redimensionarComoPIL(rgb, ancho, alto, anchoS, altoS) {
    // Pasada horizontal
    const h = coeficientes(ancho, anchoS);
    const temp = new Uint8Array(anchoS * alto * 3);
    for (let y = 0; y < alto; y++) {
      for (let xs = 0; xs < anchoS; xs++) {
        const x0 = h.inicios[xs], n = h.cuantos[xs], k = h.lista[xs];
        let r = ESCALA_ENTERA / 2, g = r, b = r;
        for (let j = 0; j < n; j++) {
          const p = ((y * ancho) + x0 + j) * 3;
          r += rgb[p] * k[j]; g += rgb[p + 1] * k[j]; b += rgb[p + 2] * k[j];
        }
        const q = (y * anchoS + xs) * 3;
        temp[q] = recortar8(r); temp[q + 1] = recortar8(g); temp[q + 2] = recortar8(b);
      }
    }
    // Pasada vertical
    const v = coeficientes(alto, altoS);
    const salida = new Uint8Array(anchoS * altoS * 3);
    for (let ys = 0; ys < altoS; ys++) {
      const y0 = v.inicios[ys], n = v.cuantos[ys], k = v.lista[ys];
      for (let x = 0; x < anchoS; x++) {
        let r = ESCALA_ENTERA / 2, g = r, b = r;
        for (let j = 0; j < n; j++) {
          const p = ((y0 + j) * anchoS + x) * 3;
          r += temp[p] * k[j]; g += temp[p + 1] * k[j]; b += temp[p + 2] * k[j];
        }
        const q = (ys * anchoS + x) * 3;
        salida[q] = recortar8(r); salida[q + 1] = recortar8(g); salida[q + 2] = recortar8(b);
      }
    }
    return salida;
  }

  // ------------------------------------------------------------------
  // Detector Haar (mismo XML y parámetros que la libreta)
  // ------------------------------------------------------------------
  function crearDetector(cv, textoXml) {
    const nombre = 'haarcascade_frontalface_default.xml';
    try { cv.FS_unlink('/' + nombre); } catch (e) { /* no existía */ }
    cv.FS_createDataFile('/', nombre, textoXml, true, false, false);
    const detector = new cv.CascadeClassifier();
    if (!detector.load(nombre)) throw new Error('No se pudo cargar el detector Haar');
    return detector;
  }

  /** gris: cv.Mat (CV_8UC1). Devuelve [{x,y,w,h}] de todos los rostros. */
  function detectarRostros(cv, detector, gris) {
    const ladoMin = Math.max(40, Math.floor(Math.min(gris.rows, gris.cols) / 10)); // ignora caras diminutas
    const caras = new cv.RectVector();
    detector.detectMultiScale(gris, caras, 1.1, 5, 0, new cv.Size(ladoMin, ladoMin), new cv.Size(0, 0));
    const lista = [];
    for (let i = 0; i < caras.size(); i++) {
      const r = caras.get(i);
      lista.push({ x: r.x, y: r.y, w: r.width, h: r.height });
    }
    caras.delete();
    return lista;
  }

  /** Caja cuadrada alrededor de la cara con margen (como recortar_rostro de la libreta). */
  function cajaConMargen(cara, ancho, alto, margen = 0.20) {
    const cx = cara.x + cara.w / 2, cy = cara.y + cara.h / 2;
    const lado = Math.max(cara.w, cara.h) * (1 + 2 * margen);
    return {
      x0: Math.trunc(Math.max(0, cx - lado / 2)), y0: Math.trunc(Math.max(0, cy - lado / 2)),
      x1: Math.trunc(Math.min(ancho, cx + lado / 2)), y1: Math.trunc(Math.min(alto, cy + lado / 2)),
    };
  }

  /** Recorta una región de una imagen RGB (Uint8Array) */
  function recortarRGB(rgb, ancho, caja) {
    const w = caja.x1 - caja.x0, h = caja.y1 - caja.y0;
    const out = new Uint8Array(w * h * 3);
    for (let y = 0; y < h; y++) {
      const ini = ((caja.y0 + y) * ancho + caja.x0) * 3;
      out.set(rgb.subarray(ini, ini + w * 3), y * w * 3);
    }
    return { rgb: out, ancho: w, alto: h };
  }

  // ------------------------------------------------------------------
  // FaceNet: imagen 160x160 RGB (Uint8Array) -> vector de 128 (norma 1)
  // ------------------------------------------------------------------
  function codificar(tf, modelo, rgb160) {
    return tf.tidy(() => {
      const x = tf.tensor(Float32Array.from(rgb160, v => v / 255), [1, 160, 160, 3]); // píxeles / 255
      const e = modelo.predict(x);
      const norma = tf.norm(e);
      return tf.div(e, norma).dataSync().slice();      // Float32Array(128)
    });
  }

  /** Igual que codificar, pero sin bloquear la página mientras la GPU calcula
   *  (se usa en la cámara en tiempo real para que el video no se congele). */
  async function codificarAsync(tf, modelo, rgb160) {
    const t = tf.tidy(() => {
      const x = tf.tensor(Float32Array.from(rgb160, v => v / 255), [1, 160, 160, 3]);
      const e = modelo.predict(x);
      return tf.div(e, tf.norm(e));
    });
    const datos = await t.data();
    t.dispose();
    return datos;
  }

  // ------------------------------------------------------------------
  // Comparación con la base de datos (who_is_it de la libreta)
  // baseDatos: { persona: [[128 números], ...] }
  // ------------------------------------------------------------------
  function distancia(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
    return Math.sqrt(s);
  }

  function quienEs(codificacion, baseDatos, umbral) {
    let minDist = 100, identidad = null;
    const ranking = [];
    for (const [nombre, lista] of Object.entries(baseDatos)) {
      // distancia a la foto MÁS parecida de esa persona
      const d = Math.min(...lista.map(v => distancia(codificacion, v)));
      ranking.push({ nombre, distancia: d });
      if (d < minDist) { minDist = d; identidad = nombre; }
    }
    ranking.sort((a, b) => a.distancia - b.distancia);
    const reconocido = minDist < umbral;
    return { identidad: reconocido ? identidad : 'desconocido', masParecido: identidad,
             distancia: minDist, reconocido, ranking };
  }

  const Nucleo = { registrarCapaResidual, redimensionarComoPIL, crearDetector, detectarRostros,
                   cajaConMargen, recortarRGB, codificar, codificarAsync, distancia, quienEs };
  if (typeof module !== 'undefined' && module.exports) module.exports = Nucleo;
  else raiz.Nucleo = Nucleo;
})(typeof self !== 'undefined' ? self : this);
