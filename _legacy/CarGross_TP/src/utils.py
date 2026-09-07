"""
utils.py — Funciones auxiliares para CarGross
===============================================
Contiene binarización, escritura de resultados,
generación de resúmenes en texto y la man page.
"""

import os
import csv
from datetime import datetime


# ---------------------------------------------------------------------------
# MAN PAGE
# ---------------------------------------------------------------------------

MAN_PAGE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║            CarGross.py — Red Neuronal ART1 (Carpenter/Grossberg)            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  NOMBRE                                                                      ║
║      CarGross.py — Clustering no supervisado con red ART1                   ║
║                                                                              ║
║  SINOPSIS                                                                    ║
║      python src/CarGross.py <archivo.csv> [opciones]                        ║
║                                                                              ║
║  OPCIONES                                                                    ║
║      --vigilance FLOAT   Umbral rho 0.0-1.0  (default: 0.5)                ║
║      --max-clusters INT  Máx. clusters       (default: 50)                  ║
║      --output FILE       CSV de salida con columna 'cluster'                ║
║      --metadata FILE     CSV de umbrales custom (ver data/metadata/)        ║
║      --save-txt FILE     Guarda resumen en .txt                             ║
║      --verbose           Detalle de cada paso del algoritmo                 ║
║      --help / -h         Muestra esta ayuda                                 ║
║                                                                              ║
║  EJEMPLOS                                                                    ║
║      python src/CarGross.py data/dataset1.csv                               ║
║      python src/CarGross.py data/dataset1.csv --vigilance 0.7 --verbose     ║
║      python src/CarGross.py data/dataset1.csv \\                             ║
║             --metadata data/metadata/metadata.csv --output results/out.csv  ║
║                                                                              ║
║  CÓDIGOS DE SALIDA                                                           ║
║      0  OK                                                                   ║
║      1  Argumento inválido o archivo no encontrado                          ║
║      2  Error de formato en CSV                                             ║
║      3  Vigilancia fuera de rango [0.0, 1.0]                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def mostrar_man():
    """Imprime la man page en stdout."""
    print(MAN_PAGE)


# ---------------------------------------------------------------------------
# DETECCIÓN DE COLUMNAS NUMÉRICAS
# ---------------------------------------------------------------------------

def detectar_columnas_numericas(encabezados, filas):
    """
    Detecta qué columnas del CSV contienen valores numéricos.

    Parámetros:
        encabezados (list[str]): Nombres de columna.
        filas (list[dict]):      Filas del CSV como dicts.

    Retorna:
        list[str]: Nombres de columnas numéricas.
    """
    numericas = []
    for col in encabezados:
        validos = 0
        es_num = True
        for fila in filas:
            v = fila.get(col, "").strip()
            if v == "":
                continue
            try:
                float(v)
                validos += 1
            except ValueError:
                es_num = False
                break
        if es_num and validos > 0:
            numericas.append(col)
    return numericas


# ---------------------------------------------------------------------------
# UMBRALES DE BINARIZACIÓN (MEDIA)
# ---------------------------------------------------------------------------

def calcular_umbrales_media(filas, cols_numericas):
    """
    Calcula la media de cada columna numérica como umbral de binarización.

    Parámetros:
        filas (list[dict]):         Filas del CSV.
        cols_numericas (list[str]): Columnas a procesar.

    Retorna:
        dict[str, float]: {columna: media}.
    """
    umbrales = {}
    for col in cols_numericas:
        valores = []
        for fila in filas:
            v = fila.get(col, "").strip()
            if v:
                try:
                    valores.append(float(v))
                except ValueError:
                    pass
        umbrales[col] = sum(valores) / len(valores) if valores else 0.0
    return umbrales


# ---------------------------------------------------------------------------
# BINARIZACIÓN
# ---------------------------------------------------------------------------

def binarizar_datos(filas, cols_numericas, umbrales):
    """
    Convierte columnas numéricas a vectores binarios.
    Regla: valor >= umbral → 1,  valor < umbral → 0.
    Valores faltantes o no numéricos → 0.

    Parámetros:
        filas (list[dict]):          Filas del CSV.
        cols_numericas (list[str]):  Columnas a binarizar.
        umbrales (dict[str,float]):  Umbral por columna.

    Retorna:
        list[list[int]]: Lista de vectores binarios.
    """
    vectores = []
    for fila in filas:
        vec = []
        for col in cols_numericas:
            v = fila.get(col, "").strip()
            try:
                vec.append(1 if float(v) >= umbrales[col] else 0)
            except (ValueError, TypeError):
                vec.append(0)
        vectores.append(vec)
    return vectores


# ---------------------------------------------------------------------------
# ESCRITURA DE RESULTADOS
# ---------------------------------------------------------------------------

def escribir_csv_resultado(filepath, encabezados, filas, asignaciones):
    """
    Escribe el CSV de salida: datos originales + columna 'cluster'.

    Parámetros:
        filepath (str):           Ruta del archivo de salida.
        encabezados (list[str]):  Encabezados originales.
        filas (list[dict]):       Datos originales.
        asignaciones (list[int]): Cluster asignado a cada fila.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    campos = list(encabezados) + ["cluster"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for fila, cluster in zip(filas, asignaciones):
            row = dict(fila)
            row["cluster"] = cluster
            writer.writerow(row)


def generar_resumen_texto(asignaciones, n_clusters, vigilance,
                          cols_numericas, archivo_entrada, umbrales,
                          tiempo_seg=None):
    """
    Genera un string con el resumen completo de la corrida.

    Parámetros:
        asignaciones (list[int]):    Clusters asignados.
        n_clusters (int):            Total de clusters formados.
        vigilance (float):           Valor de rho usado.
        cols_numericas (list[str]):  Columnas procesadas.
        archivo_entrada (str):       Nombre del archivo de entrada.
        umbrales (dict):             Umbrales usados para binarizar.
        tiempo_seg (float|None):     Tiempo de ejecución en segundos.

    Retorna:
        str: Texto formateado listo para imprimir o guardar.
    """
    conteo = {}
    for c in asignaciones:
        conteo[c] = conteo.get(c, 0) + 1

    L = []
    sep = "=" * 60
    L.append(sep)
    L.append("  RESULTADOS — Red ART1 Carpenter/Grossberg")
    L.append(sep)
    L.append(f"  Fecha/Hora:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"  Archivo de entrada:    {archivo_entrada}")
    L.append(f"  Vigilancia (rho):      {vigilance}")
    L.append(f"  Columnas procesadas:   {len(cols_numericas)}")
    L.append(f"  Registros procesados:  {len(asignaciones)}")
    L.append(f"  Clusters formados:     {n_clusters}")
    if tiempo_seg is not None:
        L.append(f"  Tiempo de ejecución:   {tiempo_seg:.3f} s")
    L.append("")
    L.append("  Umbrales de binarización:")
    for col, u in umbrales.items():
        L.append(f"    {col:<28} {u:.4f}")
    L.append("")
    L.append("-" * 60)
    L.append(f"  {'Cluster':<10} {'Cantidad':<12} {'Porcentaje'}")
    L.append("-" * 60)
    for c in sorted(conteo):
        pct = 100.0 * conteo[c] / len(asignaciones)
        L.append(f"  {c:<10} {conteo[c]:<12} {pct:.1f}%")
    L.append(sep)
    return "\n".join(L)


def guardar_resumen_txt(filepath, texto):
    """
    Guarda el texto de resumen en un archivo .txt.

    Parámetros:
        filepath (str): Ruta del archivo de salida.
        texto (str):    Contenido a escribir.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(texto + "\n")


def imprimir_resumen(texto):
    """Imprime el resumen en consola."""
    print("\n" + texto)
