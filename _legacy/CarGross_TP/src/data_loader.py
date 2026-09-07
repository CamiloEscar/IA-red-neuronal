"""
data_loader.py — Carga de datos CSV y metadatos
================================================
Funciones responsables de leer y validar los archivos
de entrada y metadatos del proyecto CarGross.
"""

import os
import csv


def leer_csv(filepath):
    """
    Lee un archivo CSV y devuelve encabezados y filas.

    Parámetros:
        filepath (str): Ruta al archivo CSV.

    Retorna:
        tuple: (encabezados: list[str], filas: list[dict])

    Lanza:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el CSV está vacío o sin encabezados válidos.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Archivo no encontrado: '{filepath}'")

    filas = []
    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        encabezados = reader.fieldnames
        if not encabezados:
            raise ValueError(f"El archivo '{filepath}' no tiene encabezados válidos.")
        for fila in reader:
            filas.append(dict(fila))

    if not filas:
        raise ValueError(f"El archivo '{filepath}' no contiene datos (solo encabezados).")

    return list(encabezados), filas


def leer_metadata(filepath):
    """
    Lee un CSV de metadatos con umbrales de binarización custom.

    Formato esperado:
        columna,umbral_binarizacion
        edad,40
        temperatura,37.5

    Parámetros:
        filepath (str): Ruta al CSV de metadatos.

    Retorna:
        dict[str, float]: {nombre_columna: umbral}

    Lanza:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si faltan columnas obligatorias o hay valores no numéricos.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metadatos no encontrados: '{filepath}'")

    umbrales = {}
    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        campos = reader.fieldnames or []

        if "columna" not in campos or "umbral_binarizacion" not in campos:
            raise ValueError(
                "El CSV de metadatos debe tener columnas: "
                "'columna' y 'umbral_binarizacion'."
            )

        for i, fila in enumerate(reader, start=2):
            col = fila.get("columna", "").strip()
            if not col:
                continue
            raw = fila.get("umbral_binarizacion", "").strip()
            try:
                umbrales[col] = float(raw)
            except ValueError:
                raise ValueError(
                    f"Línea {i}: umbral no numérico para columna '{col}' (valor='{raw}')."
                )
    return umbrales


def validar_minimo(filas, cols_numericas, min_filas=10, min_cols=2):
    """
    Valida que el dataset tenga suficientes filas y columnas numéricas.

    Parámetros:
        filas (list[dict]):          Filas del CSV.
        cols_numericas (list[str]):  Columnas numéricas detectadas.
        min_filas (int):             Mínimo de filas requerido (default: 10).
        min_cols (int):              Mínimo de columnas numéricas (default: 2).

    Lanza:
        ValueError: Si no se cumplen los mínimos.
    """
    if len(filas) < min_filas:
        raise ValueError(
            f"El dataset tiene {len(filas)} filas; se requieren al menos {min_filas}."
        )
    if len(cols_numericas) < min_cols:
        raise ValueError(
            f"Se detectaron {len(cols_numericas)} columnas numéricas; "
            f"se requieren al menos {min_cols}."
        )
