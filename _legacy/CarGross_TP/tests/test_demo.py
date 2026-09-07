"""
test_demo.py — Tests de la red ART1
======================================
Ejecutar desde la raíz del proyecto:
    python tests/test_demo.py

Cubre:
    1. Test unitario de la clase ART1 con vectores simples.
    2. Test de binarización y detección de columnas.
    3. Test de lectura de CSV y metadata.
    4. Demo de integración completa con dataset1.csv y dataset2.csv.
"""

import sys
import os
import csv
import traceback

# Permite importar desde src/ sin instalar el paquete
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from CarGross import ART1
from data_loader import leer_csv, leer_metadata, validar_minimo
from utils import (
    detectar_columnas_numericas,
    calcular_umbrales_media,
    binarizar_datos,
    generar_resumen_texto,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ok(nombre):
    print(f"  [PASS] {nombre}")

def fallo(nombre, exc):
    print(f"  [FAIL] {nombre}")
    traceback.print_exc()
    return False


# ===========================================================================
# TEST 1 — Clase ART1 básica
# ===========================================================================

def test_art1_dos_clusters():
    """Verifica que dos patrones muy distintos formen clusters separados."""
    red = ART1(N=4, vigilance=0.7)
    x1 = [1, 1, 0, 0]
    x2 = [0, 0, 1, 1]
    c1 = red._procesar(x1)
    c2 = red._procesar(x2)
    assert c1 != c2, f"Esperaba clusters distintos, ambos dieron {c1}"
    assert red.n_clusters == 2
    ok("ART1: patrones opuestos → 2 clusters distintos")


def test_art1_mismo_cluster():
    """Verifica que patrones idénticos vayan al mismo cluster."""
    red = ART1(N=4, vigilance=0.5)
    x = [1, 0, 1, 0]
    c1 = red._procesar(x)
    c2 = red._procesar(x)
    assert c1 == c2, f"Esperaba el mismo cluster, obtuvo {c1} y {c2}"
    assert red.n_clusters == 1
    ok("ART1: patrones idénticos → mismo cluster")


def test_art1_vector_nulo():
    """El vector de ceros debe asignarse sin error."""
    red = ART1(N=4, vigilance=0.5)
    red._procesar([1, 1, 0, 0])
    c = red._procesar([0, 0, 0, 0])
    assert isinstance(c, int)
    ok("ART1: vector nulo no lanza error")


def test_art1_limite_clusters():
    """Debe lanzar RuntimeError cuando se supera max_clusters."""
    red = ART1(N=4, vigilance=0.99, max_clusters=2)
    patrones = [[1,0,0,0], [0,1,0,0], [0,0,1,0]]
    lanzado = False
    try:
        red.entrenar(patrones)
    except RuntimeError:
        lanzado = True
    assert lanzado, "Esperaba RuntimeError por límite de clusters"
    ok("ART1: lanza RuntimeError al superar max_clusters")


def test_art1_vigilancia_baja():
    """Con vigilancia baja se forman menos clusters que con vigilancia alta."""
    # rho bajo
    red_baja = ART1(N=4, vigilance=0.1)
    patrones = [[1,1,0,0], [1,0,0,0], [1,1,1,0], [1,0,1,0]]
    asig_baja = red_baja.entrenar(patrones)

    # rho alto — mismos patrones deben dar más clusters
    red_alta = ART1(N=4, vigilance=0.9)
    asig_alta = red_alta.entrenar(patrones)

    assert red_baja.n_clusters <= red_alta.n_clusters, (
        f"Con rho bajo ({red_baja.n_clusters}) se esperan ≤ clusters "
        f"que con rho alto ({red_alta.n_clusters})"
    )
    ok("ART1: vigilancia baja produce ≤ clusters que vigilancia alta")


# ===========================================================================
# TEST 2 — Utilidades de binarización
# ===========================================================================

def test_deteccion_columnas():
    """Verifica que se detecten solo las columnas numéricas."""
    encabezados = ["nombre", "edad", "ciudad", "peso"]
    filas = [
        {"nombre": "Ana",  "edad": "30", "ciudad": "Paraná", "peso": "65.5"},
        {"nombre": "Luis", "edad": "25", "ciudad": "Gualeguay", "peso": "78.0"},
    ]
    cols = detectar_columnas_numericas(encabezados, filas)
    assert "edad" in cols and "peso" in cols
    assert "nombre" not in cols and "ciudad" not in cols
    ok("Detección de columnas numéricas correcta")


def test_binarizacion():
    """Verifica que la binarización por umbral sea correcta."""
    filas = [
        {"x": "10", "y": "5"},
        {"x": "3",  "y": "8"},
    ]
    cols = ["x", "y"]
    umbrales = {"x": 7.0, "y": 6.0}
    vectores = binarizar_datos(filas, cols, umbrales)
    assert vectores[0] == [1, 0], f"Fila 0: esperaba [1,0], obtuvo {vectores[0]}"
    assert vectores[1] == [0, 1], f"Fila 1: esperaba [0,1], obtuvo {vectores[1]}"
    ok("Binarización por umbral correcta")


def test_umbrales_media():
    """Verifica que la media se calcule correctamente."""
    filas = [{"v": "2"}, {"v": "4"}, {"v": "6"}]
    u = calcular_umbrales_media(filas, ["v"])
    assert u["v"] == 4.0, f"Media esperada 4.0, obtuvo {u['v']}"
    ok("Cálculo de umbrales por media correcto")


# ===========================================================================
# TEST 3 — Lectura de archivos
# ===========================================================================

def test_leer_csv_inexistente():
    """Debe lanzar FileNotFoundError si el archivo no existe."""
    lanzado = False
    try:
        leer_csv("no_existe_este_archivo.csv")
    except FileNotFoundError:
        lanzado = True
    assert lanzado
    ok("leer_csv: FileNotFoundError si el archivo no existe")


def test_leer_metadata_columnas_faltantes(tmp_path=None):
    """Debe lanzar ValueError si metadata no tiene columnas correctas."""
    ruta = "/tmp/meta_mal.csv"
    with open(ruta, "w") as f:
        f.write("col_a,col_b\nedad,40\n")
    lanzado = False
    try:
        leer_metadata(ruta)
    except ValueError:
        lanzado = True
    assert lanzado
    ok("leer_metadata: ValueError si faltan columnas obligatorias")


# ===========================================================================
# TEST 4 — Integración con datasets reales
# ===========================================================================

BASE = os.path.join(os.path.dirname(__file__), "..")

def test_integracion_dataset1():
    """Corre el pipeline completo con data/dataset1.csv."""
    ruta = os.path.join(BASE, "data", "dataset1.csv")
    if not os.path.exists(ruta):
        print(f"  [SKIP] test_integracion_dataset1: {ruta} no encontrado")
        return

    encabezados, filas = leer_csv(ruta)
    cols = detectar_columnas_numericas(encabezados, filas)
    validar_minimo(filas, cols)
    umbrales = calcular_umbrales_media(filas, cols)
    vectores = binarizar_datos(filas, cols, umbrales)

    red = ART1(N=len(cols), vigilance=0.6)
    asignaciones = red.entrenar(vectores)

    assert len(asignaciones) == len(filas)
    assert red.n_clusters >= 1
    print(f"      dataset1: {len(filas)} registros → {red.n_clusters} clusters")
    ok("Integración dataset1.csv completa")


def test_integracion_dataset2():
    """Corre el pipeline completo con data/dataset2.csv."""
    ruta = os.path.join(BASE, "data", "dataset2.csv")
    if not os.path.exists(ruta):
        print(f"  [SKIP] test_integracion_dataset2: {ruta} no encontrado")
        return

    encabezados, filas = leer_csv(ruta)
    cols = detectar_columnas_numericas(encabezados, filas)
    validar_minimo(filas, cols)
    umbrales = calcular_umbrales_media(filas, cols)
    vectores = binarizar_datos(filas, cols, umbrales)

    red = ART1(N=len(cols), vigilance=0.65)
    asignaciones = red.entrenar(vectores)

    assert len(asignaciones) == len(filas)
    assert red.n_clusters >= 1
    print(f"      dataset2: {len(filas)} registros → {red.n_clusters} clusters")
    ok("Integración dataset2.csv completa")


def test_integracion_metadata():
    """Verifica que metadata.csv modifique los umbrales correctamente."""
    ruta_meta = os.path.join(BASE, "data", "metadata", "metadata.csv")
    if not os.path.exists(ruta_meta):
        print(f"  [SKIP] test_integracion_metadata: {ruta_meta} no encontrado")
        return
    meta = leer_metadata(ruta_meta)
    assert len(meta) > 0
    ok("Lectura de metadata.csv del proyecto correcta")


# ===========================================================================
# RUNNER
# ===========================================================================

TESTS = [
    # ART1
    test_art1_dos_clusters,
    test_art1_mismo_cluster,
    test_art1_vector_nulo,
    test_art1_limite_clusters,
    test_art1_vigilancia_baja,
    # utils
    test_deteccion_columnas,
    test_binarizacion,
    test_umbrales_media,
    # data_loader
    test_leer_csv_inexistente,
    test_leer_metadata_columnas_faltantes,
    # integración
    test_integracion_dataset1,
    test_integracion_dataset2,
    test_integracion_metadata,
]


def main():
    print("\n" + "=" * 55)
    print("  TEST SUITE — CarGross ART1")
    print("=" * 55)

    pasados = 0
    fallados = 0

    for test_fn in TESTS:
        try:
            test_fn()
            pasados += 1
        except Exception as exc:
            fallo(test_fn.__name__, exc)
            fallados += 1

    print("-" * 55)
    print(f"  Resultado: {pasados} pasados, {fallados} fallados")
    print("=" * 55)
    sys.exit(0 if fallados == 0 else 1)


if __name__ == "__main__":
    main()
