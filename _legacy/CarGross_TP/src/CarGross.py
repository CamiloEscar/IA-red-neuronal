#!/usr/bin/env python3
"""
CarGross.py — Red Neuronal ART1 (Carpenter/Grossberg)
======================================================
Punto de entrada principal. Implementa el algoritmo ART1
(Adaptive Resonance Theory 1) para clustering no supervisado
sobre datos binarios.

Basado en:
    Lau, C. (Ed.), "Artificial Neural Networks", IEEE Press, 1992.
    Algoritmo Box 3, pp. 12-14.

Uso:
    python src/CarGross.py <archivo.csv> [opciones]
    python src/CarGross.py --help
"""

import sys
import os
import argparse
import time

# Asegura imports relativos tanto al correr como script o como módulo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import leer_csv, leer_metadata, validar_minimo
from utils import (
    mostrar_man,
    detectar_columnas_numericas,
    calcular_umbrales_media,
    binarizar_datos,
    escribir_csv_resultado,
    generar_resumen_texto,
    guardar_resumen_txt,
    imprimir_resumen,
)


# ===========================================================================
# CLASE ART1
# ===========================================================================

class ART1:
    """
    Red neuronal ART1 (Adaptive Resonance Theory 1).

    Implementa clustering no supervisado para entradas binarias según
    el algoritmo Box 3 de Lau (1992), pp. 12-14.

    El parámetro vigilance (rho) controla el balance
    estabilidad-plasticidad:
        rho cercano a 1.0 → clusters finos, muy específicos.
        rho cercano a 0.0 → clusters amplios, generales.

    Atributos:
        N (int):           Longitud del vector de entrada binario.
        vigilance (float): Umbral rho en [0.0, 1.0].
        max_clusters (int): Límite de clusters permitidos.
        t_weights (list):  Pesos top-down t_ij (exemplares almacenados).
        b_weights (list):  Pesos bottom-up b_ij (para matching scores).
        n_clusters (int):  Número de clusters formados hasta ahora.
    """

    def __init__(self, N, vigilance=0.5, max_clusters=50):
        """
        Inicializa la red ART1.

        Parámetros:
            N (int):           Dimensión del vector de entrada.
            vigilance (float): Umbral de vigilancia rho ∈ [0, 1].
            max_clusters (int): Máximo número de clusters.
        """
        self.N = N
        self.vigilance = vigilance
        self.max_clusters = max_clusters
        self.n_clusters = 0
        self.t_weights = []   # t_ij: pesos top-down
        self.b_weights = []   # b_ij: pesos bottom-up

    # ------------------------------------------------------------------
    # Step 1 — Pesos iniciales (Box 3)
    # ------------------------------------------------------------------
    def _pesos_iniciales(self):
        """
        Genera los pesos iniciales para un cluster nuevo.

        Fórmulas (Step 1, Box 3):
            t_ij(0) = 1             (exemplar = vector de unos)
            b_ij(0) = 1 / (1 + N)  (pesos bottom-up uniformes)

        Retorna:
            tuple: (t: list[int], b: list[float])
        """
        t = [1] * self.N
        b = [1.0 / (1.0 + self.N)] * self.N
        return t, b

    # ------------------------------------------------------------------
    # Step 3 — Matching score
    # ------------------------------------------------------------------
    def _matching_score(self, b_j, x):
        """
        Calcula el matching score mu_j del cluster j con la entrada x.

        Fórmula (Step 3):  μ_j = Σ_i  b_ij · x_i

        Parámetros:
            b_j (list[float]): Pesos bottom-up del cluster j.
            x   (list[int]):   Vector de entrada binario.

        Retorna:
            float: Matching score.
        """
        return sum(b * xi for b, xi in zip(b_j, x))

    # ------------------------------------------------------------------
    # Step 5 — Test de vigilancia
    # ------------------------------------------------------------------
    def _test_vigilancia(self, t_j, x):
        """
        Evalúa si el cluster j supera el test de vigilancia con la entrada x.

        Fórmula (Step 5):  ||T · X|| / ||X|| > ρ ?

        donde:
            ||X||    = Σ x_i              (norma L1 de la entrada)
            T · X    = AND componente a componente de t_j y x
            ||T · X|| = Σ (t_ij AND x_i)

        Parámetros:
            t_j (list[int]): Pesos top-down del cluster j.
            x   (list[int]): Vector de entrada binario.

        Retorna:
            bool: True si el cluster es suficientemente similar a x.
        """
        norma_x = sum(x)
        if norma_x == 0:
            return True   # vector nulo: acepta siempre
        tx = [ti & xi for ti, xi in zip(t_j, x)]
        return (sum(tx) / norma_x) > self.vigilance

    # ------------------------------------------------------------------
    # Step 7 — Adaptar pesos del cluster ganador
    # ------------------------------------------------------------------
    def _adaptar(self, j, x):
        """
        Actualiza los pesos del cluster ganador j.

        Fórmulas (Step 7):
            t_ij*(t+1) = t_ij(t) AND x_i
            b_ij*(t+1) = (t_ij*(t+1)) / (0.5 + ||t_ij*(t+1)||)

        Parámetros:
            j (int):       Índice del cluster ganador.
            x (list[int]): Vector de entrada binario.
        """
        tx = [ti & xi for ti, xi in zip(self.t_weights[j], x)]
        norma_tx = sum(tx)
        self.t_weights[j] = tx
        denom = 0.5 + norma_tx
        self.b_weights[j] = [v / denom for v in tx]

    # ------------------------------------------------------------------
    # Crear nuevo cluster
    # ------------------------------------------------------------------
    def _crear_cluster(self, x, verbose=False):
        """
        Crea un nuevo cluster e inicializa sus pesos adaptados a x.

        Parámetros:
            x (list[int]):  Vector de entrada.
            verbose (bool): Modo detallado.

        Retorna:
            int: Índice del nuevo cluster.

        Lanza:
            RuntimeError: Si se alcanzó el límite máximo de clusters.
        """
        if self.n_clusters >= self.max_clusters:
            raise RuntimeError(
                f"Límite de clusters alcanzado ({self.max_clusters}). "
                "Use --max-clusters para aumentarlo o baje --vigilance."
            )
        t, b = self._pesos_iniciales()
        self.t_weights.append(t)
        self.b_weights.append(b)
        j = self.n_clusters
        self.n_clusters += 1
        self._adaptar(j, x)
        if verbose:
            print(f"      → Nuevo cluster creado: {j}")
        return j

    # ------------------------------------------------------------------
    # Procesar una entrada — Steps 2 a 8
    # ------------------------------------------------------------------
    def _procesar(self, x, verbose=False):
        """
        Clasifica un vector x asignándolo al cluster correspondiente.
        Implementa los Steps 2-8 del algoritmo Box 3.

        Parámetros:
            x (list[int]):  Vector de entrada binario.
            verbose (bool): Modo detallado.

        Retorna:
            int: Índice del cluster asignado.
        """
        if self.n_clusters == 0:
            return self._crear_cluster(x, verbose)

        deshabilitados = set()

        while True:
            # Step 3: calcular scores (−1 para deshabilitados)
            scores = [
                -1.0 if j in deshabilitados
                else self._matching_score(self.b_weights[j], x)
                for j in range(self.n_clusters)
            ]

            # Step 4: seleccionar el mejor
            mejor_j = max(range(len(scores)), key=lambda j: scores[j])
            if scores[mejor_j] < 0:
                return self._crear_cluster(x, verbose)

            if verbose:
                fmt = [f"{s:.3f}" if s >= 0 else "off" for s in scores]
                print(f"      scores={fmt}  mejor={mejor_j}({scores[mejor_j]:.3f})")

            # Step 5: test de vigilancia
            if self._test_vigilancia(self.t_weights[mejor_j], x):
                self._adaptar(mejor_j, x)           # Step 7
                if verbose:
                    print(f"      Vigilancia OK → cluster {mejor_j} actualizado")
                return mejor_j
            else:
                if verbose:
                    print(f"      Vigilancia FALLÓ → cluster {mejor_j} deshabilitado")
                deshabilitados.add(mejor_j)          # Step 6
                if len(deshabilitados) >= self.n_clusters:
                    return self._crear_cluster(x, verbose)

    # ------------------------------------------------------------------
    # Entrenar con todos los vectores
    # ------------------------------------------------------------------
    def entrenar(self, vectores, verbose=False):
        """
        Presenta todos los vectores de entrada a la red (Steps 2-8 en loop).

        Parámetros:
            vectores (list[list[int]]): Vectores binarios de entrada.
            verbose (bool):             Modo detallado.

        Retorna:
            list[int]: Cluster asignado a cada vector.
        """
        asignaciones = []
        for idx, x in enumerate(vectores):
            if verbose:
                print(f"\n  [#{idx+1:>3}] {x}")
            c = self._procesar(x, verbose)
            asignaciones.append(c)
            if verbose:
                print(f"      → cluster {c}")
        return asignaciones


# ===========================================================================
# CLI
# ===========================================================================

def construir_parser():
    """Construye el parser de argumentos."""
    p = argparse.ArgumentParser(
        prog="CarGross.py",
        description="Red ART1 Carpenter/Grossberg — Clustering no supervisado",
        add_help=False,
    )
    p.add_argument("input",           nargs="?",
                   help="Archivo CSV de entrada")
    p.add_argument("--vigilance",     type=float, default=0.5, metavar="FLOAT",
                   help="Umbral rho 0.0-1.0 (default: 0.5)")
    p.add_argument("--max-clusters",  type=int,   default=50,  metavar="INT",
                   help="Máx. clusters (default: 50)")
    p.add_argument("--output",        type=str,   default=None, metavar="FILE",
                   help="CSV de salida con columna 'cluster'")
    p.add_argument("--metadata",      type=str,   default=None, metavar="FILE",
                   help="CSV de metadatos con umbrales custom")
    p.add_argument("--save-txt",      type=str,   default=None, metavar="FILE",
                   help="Guarda resumen en archivo .txt")
    p.add_argument("--verbose",       action="store_true",
                   help="Muestra detalle de cada iteración")
    p.add_argument("--help", "-h",    action="store_true",
                   help="Muestra el manual y sale")
    return p


def main():
    """Punto de entrada principal del programa."""

    if len(sys.argv) < 2:
        mostrar_man()
        sys.exit(1)

    parser = construir_parser()
    try:
        args = parser.parse_args()
    except SystemExit:
        mostrar_man()
        sys.exit(1)

    if args.help:
        mostrar_man()
        sys.exit(0)

    if not args.input:
        print("ERROR: Falta el archivo CSV de entrada.")
        print("  Uso:   python src/CarGross.py <archivo.csv> [opciones]")
        print("  Ayuda: python src/CarGross.py --help")
        sys.exit(1)

    if not (0.0 <= args.vigilance <= 1.0):
        print(f"ERROR: --vigilance debe estar entre 0.0 y 1.0. "
              f"Recibido: {args.vigilance}")
        sys.exit(3)

    # --- encabezado ---------------------------------------------------------
    print(f"\nCarGross.py — Red ART1 Carpenter/Grossberg")
    print("=" * 52)
    print(f"  Entrada:        {args.input}")
    print(f"  Vigilancia:     {args.vigilance}")
    print(f"  Máx. clusters:  {args.max_clusters}")

    # 1. Leer CSV ------------------------------------------------------------
    try:
        encabezados, filas = leer_csv(args.input)
    except FileNotFoundError as e:
        print(f"ERROR: {e}"); sys.exit(1)
    except ValueError as e:
        print(f"ERROR de formato: {e}"); sys.exit(2)

    print(f"  Registros:      {len(filas)}")

    # 2. Columnas numéricas --------------------------------------------------
    cols_num = detectar_columnas_numericas(encabezados, filas)
    if not cols_num:
        print("ERROR: No se encontraron columnas numéricas en el CSV.")
        sys.exit(2)
    print(f"  Columnas (num): {', '.join(cols_num)}")

    try:
        validar_minimo(filas, cols_num)
    except ValueError as e:
        print(f"ERROR: {e}"); sys.exit(2)

    # 3. Umbrales de binarización -------------------------------------------
    umbrales = calcular_umbrales_media(filas, cols_num)

    if args.metadata:
        try:
            meta = leer_metadata(args.metadata)
            print(f"  Metadatos:      {args.metadata} ({len(meta)} umbrales custom)")
            umbrales.update({k: v for k, v in meta.items() if k in umbrales})
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR en metadatos: {e}"); sys.exit(2)

    if args.verbose:
        print("\n  Umbrales de binarización:")
        for col, u in umbrales.items():
            print(f"    {col:<28} {u:.4f}")

    # 4. Binarizar -----------------------------------------------------------
    vectores = binarizar_datos(filas, cols_num, umbrales)
    N = len(cols_num)

    # 5. Entrenar ART1 -------------------------------------------------------
    print(f"\nEntrenando ART1  (N={N}, rho={args.vigilance})...")
    if args.verbose:
        print("\nDetalle de procesamiento:")

    red = ART1(N=N, vigilance=args.vigilance, max_clusters=args.max_clusters)
    t0 = time.time()
    try:
        asignaciones = red.entrenar(vectores, verbose=args.verbose)
    except RuntimeError as e:
        print(f"\nERROR durante entrenamiento: {e}"); sys.exit(1)
    elapsed = time.time() - t0

    # 6. Guardar CSV ---------------------------------------------------------
    if args.output:
        try:
            escribir_csv_resultado(args.output, encabezados, filas, asignaciones)
            print(f"\nCSV de salida → {args.output}")
        except IOError as e:
            print(f"ERROR al escribir CSV: {e}"); sys.exit(1)

    # 7. Resumen -------------------------------------------------------------
    resumen = generar_resumen_texto(
        asignaciones, red.n_clusters, args.vigilance,
        cols_num, args.input, umbrales, elapsed
    )
    imprimir_resumen(resumen)

    # 8. Guardar .txt --------------------------------------------------------
    if args.save_txt:
        try:
            guardar_resumen_txt(args.save_txt, resumen)
            print(f"Resumen .txt   → {args.save_txt}")
        except IOError as e:
            print(f"ERROR al guardar .txt: {e}"); sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
