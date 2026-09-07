# SPDX-License-Identifier: MIT
"""
CarGross.py
===========

Implementacion del clasificador Carpenter/Grossberg (ART1, Adaptive Resonance
Theory 1) segun el algoritmo del Box 3 de Lau (1992), pp. 12-14.

Mapeo de notacion matematica (paper Lau 1992 Box 3) a variables del codigo:

+--------------------+-------------------------------------------------+----------------------------------+
| Simbolo matematico | Significado                                      | Variable en codigo               |
+====================+=================================================+==================================+
| N                  | Dimension del vector de entrada                  | self.dimension_entrada           |
+--------------------+-------------------------------------------------+----------------------------------+
| M                  | Cantidad maxima de clusters                      | self.maximo_clusters             |
+--------------------+-------------------------------------------------+----------------------------------+
| x                  | Vector de entrada binario                        | entrada (lista de ints)          |
+--------------------+-------------------------------------------------+----------------------------------+
| X                  | Matriz de entrada (N filas x M cols)             | entradas (lista de listas)       |
+--------------------+-------------------------------------------------+----------------------------------+
| t_ij               | Peso top-down entre input i y cluster j         | self.pesos_descendentes[i][j]    |
+--------------------+-------------------------------------------------+----------------------------------+
| b_ij               | Peso bottom-up entre input i y cluster j        | self.pesos_ascendentes[i][j]     |
+--------------------+-------------------------------------------------+----------------------------------+
| mu_j               | Puntaje de coincidencia del cluster j            | puntajes[j]                      |
+--------------------+-------------------------------------------------+----------------------------------+
| j*                 | Indice del cluster con mejor coincidencia        | indice_mejor                     |
+--------------------+-------------------------------------------------+----------------------------------+
| rho                | Umbral de vigilancia                             | self.vigilancia                  |
+--------------------+-------------------------------------------------+----------------------------------+
| ||X||              | Norma L1 del vector X (= suma de sus bits)      | sum(x)                           |
+--------------------+-------------------------------------------------+----------------------------------+
| ||T*X||            | Suma de productos t_ij * x_i                    | _test_de_vigilancia numerator    |
+--------------------+-------------------------------------------------+----------------------------------+
| phi (relacion)     | ||T*X|| / ||X|| - pasa si > rho                 | (computed in _test_de_vigilancia)|
+--------------------+-------------------------------------------------+----------------------------------+
| AND logico         | Operacion bitwise AND para adaptacion           | (computed in _adaptar)           |
+--------------------+-------------------------------------------------+----------------------------------+

Algoritmo (Box 3 de Lau 1992, pp. 12-14):

  Step 1: inicializar t_ij = 1, b_ij = 1/(1+N), set rho
  Step 2: aplicar nueva entrada x
  Step 3: computar mu_j = sum b_ij * x_i para todos los clusters activos
  Step 4: seleccionar j* = argmax(mu_j) (via MAXNET/inhibicion lateral)
  Step 5: test de vigilancia - phi = ||T*X|| / ||X|| - phi > rho?
          SI -> Step 7 (resonancia, adaptar)
          NO -> Step 6 (desactivar j*, volver a Step 3)
  Step 6: desactivar el mejor cluster temporalmente
  Step 7: adaptar t_ij* = t_ij* * x_i (AND), renormalizar b
  Step 8: rehabilitar clusters desactivados, repetir desde Step 2
"""

import argparse
import csv
import math
import random
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Excepciones
# ---------------------------------------------------------------------------

class CarGrossError(Exception):
    pass


class FileNotFoundCarGrossError(CarGrossError):
    pass


class MetadataError(CarGrossError):
    pass


class BinarizationError(CarGrossError):
    pass


class VigilanceError(CarGrossError):
    pass


class DatasetError(CarGrossError):
    pass


# ---------------------------------------------------------------------------
# ART1
# ---------------------------------------------------------------------------

class ART1:
    """Red neuronal ART1 (Adaptive Resonance Theory 1) para clustering
    no supervisado de patrones binarios. ``vigilancia`` (rho) controla el
    dilema estabilidad-plasticidad y ``maximo_clusters`` acota el crecimiento.
    Implementacion fiel del Box 3 de Lau (1992), pp. 12-14.
    """

    def __init__(self, dimension_entrada, vigilancia=0.5, maximo_clusters=1000):
        if not isinstance(dimension_entrada, int) or dimension_entrada <= 0:
            raise DatasetError(
                f"dimension_entrada debe ser entero positivo, recibio {dimension_entrada!r}"
            )
        if not 0.0 <= vigilancia <= 1.0:
            raise VigilanceError(
                f"vigilancia debe estar en [0,1], recibio {vigilancia!r}"
            )
        if not isinstance(maximo_clusters, int) or maximo_clusters <= 0:
            raise DatasetError(
                f"maximo_clusters debe ser entero positivo, recibio {maximo_clusters!r}"
            )

        self.dimension_entrada = dimension_entrada
        self.vigilancia = vigilancia
        self.maximo_clusters = maximo_clusters
        self._inicializar_pesos()

    def _inicializar_pesos(self):
        n, m = self.dimension_entrada, self.maximo_clusters
        self.pesos_descendentes = [[1] * m for _ in range(n)]
        self.pesos_ascendentes = [[1.0 / (1.0 + n)] * m for _ in range(n)]
        self.cantidad_clusters = 0
        self.desactivados = set()

    def _calcular_puntajes(self, entrada):
        puntajes = []
        for j in range(self.cantidad_clusters):
            if j in self.desactivados:
                puntajes.append(-math.inf)
            else:
                mu = sum(self.pesos_ascendentes[i][j] * entrada[i] for i in range(self.dimension_entrada))
                puntajes.append(mu)
        return puntajes

    def _seleccionar_mejor(self, puntajes):
        if not puntajes:
            return None
        mejor_j, mejor_mu = -1, -math.inf
        for j, mu in enumerate(puntajes):
            if mu > mejor_mu:
                mejor_mu = mu
                mejor_j = j
        return mejor_j if mejor_j >= 0 else None

    def _test_de_vigilancia(self, entrada, indice_mejor):
        """Test de vigilancia: phi = ||T*x|| / ||x|| > rho."""
        # > ESTRICTO (no >=) siguiendo literalmente el Box 3 de Lau (1992):
        # phi == rho cae en el lado del rechazo.
        norm_x = sum(entrada)
        if norm_x == 0:
            return True
        norm_tx = sum(self.pesos_descendentes[i][indice_mejor] * entrada[i] for i in range(self.dimension_entrada))
        return (norm_tx / norm_x) > self.vigilancia

    def _adaptar(self, entrada, indice_mejor):
        """Adapta los pesos del cluster ganador (Step 7: t AND x, renormalizar b)."""
        # El AND colapsa el exemplar a la interseccion con entrada: solo
        # sobreviven los bits prendidos en AMBOS vectores (sin AND logico no
        # podriamos representar la operacion de generalizacion).
        for i in range(self.dimension_entrada):
            self.pesos_descendentes[i][indice_mejor] = self.pesos_descendentes[i][indice_mejor] * entrada[i]
        norm_tx = sum(self.pesos_descendentes[i][indice_mejor] for i in range(self.dimension_entrada))
        denom = 0.5 + norm_tx
        for i in range(self.dimension_entrada):
            self.pesos_ascendentes[i][indice_mejor] = self.pesos_descendentes[i][indice_mejor] / denom

    def _create_cluster(self, entrada):
        if self.cantidad_clusters >= self.maximo_clusters:
            raise DatasetError(
                f"Se alcanzo el limite maximo_clusters={self.maximo_clusters}; "
                f"no se puede crear un cluster nuevo. Suba --max-clusters "
                f"o baje la vigilancia."
            )
        j_new = self.cantidad_clusters
        norm_x = sum(entrada)
        denom = 0.5 + norm_x if norm_x > 0 else 1.0
        for i in range(self.dimension_entrada):
            self.pesos_descendentes[i][j_new] = int(entrada[i])
            self.pesos_ascendentes[i][j_new] = entrada[i] / denom if norm_x > 0 else 0.0
        self.cantidad_clusters += 1

    def entrenar(self, entradas):
        """Entrena la red ART1 sobre la matriz de entradas (Steps 2-8)."""
        if not entradas:
            raise DatasetError("entradas esta vacio; nada que entrenar.")
        for idx, entrada in enumerate(entradas):
            if len(entrada) != self.dimension_entrada:
                raise DatasetError(
                    f"Fila {idx}: longitud {len(entrada)} != dimension_entrada={self.dimension_entrada}"
                )
            if any(v not in (0, 1) for v in entrada):
                raise BinarizationError(
                    f"Fila {idx}: contiene valores no binarios {set(v for v in entrada if v not in (0,1))}"
                )
            self.desactivados.clear()
            if sum(entrada) == 0:
                self._create_cluster(entrada)
                continue
            while True:
                puntajes = self._calcular_puntajes(entrada)
                indice_mejor = self._seleccionar_mejor(puntajes)
                # Fallback del Step 6: si todos los clusters activos fueron
                # rechazados por vigilancia, indice_mejor viene como None y la
                # unica opcion es crear un cluster nuevo con entrada como
                # exemplar (cara "plasticidad" de ART1).
                if indice_mejor is None:
                    self._create_cluster(entrada)
                    break
                if self._test_de_vigilancia(entrada, indice_mejor):
                    self._adaptar(entrada, indice_mejor)
                    break
                self.desactivados.add(indice_mejor)
        return self

    def predecir(self, entrada):
        if self.cantidad_clusters == 0:
            return (-1, 0.0)
        if len(entrada) != self.dimension_entrada:
            raise DatasetError(
                f"predecir: longitud {len(entrada)} != dimension_entrada={self.dimension_entrada}"
            )
        norm_x = sum(entrada)
        if norm_x == 0:
            return (0, 0.0)
        best_j, best_mu = -1, -math.inf
        for j in range(self.cantidad_clusters):
            mu = sum(self.pesos_ascendentes[i][j] * entrada[i] for i in range(self.dimension_entrada))
            if mu > best_mu:
                best_mu = mu
                best_j = j
        norm_tx = sum(self.pesos_descendentes[i][best_j] * entrada[i] for i in range(self.dimension_entrada))
        return (best_j, norm_tx / norm_x)

    def obtener_exemplares(self):
        return [
            [self.pesos_descendentes[i][j] for i in range(self.dimension_entrada)]
            for j in range(self.cantidad_clusters)
        ]


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

_ID_COLUMNS = {"id", "sensor_id"}


class DataLoader:
    """Carga un CSV de features continuas y lo binariza segun el metadata.
    El header debe tener una columna ID (``id`` o ``sensor_id``) y el resto
    son features numericas. El metadata CSV define las reglas de binarizacion
    (threshold + rule: gte/lte/gt/lt) por feature.
    """

    def __init__(self, csv_path, metadata_path=None):
        self.csv_path = Path(csv_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self._metadata = None

    def load_metadata(self):
        if self.metadata_path is None:
            raise MetadataError(
                "No se proporciono metadata_path; la binarizacion lo requiere."
            )
        if not self.metadata_path.exists():
            raise FileNotFoundCarGrossError(
                f"No existe metadata: {self.metadata_path}"
            )
        rows = []
        with self.metadata_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            required = {"dataset", "feature", "threshold", "rule"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise MetadataError(
                    f"metadata le faltan columnas: {sorted(missing)}"
                )
            dataset_name = self.csv_path.stem
            for row in reader:
                if row["dataset"].strip() != dataset_name:
                    continue
                try:
                    threshold = float(row["threshold"])
                except ValueError as exc:
                    raise MetadataError(
                        f"threshold invalido en metadata: {row['threshold']!r}"
                    ) from exc
                rows.append({
                    "feature": row["feature"].strip(),
                    "threshold": threshold,
                    "rule": row["rule"].strip(),
                })
        if not rows:
            raise MetadataError(
                f"metadata sin filas para dataset={self.csv_path.stem!r}"
            )
        self._metadata = rows
        return rows

    @staticmethod
    def binarize(value, rule, threshold):
        if rule == "gte":
            return int(value >= threshold)
        if rule == "lte":
            return int(value <= threshold)
        if rule == "gt":
            return int(value > threshold)
        if rule == "lt":
            return int(value < threshold)
        raise BinarizationError(f"Regla desconocida: {rule!r}")

    def load_and_binarize(self):
        if not self.csv_path.exists():
            raise FileNotFoundCarGrossError(
                f"No existe CSV de entrada: {self.csv_path}"
            )
        metadata = self.load_metadata()
        rules_by_feature = {}
        for entry in metadata:
            rules_by_feature.setdefault(entry["feature"], []).append(entry)
        with self.csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                raise DatasetError(f"CSV vacio: {self.csv_path}")
            columns = list(reader.fieldnames)
            id_col = next((c for c in columns if c in _ID_COLUMNS), None)
            if id_col is None:
                raise DatasetError(
                    f"CSV sin columna ID (esperaba una de {_ID_COLUMNS}): "
                    f"{columns}"
                )
            feature_cols = [c for c in columns if c != id_col]
            missing_meta = [c for c in feature_cols if c not in rules_by_feature]
            if missing_meta:
                raise BinarizationError(
                    f"Features sin entrada en metadata: {missing_meta}"
                )
            row_ids = []
            binary_matrix = []
            for row_idx, row in enumerate(reader):
                row_ids.append(row[id_col])
                bin_row = []
                for col in feature_cols:
                    raw = row[col]
                    try:
                        value = float(raw)
                    except ValueError as exc:
                        raise BinarizationError(
                            f"Fila {row_idx}, columna {col!r}: valor no numerico {raw!r}"
                        ) from exc
                    for entry in rules_by_feature[col]:
                        bin_row.append(self.binarize(value, entry["rule"], entry["threshold"]))
                binary_matrix.append(bin_row)
        if not binary_matrix:
            raise DatasetError(f"CSV sin filas de datos: {self.csv_path}")
        return row_ids, binary_matrix


# ---------------------------------------------------------------------------
# Manual (--man)
# ---------------------------------------------------------------------------

_MANUAL = """\
SINOPSIS
  python src/CarGross.py <csv_file> [opciones]

DESCRIPCION
  Implementa el clasificador Carpenter/Grossberg (ART1) para hacer clustering
  no supervisado sobre patrones binarios. ART1 resuelve el dilema
  estabilidad-plasticidad: la red puede aprender patrones nuevos sin olvidar
  los antiguos gracias al parametro de vigilancia (rho).

  Algoritmo de referencia: Box 3 de Lau, C. (Ed.) (1992). "Artificial Neural
  Networks". IEEE Press, pp. 12-14. Transcripcion completa en
  _legacy/CarGross_TP/lau_contenido.md dentro de este repositorio.

  La red opera en ocho pasos (Box 3, Lau 1992):
    1. Inicializa pesos top-down en 1 y bottom-up en 1/(1+N).
    2. Presenta una nueva entrada binaria x.
    3. Calcula puntajes de coincidencia mu_j para cada cluster activo.
    4. Selecciona el mejor cluster j* por inhibicion lateral (MAXNET-like).
    5. Test de vigilancia: ||T*X|| / ||X|| > rho ?
    6. Si NO, deshabilita j* y vuelve a 3. Si todos fallan, crea cluster.
    7. Si SI, adapta j*: t <- t AND x, b <- t / (0.5 + sum(t)).
    8. Rehabilita los deshabilitados y vuelve a 2.

  La inicializacion t_ij = 1 representa "no exemplar" (vector de todos unos).
  Tras el primer match, el AND con x colapsa t al exemplar real.

ARGUMENTOS
  csv_file                  (posicional) CSV de entrada con features continuas.
  -r, --vigilance RHO       Vigilancia rho en [0,1]. Default: 0.5.
                            Cerca de 1 = coincidencia estricta (mas clusters).
                            Cerca de 0 = coincidencia laxa (menos clusters).
  -m, --max-clusters M      Maximo de clusters a crear. Default: 1000.
  --metadata PATH           Metadata CSV. Default: data/metadata.csv.
  -o, --output PATH         CSV de salida. Default: results/resultado.csv.
  --save-txt PATH           TXT de salida. Default: results/resultado.txt.
  --shuffle N               Repite el entrenamiento N veces con orden aleatorio.
                            Reporta estabilidad y variacion de # clusters.
  --seed S                  Semilla aleatoria. Default: 42.
  -v, --verbose             Logging detallado paso a paso.
  --man                     Imprime este manual y sale.
  --test                    Corre el smoke test y sale.

EJEMPLOS
  # Dataset 1: pacientes (7 features -> 7 bits con metadata por defecto).
  python src/CarGross.py data/dataset1_pacientes.csv --vigilance 0.7 -v

  # Dataset 2: sensores (8 features -> 8 bits, voltaje produce 2).
  python src/CarGross.py data/dataset2_sensores.csv --vigilance 0.6

  # Evaluar estabilidad frente al orden de presentacion.
  python src/CarGross.py data/dataset1_pacientes.csv --shuffle 20 --seed 7

FORMATO DE SALIDA
  CSV (--output): tres columnas, una fila por patron de entrada.
    id          ID original (id o sensor_id segun dataset).
    cluster     ID de cluster 0-indexed.
    match_score ||T*X|| / ||X|| con 3 decimales.

  TXT (--save-txt): reporte legible.
    Encabezado con parametros y resumen.
    Bloque por cluster: id, tamano, exemplar binario, score medio.
    Pie con referencia al metadata usado.

ALGORITMO DE REFERENCIA
  Box 3, Lau (1992) pp. 12-14. La transcripcion completa, en espanol, vive
  en _legacy/CarGross_TP/lau_contenido.md. Las ecuaciones se implementan literalmente.

LIMITACIONES
  - Solo entradas binarias. La binarizacion previa corre por metadata.
  - El algoritmo es sensible al orden de presentacion (use --shuffle).
  - Con rho alto y datos ruidosos, el numero de clusters crece rapido.
  - Sin modificacion por "slow learning", la red no maneja bien el ruido
    (ver discusion en Lau 1992, p. 13, "Comportamiento del clasificador").
  - max_clusters limita la capacidad; agotarlo levanta DatasetError.

REFERENCIAS
  [1] Lau, C. (Ed.) (1992). Artificial Neural Networks. IEEE Press.
      Box 3, pp. 12-14 ("El clasificador Carpenter/Grossberg").
  [2] Carpenter, G.A. & Grossberg, S. (1987). A massively parallel
      architecture for a self-organizing neural pattern recognition
      machine. CVGIP, 37, 54-115.
  [3] Lippmann, R.P. (1987). An Introduction to Computing with Neural
      Nets. IEEE ASSP Magazine, April 1987, pp. 4-22. Reproducido en [1].
"""


def _imprimir_manual():
    print(_MANUAL)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _ejecutar_smoke_test():
    here = Path(__file__).resolve().parent
    project_root = here.parent
    csv_path = project_root / "data" / "dataset1_pacientes.csv"
    meta_path = project_root / "data" / "metadata.csv"
    try:
        loader = DataLoader(csv_path, meta_path)
        row_ids, entradas = loader.load_and_binarize()
        net = ART1(dimension_entrada=len(entradas[0]), vigilancia=0.6)
        net.entrenar(entradas)
        if net.cantidad_clusters < 1:
            print(f"TEST FAILED: cantidad_clusters={net.cantidad_clusters} (esperaba >=1)")
            return
        for idx, entrada in enumerate(entradas):
            j, _ = net.predecir(entrada)
            if j < 0 or j >= net.cantidad_clusters:
                print(f"TEST FAILED: fila {idx} (id={row_ids[idx]}) sin cluster valido (j={j})")
                return
        print("TEST PASSED")
    except Exception as exc:
        print(f"TEST FAILED: {exc}")


# ---------------------------------------------------------------------------
# Salida
# ---------------------------------------------------------------------------

def _escribir_salida_csv(path, row_ids, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "cluster", "match_score"])
        for rid, (cluster, score) in zip(row_ids, results):
            writer.writerow([rid, cluster, f"{score:.3f}"])


def _miembros_del_cluster(row_ids, results):
    members = {}
    for rid, (cluster, _) in zip(row_ids, results):
        members.setdefault(cluster, []).append(rid)
    return members


def _formatear_exemplar(ex):
    return "[" + " ".join(str(b) for b in ex) + "]"


def _escribir_salida_txt(path, args, net, row_ids, results, feature_count):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    members = _miembros_del_cluster(row_ids, results)
    exemplars = net.obtener_exemplares()
    n_assigned = sum(1 for _, (c, _) in zip(row_ids, results) if c >= 0)
    avg_score = (
        sum(s for _, (_, s) in zip(row_ids, results) if s is not None) / len(results)
        if results else 0.0
    )
    total_created = net.cantidad_clusters
    nonempty_clusters = [j for j in range(total_created) if len(members.get(j, [])) > 0]
    skipped = total_created - len(nonempty_clusters)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("Reporte ART1 (Carpenter/Grossberg)\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"Dataset:          {args.csv_file}\n")
        fh.write(f"Metadata:         {args.metadata}\n")
        fh.write(f"Vigilancia (rho): {args.vigilance}\n")
        fh.write(f"Max clusters:     {args.max_clusters or feature_count}\n")
        fh.write(f"N (features bin): {feature_count}\n")
        fh.write(f"Filas totales:    {len(row_ids)}\n")
        fh.write(
            f"Cantidad de clusters con miembros: "
            f"{len(nonempty_clusters)} (de {total_created} totales creados)\n"
        )
        fh.write(f"Score medio:      {avg_score:.3f}\n")
        fh.write("=" * 60 + "\n\n")
        if skipped > 0:
            fh.write(
                f"(Se omiten {skipped} clusters con exemplar vacio; "
                "ver doc 06_limitaciones_y_etica.md para contexto.)\n\n"
            )
        for j in nonempty_clusters:
            ids = members.get(j, [])
            scores = [s for rid, (c, s) in zip(row_ids, results) if c == j]
            mean_s = (sum(scores) / len(scores)) if scores else 0.0
            fh.write(f"Cluster {j}\n")
            fh.write(f"  Tamano:    {len(ids)}\n")
            fh.write(f"  Exemplar:  {_formatear_exemplar(exemplars[j])}\n")
            fh.write(f"  Score med: {mean_s:.3f}\n")
            fh.write(f"  IDs:       {', '.join(ids)}\n\n")
        fh.write("Referencia algoritmica: Box 3, Lau (1992) pp. 12-14.\n")
        fh.write("Ver _legacy/CarGross_TP/lau_contenido.md para la transcripcion completa.\n")


def _ejecutar_barajado(net_factory, entradas, n_runs, base_seed):
    rng = random.Random(base_seed)
    base_assignment = None
    agreements = []
    cluster_counts = []
    for _ in range(n_runs):
        indices = list(range(len(entradas)))
        rng.shuffle(indices)
        entradas_shuf = [entradas[i] for i in indices]
        net = net_factory()
        net.entrenar(entradas_shuf)
        cluster_counts.append(net.cantidad_clusters)
        shuffled_assign = [net.predecir(entrada)[0] for entrada in entradas_shuf]
        assign = [0] * len(entradas)
        for new_idx, old_idx in enumerate(indices):
            assign[old_idx] = shuffled_assign[new_idx]
        if base_assignment is None:
            base_assignment = assign
        else:
            agree = sum(1 for a, b in zip(assign, base_assignment) if a == b)
            agreements.append(agree / len(assign))
    mean_clusters = sum(cluster_counts) / len(cluster_counts) if cluster_counts else 0.0
    mean_agreement = (sum(agreements) / len(agreements)) if agreements else 1.0
    return mean_clusters, mean_agreement, cluster_counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _construir_parser():
    p = argparse.ArgumentParser(
        prog="CarGross",
        description="Clustering no supervisado con ART1 (Carpenter-Grossberg, 1987).",
        epilog="Usa --man para el manual completo.",
    )
    p.add_argument("csv_file", nargs="?", help="CSV de entrada (features continuas).")
    p.add_argument("--vigilance", "-r", type=float, default=0.5,
                   help="Vigilancia rho en [0,1]. Default: 0.5.")
    p.add_argument("--max-clusters", "-m", type=int, default=1000,
                   help="Maximo de clusters. Default: 1000.")
    p.add_argument("--metadata", default="data/metadata.csv",
                   help="Metadata CSV. Default: data/metadata.csv.")
    p.add_argument("--output", "-o", default="results/resultado.csv",
                   help="CSV de salida. Default: results/resultado.csv.")
    p.add_argument("--save-txt", default="results/resultado.txt",
                   help="TXT de salida. Default: results/resultado.txt.")
    p.add_argument("--shuffle", type=int, default=0,
                   help="Repite N veces con orden aleatorio. Default: 0 (off).")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria. Default: 42.")
    p.add_argument("--verbose", "-v", action="store_true", help="Logging detallado.")
    p.add_argument("--man", action="store_true", help="Imprime manual y sale.")
    p.add_argument("--test", action="store_true", help="Smoke test y sale.")
    return p


def main(argv=None):
    parser = _construir_parser()
    args = parser.parse_args(argv)
    if args.man:
        _imprimir_manual()
        return
    if args.test:
        _ejecutar_smoke_test()
        return
    if not args.csv_file:
        parser.error("csv_file es obligatorio (o usa --man / --test).")
    try:
        loader = DataLoader(Path(args.csv_file), Path(args.metadata))
        row_ids, entradas = loader.load_and_binarize()
        feature_count = len(entradas[0])
        if args.verbose:
            print(f"[INFO] Cargadas {len(row_ids)} filas, {feature_count} features binarias.")
            print(f"[INFO] Metadata: {args.metadata}")
        net = ART1(
            dimension_entrada=feature_count,
            vigilancia=args.vigilance,
            maximo_clusters=args.max_clusters,
        )
        net.entrenar(entradas)
        results = [net.predecir(entrada) for entrada in entradas]
        _escribir_salida_csv(args.output, row_ids, results)
        _escribir_salida_txt(args.save_txt, args, net, row_ids, results, feature_count)
        if args.verbose:
            print(f"[INFO] Clusters formados: {net.cantidad_clusters}")
            print(f"[INFO] CSV: {args.output}")
            print(f"[INFO] TXT: {args.save_txt}")
        if args.shuffle and args.shuffle > 0:
            mean_c, mean_a, counts = _ejecutar_barajado(
                net_factory=lambda: ART1(
                    dimension_entrada=feature_count,
                    vigilancia=args.vigilance,
                    maximo_clusters=args.max_clusters,
                ),
                entradas=entradas,
                n_runs=args.shuffle,
                base_seed=args.seed,
            )
            print()
            print("Reporte de estabilidad (--shuffle)")
            print("-" * 40)
            print(f"Ejecuciones:       {args.shuffle}")
            print(f"Clusters por run:  {counts}")
            print(f"Media # clusters:  {mean_c:.2f}")
            if args.shuffle > 1:
                print(f"Acuerdo medio:     {mean_a:.3f}  (vs. run 0)")
    except CarGrossError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        print("Usa --man para ver el manual completo.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] Inesperado: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
