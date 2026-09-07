# SPDX-License-Identifier: MIT
"""
CarGross.py
===========

Implementacion del clasificador Carpenter/Grossberg (ART1, Adaptive Resonance
Theory 1) segun el algoritmo del Box 3 de Lau (1992), pp. 12-14.

Trabajo Final Integrador (TFI) de la materia Inteligencia Artificial.
UADER - Instituto de Desarrollo Territorial e Ingenieria (IDTI).
Laboratorio de Redes Neuronales Artificiales.

Este modulo provee una red ART1 de aprendizaje no supervisado que realiza
clustering sobre patrones binarios. Acepta datos continuos desde CSV y los
binariza automaticamente usando reglas declaradas en un metadata CSV
separado. La CLI procesa un dataset y emite dos archivos de salida: un CSV
con la asignacion de cluster por fila y un TXT con un reporte legible por
humano. Tambien incluye modo ``--man`` (manual extendido), ``--test``
(smoke test) y ``--shuffle N`` para evaluar la estabilidad del algoritmo
frente al orden de presentacion de los patrones.

Mapeo de notacion matematica (paper Lau 1992 Box 3) a variables del codigo:

+--------------------+-------------------------------------------------+----------------------------------+
| Simbolo matematico | Significado                                      | Variable en codigo               |
+====================+=================================================+==================================+
| N                  | Dimension del vector de entrada                  | self.n_inputs                    |
+--------------------+-------------------------------------------------+----------------------------------+
| M                  | Cantidad maxima de clusters                      | self.max_clusters                |
+--------------------+-------------------------------------------------+----------------------------------+
| x                  | Vector de entrada binario                        | x (lista de ints)                |
+--------------------+-------------------------------------------------+----------------------------------+
| X                  | Matriz de entrada (N filas x M cols)             | X (lista de listas de ints)      |
+--------------------+-------------------------------------------------+----------------------------------+
| t_ij               | Peso top-down entre input i y cluster j         | self.t_weights[i][j]             |
+--------------------+-------------------------------------------------+----------------------------------+
| b_ij               | Peso bottom-up entre input i y cluster j        | self.b_weights[i][j]             |
+--------------------+-------------------------------------------------+----------------------------------+
| mu_j               | Puntaje de coincidencia del cluster j            | matching_scores[j]               |
+--------------------+-------------------------------------------------+----------------------------------+
| j*                 | Indice del cluster con mejor coincidencia        | j_star                           |
+--------------------+-------------------------------------------------+----------------------------------+
| rho                | Umbral de vigilancia                             | self.vigilance                   |
+--------------------+-------------------------------------------------+----------------------------------+
| ||X||              | Norma L1 del vector X (= suma de sus bits)      | norm(x) = sum(x)                 |
+--------------------+-------------------------------------------------+----------------------------------+
| ||T*X||            | Suma de productos t_ij * x_i                    | vigilance_test numerator         |
+--------------------+-------------------------------------------------+----------------------------------+
| phi (relacion)     | ||T*X|| / ||X|| - pasa si > rho                 | (computed in _vigilance_test)    |
+--------------------+-------------------------------------------------+----------------------------------+
| AND logico         | Operacion bitwise AND para adaptacion           | (computed in _adapt)             |
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

Ver ``docs/04_algoritmo.md`` para la descripcion completa.
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
    """Excepcion base para todos los errores especificos del modulo."""


class FileNotFoundCarGrossError(CarGrossError):
    """Archivo requerido no encontrado en el sistema de archivos."""


class MetadataError(CarGrossError):
    """Metadata invalido, incompleto o inconsistente con el dataset."""


class BinarizationError(CarGrossError):
    """Error durante el proceso de binarizacion de una feature."""


class VigilanceError(CarGrossError):
    """Parametro de vigilancia fuera del rango [0,1] o invalido."""


class DatasetError(CarGrossError):
    """Error en la estructura o contenido del dataset de entrada."""


# ---------------------------------------------------------------------------
# ART1
# ---------------------------------------------------------------------------

class ART1:
    """Red neuronal ART1 (Adaptive Resonance Theory 1).

    ART1 es un modelo de aprendizaje no supervisado para clustering de
    patrones binarios. Resuelve el dilema estabilidad-plasticidad: puede
    aprender patrones nuevos sin olvidar los antiguos.

    Implementacion fiel del Box 3 de Lau (1992). Los pesos top-down ``t``
    almacenan los exemplares activos (vector binario de longitud N) y los
    pesos bottom-up ``b`` se usan para calcular los puntajes de coincidencia.

    Parameters
    ----------
    n_inputs : int
        Dimension del vector de entrada binario (N).
    vigilance : float
        Umbral de vigilancia rho en [0,1]. Controla el compromiso
        estabilidad-plasticidad: cerca de 1 exige coincidencia estricta y
        favorece la creacion de clusters nuevos; cerca de 0 acepta
        coincidencias parciales y consolida los clusters existentes.
    max_clusters : int, default 1000
        Tope de seguridad para evitar crecimiento descontrolado cuando
        la vigilancia es alta y los datos son ruidosos. Si la red intenta
        crear un cluster nuevo habiendo alcanzado este limite, levanta
        DatasetError.

    Examples
    --------
    >>> net = ART1(n_inputs=7, vigilance=0.6)
    >>> net.fit([[1,0,1,1,0,0,1], [1,0,1,1,0,0,1], [0,1,0,0,1,1,0]])
    >>> net.n_clusters
    2
    >>> net.predict([1,0,1,1,0,0,1])
    (0, 1.0)
    """

    def __init__(self, n_inputs, vigilance=0.5, max_clusters=1000):
        """Inicializa la red y crea las matrices de pesos vacias.

        Valida que ``n_inputs`` y ``max_clusters`` sean enteros positivos
        y que ``vigilance`` este en [0,1]. Luego invoca ``_init_weights``
        para materializar las matrices ``t`` (top-down) y ``b`` (bottom-up)
        con la forma N x M indicada por el Box 3 de Lau (1992).

        Raises
        ------
        DatasetError
            Si ``n_inputs`` o ``max_clusters`` no son enteros positivos.
        VigilanceError
            Si ``vigilance`` esta fuera de [0,1].
        """
        if not isinstance(n_inputs, int) or n_inputs <= 0:
            raise DatasetError(
                f"n_inputs debe ser entero positivo, recibio {n_inputs!r}"
            )
        if not 0.0 <= vigilance <= 1.0:
            raise VigilanceError(
                f"vigilance debe estar en [0,1], recibio {vigilance!r}"
            )
        if not isinstance(max_clusters, int) or max_clusters <= 0:
            raise DatasetError(
                f"max_clusters debe ser entero positivo, recibio {max_clusters!r}"
            )

        self.n_inputs = n_inputs
        self.vigilance = vigilance
        self.max_clusters = max_clusters
        self._init_weights()

    def _init_weights(self):
        """Inicializa las matrices de pesos (Step 1 del Box 3).

        Crea ``t_weights`` (top-down) como matriz N x M inicializada en 1
        y ``b_weights`` (bottom-up) como matriz N x M inicializada en
        1/(1+N). El vector de todos unos en ``t`` representa "no exemplar"
        todavia; tras el primer match el AND con la entrada colapsa ``t``
        al exemplar real. ``n_clusters`` arranca en 0 (sin categorias
        creadas) y ``disabled`` como set vacio (sin clusters desactivados).
        """
        n, m = self.n_inputs, self.max_clusters
        self.t_weights = [[1] * m for _ in range(n)]
        self.b_weights = [[1.0 / (1.0 + n)] * m for _ in range(n)]
        self.n_clusters = 0
        self.disabled = set()

    def _matching_scores(self, x):
        """Calcula el puntaje de coincidencia de cada cluster activo (Step 3).

        Implementa la formula ``mu_j = sum_i b_ij * x_i`` del Box 3:
        cada cluster j recibe como puntaje la suma de sus pesos bottom-up
        ``b_ij`` pesados por los bits activos de ``x``. Los clusters que
        estan en ``self.disabled`` (fueron rechazados por vigilancia en
        iteraciones previas de este mismo input) reciben ``-infinity``
        para que ``argmax`` los ignore sin necesidad de filtrar la lista.

        Parameters
        ----------
        x : list[int]
            Vector de entrada binario de longitud ``n_inputs``.

        Returns
        -------
        list[float]
            Puntajes ``mu_j`` por cluster; los deshabilitados valen
            ``-math.inf``.
        """
        # Step 3: mu_j = sum_i b_ij * x_i. Clusters deshabilitados => -inf.
        # Usamos -inf (no 0 ni -1) para garantizar que NUNCA ganen el argmax
        # aunque x tenga bits altos: es una exclusion dura, no un castigo suave.
        scores = []
        for j in range(self.n_clusters):
            if j in self.disabled:
                scores.append(-math.inf)
            else:
                mu = sum(self.b_weights[i][j] * x[i] for i in range(self.n_inputs))
                scores.append(mu)
        return scores

    def _select_best(self, scores):
        """Selecciona el cluster ganador j* por argmax (Step 4 del Box 3).

        Implementacion directa del ``argmax(mu_j)``: recorre la lista de
        puntajes y devuelve el indice del maximo. Equivalente a un MAXNET
        con inhibicion lateral ya resuelta en una sola pasada, valido
        porque los puntajes ya son escalares independientes por cluster.

        Parameters
        ----------
        scores : list[float]
            Puntajes ``mu_j`` (los deshabilitados vienen como ``-inf``).

        Returns
        -------
        int or None
            Indice del cluster ganador, o ``None`` si no hay candidatos
        (lista vacia o todos en ``-inf``).
        """
        # Step 4: argmax via seleccion directa (equivalente a MAXNET).
        if not scores:
            return None
        best_j, best_mu = -1, -math.inf
        for j, mu in enumerate(scores):
            if mu > best_mu:
                best_mu = mu
                best_j = j
        return best_j if best_j >= 0 else None

    def _vigilance_test(self, x, j_star):
        """Test de vigilancia para el candidato j* (Step 5 del Box 3).

        Computa ``phi = ||T*X|| / ||X||`` donde ``||T*X||`` es la cantidad
        de bits del exemplar ``t_j*`` que estan prendidos en ``x`` (AND),
        y ``||X||`` es la cantidad de bits prendidos en ``x``. Si
        ``phi > rho`` el candidato pasa (resonancia); si no, sera
        desactivado y se probara el siguiente argmax.

        Caso borde: si ``x`` es el vector cero (``norm_x == 0``), devuelve
        ``True`` para no atascar la red; el caller trata esos casos como
        "entrada degenerada" de todos modos.

        Parameters
        ----------
        x : list[int]
            Vector de entrada binario.
        j_star : int
            Indice del cluster candidato a ganador.

        Returns
        -------
        bool
            ``True`` si ``phi > rho`` (resonancia), ``False`` si no.
        """
        # Step 5: ||T*X|| / ||X|| > rho.
        # Usamos > ESTRICTO (no >=) siguiendo literalmente el Box 3 de Lau
        # (1992). phi == rho cae en el lado del rechazo, lo cual es la
        # interpretacion canonica del test de vigilancia.
        norm_x = sum(x)
        if norm_x == 0:
            return True
        norm_tx = sum(self.t_weights[i][j_star] * x[i] for i in range(self.n_inputs))
        return (norm_tx / norm_x) > self.vigilance

    def _adapt(self, x, j_star):
        """Adapta los pesos del cluster ganador j* (Step 7 del Box 3).

        Dos operaciones:

        1. AND logico: ``t_ij* = t_ij* AND x_i``. Como ambos operandos son
           0 o 1, ``t * x`` en Python produce exactamente el AND binario
           (1*1=1, todo lo demas es 0). Esto apaga los bits del exemplar
           que NO estaban en ``x``, reforzando solo la interseccion.
        2. Renormalizacion: ``b_ij* = t_ij* / (0.5 + sum_k t_kj*)``. La
           constante 0.5 en el denominador evita que un cluster con
           exemplar muy chico (pocos bits) reciba pesos bottom-up
           desproporcionadamente grandes, lo que lo haria ganar siempre
           el argmax y dejaria de aprender.

        Parameters
        ----------
        x : list[int]
            Vector de entrada binario aceptado.
        j_star : int
            Indice del cluster ganador (resonancia confirmada).
        """
        # Step 7: t <- t AND x, b <- t / (0.5 + sum(t)).
        # El AND colapsa el exemplar a la interseccion con x:
        # solo sobreviven los bits prendidos en AMBOS vectores.
        for i in range(self.n_inputs):
            self.t_weights[i][j_star] = self.t_weights[i][j_star] * x[i]
        norm_tx = sum(self.t_weights[i][j_star] for i in range(self.n_inputs))
        # Denominador 0.5 + ||T*X||: la constante 0.5 estabiliza clusters
        # pequenos para que no acaparen todos los argmax futuros.
        denom = 0.5 + norm_tx
        for i in range(self.n_inputs):
            self.b_weights[i][j_star] = self.t_weights[i][j_star] / denom

    def _create_cluster(self, x):
        # Step 6 (caso final): exemplar es x.
        if self.n_clusters >= self.max_clusters:
            raise DatasetError(
                f"Se alcanzo el limite max_clusters={self.max_clusters}; "
                f"no se puede crear un cluster nuevo. Suba --max-clusters "
                f"o baje la vigilancia."
            )
        j_new = self.n_clusters
        norm_x = sum(x)
        denom = 0.5 + norm_x if norm_x > 0 else 1.0
        for i in range(self.n_inputs):
            self.t_weights[i][j_new] = int(x[i])
            self.b_weights[i][j_new] = x[i] / denom if norm_x > 0 else 0.0
        self.n_clusters += 1

    def fit(self, X):
        """Entrena la red ART1 sobre la matriz de entradas ``X`` (Steps 2-8).

        Recorre ``X`` en orden de presentacion y, para cada vector ``x``,
        ejecuta el ciclo completo de resonancia del Box 3 de Lau (1992).
        Internamente reutiliza ``_matching_scores``, ``_select_best``,
        ``_vigilance_test`` y ``_adapt``. Si tras probar todos los clusters
        existentes ninguno pasa la vigilancia, crea un cluster nuevo con
        ``_create_cluster``.

        Diagrama del flujo por cada vector de entrada:

            +---------------------------------------------------------+
            |  Para cada vector x en X (en orden de presentacion):   |
            |                                                         |
            |   +-------------------------------------------------+   |
            |   | Step 2: aplicar x                              |   |
            |   +--------------------+---------------------------+   |
            |                        v                               |
            |   +-------------------------------------------------+   |
            |   | Step 3: matching scores mu_j = sum b_ij * x_i |   |
            |   +--------------------+---------------------------+   |
            |                        v                               |
            |   +-------------------------------------------------+   |
            |   | Step 4: j* = argmax(mu_j) sobre clusters activos| |
            |   +--------------------+---------------------------+   |
            |                        v                               |
            |   +-------------------------------------------------+   |
            |   | Step 5: es ||T*X||/||X|| > rho ?               |   |
            |   +----+--------------------------+-----------------+   |
            |     SI |                          | NO               |
            |        v                          v                   |
            |   +-------------+        +------------------------+   |
            |   | Step 7:     |        | Step 6: desactivar j*, |   |
            |   | adaptar j*  |        | volver a Step 3        |   |
            |   | (AND + renorm)       +------------------------+   |
            |   +-------------+                  |                   |
            |        |                           |                   |
            |        |  (si no quedan clusters activos: crear nuevo) |
            |        v                                               |
            |   Step 8: rehabilitar disabled, siguiente input        |
            +---------------------------------------------------------+

        Parameters
        ----------
        X : list[list[int]]
            Matriz de N filas, cada una un vector binario de longitud
            ``n_inputs``.

        Returns
        -------
        ART1
            ``self``, para permitir encadenamiento ``net.fit(X).predict(x)``.

        Raises
        ------
        DatasetError
            Si ``X`` esta vacio o si alguna fila tiene longitud != ``n_inputs``.
        BinarizationError
            Si alguna fila contiene valores fuera de ``{0, 1}``.
        """
        # Loop principal: Steps 2-8.
        if not X:
            raise DatasetError("X esta vacio; nada que entrenar.")
        for idx, x in enumerate(X):
            if len(x) != self.n_inputs:
                raise DatasetError(
                    f"Fila {idx}: longitud {len(x)} != n_inputs={self.n_inputs}"
                )
            if any(v not in (0, 1) for v in x):
                raise BinarizationError(
                    f"Fila {idx}: contiene valores no binarios {set(v for v in x if v not in (0,1))}"
                )
            # Step 8: rehabilitar clusters (cada input empieza limpio).
            self.disabled.clear()
            # Entrada degenerada (sum=0): crear cluster zero y seguir.
            if sum(x) == 0:
                self._create_cluster(x)
                continue
            # Loop Steps 3-6.
            while True:
                scores = self._matching_scores(x)
                j_star = self._select_best(scores)
                # Fallback del Step 6: si todos los clusters activos fueron
                # rechazados por vigilancia, j_star viene como None y la
                # unica opcion es crear un cluster nuevo con x como exemplar.
                # Esto es la cara "plasticidad" de ART1: el sistema admite
                # categorias que no encajan en ningun cluster previo.
                if j_star is None:
                    self._create_cluster(x)
                    break
                if self._vigilance_test(x, j_star):
                    self._adapt(x, j_star)
                    break
                # Step 6: deshabilitar j* y volver a Step 3.
                self.disabled.add(j_star)
        return self

    def predict(self, x):
        """Clasifica ``x`` contra los clusters existentes (paso de inferencia).

        Recorre todos los clusters ya entrenados, calcula ``mu_j`` para
        cada uno y se queda con el argmax. Luego devuelve tambien el
        ``match_score = ||T*X|| / ||X||`` para esa asignacion, que es la
        proporcion de bits del exemplar que encendieron con ``x``.

        A diferencia de ``fit``, ``predict`` NO aplica el test de vigilancia:
        la categoria ya fue creada, asi que un input se asigna al cluster
        mas cercano aunque la coincidencia sea pobre.

        Parameters
        ----------
        x : list[int]
            Vector de entrada binario de longitud ``n_inputs``.

        Returns
        -------
        tuple[int, float]
            ``(cluster_id, match_score)``. ``cluster_id`` es 0-indexed y
            ``match_score = ||T*X|| / ||X||`` redondeado a 3 decimales
            por el caller.

        Raises
        ------
        DatasetError
            Si ``len(x) != n_inputs``.
        """
        if self.n_clusters == 0:
            # -1 es la senal convencional de "no hay clasificacion posible"
            # (la red aun no fue entrenada). El caller lo distingue de un
            # cluster real (>=0) revisando el signo.
            return (-1, 0.0)
        if len(x) != self.n_inputs:
            raise DatasetError(
                f"predict: longitud {len(x)} != n_inputs={self.n_inputs}"
            )
        norm_x = sum(x)
        if norm_x == 0:
            return (0, 0.0)
        best_j, best_mu = -1, -math.inf
        for j in range(self.n_clusters):
            mu = sum(self.b_weights[i][j] * x[i] for i in range(self.n_inputs))
            if mu > best_mu:
                best_mu = mu
                best_j = j
        norm_tx = sum(self.t_weights[i][best_j] * x[i] for i in range(self.n_inputs))
        return (best_j, norm_tx / norm_x)

    def get_exemplars(self):
        """Devuelve la lista de exemplares activos, uno por cluster.

        El exemplar del cluster j es la columna ``t_weights[:, j]``:
        el vector binario que resulta de aplicar sucesivos AND con los
        inputs que resonaron en ese cluster. Es el prototipo de la
        categoria.

        Returns
        -------
        list[list[int]]
            Lista de ``n_clusters`` vectores binarios, cada uno de
            longitud ``n_inputs``.
        """
        return [
            [self.t_weights[i][j] for i in range(self.n_inputs)]
            for j in range(self.n_clusters)
        ]


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

_ID_COLUMNS = {"id", "sensor_id"}


class DataLoader:
    """Carga un CSV de features continuas y lo binariza segun el metadata.

    Separa la I/O de la red: la red ART1 trabaja solo con vectores
    binarios, asi que toda la conversion continuo -> binario vive aqui.

    Formato del CSV de entrada
    --------------------------
    Un header con una columna ID (cuyo nombre debe ser ``id`` o
    ``sensor_id``) y el resto son features continuas numericas. Ejemplo::

        id,edad,presion,colesterol
        1,55,140,210
        2,30,120,180

    Formato del metadata CSV
    -------------------------
    Header obligatorio: ``dataset, feature, threshold, rule``. Se filtran
    las filas cuya columna ``dataset`` coincide con el ``stem`` del CSV
    de entrada (ej. ``dataset1_pacientes``). El resto define como
    binarizar cada feature:

    - ``feature``:  nombre de la columna en el CSV de entrada.
    - ``threshold`` (float): umbral de decision.
    - ``rule``: uno de ``gte``, ``lte``, ``gt``, ``lt``
      (mayor-igual, menor-igual, mayor, menor contra el umbral).

    Proceso de binarizacion
    -----------------------
    Para cada celda ``(fila, columna)`` se aplica la regla del metadata
    correspondiente y se emite 1 o 0. Si una feature tiene multiples
    reglas en el metadata (por ejemplo ``edad >= 50`` y ``edad >= 70``),
    se aplican en orden y se concatenan los bits, expandiendo la
    dimension efectiva del vector.

    Attributes
    ----------
    csv_path : Path
        Ruta al CSV de features continuas.
    metadata_path : Path or None
        Ruta al metadata CSV. Si es ``None``, ``load_and_binarize``
        lanzara ``MetadataError``.
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
        # Reglas agrupadas por feature (orden estable segun metadata).
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


def _print_manual():
    """Imprime el manual extendido (modo ``--man``) a stdout."""
    print(_MANUAL)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _run_smoke_test():
    """Ejecuta el smoke test sobre ``dataset1_pacientes.csv`` con rho=0.6.

    Carga el dataset canonico, entrena la red y verifica que produzca
    al menos un cluster y que ``predict`` asigne IDs validos a todas
    las filas. Imprime ``TEST PASSED`` o ``TEST FAILED: <razon>``.
    Sirve como regresion rapida: si falla, el modulo entero esta roto.
    """
    here = Path(__file__).resolve().parent
    project_root = here.parent
    csv_path = project_root / "data" / "dataset1_pacientes.csv"
    meta_path = project_root / "data" / "metadata.csv"
    try:
        loader = DataLoader(csv_path, meta_path)
        row_ids, X = loader.load_and_binarize()
        net = ART1(n_inputs=len(X[0]), vigilance=0.6)
        net.fit(X)
        if net.n_clusters < 1:
            print(f"TEST FAILED: n_clusters={net.n_clusters} (esperaba >=1)")
            return
        for idx, x in enumerate(X):
            j, _ = net.predict(x)
            if j < 0 or j >= net.n_clusters:
                print(f"TEST FAILED: fila {idx} (id={row_ids[idx]}) sin cluster valido (j={j})")
                return
        print("TEST PASSED")
    except Exception as exc:
        print(f"TEST FAILED: {exc}")


# ---------------------------------------------------------------------------
# Salida
# ---------------------------------------------------------------------------

def _write_csv_output(path, row_ids, results):
    """Escribe el CSV de salida con ``id, cluster, match_score`` por fila."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "cluster", "match_score"])
        for rid, (cluster, score) in zip(row_ids, results):
            writer.writerow([rid, cluster, f"{score:.3f}"])


def _cluster_members(row_ids, results):
    """Agrupa los IDs por cluster: ``{cluster_id: [ids...]}``."""
    members = {}
    for rid, (cluster, _) in zip(row_ids, results):
        members.setdefault(cluster, []).append(rid)
    return members


def _format_exemplar(ex):
    """Formatea un exemplar binario como ``[0 1 1 0 1]`` para el TXT."""
    return "[" + " ".join(str(b) for b in ex) + "]"


def _write_txt_output(path, args, net, row_ids, results, feature_count):
    """Escribe el TXT legible con resumen, clusters, exemplares e IDs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    members = _cluster_members(row_ids, results)
    exemplars = net.get_exemplars()
    n_assigned = sum(1 for _, (c, _) in zip(row_ids, results) if c >= 0)
    avg_score = (
        sum(s for _, (_, s) in zip(row_ids, results) if s is not None) / len(results)
        if results else 0.0
    )
    total_created = net.n_clusters
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
            fh.write(f"  Exemplar:  {_format_exemplar(exemplars[j])}\n")
            fh.write(f"  Score med: {mean_s:.3f}\n")
            fh.write(f"  IDs:       {', '.join(ids)}\n\n")
        fh.write("Referencia algoritmica: Box 3, Lau (1992) pp. 12-14.\n")
        fh.write("Ver _legacy/CarGross_TP/lau_contenido.md para la transcripcion completa.\n")


def _run_shuffle(net_factory, X, n_runs, base_seed):
    """Entrena N veces con orden aleatorio y mide estabilidad.

    "Pairwise agreement vs run 0" significa: para cada corrida i > 0,
    contamos cuantas filas quedaron en el mismo cluster que en la corrida
    0 (tomada como baseline). El promedio sobre i = 1..N-1 da una
    fraccion en [0,1] que indica que tan estable es la asignacion de
    clusters frente al orden de presentacion.

    Cada corrida usa la misma semilla base + el avance del RNG, asi que
    es reproducible. ``net_factory`` es un callable que devuelve una red
    ART1 fresca para que cada corrida arranque con los mismos pesos
    iniciales (t=1, b=1/(1+N)) y solo cambie el orden de ``X``.

    Returns
    -------
    tuple[float, float, list[int]]
        ``(media_clusters, acuerdo_medio, lista_de_clusters_por_run)``.
    """
    rng = random.Random(base_seed)
    base_assignment = None
    agreements = []
    cluster_counts = []
    for _ in range(n_runs):
        indices = list(range(len(X)))
        rng.shuffle(indices)
        X_shuf = [X[i] for i in indices]
        net = net_factory()
        net.fit(X_shuf)
        cluster_counts.append(net.n_clusters)
        # Reordenar asignaciones al orden original.
        shuffled_assign = [net.predict(x)[0] for x in X_shuf]
        assign = [0] * len(X)
        for new_idx, old_idx in enumerate(indices):
            assign[old_idx] = shuffled_assign[new_idx]
        if base_assignment is None:
            # La primera corrida define la baseline de asignaciones.
            # Las siguientes se comparan con esta, NO entre si, para
            # obtener un numero de "estabilidad absoluta" reproducible.
            base_assignment = assign
        else:
            # Cuenta de filas que mantienen el mismo cluster_id que run 0.
            agree = sum(1 for a, b in zip(assign, base_assignment) if a == b)
            agreements.append(agree / len(assign))
    mean_clusters = sum(cluster_counts) / len(cluster_counts) if cluster_counts else 0.0
    mean_agreement = (sum(agreements) / len(agreements)) if agreements else 1.0
    return mean_clusters, mean_agreement, cluster_counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    """Construye el ``ArgumentParser`` con todas las opciones de la CLI."""
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
    """Punto de entrada de la CLI. Orquesta carga, fit, predict y salida.

    Flujo:

    1. Parsea argumentos (``_build_parser``).
    2. Si ``--man``: imprime manual y sale.
    3. Si ``--test``: corre smoke test y sale.
    4. Carga el CSV + metadata via ``DataLoader`` y binariza.
    5. Entrena ``ART1`` sobre la matriz binaria.
    6. Predice para cada fila y emite CSV (``_write_csv_output``) y TXT
       (``_write_txt_output``).
    7. Si ``--shuffle N``: ejecuta ``_run_shuffle`` y reporta estabilidad.

    Los errores especificos del modulo (``CarGrossError`` y derivados)
    se imprimen a stderr con exit code 1; cualquier otra excepcion se
    considera bug y sale con exit code 2.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.man:
        _print_manual()
        return
    if args.test:
        _run_smoke_test()
        return
    if not args.csv_file:
        parser.error("csv_file es obligatorio (o usa --man / --test).")
    try:
        loader = DataLoader(Path(args.csv_file), Path(args.metadata))
        row_ids, X = loader.load_and_binarize()
        feature_count = len(X[0])
        if args.verbose:
            print(f"[INFO] Cargadas {len(row_ids)} filas, {feature_count} features binarias.")
            print(f"[INFO] Metadata: {args.metadata}")
        net = ART1(
            n_inputs=feature_count,
            vigilance=args.vigilance,
            max_clusters=args.max_clusters,
        )
        net.fit(X)
        results = [net.predict(x) for x in X]
        _write_csv_output(args.output, row_ids, results)
        _write_txt_output(args.save_txt, args, net, row_ids, results, feature_count)
        if args.verbose:
            print(f"[INFO] Clusters formados: {net.n_clusters}")
            print(f"[INFO] CSV: {args.output}")
            print(f"[INFO] TXT: {args.save_txt}")
        # Bloque de shuffle si se pidio.
        if args.shuffle and args.shuffle > 0:
            mean_c, mean_a, counts = _run_shuffle(
                net_factory=lambda: ART1(
                    n_inputs=feature_count,
                    vigilance=args.vigilance,
                    max_clusters=args.max_clusters,
                ),
                X=X,
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
