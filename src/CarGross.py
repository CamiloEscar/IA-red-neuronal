"""CarGross.py - Red neuronal ART1 (Carpenter-Grossberg) para agrupar síntomas de pacientes. TFI."""

import sys
import os
import csv
import random
import argparse
import unicodedata
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any


class ART1Network:
    def __init__(self, num_inputs: int, max_categories: int = 50, rho: float = 0.65):
        if not (0.0 <= rho <= 1.0):
            raise ValueError("El parámetro de vigilancia (rho) debe estar entre 0.0 y 1.0.")
        if num_inputs <= 0:
            raise ValueError("El número de variables de entrada (num_inputs) debe ser mayor a 0.")

        self.N = num_inputs
        self.M = max_categories
        self.rho = rho

        self.num_committed_categories = 0

        # top-down t_ji (M x N), en 1
        self.t = [[1.0 for _ in range(self.N)] for _ in range(self.M)]

        # bottom-up b_ij (N x M), en 1 / (1 + N)
        initial_b = 1.0 / (1.0 + self.N)
        self.b = [[initial_b for _ in range(self.M)] for _ in range(self.N)]

        self.history_assignments: List[int] = []
        self.history_ratios: List[float] = []

    def _compute_matching_scores(self, x: List[int], active_mask: List[bool]) -> List[float]:
        scores = []
        for j in range(self.M):
            if not active_mask[j]:
                scores.append(-1.0)
            else:
                score = sum(self.b[i][j] * x[i] for i in range(self.N))
                scores.append(score)
        return scores

    def train_pattern(self, x: List[int]) -> Tuple[int, float]:
        """Asigna un patrón binario a una categoría. Devuelve (cluster_id, match_ratio)."""
        if len(x) != self.N:
            raise ValueError(f"Dimensión de entrada inválida ({len(x)}). Se esperaban {self.N} elementos.")
        if any(bit not in (0, 1) for bit in x):
            raise ValueError("El vector de entrada contiene valores no binarios (diferentes de 0 o 1).")

        norm_x = sum(x)
        if norm_x == 0:
            return 0, 1.0

        # categorías que ya fallaron la vigilancia
        active_mask = [True] * self.M

        while True:
            scores = self._compute_matching_scores(x, active_mask)
            max_score = max(scores)

            if max_score < 0:
                raise RuntimeError("Capacidad de la red alcanzada: No quedan categorías disponibles.")

            j_star = scores.index(max_score)

            # test de vigilancia: T = plantilla AND x
            T = [self.t[j_star][i] * x[i] for i in range(self.N)]
            norm_T = sum(T)

            match_ratio = norm_T / float(norm_x)

            if match_ratio >= self.rho:
                self._adapt_weights(j_star, T)
                if j_star >= self.num_committed_categories:
                    self.num_committed_categories = j_star + 1
                return j_star, match_ratio
            else:
                active_mask[j_star] = False

    def _adapt_weights(self, j_star: int, T: List[float]):
        """Actualiza los pesos de la categoría ganadora."""
        for i in range(self.N):
            self.t[j_star][i] = T[i]

        sum_t = sum(self.t[j_star][k] for k in range(self.N))
        denominator = 0.5 + sum_t

        for i in range(self.N):
            self.b[i][j_star] = self.t[j_star][i] / denominator

    def fit(self, dataset: List[List[int]]) -> List[Dict[str, float]]:
        """Entrena la red patrón por patrón."""
        results = []
        for idx, pattern in enumerate(dataset):
            cluster_id, match_ratio = self.train_pattern(pattern)
            self.history_assignments.append(cluster_id)
            self.history_ratios.append(match_ratio)
            results.append({
                "patient_index": idx,
                "cluster_assigned": cluster_id,
                "matching_ratio": match_ratio
            })
        return results

    def get_cluster_templates(self) -> Dict[int, List[int]]:
        """Plantillas binarias de cada cluster."""
        templates = {}
        for j in range(self.num_committed_categories):
            templates[j] = [int(self.t[j][i]) for i in range(self.N)]
        return templates

    def compute_metrics(self, validation_labels: Optional[List[str]] = None) -> Dict[str, float]:
        """Métricas de la corrida."""
        if not self.history_ratios:
            return {}

        avg_matching_ratio = sum(self.history_ratios) / len(self.history_ratios)
        templates = self.get_cluster_templates()
        avg_template_stability = (
            sum(sum(temp) for temp in templates.values()) / float(len(templates))
            if templates else 0.0
        )

        metrics = {
            "num_clusters": float(self.num_committed_categories),
            "avg_matching_ratio": avg_matching_ratio,
            "avg_template_active_bits": avg_template_stability,
        }

        if validation_labels and len(validation_labels) == len(self.history_assignments):
            metrics["cluster_purity"] = self._calculate_purity(validation_labels)

        return metrics

    def _calculate_purity(self, labels: List[str]) -> float:
        """Calcula la pureza de los clusters."""
        cluster_counts: Dict[int, Dict[str, int]] = {}
        for cluster_id, label in zip(self.history_assignments, labels):
            if cluster_id not in cluster_counts:
                cluster_counts[cluster_id] = {}
            cluster_counts[cluster_id][label] = cluster_counts[cluster_id].get(label, 0) + 1

        correct_count = sum(max(counts.values()) for counts in cluster_counts.values())
        return correct_count / float(len(labels))


def infer_specialty_and_recommendations(active_symptoms: List[str]) -> Tuple[str, List[str]]:
    """Devuelve especialidad y recomendaciones según los síntomas activos."""
    # sin tildes ni mayúsculas: 'Presión alta' == 'presion alta'
    symptoms_set = set(
        unicodedata.normalize("NFD", s.lower()).encode("ascii", "ignore").decode()
        for s in active_symptoms
    )

    if "disnea" in symptoms_set or "tos" in symptoms_set:
        spec = "NEUMONOLOGÍA"
        recs = [
            "[+] Realizar espirometría y placa de tórax (Rx).",
            "[+] Evaluación de función respiratoria y saturación de oxígeno.",
            "[+] Monitoreo de disnea y expectoración."
        ]
    elif "presion_alta" in symptoms_set or "presion alta" in symptoms_set:
        spec = "CARDIOLOGÍA"
        recs = [
            "[+] Electrocardiograma (ECG) y Monitoreo Holter 24hs.",
            "[+] Medición periódica de presión arterial y ecocardiograma.",
            "[+] Perfil lipídico y laboratorio cardiovascular."
        ]
    elif "dolor_abdominal" in symptoms_set or "dolor abdominal" in symptoms_set:
        spec = "GASTROENTEROLOGÍA"
        recs = [
            "[+] Ecografía abdominal y ecografía hepatobiliar.",
            "[+] Evaluación de laboratorio hepático y digestivo.",
            "[+] Control nutricional y dieta blanda."
        ]
    elif "fiebre" in symptoms_set or "moco" in symptoms_set:
        spec = "INFECTOLOGÍA / CLÍNICA MÉDICA"
        recs = [
            "[+] Hemograma completo y reactivos de fase aguda (PCR / VSG).",
            "[+] Hisopado nasofaríngeo o cultivo de control.",
            "[+] Hidratación adecuada y control térmico."
        ]
    else:
        spec = "CLÍNICA MÉDICA GENERAL"
        recs = [
            "[+] Chequeo clínico general y laboratorio de rutina.",
            "[+] Monitoreo preventivo de signos vitales."
        ]
    return spec, recs


def _normalize_numeric_str(text: str) -> str:
    """'1.0' -> '1' (así CSV y XLSX dan lo mismo). No toca IDs tipo '007'."""
    if not text or text[0] not in '0123456789.-':
        return text
    try:
        num = float(text)
    except ValueError:
        return text
    if not num.is_integer():
        return text
    canonical = str(int(num))
    return canonical if text in (canonical, canonical + '.0') else text


def _cell_to_str(value: Any) -> str:
    """Pasa una celda a texto: None -> "", True/False -> "1"/"0", 1.0 -> "1"."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    return _normalize_numeric_str(str(value).strip())


def _read_csv_rows(file_path: str) -> List[List[str]]:
    """Lee las filas de un .csv, saltando comentarios (#) y filas vacías."""
    # utf-8-sig por el BOM de Excel; si falla se prueba cp1252 y después latin-1
    for encoding in ('utf-8-sig', 'cp1252', 'latin-1'):
        try:
            with open(file_path, mode='r', encoding=encoding, newline='') as f:
                rows = []
                for row in csv.reader(f):
                    if not row or row[0].lstrip().startswith('#'):
                        continue
                    if all(cell.strip() == "" for cell in row):
                        continue
                    rows.append([_cell_to_str(cell) for cell in row])
                return rows
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo decodificar el archivo '{file_path}'. Se probaron UTF-8, CP1252 y Latin-1.")


def _xlsx_col_index(cell_ref: str) -> int:
    """'B3' -> 1 (columna base 0)."""
    idx = 0
    letters = 0
    for ch in cell_ref:
        if not ch.isalpha():
            break
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
        letters += 1
    return idx - 1 if letters else -1


def _xlsx_cell_value(cell: Any, shared_strings: List[str], ns: str) -> str:
    """Valor de una celda .xlsx como texto."""
    cell_type = cell.get("t", "n")
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.iter(f"{ns}t")).strip()
    v = cell.find(f"{ns}v")
    if v is None or v.text is None:
        return ""
    if cell_type == "s":  # índice a shared strings
        return _normalize_numeric_str(shared_strings[int(v.text)].strip())
    if cell_type == "b":  # booleano
        return "1" if v.text.strip() == "1" else "0"
    return _normalize_numeric_str(v.text.strip())


def _read_xlsx_rows_stdlib(file_path: str) -> List[List[str]]:
    """Lector .xlsx mínimo (primera hoja). Un xlsx es un ZIP con XML adentro."""
    import zipfile
    import xml.etree.ElementTree as ET

    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    ns_rel = "{http://schemas.openxmlformats.org/package/2006/relationships}"
    ns_rid = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

    with zipfile.ZipFile(file_path) as z:
        names = z.namelist()

        # shared strings
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in names:
            sst = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in sst.findall(f"{ns}si"):
                shared_strings.append("".join(t.text or "" for t in si.iter(f"{ns}t")))

        # primera hoja
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        first_sheet = wb.find(f"{ns}sheets/{ns}sheet")
        if first_sheet is None:
            raise ValueError(f"No se encontró ninguna hoja de cálculo en '{file_path}'.")
        rel_id = first_sheet.get(f"{ns_rid}id")
        target = None
        for rel in ET.fromstring(z.read("xl/_rels/workbook.xml.rels")).findall(f"{ns_rel}Relationship"):
            if rel.get("Id") == rel_id:
                target = rel.get("Target")
                break
        if not target:
            raise ValueError(f"No se pudo ubicar la hoja de cálculo en '{file_path}'.")
        sheet_path = target.lstrip("/") if target.startswith("/") else "xl/" + target

        # filas; las celdas que faltan se rellenan con ""
        sheet = ET.fromstring(z.read(sheet_path))
        rows = []
        for row_el in sheet.iter(f"{ns}row"):
            row: List[str] = []
            for cell in row_el.findall(f"{ns}c"):
                col_idx = _xlsx_col_index(cell.get("r", ""))
                if col_idx < 0:
                    col_idx = len(row)
                while len(row) < col_idx:
                    row.append("")
                row.append(_xlsx_cell_value(cell, shared_strings, ns))
            rows.append(row)
        return rows


def _read_xlsx_rows(file_path: str) -> List[List[str]]:
    """Lee las filas de un .xlsx (openpyxl si está, si no el lector stdlib)."""
    try:
        import openpyxl
    except ImportError:
        openpyxl = None

    if openpyxl is not None:
        wb = openpyxl.load_workbook(file_path, data_only=True)
        rows = [[_cell_to_str(cell) for cell in r] for r in wb.active.iter_rows(values_only=True)]
    else:
        rows = _read_xlsx_rows_stdlib(file_path)

    return [row for row in rows if any(cell != "" for cell in row)]


def _drop_phantom_columns(rows: List[List[str]]) -> List[List[str]]:
    """Saca columnas fantasma de pandas ('Unnamed: 0' o header vacío)."""
    if not rows:
        return rows
    width = max(len(r) for r in rows)
    phantom = []
    for col in range(width):
        header = rows[0][col].strip() if col < len(rows[0]) else ""
        is_unnamed = (header == "" or header.lower().startswith("unnamed:"))
        if not is_unnamed:
            phantom.append(False)
            continue
        all_empty = all(col >= len(r) or r[col].strip() == "" for r in rows[1:])
        phantom.append(col == 0 or all_empty)

    if not any(phantom):
        return rows
    return [[cell for col, cell in enumerate(r) if col < len(phantom) and not phantom[col]] for r in rows]


def load_dataset(file_path: str) -> Tuple[List[List[int]], List[str], Optional[List[str]], List[str]]:
    """Carga un dataset .xlsx o .csv."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo '{file_path}' no fue encontrado.")

    patient_ids = []

    if file_path.endswith('.xlsx'):
        all_rows = _read_xlsx_rows(file_path)
    else:
        all_rows = _drop_phantom_columns(_read_csv_rows(file_path))

    if not all_rows:
        raise ValueError("El archivo está vacío o no contiene datos válidos.")

    raw_headers = [col.strip() for col in all_rows[0]]

    id_col_idx = None
    label_col_idx = None
    feature_indices = []

    for idx, header_name in enumerate(raw_headers):
        h_lower = header_name.lower()
        if idx == 0 and ('id' in h_lower or 'paciente' in h_lower):
            id_col_idx = idx
        elif idx == len(raw_headers) - 1 and ('especialidad' in h_lower or 'label' in h_lower or 'diag' in h_lower):
            label_col_idx = idx
        else:
            feature_indices.append(idx)

    if label_col_idx is None and len(all_rows) > 1:
        last_val = all_rows[1][-1].strip()
        if not last_val.isdigit():
            label_col_idx = len(raw_headers) - 1
            if label_col_idx in feature_indices:
                feature_indices.remove(label_col_idx)

    feature_names = [raw_headers[i] for i in feature_indices]
    data_patterns = []
    validation_labels = []

    for row_idx, row in enumerate(all_rows[1:], start=2):
        cleaned_row = [str(c).strip() for c in row]
        if not cleaned_row or len(cleaned_row) < len(raw_headers):
            continue

        p_id = cleaned_row[id_col_idx] if id_col_idx is not None else f"PAC_{row_idx-1:03d}"
        patient_ids.append(p_id)

        binary_vector = []
        for f_idx in feature_indices:
            val_str = cleaned_row[f_idx]
            try:
                val = int(float(val_str))
            except ValueError:
                raise ValueError(f"Fila {row_idx}, columna '{raw_headers[f_idx]}': Valor '{val_str}' no es binario (0 o 1).")
            if val not in (0, 1):
                raise ValueError(f"Fila {row_idx}, columna '{raw_headers[f_idx]}': Valor '{val}' no es binario (0 o 1).")
            binary_vector.append(val)

        data_patterns.append(binary_vector)

        if label_col_idx is not None:
            validation_labels.append(cleaned_row[label_col_idx])

    if not data_patterns:
        raise ValueError("El archivo tiene encabezado pero no tiene filas de datos válidas.")

    return data_patterns, feature_names, (validation_labels if label_col_idx is not None else None), patient_ids


def generate_individual_patient_report(
    patient_name_or_id: str,
    symptoms_vector: List[int],
    feature_names: List[str],
    art1: ART1Network,
    cluster_assigned: int,
    matching_ratio: float,
    reference_label: Optional[str] = None
) -> str:
    """Arma la ficha individual de un paciente."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    active_symptoms = [feature_names[i] for i, bit in enumerate(symptoms_vector) if bit == 1]
    absent_symptoms = [feature_names[i] for i, bit in enumerate(symptoms_vector) if bit == 0]

    spec, recs = infer_specialty_and_recommendations(active_symptoms)
    if reference_label:
        ref_text = f"    • ESPECIALIDAD REGISTRADA (REF) : {reference_label}\n"
    else:
        ref_text = ""

    template = art1.get_cluster_templates().get(cluster_assigned, [])
    template_str = str(template)

    report = f"""============================================================================
                    RED NEURONAL ART1 (CARPENTER-GROSSBERG)
                    FICHA DE EVALUACIÓN CLÍNICA INDIVIDUAL
============================================================================
 FECHA Y HORA DE EVALUACIÓN : {now_str}
 NOMBRE / ID DEL PACIENTE   : {patient_name_or_id}
 UMBRAL DE VIGILANCIA (rho)  : {art1.rho:.2f}
----------------------------------------------------------------------------

 1. PERFIL SINTOMÁTICO DEL PACIENTE / CLIENTE
----------------------------------------------------------------------------
    • Síntomas Presentes (1)  : {', '.join(active_symptoms) if active_symptoms else 'Ninguno'}
    • Síntomas Ausentes  (0)  : {', '.join(absent_symptoms) if absent_symptoms else 'Ninguno'}
    • Vector Binario de Entrada (X) : {symptoms_vector}

 2. DIAGNÓSTICO Y CLASIFICACIÓN NO SUPERVISADA (RED ART1)
----------------------------------------------------------------------------
    • CATEGORÍA / CLUSTER ASIGNADO  : Cluster #{cluster_assigned}
    • PORCENTAJE DE COINCIDENCIA    : {matching_ratio * 100:.2f}% (Matching Ratio: {matching_ratio:.4f})
    • ESTADO DE VIGILANCIA          : PASADO (Coincidencia >= {art1.rho:.2f})
    • PATRÓN REPRESENTANTE (t_ji)   : {template_str}
    • TOTAL CLUSTERS EN RED         : {art1.num_committed_categories} categorías aprendidas
{ref_text}
 3. DERIVACIÓN Y SUGERENCIA DE ESPECIALIDAD MÉDICA
----------------------------------------------------------------------------
    • ESPECIALIDAD RECOMENDADA     : {spec}
    • RECOMENDACIONES Y PASOS A SEGUIR:
"""
    for r in recs:
        report += f"       {r}\n"

    report += """
----------------------------------------------------------------------------
 NOTA: Resultado orientativo del agrupamiento ART1 (aprendizaje no supervisado).
 Debe ser revisado por un profesional médico. TFI - UADER.
============================================================================
"""
    return report


def run_interactive_mode(art1: ART1Network, feature_names: List[str], save_txt_path: Optional[str] = None):
    """Pide nombre y síntomas por consola y arma la ficha."""
    print("\n============================================================================")
    print("      MODO CONSULTA INTERACTIVA DE PACIENTE / CLIENTE EN VIVO")
    print("============================================================================")

    patient_name = input("\n[?] Ingrese el Nombre o ID del Paciente / Cliente: ").strip()
    if not patient_name:
        patient_name = "PAC_CLIENTE_INTERACTIVO"

    print(f"\n[+] Evaluando síntomas para: '{patient_name}'")
    print("   Responda con '1' (Presente) o '0' (Ausente) para cada síntoma:")

    vector = []
    for feat in feature_names:
        while True:
            val_str = input(f"   -> ¿Presenta {feat}? (1=Sí / 0=No): ").strip()
            if val_str in ("0", "1"):
                vector.append(int(val_str))
                break
            print("      [!] Entrada inválida. Ingrese exclusivamente 1 o 0.")

    cluster_id, match_ratio = art1.train_pattern(vector)
    report_text = generate_individual_patient_report(
        patient_name_or_id=patient_name,
        symptoms_vector=vector,
        feature_names=feature_names,
        art1=art1,
        cluster_assigned=cluster_id,
        matching_ratio=match_ratio
    )

    print("\n" + report_text)

    if save_txt_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_txt_path)), exist_ok=True)
        with open(save_txt_path, mode='w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"[ÉXITO] Informe individual guardado en '{save_txt_path}'.")


def print_manual():
    """Imprime el manual (--man)."""
    man_text = """================================================================================
   MANUAL DE REFERENCIA - RED ART1 (CarGross.py)
================================================================================

1. DESCRIPCIÓN
--------------------------------------------------------------------------------
CarGross.py implementa la red ART1 (Adaptive Resonance Theory 1) de Carpenter y
Grossberg (1987), siguiendo el Box 3 de Lau (1992). Agrupa pacientes a partir de
síntomas binarios (1 = presente, 0 = ausente), sin etiquetas previas.

ART1 puede crear categorías nuevas cuando aparece un patrón distinto, sin
borrar las que ya aprendió. Eso es lo que se conoce como el dilema
estabilidad-plasticidad.

2. ALGORITMO
--------------------------------------------------------------------------------
- Capa F1 (comparación): recibe el vector binario X de N elementos.
- Capa F2 (reconocimiento): tiene hasta M nodos de categoría. Gana el que
  tiene mayor puntaje.
- Pesos bottom-up (b_ij): arrancan en 1 / (1 + N).
- Pesos top-down (t_ji): arrancan en 1. Son la plantilla de cada categoría.
- Test de vigilancia: se calcula ||T|| / ||X||, donde T = plantilla AND X.
  * Si es mayor o igual que rho, el paciente entra a esa categoría y la
    plantilla se actualiza con el AND.
  * Si es menor, se descarta ese nodo y se prueba con el siguiente mejor. Si
    no queda ninguno, se crea una categoría nueva.

3. USO
--------------------------------------------------------------------------------
  python src/CarGross.py <archivo_dataset> [opciones]
  python src/CarGross.py --input <archivo_dataset> [opciones]

Opciones:
  <archivo> / -i, --input <ruta>  Dataset de entrada. Formato CSV (.csv); también
                                  se acepta Excel (.xlsx).
  -r, --rho <float>      Parámetro de vigilancia, entre 0.0 y 1.0 (defecto 0.65).
  -o, --output <ruta>    CSV con el resultado por paciente (defecto:
                         resultados_cargross.csv).
  --save-txt <ruta>      Guarda el reporte en un TXT.
  --max_cat <N>          Máximo de categorías (defecto 50).
  -v, --verbose          Muestra también los síntomas detectados.
  --interactive          Carga un paciente por consola y lo clasifica.
  --patient-id <ID>      Ficha individual de un paciente del dataset (por ID o
                         por número de fila).
  --shuffle <N>          Repite la corrida N veces con el orden mezclado.
  --seed <S>             Semilla del barajado (defecto 42).
  --test                 Autotest rápido.
  --man                  Muestra este manual.

4. EJEMPLOS
--------------------------------------------------------------------------------
a) Autotest:
   python src/CarGross.py --test

b) Dataset de entrenamiento:
   python src/CarGross.py data/dataset_entrenamiento_ART1.csv -o results/res_entrenamiento.csv --save-txt results/res_entrenamiento.txt

c) Dataset de validación:
   python src/CarGross.py data/dataset_validacion_ART1.csv -o results/res_validacion.csv --save-txt results/res_validacion.txt

d) Con un rho más exigente (0.80):
   python src/CarGross.py data/dataset_entrenamiento_ART1.csv -r 0.80 -o results/res_entrenamiento_rho080.csv --save-txt results/res_entrenamiento_rho080.txt

e) Modo interactivo (pide nombre y síntomas):
   python src/CarGross.py data/dataset_entrenamiento_ART1.csv --interactive --save-txt results/informe_paciente.txt

f) Estabilidad con el orden mezclado:
   python src/CarGross.py data/dataset_entrenamiento_ART1.csv --shuffle 5 --seed 42 -o results/estable.csv --save-txt results/estable.txt

5. INSTALACIÓN
--------------------------------------------------------------------------------
Hace falta Python 3.10 o superior.

  git clone <url-del-repo>
  python -m venv venv
  .\\venv\\Scripts\\Activate.ps1        (en bash: source venv/bin/activate)
  pip install -r requirements.txt

Para chequear que anda:

  python src/CarGross.py --test

Si imprime "TEST PASSED", está todo bien.

Los .csv se leen con la biblioteca estándar, así que no hace falta instalar
nada para usarlos. openpyxl se usa solo para .xlsx (si no está, hay un lector
propio) y pandas/matplotlib solo para el notebook.

6. DATASETS Y SALIDA ESPERADA
--------------------------------------------------------------------------------
Los dos datasets están en data/ y tienen 100 registros y 5 síntomas:

  data/dataset_entrenamiento_ART1.csv     100 pacientes, 5 síntomas
  data/dataset_validacion_ART1.csv        100 pacientes, 5 síntomas + etiqueta

Cada CSV tiene el ID del paciente en la primera columna, una columna 1/0 por
síntoma y, opcionalmente, la especialidad esperada en la última (se usa solo
para calcular la pureza).

Por consola se ve: la ruta del archivo y el rho, cuántos registros y síntomas
se cargaron, el resumen (clusters formados, matching ratio, estabilidad y
pureza si hay etiquetas) y la plantilla de cada cluster con su especialidad.

El CSV de salida (-o) tiene una fila por paciente con ID, cluster, matching
ratio y especialidad sugerida. El TXT (--save-txt) tiene el reporte completo.
Si la carpeta de salida no existe, se crea.

Con rho 0.65 se obtiene:

  - Entrenamiento: 14 clusters, matching ratio 0.9500.
  - Validación: 12 clusters, matching ratio 0.9600, pureza 69%.

Las corridas completas están en docs/informe.md.

7. LIMITACIONES
--------------------------------------------------------------------------------
- Es un trabajo académico. No diagnostica: la especialidad sugerida sale de las
  plantillas del cluster y la tiene que revisar un profesional de la salud.
- Solo acepta valores 0 y 1. Si el archivo trae otra cosa, termina con [ERROR].
- El resultado depende del orden de los registros. Con --shuffle se puede ver
  cuánto cambia.
- El rho se elige a mano. Con rho bajo los clusters quedan pocos y mezclados;
  con rho alto quedan muchos y chicos.
- La cantidad de clusters está limitada por --max_cat (50 por defecto).
- No guarda nada por sí solo: para tener archivos hay que usar -o y/o
  --save-txt.

8. PREGUNTAS FRECUENTES
--------------------------------------------------------------------------------
P: ¿Para qué es el pip install si los .csv no necesitan nada?
R: Solo para leer .xlsx (openpyxl) y para correr el notebook (pandas,
   matplotlib).

P: ¿Cómo elijo el rho?
R: No hay un valor exacto. Con 0.65 se obtienen 14 y 12 clusters y una pureza
   de 69% en validación. Subirlo fragmenta los grupos y bajarlo los mezcla, así
   que conviene probar varios valores y mirar clusters y pureza.

P: ¿Por qué cambian los clusters si mezclo los datos?
R: ART1 aprende un registro por vez, entonces el orden en que llegan los
   pacientes influye en qué plantillas se crean. Con --shuffle se mide.

P: ¿Puedo usar mis propios datos?
R: Sí, con un CSV con ID, síntomas en 0/1 y (opcional) la etiqueta al final.

P: ¿Diagnostica pacientes?
R: No, solo agrupa. Es un trabajo práctico de la materia Inteligencia
   Artificial y cualquier resultado tiene que revisarlo un profesional médico.

================================================================================
"""
    print(man_text)


def run_smoke_test() -> bool:
    """Autotest rápido (--test). Devuelve True si pasan todos los chequeos."""
    print("[TEST] Iniciando autotest de la Red ART1...")
    patrones = [[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1]]

    def clusters(rho):
        red = ART1Network(num_inputs=4, max_categories=10, rho=rho)
        red.fit(patrones)
        return red

    def vacio_va_a_0():
        return ART1Network(4, 10, 0.65).train_pattern([0, 0, 0, 0]) == (0, 1.0)

    def aprende_con_and():
        red = ART1Network(4, 10, 0.5)
        red.train_pattern([1, 1, 1, 0])
        red.train_pattern([1, 1, 0, 0])
        return red.get_cluster_templates() == {0: [1, 1, 0, 0]}

    chequeos = [
        ("rho bajo (0.3) da 2 categorías", lambda: clusters(0.3).num_committed_categories == 2),
        ("rho alto (0.9) da 3 categorías", lambda: clusters(0.9).num_committed_categories == 3),
        ("patrón vacío va a la categoría 0", vacio_va_a_0),
        ("el aprendizaje achica la plantilla con AND", aprende_con_and),
        ("presión alta (con tilde) deriva a cardiología",
         lambda: infer_specialty_and_recommendations(["Presión alta"])[0] == "CARDIOLOGÍA"),
    ]
    fallos = 0
    for nombre, chequeo in chequeos:
        try:
            ok = chequeo()
        except Exception as exc:
            ok = False
            nombre += f" ({exc})"
        print(f"[TEST] {'ok' if ok else 'FALLO'}: {nombre}")
        fallos += not ok
    if fallos:
        print(f"[ERROR] Autotest fallido: {fallos} chequeo(s) no pasaron.", file=sys.stderr)
        return False
    print("TEST PASSED")
    return True


def main():
    """Punto de entrada: los errores previsibles salen como [ERROR]."""
    try:
        _run()
    except (FileNotFoundError, FileExistsError, ValueError, ImportError, OSError, RuntimeError) as exc:
        motivo = str(exc).strip() or exc.__class__.__name__
        print(f"[ERROR] {motivo}", file=sys.stderr)
        print("Corrida interrumpida. Use --man para el manual de referencia "
              "o --help para ver la sintaxis.", file=sys.stderr)
        sys.exit(1)


def _run():
    parser = argparse.ArgumentParser(
        description="CarGross.py - Red Neuronal ART1 (Carpenter-Grossberg) para Diagnóstico Médico.",
        add_help=True
    )

    parser.add_argument("pos_input", nargs="?", type=str, default=None, help="Ruta al archivo dataset (.xlsx / .csv)")
    parser.add_argument("--input", "-i", type=str, default=None, help="Ruta al archivo dataset (.xlsx / .csv)")
    parser.add_argument("--rho", "-r", type=float, default=0.65, help="Parámetro de vigilancia (0.0 a 1.0, defecto: 0.65)")
    parser.add_argument("--output", "-o", type=str, default="resultados_cargross.csv", help="Ruta del CSV de salida")
    parser.add_argument("--save-txt", type=str, default=None, help="Ruta del reporte TXT de salida")
    parser.add_argument("--max_cat", type=int, default=50, help="Máximo de categorías (defecto: 50)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Modo detallado")
    parser.add_argument("--interactive", action="store_true", help="Modo consulta interactiva de paciente en vivo")
    parser.add_argument("--patient-id", type=str, default=None, help="ID o número de paciente para ficha individual")
    parser.add_argument("--shuffle", type=int, default=0, help="Número de corridas barajadas para test de estabilidad")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad")
    parser.add_argument("--test", action="store_true", help="Ejecutar autotest")
    parser.add_argument("--man", action="store_true", help="Mostrar manual de referencia técnica")

    args = parser.parse_args()

    if args.man:
        print_manual()
        return

    if args.test:
        if not run_smoke_test():
            sys.exit(1)
        return

    input_path = args.pos_input or args.input
    if not input_path:
        print("[ERROR] Debe especificar el archivo dataset de entrada. Use --help para ver las opciones.", file=sys.stderr)
        sys.exit(1)

    print("===============================================================")
    print("   RED NEURONAL CARPENTER-GROSSBERG (ART1) - DIAGNÓSTICO MÉDICO ")
    print("===============================================================")
    print(f"-> Archivo de entrada: {input_path}")
    print(f"-> Parámetro de Vigilancia (rho): {args.rho:.2f}")

    patterns, feature_names, labels, patient_ids = load_dataset(input_path)
    num_patients = len(patterns)
    num_features = len(patterns[0])

    print(f"-> Carga completada: {num_patients} registros, {num_features} síntomas.")
    if args.verbose:
        print(f"-> Síntomas detectados: {', '.join(feature_names)}")

    art1 = ART1Network(num_inputs=num_features, max_categories=args.max_cat, rho=args.rho)
    results = art1.fit(patterns)
    metrics = art1.compute_metrics(validation_labels=labels)

    # modo interactivo
    if args.interactive:
        run_interactive_mode(art1, feature_names, save_txt_path=args.save_txt)
        return

    # ficha de un paciente
    if args.patient_id:
        target_idx = None
        for idx, pid in enumerate(patient_ids):
            if str(pid).strip().lower() == str(args.patient_id).strip().lower():
                target_idx = idx
                break
        if target_idx is None:
            # si no es un ID, probar como número de paciente ("1" -> índice 0)
            if args.patient_id.isdigit():
                idx_num = int(args.patient_id) - 1
                if 0 <= idx_num < num_patients:
                    target_idx = idx_num

        if target_idx is None:
            print(f"[!] No se encontró el paciente con ID '{args.patient_id}'. Mostrando informe del primer paciente.")
            target_idx = 0

        p_id = patient_ids[target_idx]
        p_vec = patterns[target_idx]
        p_res = results[target_idx]
        p_label = labels[target_idx] if labels else None

        indiv_report = generate_individual_patient_report(
            patient_name_or_id=p_id,
            symptoms_vector=p_vec,
            feature_names=feature_names,
            art1=art1,
            cluster_assigned=p_res["cluster_assigned"],
            matching_ratio=p_res["matching_ratio"],
            reference_label=p_label
        )

        print("\n" + indiv_report)

        if args.save_txt:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_txt)), exist_ok=True)
            with open(args.save_txt, mode='w', encoding='utf-8') as f:
                f.write(indiv_report)
            print(f"[ÉXITO] Ficha del paciente '{p_id}' guardada en '{args.save_txt}'.")
        return

    # reporte general
    txt_lines = []
    header_info = f"""================================================================
   RED NEURONAL CARPENTER-GROSSBERG (ART1) - DIAGNÓSTICO MÉDICO
================================================================
-> Archivo de entrada: {input_path}
-> Parámetro de Vigilancia (rho): {args.rho:.2f}
"""
    txt_lines.append(header_info)

    resumen = "--- RESUMEN DE EVALUACIÓN Y MÉTRICAS ---\n"
    resumen += f"1. Categorías / Clusters Formados (M): {int(metrics.get('num_clusters', 0))}\n"
    resumen += f"2. Matching Ratio Promedio (||T||/||X||): {metrics.get('avg_matching_ratio', 0.0):.4f}\n"
    resumen += f"3. Estabilidad (Bits '1' activos promedio por plantilla): {metrics.get('avg_template_active_bits', 0.0):.2f}\n"
    if "cluster_purity" in metrics:
        resumen += f"4. Pureza de Clusters (Validación Externa): {metrics.get('cluster_purity', 0.0) * 100:.2f}%\n"

    txt_lines.append(resumen)

    templates_str = "--- PLANTILLAS REPRESENTATIVAS DE CADA CLUSTER ---\n"
    templates = art1.get_cluster_templates()
    for c_id, temp in templates.items():
        active_syms = [feature_names[i] for i, bit in enumerate(temp) if bit == 1]
        spec, _ = infer_specialty_and_recommendations(active_syms)
        sym_names_str = f" ({', '.join(active_syms)})" if active_syms else ""
        templates_str += f" Cluster #{c_id}{sym_names_str}: {temp} -> {spec}\n"

    txt_lines.append(templates_str)

    # test de estabilidad con barajado
    if args.shuffle > 0:
        print("\n--- TEST DE ESTABILIDAD CON BARAJADO ALEATORIO ---")
        st_summary = f"--- TEST DE ESTABILIDAD CON BARAJADO ALEATORIO ({args.shuffle} CORRIDAS) ---\n"

        clusters_list = []
        ratios_list = []
        rng = random.Random(args.seed)

        for run_idx in range(1, args.shuffle + 1):
            shuffled_indices = list(range(len(patterns)))
            rng.shuffle(shuffled_indices)
            shuffled_patterns = [patterns[i] for i in shuffled_indices]

            art1_shuffled = ART1Network(num_inputs=num_features, max_categories=args.max_cat, rho=args.rho)
            shuffled_res = art1_shuffled.fit(shuffled_patterns)
            shuffled_metrics = art1_shuffled.compute_metrics()

            num_c = int(shuffled_metrics.get('num_clusters', 0))
            avg_m = shuffled_metrics.get('avg_matching_ratio', 0.0)

            clusters_list.append(num_c)
            ratios_list.append(avg_m)

            line_run = f"  Corrida Barajada #{run_idx}: {num_c} clusters formados | Matching Ratio: {avg_m:.4f}"
            print(line_run)
            st_summary += line_run + "\n"

        avg_c = sum(clusters_list) / float(len(clusters_list))
        avg_r = sum(ratios_list) / float(len(ratios_list))
        line_avg = f" -> Promedio post-barajado: {avg_c:.2f} clusters (Base original: {int(metrics.get('num_clusters', 0))}) | Matching Ratio: {avg_r:.4f}"
        print(line_avg)
        st_summary += line_avg + "\n"
        txt_lines.append(txt_lines.pop() + "\n" if txt_lines else "")
        txt_lines.append(st_summary)

    for block in txt_lines[1:]:
        print(block)

    # CSV de salida
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, mode='w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(["ID_Paciente", "Cluster_Asignado", "Matching_Ratio", "Especialidad_Sugerida"])
            for idx, r in enumerate(results):
                p_id = patient_ids[idx]
                p_vec = patterns[idx]
                act_s = [feature_names[i] for i, bit in enumerate(p_vec) if bit == 1]
                spec, _ = infer_specialty_and_recommendations(act_s)
                writer.writerow([p_id, r["cluster_assigned"], f"{r['matching_ratio']:.4f}", spec])
        print(f"[ÉXITO] Resultados guardados en CSV '{args.output}'.")

    # TXT de salida
    if args.save_txt:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_txt)), exist_ok=True)
        with open(args.save_txt, mode='w', encoding='utf-8') as f_txt:
            f_txt.write("\n".join(txt_lines))
            f_txt.write(f"\n================================================================\n Reporte generado el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n================================================================\n")
        print(f"[ÉXITO] Reporte TXT completo guardado en '{args.save_txt}'.")


if __name__ == "__main__":
    main()
