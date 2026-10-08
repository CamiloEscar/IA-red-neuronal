"""
CarGross.py - Implementación de la Red Neuronal ART1 (Carpenter-Grossberg)
Cátedra Inteligencia Artificial - Trabajo Final Integrador (TFI)

Este módulo implementa el algoritmo de la Red Neuronal ART1 (Adaptive Resonance Theory 1)
para el agrupamiento no supervisado de patrones binarios, aplicado al diagnóstico médico
y categorización de síntomas de pacientes.
"""

import sys
import os
import csv
import random
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any



class ART1Network:
    """
    Inicializa la red ART1.

    Parámetros:
        num_inputs: cantidad de variables de entrada del patrón binario.
        max_categories: número máximo de categorías o clusters que la red puede crear.
        rho: umbral de vigilancia que controla la aceptación de un patrón en un cluster.
    """
    # Validacion del parametro de vigilancia (rho) y del numero de entradas (num_inputs)
    # Debe estar en el rango [0, 1], ya que se usa como umbral de comparación.

    def __init__(self, num_inputs: int, max_categories: int = 50, rho: float = 0.65):
        if not (0.0 <= rho <= 1.0):
            raise ValueError("El parámetro de vigilancia (rho) debe estar entre 0.0 y 1.0.")
        if num_inputs <= 0:
            raise ValueError("El número de variables de entrada (num_inputs) debe ser mayor a 0.")

        self.N = num_inputs
        self.M = max_categories
        self.rho = rho

        self.num_committed_categories = 0

        # Paso 1: Inicialización de Pesos Synápticos
        # Pesos Top-Down t_ji (M x N), inicializados en 1
        self.t = [[1.0 for _ in range(self.N)] for _ in range(self.M)]

        # Pesos Bottom-Up b_ij (N x M), inicializados en 1 / (1 + N)
        initial_b = 1.0 / (1.0 + self.N)
        self.b = [[initial_b for _ in range(self.M)] for _ in range(self.N)]

        self.history_assignments: List[int] = []
        self.history_ratios: List[float] = []
    
    def _compute_matching_scores(self, x: List[int], active_mask: List[bool]) -> List[float]:
        """Calcula las puntuaciones de coincidencia (mu_j) para cada categoría j."""
        # Lista que guarda el score final para cada categoría j.
        scores = []
        for j in range(self.M):
            # Si la categoría está desactivada por no superar el test de vigilancia,
            # se descarta para esta iteración.
            if not active_mask[j]:
                scores.append(-1.0)
            else:
                # Score de coincidencia:
                # suma de b[i][j] * x[i] para todas las entradas i.
                # Esto mide qué tan bien la categoría actual representa el patrón.
                score = sum(self.b[i][j] * x[i] for i in range(self.N))
                scores.append(score)
        return scores

    def train_pattern(self, x: List[int]) -> Tuple[int, float]:
        """
            Procesa un único patrón binario de entrada y lo asigna a una categoría ART1.

            Este método implementa la lógica principal del algoritmo:
            1. valida que el vector sea válido,
            2. calcula la categoría más compatible,
            3. aplica el test de vigilancia,
            4. si supera la prueba, adapta los pesos y devuelve el cluster asignado.

            Parámetros:
                x: vector binario de entrada de tamaño N.

            Retorna:
                (cluster_id, match_ratio)
                - cluster_id: índice de la categoría ganadora.
                - match_ratio: proporción de coincidencia con la plantilla.
        
        """
        # Verifica que la longitud del vector coincida con la dimensión de entrada de la red.
        if len(x) != self.N:
            raise ValueError(f"Dimensión de entrada inválida ({len(x)}). Se esperaban {self.N} elementos.")
        # Verifica que el patrón sea binario, es decir, que cada valor sea 0 o 1.
        if any(bit not in (0, 1) for bit in x):
            raise ValueError("El vector de entrada contiene valores no binarios (diferentes de 0 o 1).")

        # Norma del vector de entrada: cantidad de bits activos (1s).
        norm_x = sum(x)
        if norm_x == 0:
            return 0, 1.0

        # En cada iteración, si una categoría falla la prueba de vigilancia,
        # se desactiva para no volver a elegirla.
        active_mask = [True] * self.M

        while True:
            # 1) Calcula el score de coincidencia para todas las categorías activas.
            scores = self._compute_matching_scores(x, active_mask)
            
            # 2) Selecciona la categoría con mayor puntaje.
            max_score = max(scores)
            
            # Si todas las categorías quedaron desactivadas, la red ya no puede clasificar.
            if max_score < 0:
                raise RuntimeError("Capacidad de la red alcanzada: No quedan categorías disponibles.")

            # Índice de la categoría ganadora.
            j_star = scores.index(max_score)

            # 3) Test de vigilancia ART1.
            # T = plantilla top-down de la categoría elegida intersectada con el patrón x.
            T = [self.t[j_star][i] * x[i] for i in range(self.N)]
            norm_T = sum(T)


            # Ratio de coincidencia: ||T|| / ||X||
            match_ratio = norm_T / float(norm_x)

            # Si el patrón es suficientemente compatible con la categoría,
            # se acepta y se actualizan los pesos.
            if match_ratio >= self.rho:
                self._adapt_weights(j_star, T)
                if j_star >= self.num_committed_categories:
                    self.num_committed_categories = j_star + 1
                return j_star, match_ratio
            
            # Si la categoría no cumple el umbral de vigilancia,
            # se desactiva y se intenta otra.
            else:
                active_mask[j_star] = False

    def _adapt_weights(self, j_star: int, T: List[float]):
        """
            Actualiza los pesos de la categoría ganadora después de aceptar un patrón.

            En ART1, cuando una categoría gana y supera el test de vigilancia,
            se adapta su plantilla prototipo para reflejar mejor el nuevo patrón
            de entrada. Esto permite que la red aprenda sin perder estabilidad.

            Parámetros:
                j_star: índice de la categoría ganadora.
                T: vector resultante de la intersección entre la plantilla top-down
                y el patrón de entrada, es decir, T = t_ji * x_i.
        """
        for i in range(self.N):
            self.t[j_star][i] = T[i]

        sum_t = sum(self.t[j_star][k] for k in range(self.N))
        denominator = 0.5 + sum_t

        for i in range(self.N):
            self.b[i][j_star] = self.t[j_star][i] / denominator

    def fit(self, dataset: List[List[int]]) -> List[Dict[str, float]]:
        """Entrena la red procesando secuencialmente cada patrón del dataset."""
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
        """Retorna las plantillas binarias de cada cluster formado."""
        templates = {}
        for j in range(self.num_committed_categories):
            templates[j] = [int(self.t[j][i]) for i in range(self.N)]
        return templates

    def compute_metrics(self, validation_labels: Optional[List[str]] = None) -> Dict[str, float]:
        """Calcula métricas clave de evaluación."""
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
    """Infiere especialidad médica y estudios recomendados según síntomas activos."""
    symptoms_set = set(s.lower() for s in active_symptoms)
    
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
    """Unifica la representación de números entre formatos: '1.0' -> '1'.

    Es el mismo resultado que queda al leer una celda numérica desde XLSX,
    de modo que un CSV y un XLSX del mismo dataset den exactamente lo mismo.
    Solo actúa sobre números canónicos, para no alterar IDs con formato '007'.
    """
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
    """Normaliza un valor de celda (Excel o CSV) a texto plano.

    De esta forma ambos formatos producen exactamente los mismos strings:
    None -> "", True/False -> "1"/"0", 1.0 -> "1" (mismo resultado que lee un CSV).
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    return _normalize_numeric_str(str(value).strip())


def _read_csv_rows(file_path: str) -> List[List[str]]:
    """Lee todas las filas de un archivo .csv como texto plano.

    Ignora comentarios (líneas que empiezan con '#') y filas totalmente vacías
    (típicas al final del archivo).
    """
    # utf-8-sig absorbe el BOM que dejan Excel/Windows al exportar;
    # si el archivo no decodifica como UTF-8, se reintenta con la codificación local (cp1252).
    for encoding in ('utf-8-sig', 'cp1252'):
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
    raise ValueError(f"No se pudo decodificar el archivo '{file_path}'. Se probaron UTF-8 y CP1252.")


def _xlsx_col_index(cell_ref: str) -> int:
    """Convierte la referencia de celda ('B3') en índice de columna base 0 (1)."""
    idx = 0
    letters = 0
    for ch in cell_ref:
        if not ch.isalpha():
            break
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
        letters += 1
    return idx - 1 if letters else -1


def _xlsx_cell_value(cell: Any, shared_strings: List[str], ns: str) -> str:
    """Extrae el valor de una celda .xlsx (lector stdlib) normalizado a texto."""
    cell_type = cell.get("t", "n")
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.iter(f"{ns}t")).strip()
    v = cell.find(f"{ns}v")
    if v is None or v.text is None:
        return ""
    if cell_type == "s":  # índice dentro de las cadenas compartidas
        return _normalize_numeric_str(shared_strings[int(v.text)].strip())
    if cell_type == "b":  # booleano de Excel -> 1 / 0
        return "1" if v.text.strip() == "1" else "0"
    # Valores numéricos: '1.0' se normaliza a '1' para que coincida con lo que lee un CSV.
    return _normalize_numeric_str(v.text.strip())


def _read_xlsx_rows_stdlib(file_path: str) -> List[List[str]]:
    """Lector .xlsx mínimo sin dependencias externas (lee la primera hoja del libro).

    Un archivo .xlsx es un ZIP con XML adentro: se leen las cadenas compartidas
    y la hoja de cálculo con xml.etree. Cubre hojas simples de celdas de texto/número.
    """
    import zipfile
    import xml.etree.ElementTree as ET

    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    ns_rel = "{http://schemas.openxmlformats.org/package/2006/relationships}"
    ns_rid = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

    with zipfile.ZipFile(file_path) as z:
        names = z.namelist()

        # 1) Cadenas compartidas: el texto de las celdas vive acá, indexado.
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in names:
            sst = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in sst.findall(f"{ns}si"):
                shared_strings.append("".join(t.text or "" for t in si.iter(f"{ns}t")))

        # 2) Primera hoja declarada en el libro, resuelta vía sus relaciones.
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

        # 3) Filas y celdas (las celdas ausentes se rellenan con "" para alinear columnas).
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
    """Lee todas las filas de una hoja .xlsx como texto plano.

    Usa openpyxl si está instalado; si no, el lector stdlib (_read_xlsx_rows_stdlib).
    """
    try:
        import openpyxl
    except ImportError:
        openpyxl = None

    if openpyxl is not None:
        wb = openpyxl.load_workbook(file_path, data_only=True)
        rows = [[_cell_to_str(cell) for cell in r] for r in wb.active.iter_rows(values_only=True)]
    else:
        rows = _read_xlsx_rows_stdlib(file_path)

    # Se descartan las filas totalmente vacías (formato residual al final de la hoja).
    return [row for row in rows if any(cell != "" for cell in row)]


def _drop_phantom_columns(rows: List[List[str]]) -> List[List[str]]:
    """Elimina columnas fantasma del índice de pandas ('Unnamed: 0' o header vacío).

    Solo se descartan si están al principio de la fila o si todos sus valores son vacíos,
    para que un CSV y un XLSX exportados desde el mismo DataFrame den exactamente lo mismo.
    """
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
    """Carga dataset Excel (.xlsx) o CSV (.csv)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo '{file_path}' no fue encontrado.")

    patient_ids = []
    
    if file_path.endswith('.xlsx'):
        # openpyxl si está instalado; si no, lector stdlib (zipfile + xml.etree).
        all_rows = _read_xlsx_rows(file_path)
    else:
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            all_rows = [row for row in reader if row and not row[0].startswith('#')]

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
    """Genera la Ficha e Informe Clínico Individual detallado del paciente/cliente."""
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
 NOTA: Este informe es generado mediante Inteligencia Artificial (Red ART1)
 con Aprendizaje No Supervisado. Debe ser validado por un profesional médico.
============================================================================
"""
    return report


def run_interactive_mode(art1: ART1Network, feature_names: List[str], save_txt_path: Optional[str] = None):
    """Ejecuta la evaluación interactiva solicitando nombre y síntomas de un paciente en vivo."""
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
    """Imprime el Manual de Referencia Técnica (--man)."""
    man_text = """================================================================================
   MANUAL DE REFERENCIA TÉCNICA Y DE USUARIO - RED NEURONAL ART1 (CarGross.py)
================================================================================

1. DESCRIPCIÓN DEL SISTEMA
--------------------------------------------------------------------------------
Este módulo implementa el clasificador no supervisado ART1 (Adaptive Resonance
Theory 1) desarrollado por Gail Carpenter y Stephen Grossberg (1987), siguiendo
la formulación matemática de Richard Lippmann (1987) y Lau (1992, Box 3).

La red resuelve el Dilema de Estabilidad-Plasticidad, permitiendo aprender nuevos
patrones sintomáticos (plasticidad) sin destruir las categorías previamente
aprendidas (estabilidad).

2. ARQUITECTURA Y ALGORITMO (8 PASOS DE LAU, 1992)
--------------------------------------------------------------------------------
- Capa F1 (Capa de Comparación): Recibe el vector binario de entrada X de dimensión N.
- Capa F2 (Capa de Reconocimiento): Contiene M nodos de categoría con inhibición lateral.
- Pesos Bottom-Up (b_ij): Inicializados en 1 / (1 + N). Filtran la entrada hacia F2.
- Pesos Top-Down (t_ji): Inicializados en 1. Representan las plantillas prototipo.
- Test de Vigilancia: Se evalúa ||T|| / ||X|| >= p.
  * Si cumple: Se acepta la categoría j* y se adaptan los pesos mediante la regla AND.
  * Si no cumple: Se deshabilita el nodo j* y se busca el siguiente mejor candidato.

3. SINTAXIS Y COMANDOS DE EJECUCIÓN
--------------------------------------------------------------------------------
Sintaxis básica:
  python src/CarGross.py <archivo_dataset> [opciones]
  python src/CarGross.py --input <archivo_dataset> [opciones]

Argumentos Principales:
  pos_input / --input <ruta> Ruta al dataset. Formato de entrada: CSV (.csv);
                      también se admiten archivos Excel (.xlsx).
  -r, --rho <float>          Parámetro de vigilancia entre 0.0 y 1.0 (Defecto: 0.65).
  -o, --output <ruta>        Ruta del archivo CSV para exportar resultados globales.
  --save-txt <ruta>          Ruta para exportar el reporte descriptivo TXT.
  -v, --verbose              Muestra el detalle del procesamiento en consola.
  --interactive              Abre el modo interactivo de consulta para un paciente.
  --patient-id <ID>          Genera la ficha médica individual de un paciente por su ID.
  --shuffle <N>              Ejecuta N corridas con barajado aleatorio para test de estabilidad.
  --seed <S>                 Semilla para reproducibilidad del barajado (Defecto: 42).
  --test                     Ejecuta el Smoke Test automático de autoverificación.
  --man                      Muestra este manual de referencia técnica.

4. EJEMPLOS DE USO PRÁCTICO
--------------------------------------------------------------------------------
a) ejecutar el Smoke Test de verificación:
   python src/CarGross.py --test

b) Procesar Dataset de Entrenamiento:
   python src/CarGross.py data/dataset_entrenamiento_ART1.csv -o results/res_entrenamiento.csv --save-txt results/res_entrenamiento.txt

c) Procesar el Dataset de validacion:
   python src/CarGross.py data/dataset_validacion_ART1.csv -o results/res_validacion.csv --save-txt results/res_validacion.txt

d) procesar con un parámetro de vigilancia más estricto (r=0.80):
   python src/CarGross.py data/dataset_entrenamiento_ART1.csv -r 0.80 -o results/res_entrenamiento_rho080.csv --save-txt results/res_entrenamiento_rho080.txt 

e) Modo Interactivo en Vivo (Solicita Nombre y Síntomas):
   python src/CarGross.py data/dataset_entrenamiento_ART1.csv --interactive --save-txt results/informe_paciente.txt

f) Test de Estabilidad con Barajado Aleatorio:
   python src/CarGross.py data/dataset_entrenamiento_ART1.csv --shuffle 5 --seed 42 -o results/estable.csv --save-txt results/estable.txt

   
5. INSTALACIÓN
--------------------------------------------------------------------------------
Requisitos: Python 3.10 o superior.

  git clone <url-del-repo>
  python -m venv venv
  .\\venv\\Scripts\\Activate.ps1        (en bash: source venv/bin/activate)
  pip install -r requirements.txt

Para verificar que quedó bien:

  python src/CarGross.py --test

Si imprime "TEST PASSED", todo listo.

Nota: los .csv se leen con la librería estándar de Python, no hace falta
instalar nada extra. openpyxl solo se usa para .xlsx y pandas/matplotlib solo
para el notebook.

6. DATASETS, CÓMO CORRERLOS Y SALIDA ESPERADA
--------------------------------------------------------------------------------
Los datasets incluidos están en formato CSV (el formato de entrada del
módulo) y tienen 100 registros y 5 variables cada uno:

  data/dataset_entrenamiento_ART1.csv     100 pacientes, 5 síntomas
  data/dataset_validacion_ART1.csv        100 pacientes, 5 síntomas + etiqueta

Cada CSV tiene: primera columna con el ID del paciente, una columna por
síntoma con 1/0, y opcionalmente una última columna con la especialidad
esperada (solo se usa para calcular la pureza).

Comandos:

  python src/CarGross.py data/dataset_entrenamiento_ART1.csv -o results/res_entrenamiento.csv --save-txt results/res_entrenamiento.txt
  python src/CarGross.py data/dataset_validacion_ART1.csv -o results/res_validacion.csv --save-txt results/res_validacion.txt

En consola se ve: la ruta del archivo y el rho, la cantidad de registros y
síntomas cargados, el resumen con clusters formados (M), matching ratio y
estabilidad (y pureza si el dataset trae etiquetas), y las plantillas de cada
cluster con su especialidad sugerida.

En archivos: el CSV (-o) deja una fila por paciente con ID, cluster, matching
ratio y especialidad; el TXT (--save-txt) deja el reporte completo. Si la
carpeta de salida no existe, se crea sola.

Con rho 0.65 los resultados de referencia son:

  - Entrenamiento: 14 clusters, matching ratio 0.9500.
  - Validación: 12 clusters, matching ratio 0.9600, pureza 69%.

El detalle de las corridas está en docs/informe.md.

7. ALCANCES Y LIMITACIONES
--------------------------------------------------------------------------------
- Es una herramienta académica: no diagnostica pacientes. La especialidad que
  sugiere es una traducción de la plantilla del cluster y hay que validarla
  con un profesional de la salud.
- Solo acepta valores 0 y 1. Si el archivo trae otra cosa, corta con [ERROR].
- ART1 depende del orden de los registros: si se barajan los pacientes pueden
  cambiar los clusters. Es una propiedad conocida del algoritmo (se puede
  medir con --shuffle).
- El rho se elige a mano. Con rho bajo los clusters quedan grandes y
  mezclados; con rho alto quedan muchos y chicos.
- El máximo de clusters está limitado por --max_cat (50 por defecto).
- No hay interfaz gráfica ni guarda nada por sí solo: hay que pasar -o y/o
  --save-txt para generar archivos.

8. PREGUNTAS FRECUENTES
--------------------------------------------------------------------------------
P: ¿Por qué hay que hacer pip install si dice que no hay dependencias?
R: Para los .csv no hace falta nada: se leen con la biblioteca estándar. El
   pip install es solo para .xlsx (openpyxl) y para el notebook (pandas,
   matplotlib).

P: ¿Cómo elijo el rho?
R: No hay un valor exacto. Con 0.65 los clusters quedan razonables y la
   pureza da 69%. Si se sube mucho, se fragmentan los grupos; si se baja
   mucho, se mezclan. Lo mejor es probar varios valores y mirar la cantidad
   de clusters y la pureza.

P: ¿Por qué cambian los clusters si barajo los datos?
R: ART1 aprende registro por registro, y el orden en que llegan los pacientes
   define qué plantillas se crean. Es propio del algoritmo, y por eso existe
   el modo --shuffle para medirlo.

P: ¿Puedo usar mis propios datos?
R: Sí, con un CSV con ID, variables en 0/1 y (opcional) la etiqueta al final.
   Si algún valor no es 0 o 1, la corrida corta con [ERROR].

P: ¿Esto diagnostica pacientes?
R: No, solo agrupa. Es un trabajo de la materia Redes Neuronales y la salida
   tiene que ser revisada por un profesional médico.

================================================================================
"""
    print(man_text)


def run_smoke_test():
    """Ejecuta autotest de verificación."""
    print("[TEST] Iniciando Smoke Test de la Red ART1...")
    sample_patterns = [
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [0, 1, 1, 0, 0, 0]
    ]
    art1 = ART1Network(num_inputs=6, max_categories=10, rho=0.65)
    res = art1.fit(sample_patterns)
    print(f"[TEST] Éxito: {art1.num_committed_categories} clusters creados.")
    print("TEST PASSED")


def main():
    """Punto de entrada con manejo de errores de corrida: todo fallo
    previsible se informa con [ERROR] y una pista hacia --man / --help."""
    try:
        _run()
    except (FileNotFoundError, FileExistsError, ValueError, ImportError, OSError) as exc:
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
        run_smoke_test()
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

    # 1. Modo Interactivo
    if args.interactive:
        run_interactive_mode(art1, feature_names, save_txt_path=args.save_txt)
        return

    # 2. Modo Ficha de Paciente por ID
    if args.patient_id:
        target_idx = None
        for idx, pid in enumerate(patient_ids):
            if str(pid).strip().lower() == str(args.patient_id).strip().lower():
                target_idx = idx
                break
        if target_idx is None:
            # Intento por índice numérico de paciente (ej. "1" -> índice 0)
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

    # 3. Reporte General Normal
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

    # 4. Test de Estabilidad (Shuffle) si corresponde
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

    # Imprimir resumen en consola
    for block in txt_lines[1:]:
        print(block)

    # Exportar CSV de asignación general
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

    # Exportar Reporte TXT general
    if args.save_txt:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_txt)), exist_ok=True)
        with open(args.save_txt, mode='w', encoding='utf-8') as f_txt:
            f_txt.write("\n".join(txt_lines))
            f_txt.write(f"\n================================================================\n Reporte generado automáticamente: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n================================================================\n")
        print(f"[ÉXITO] Reporte TXT completo guardado en '{args.save_txt}'.")


if __name__ == "__main__":
    main()
