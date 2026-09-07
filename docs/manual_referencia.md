# 09 · Manual de Referencia

> **Estado**: completo y alineado con la consigna oficial.
> **Actividad obligatoria cubierta**: #4 del Trabajo Final Integrador (TFI) — *"Manual de referencia"* (`_legacy/CarGross_TP/consignas_TP.md`, líneas 50–56).
> **Audiencia primaria**: estudiantes de la materia *Redes Neuronales*, docentes, y lectores técnicos que deseen reproducir las corridas o extender la implementación.

---

## 1. Introducción

### 1.1. Contexto del TFI

Este Trabajo Final Integrador corresponde a la materia **Redes Neuronales** de la carrera de Ingeniería en la **UADER — Facultad de Ingeniería**, en el marco del **IDTI Lab**. La consigna oficial, conservada en `_legacy/CarGross_TP/consignas_TP.md`, exige la implementación de la red neuronal **ART1** (Adaptive Resonance Theory 1) de Carpenter y Grossberg, en el lenguaje Python, con entrega de datasets de prueba, informe de corridas y un **manual de referencia** para el módulo `CarGross.py`. Este documento cubre específicamente esa última pieza.

El TFI en su conjunto se documenta en `docs/`:

- `docs/01_marco_teorico.md` — qué es ART1 y por qué se eligió sobre alternativas.
- `docs/02_problema_y_alcance.md` — qué hace y qué NO hace el sistema.
- `docs/03_dataset_y_preprocesamiento.md` — datasets canónicos y reglas de binarización.
- `docs/04_algoritmo.md` — transcripción y análisis del algoritmo Box 3 (Lau 1992).
- `docs/05_corridas_y_evaluacion.md` — diseño experimental y métricas no supervisadas.
- `docs/06_limitaciones_y_etica.md` — limitaciones técnicas, consideraciones éticas y disclaimer formal.
- `docs/informe_corridas.md` — informe narrativo de las 30 corridas realizadas.
- `docs/manual_referencia.md` — **este documento**.

### 1.2. Qué cubre este manual

Este manual de referencia está pensado como el documento que un tercero lee **antes** de ejecutar el módulo por primera vez. Cubre:

1. **Alcances y limitaciones** del sistema (qué hace y qué no hace).
2. **Proceso de instalación** completo, paso por paso, multiplataforma.
3. **Modo de correr al menos un test demo**, con la salida esperada.
4. **Uso general del CLI**, incluyendo todos los flags de `argparse` y el manual extendido `--man`.
5. **Formatos de archivos** de entrada (CSV, `metadata.csv`) y salida (CSV, TXT).
6. **Manejo de errores**, con tabla de excepciones custom y códigos de salida.
7. **FAQ** con las preguntas más frecuentes sobre el comportamiento de ART1.
8. **Referencias** cruzadas al resto del proyecto.

### 1.3. Referencia a la consigna oficial

La actividad #4 de la consigna oficial (`_legacy/CarGross_TP/consignas_TP.md`) exige explícitamente que el manual de referencia contenga:

- Alcances y limitaciones.
- Proceso de instalación.
- Modo de correr al menos un test demo.
- FAQ con al menos 3 preguntas.

Las secciones §2, §3, §4 y §8 de este documento cubren esos cuatro puntos en ese orden, agregando además — por completitud y para que el lector no necesite saltar a otros documentos — el detalle de flags, formatos y errores que un usuario realmente necesita la primera vez que ejecuta el módulo.

---

## 2. Alcances y limitaciones

### 2.1. Alcances (qué hace el sistema)

El sistema, implementado íntegramente en `src/CarGross.py`, cumple los siguientes puntos:

- **Implementa ART1** según el algoritmo del Box 3 de Lau (1992, pp. 12–14). La transcripción comentada del algoritmo vive en `docs/04_algoritmo.md` y la fuente primaria en `_legacy/CarGross_TP/lau_contenido.md`.
- **Está escrito en Python puro** (sólo *standard library*: `argparse`, `csv`, `math`, `random`, `sys`, `pathlib`). El núcleo de ART1 **no depende** de `numpy`, `pandas`, `matplotlib`, `scikit-fuzzy` ni de ningún paquete externo. Las dependencias listadas en `requirements.txt` se utilizan únicamente para análisis posterior de los CSV/TXT de salida, no para correr el módulo.
- **Lee CSV de entrada** con features numéricos continuos y los **binariza automáticamente** según las reglas declaradas en `data/metadata.csv` (un umbral por fila, con regla `gte`, `lte`, `gt`, `lt`).
- **Produce dos archivos de salida**: un CSV con la asignación de cluster por fila (`id`, `cluster`, `match_score`) y un TXT con un reporte legible por humanos (clusters, tamaños, exemplares, scores, IDs).
- **Soporta el barrido de vigilancia** $\rho \in [0.0, 1.0]$ vía el flag `--vigilance` / `-r`, lo que permite estudiar el compromiso plasticidad–estabilidad.
- **Soporta análisis de estabilidad** frente al orden de presentación mediante el flag `--shuffle N`, que ejecuta N corridas con orden aleatorio y reporta el acuerdo *pairwise* medio contra la corrida base.
- **Trabaja con los dos datasets provistos** (`data/dataset1_pacientes.csv` con 55 pacientes y `data/dataset2_sensores.csv` con 55 lecturas de sensores), tal como exige la actividad #5 de la consigna.
- **Se autodocumenta** mediante el flag `--man`, que imprime a `stdout` el manual extendido con descripción del algoritmo, semántica de cada flag, ejemplos y referencias bibliográficas.
- **Maneja errores** con mensajes en español, excepciones custom por categoría (`FileNotFoundCarGrossError`, `MetadataError`, `BinarizationError`, `VigilanceError`, `DatasetError`) y códigos de salida diferenciados (1 para errores del módulo, 2 para excepciones inesperadas; ver §7).

### 2.2. Limitaciones (qué NO hace o hace con reservas)

Las siguientes limitaciones son **propias del modelo** o **deliberadas** en el alcance del TFI. No son bugs sino decisiones de diseño documentadas:

- **Sólo entradas binarias.** ART1, según el paper Lau (1992), opera sobre vectores con componentes en $\{0, 1\}$. Los valores continuos del CSV de entrada se binarizan antes del `fit` usando los umbrales del metadata. Esto descarta matices cuantitativos (dos pacientes con colesterol 239 y 241 mg/dL caen en categorías distintas; dos con 90 y 109, en la misma). Ver `docs/06_limitaciones_y_etica.md` §1.
- **Sensibilidad al orden de presentación.** ART1 procesa las entradas en serie, una a una, en el orden en que aparecen. Sin barajado ni mecanismos de estabilización adicionales, dos corridas del mismo dataset con el mismo $\rho$ pueden producir particiones distintas si el orden cambia. Esto es comportamiento esperado del algoritmo, no un bug. Se mitiga parcialmente con el flag `--shuffle N` y se reporta cuantitativamente en `results/resumen_corridas.md`.
- **Sin persistencia de exemplares entrenados entre corridas.** Cada invocación del módulo entrena la red desde cero sobre el CSV provisto. No hay un formato de "modelo guardado" (`.pkl`, `.json`): el sistema se concibe como *pipeline* (CSV → binarización → fit → reporte), no como librería con `pickle`.
- **Sin interfaz gráfica.** La consigna oficial (actividad #3, punto "No se exige interfaz gráfica") fija el alcance como CLI. El módulo no provee GUI, ni servidor HTTP, ni integración con notebooks Jupyter más allá de la importación estándar de la clase `ART1`.
- **Sin paralelismo ni aceleración por GPU.** La implementación es secuencial y single-threaded. No usa `numpy`, `numba`, `cython` ni `torch`. Para N grande (> 10⁴ filas) el runtime crece linealmente con `len(X) × max_clusters` por la estructura interna de las matrices de pesos.
- **Sin comparación con otros algoritmos de clustering.** La consigna exige ART1; el módulo no incluye K-means, DBSCAN, clustering jerárquico ni Spectral Clustering. La comparación cuantitativa con algoritmos alternativos queda fuera del alcance (ver FAQ §8.6).
- **Sin validación supervisada.** ART1 es no supervisada. El módulo no calcula accuracy, F1, precision ni recall porque no hay etiquetas verdaderas en el dataset. Las métricas que produce (número de clusters, score medio, acuerdo entre barajados) son no supervisadas, tal como exige la actividad #6 y se discute en `docs/05_corridas_y_evaluacion.md`.
- **Sólo apto para datasets pequeños.** Con $N \leq 10^4$ filas y `max_clusters` por defecto (1000), el módulo funciona en tiempos razonables. Más allá, hay que ajustar `--max-clusters` con cuidado y considerar reescribir el núcleo en `numpy` (lo cual está fuera del alcance del TFI).
- **Dataset `dataset2_sensores.csv` estructuralmente rígido.** Este dataset tiene sólo 6 vectores únicos en el espacio binario de 8 dimensiones; ART1 produce 5 clusters efectivos (un vector es absorbido por un cluster existente durante el `fit`). El barrido de $\rho$ no discrimina nuevos clusters para este dataset: el número de particiones, los tamaños y el score medio permanecen prácticamente constantes en $\rho = 0.50, 0.65, 0.80$. Ver `docs/informe_corridas.md` §4.1 para el análisis completo.
- **Clusters con exemplar vacío (`[0 0 0 0 ...]`).** Cuando una fila del CSV binariza a todos-ceros (todos los features bajo sus umbrales), ART1 crea un cluster con ese vector. Estos clusters suelen tener un solo miembro y representan el caso "ningún factor de riesgo activo" o "ningún indicador fuera de rango". Se reportan **por separado** en el TXT (línea "Se omiten N clusters con exemplar vacío") para no contaminar la lectura principal. Ver `docs/06_limitaciones_y_etica.md` §1 y `docs/informe_corridas.md` §6.1.

### 2.3. Lo que el sistema NO afirma

Para evitar malas interpretaciones, se reproducen acá — adaptadas de `docs/02_problema_y_alcance.md` §2 — las afirmaciones que el sistema **no** hace:

- No diagnostica condiciones médicas.
- No prescribe tratamientos ni medicaciones.
- No decide a qué especialista debe derivarse un paciente.
- No reemplaza el juicio clínico de un profesional matriculado.
- No opera en tiempo real ni se integra con historias clínicas electrónicas (EHR).
- No tiene aprobación regulatoria de ANMAT, FDA ni EMA.

El disclaimer formal completo se reproduce en `docs/06_limitaciones_y_etica.md` §3 y debe acompañar cualquier entrega derivada de este TFI.

---

## 3. Proceso de instalación

### 3.1. Requisitos

Antes de instalar, asegurarse de contar con:

| Requisito | Detalle |
|-----------|---------|
| Python | 3.10 o superior (la implementación usa `match`/`case` *internamente* y *type hints* modernos; el argparse y la API csv funcionan desde 3.6 pero se recomienda 3.10+) |
| Sistema operativo | Windows 10/11, macOS 11+, o Linux (cualquier distribución con Python 3.10+) |
| Espacio en disco | ~10 MB para el proyecto completo (código + datasets + docs) |
| Conexión a internet | Sólo si se decide instalar las dependencias opcionales (`pip install -r requirements.txt`) |
| Permisos | Ninguno especial: el módulo no requiere root ni escritura fuera de `results/` |

**Dependencias del núcleo ART1**: ninguna. `src/CarGross.py` usa exclusivamente módulos de la *standard library* de Python (`argparse`, `csv`, `math`, `random`, `sys`, `pathlib`). Esto cumple la actividad #3 de la consigna ("debe ser sencillo de instalar", "funcionar sin errores") sin requerir un *virtual environment* obligatorio.

**Dependencias opcionales** (listadas en `requirements.txt`):

- `numpy` — para análisis numérico de los CSV de salida.
- `pandas` — para abrir `results/resultado.csv` como `DataFrame` y aplicar `groupby('cluster')` u operaciones similares.
- `matplotlib` — para graficar la distribución de tamaños de cluster o el score por cluster.
- `scikit-fuzzy` — para cálculos de membresía fuzzy sobre los mismos vectores binarizados (no usado por `CarGross.py`, sí por extensiones experimentales).

Estas dependencias **no se requieren** para que el módulo funcione. Sólo son necesarias si el lector desea analizar las salidas con `pandas`/`matplotlib`.

### 3.2. Pasos de instalación

#### Paso 1 — Obtener el repositorio

Clonar el repositorio desde el control de versiones del proyecto, o copiar la carpeta del proyecto al equipo local:

```bash
git clone <URL-del-repo> ia2026
cd ia2026
```

Si se descarga como archivo ZIP, descomprimirlo y ubicarse en la carpeta descomprimida.

#### Paso 2 — (Opcional) Crear un entorno virtual

Aunque el núcleo ART1 no requiere dependencias externas, se recomienda crear un *venv* para mantener aisladas las dependencias opcionales de análisis:

```bash
python -m venv venv
```

Activación del entorno:

```powershell
# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

```bash
# Windows (cmd)
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

#### Paso 3 — (Opcional) Instalar dependencias de análisis

Si se quieren usar `pandas`, `matplotlib`, etc., para inspeccionar los CSV de salida:

```bash
pip install -r requirements.txt
```

Si no se instalan, el módulo sigue funcionando al 100%; sólo los scripts de análisis externo fallarán al importar.

#### Paso 4 — Verificar la instalación

Ejecutar el smoke test incluido en el módulo (ver §4 para la salida esperada):

```powershell
# Windows (PowerShell o cmd)
cd C:\ruta\al\proyecto\ia2026
python src\CarGross.py --test
```

```bash
# macOS / Linux
cd /ruta/al/proyecto/ia2026
python src/CarGross.py --test
```

Si la salida es `TEST PASSED` (exit code 0), la instalación está completa y el módulo funciona. Si hay un mensaje de error, ver §7.

---

## 4. Test demo

### 4.1. Smoke test incluido (`--test`)

El módulo incluye un smoke test que ejercita el camino crítico: cargar `data/dataset1_pacientes.csv`, binarizarlo con `data/metadata.csv`, entrenar ART1 con $\rho = 0.6$, predecir sobre cada fila y validar que cada una reciba un cluster válido.

**Comando:**

```powershell
cd C:\Users\camil\Proyectos\ia2026
python src\CarGross.py --test
```

**Salida esperada:**

```
TEST PASSED
```

**Exit code:** 0 (en PowerShell: `$LASTEXITCODE -eq 0`).

**Qué verifica internamente:**

1. Que el CSV de entrada exista y se pueda abrir.
2. Que el metadata exista y tenga las columnas requeridas (`dataset`, `feature`, `threshold`, `rule`).
3. Que la binarización produzca una matriz no vacía.
4. Que ART1 entrene al menos 1 cluster (`net.n_clusters >= 1`).
5. Que cada fila del CSV original reciba un cluster válido en `predict` (`j >= 0` y `j < net.n_clusters`).

Si cualquiera de estas verificaciones falla, el smoke test imprime un mensaje específico (`TEST FAILED: <motivo>`) y devuelve exit code no-cero. Ver §7 para los mensajes típicos.

### 4.2. Demo adicional con datos reales

Una vez que el smoke test pasa, se puede correr una demo completa sobre los datasets provistos. **Esta demo es la que produce los archivos `results/resultado.csv` y `results/resultado.txt`** con la asignación de clusters.

**Comando para el dataset de pacientes:**

```powershell
cd C:\Users\camil\Proyectos\ia2026
python src\CarGross.py data\dataset1_pacientes.csv -r 0.6 --output results\demo_pacientes.csv --save-txt results\demo_pacientes.txt --verbose
```

**Qué produce cada flag:**

| Flag | Efecto |
|------|--------|
| `data\dataset1_pacientes.csv` | Argumento posicional: CSV de entrada. 55 pacientes, 7 features continuos. |
| `-r 0.6` | Vigilancia $\rho = 0.6$. Equilibrio entre agrupar y separar. |
| `--output results\demo_pacientes.csv` | Sobrescribe el CSV de salida por defecto con un nombre explícito. |
| `--save-txt results\demo_pacientes.txt` | Sobrescribe el TXT de salida por defecto con un nombre explícito. |
| `--verbose` | Imprime logs adicionales a `stdout` durante la corrida. |

**Salida esperada por stdout (modo verbose):**

```
[INFO] Cargadas 55 filas, 7 features binarias.
[INFO] Metadata: data/metadata.csv
[INFO] Clusters formados: 26
[INFO] CSV: results/demo_pacientes.csv
[INFO] TXT: results/demo_pacientes.txt
```

**Salida esperada en `results/demo_pacientes.csv`** (tres columnas: `id`, `cluster`, `match_score`):

```
id,cluster,match_score
1,0,0.300
2,0,0.300
3,6,0.800
...
```

**Salida esperada en `results/demo_pacientes.txt`** (encabezado + bloque por cluster):

```
Reporte ART1 (Carpenter/Grossberg)
============================================================
Dataset:          data\dataset1_pacientes.csv
Metadata:         data/metadata.csv
Vigilancia (rho): 0.6
Max clusters:     1000
N (features bin): 7
Filas totales:    55
Cantidad de clusters con miembros: 3 (de 26 totales creados)
Score medio:      0.517
============================================================

(Se omiten 23 clusters con exemplar vacio; ver doc 06_limitaciones_y_etica.md para contexto.)

Cluster 0
  Tamano:    35
  Exemplar:  [1 0 0 0 0 0 0]
  Score med: 0.343
  IDs:       1, 2, 3, ...

Cluster 2
  Tamano:    4
  Exemplar:  [1 0 0 1 0 0 0]
  Score med: 0.917
  IDs:       ...

Cluster 6
  Tamano:    16
  Exemplar:  [1 1 1 1 0 1 0]
  Score med: 0.798
  IDs:       ...

Referencia algoritmica: Box 3, Lau (1992) pp. 12-14.
Ver _legacy/CarGross_TP/lau_contenido.md para la transcripcion completa.
```

**Cómo interpretar el TXT:**

- **Encabezado**: confirma el dataset, el metadata, la vigilancia usada, el número de features binarios y el total de filas.
- **"Cantidad de clusters con miembros"**: es la cifra interpretable. En este ejemplo, 3 clusters contienen todas las filas. Los otros 23 están "vacíos" (exemplar todo en 0) y se reportan por separado.
- **Por cada cluster**: el tamaño (cuántos pacientes asignados), el exemplar (vector binario que define al cluster — la *firma* del cluster), el score medio (qué tan compactos son los miembros: 1.0 significa bit-exactos), y la lista de IDs asignados.
- **Exemplar activo**: si la posición $i$ del vector binario es 1, esa feature está "activa" en el cluster. Por ejemplo, `[1 1 1 1 0 1 0]` significa "edad + presion_sistolica + presion_diastolica + colesterol + imc", que corresponde al perfil cardiometabólico clásico (sin glucosa ni taquicardia).
- **Score medio**: cuanto más cercano a 1.0, más homogéneo es el cluster (todos los miembros coinciden casi exactamente con el exemplar). Valores cercanos a 0.3 o 0.4 indican clusters más heterogéneos.

### 4.3. Demo equivalente sobre el dataset de sensores

```powershell
python src\CarGross.py data\dataset2_sensores.csv -r 0.65 --output results\demo_sensores.csv --save-txt results\demo_sensores.txt --verbose
```

Produce 5 clusters con miembros (sobre 5 totales creados) y score medio ≈ 0.994 — los vectores binarizados son tan discretos que las asignaciones son casi bit-exactas. La interpretación operativa de cada cluster está en `docs/informe_corridas.md` §4.2.

---

## 5. Uso general del CLI

### 5.1. Sinopsis

```
python src/CarGross.py <csv_file> [opciones]
```

Si se omite `<csv_file>` y no se pasan `--man` ni `--test`, el módulo imprime el `usage` de `argparse` y sale con código 2 (error de argumentos).

### 5.2. Tabla completa de flags

| Flag | Short | Tipo | Default | Descripción |
|------|-------|------|---------|-------------|
| `csv_file` | (posicional) | Path | requerido | CSV de entrada con features numéricos continuos. |
| `--vigilance` | `-r` | float | 0.5 | Vigilancia $\rho \in [0.0, 1.0]$. Cerca de 1 = coincidencia estricta (más clusters). Cerca de 0 = coincidencia laxa (menos clusters). |
| `--max-clusters` | `-m` | int | 1000 | Máximo de clusters a crear. Si ART1 excede este número durante el fit, se lanza `DatasetError`. |
| `--metadata` | — | Path | `data/metadata.csv` | Ruta al CSV de metadata con reglas de binarización. |
| `--output` | `-o` | Path | `results/resultado.csv` | CSV de salida con tres columnas: `id`, `cluster`, `match_score`. |
| `--save-txt` | — | Path | `results/resultado.txt` | TXT de salida con el reporte legible por humanos. |
| `--shuffle` | — | int | 0 | Cantidad de barajados (estabilidad). 0 = desactivado. N > 0 ejecuta N corridas con orden aleatorio y reporta el acuerdo pairwise medio. |
| `--seed` | — | int | 42 | Semilla aleatoria para el generador interno (usado por `--shuffle`). |
| `--verbose` | `-v` | flag | False | Activa logging detallado paso a paso a `stdout`. |
| `--man` | — | flag | False | Imprime el manual extendido a `stdout` y sale con código 0. |
| `--test` | — | flag | False | Ejecuta el smoke test (`data/dataset1_pacientes.csv` con $\rho = 0.6$) y sale con código 0. |

### 5.3. El flag `--man`

El flag `--man` imprime a `stdout` el manual extendido del módulo, redactado por el propio `src/CarGross.py` (constante `_MANUAL`). Incluye:

- **SINOPSIS** y **DESCRIPCION** del módulo.
- Los **ocho pasos** del algoritmo Box 3 (Lau 1992) con la notación exacta del paper.
- **ARGUMENTOS**: descripción detallada de cada flag.
- **EJEMPLOS** de invocación para los dos datasets.
- **FORMATO DE SALIDA**: especificación del CSV y del TXT.
- **LIMITACIONES** propias del algoritmo.
- **REFERENCIAS** bibliográficas (Lau 1992, Carpenter & Grossberg 1987, Lippmann 1987).

Es útil cuando el lector está en una terminal sin acceso a este manual de referencia: `python src/CarGross.py --man` da toda la información operativa sin necesidad de abrir un browser o un editor.

### 5.4. El flag `--shuffle`

El flag `--shuffle N` ejecuta el `fit` N veces con órdenes aleatorios del dataset, usando `random.Random(seed)` con la semilla indicada por `--seed`. Para cada corrida barajada:

1. Se mide el número de clusters resultantes.
2. Se calcula la asignación predicha fila por fila.
3. Se compara contra la asignación de la corrida base (run 0) y se acumula el *acuerdo pairwise*.

Al final se imprime un bloque de **Reporte de estabilidad** con:

```
Reporte de estabilidad (--shuffle)
----------------------------------------
Ejecuciones:       N
Clusters por run:  [k1, k2, ..., kN]
Media # clusters:  K.Media
Acuerdo medio:     A.FFF  (vs. run 0)
```

Esta métrica (acuerdo pairwise vs run 0) se eligió por simplicidad sobre el ARI formal, suficiente para $N = 55$. La decisión está documentada como D1.4 en `docs/07_iteraciones.md`.

### 5.5. Exit codes

| Código | Significado |
|--------|-------------|
| 0 | Corrida exitosa (incluye `--man` y `--test`). |
| 1 | Error del módulo: `CarGrossError` o subclase (ver §7). |
| 2 | Error inesperado: cualquier `Exception` que no herede de `CarGrossError`. En este caso, con `--verbose` activo se imprime el traceback completo; sin verbose, sólo el mensaje genérico. |

---

## 6. Formato de archivos

### 6.1. Formato del CSV de entrada

- **Header row**: requerido. Primera fila del archivo con los nombres de columna.
- **Columna de ID**: el módulo detecta automáticamente una columna llamada `id` o `sensor_id` (configurable en `src/CarGross.py` constante `_ID_COLUMNS = {"id", "sensor_id"}`) y la usa como identificador de fila en la salida. Si no hay columna de ID, se lanza `DatasetError`.
- **Columnas de features**: el resto de las columnas, una por feature numérico continuo. Cada feature debe tener una entrada en `data/metadata.csv` con la regla de binarización correspondiente. Si falta una entrada, se lanza `BinarizationError`.
- **Filas de datos**: mínimo 1 fila. Si el CSV tiene sólo header, se lanza `DatasetError("CSV sin filas de datos")`.
- **Codificación**: UTF-8 (sin BOM). El módulo abre los archivos con `encoding="utf-8"`.
- **Separador**: coma (`,`). El módulo usa el `csv.DictReader` de la standard library, que acepta coma por defecto.
- **Valores faltantes o no numéricos**: lanzan `BinarizationError` con el índice de fila y el nombre de columna.

**Ejemplo mínimo:**

```csv
id,edad,presion_sistolica,colesterol
1,45,130,220
2,58,145,260
3,27,112,165
```

### 6.2. Formato del `metadata.csv`

El `metadata.csv` declara, una fila por feature binaria, cómo se binariza cada columna del CSV de entrada.

**Header requerido** (orden libre, pero todas presentes):

```csv
dataset,feature,threshold,rule,unit,justification
```

**Significado de cada columna:**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `dataset` | string | Nombre del CSV de entrada **sin extensión**. Debe coincidir con el `stem` del archivo pasado como `csv_file`. Por ejemplo, `dataset1_pacientes` para `data/dataset1_pacientes.csv`. |
| `feature` | string | Nombre exacto de la columna en el CSV de entrada. |
| `threshold` | float | Umbral numérico para la binarización. Se interpreta según la `rule`. |
| `rule` | string ∈ {`gte`, `lte`, `gt`, `lt`} | Operador de comparación. `gte`: $\geq$. `lte`: $\leq$. `gt`: $>$. `lt`: $<$. |
| `unit` | string | Unidad de medida del feature (sólo documental; no se usa para binarizar). Ejemplos: `mmHg`, `mg/dL`, `kg/m²`, `V`, `A`. |
| `justification` | string | Justificación clínica/operativa del umbral elegido, con referencia a guía o norma cuando aplique. ~80 caracteres. |

**Reglas adicionales:**

- Una feature puede aparecer **más de una vez** si se desea modelar más de una condición sobre la misma columna. Caso de uso típico: el voltaje del dataset sensores se desdobla en `voltaje` con `rule=lt threshold=219` y `voltaje` con `rule=gt threshold=221`, para capturar **fuera de rango nominal 220V ±1V** sin firmar el umbral.
- Las filas del metadata se procesan **en el orden en que aparecen**. Si una feature tiene múltiples reglas, se respeta ese orden y se concatenan los bits resultantes.
- Filas con `dataset` distinto al del CSV de entrada se **ignoran** silenciosamente (esto permite tener un único `metadata.csv` con reglas para varios datasets en el mismo archivo, como ocurre en este proyecto).
- `threshold` debe ser parseable como `float`. Si no, se lanza `MetadataError`.

**Ejemplo (extracto de `data/metadata.csv` real):**

```csv
dataset,feature,threshold,rule,unit,justification
dataset1_pacientes,edad,40,gte,años,Mediana adultez, factor de riesgo cardiovascular
dataset1_pacientes,presion_sistolica,140,gte,mmHg,Hipertensión grado 1 (guías AHA)
dataset1_pacientes,colesterol,240,gte,mg/dL,Colesterol alto (guías ATP III)
dataset2_sensores,voltaje,219,lt,V,Fuera de rango nominal 220V ±1V (umbral bajo: V < 219)
dataset2_sensores,voltaje,221,gt,V,Fuera de rango nominal 220V ±1V (umbral alto: V > 221)
```

### 6.3. Formato del CSV de salida

Archivo generado por el módulo (default `results/resultado.csv`, configurable con `--output`).

**Header:**

```
id,cluster,match_score
```

**Una fila por cada fila del CSV de entrada**, en el mismo orden:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | string | Valor de la columna `id` o `sensor_id` del CSV original. Se preserva tal cual. |
| `cluster` | int | ID del cluster asignado (0-indexed). Rango: 0 a `n_clusters - 1`. |
| `match_score` | float | Score de coincidencia entre la fila y el exemplar del cluster. Se calcula como $\|\mathbf{T} \odot \mathbf{X}\| / \|\mathbf{X}\|$ (norma L1 de la AND entre el peso top-down y la entrada, dividida por la norma L1 de la entrada). Rango: 0.000 a 1.000, con 3 decimales. |

**Codificación y separador:** UTF-8, coma.

### 6.4. Formato del TXT de salida

Archivo generado por el módulo (default `results/resultado.txt`, configurable con `--save-txt`).

**Estructura:**

```
Reporte ART1 (Carpenter/Grossberg)
============================================================
Dataset:          <ruta del CSV de entrada>
Metadata:         <ruta del metadata>
Vigilancia (rho): <float>
Max clusters:     <int>
N (features bin): <int>          ← cantidad de columnas binarias
Filas totales:    <int>           ← cantidad de filas del CSV
Cantidad de clusters con miembros: K1 (de K2 totales creados)
Score medio:      <float, 3 decimales>
============================================================

(Si hay clusters vacíos:)
(Se omiten N clusters con exemplar vacio; ver doc 06_limitaciones_y_etica.md para contexto.)

(Para cada cluster con miembros:)
Cluster J
  Tamano:    <int>               ← cantidad de filas asignadas
  Exemplar:  [b1 b2 b3 ... bN]   ← vector binario (N = N features bin)
  Score med: <float, 3 decimales>
  IDs:       id1, id2, id3, ...  ← lista de IDs asignados, separados por ", "

(Pie:)
Referencia algoritmica: Box 3, Lau (1992) pp. 12-14.
Ver _legacy/CarGross_TP/lau_contenido.md para la transcripcion completa.
```

**Notas operativas:**

- Los clusters se reportan **en orden de ID** (0, 1, 2, ...).
- Los clusters con exemplar vacío (vector todo en 0) **no se listan** individualmente; se mencionan en una sola línea ("Se omiten N clusters con exemplar vacio").
- El score medio es la **media aritmética** de los `match_score` de las filas asignadas a cada cluster.
- Los IDs se preservan tal cual vienen del CSV original (pueden ser enteros como `1`, `2`, strings como `S001`, etc.).

---

## 7. Manejo de errores

El módulo define una jerarquía de excepciones custom que permiten al usuario distinguir **por categoría** el tipo de error y, cuando corresponde, actuar en consecuencia.

### 7.1. Jerarquía

```
CarGrossError                   ← excepción base
├── FileNotFoundCarGrossError   ← archivo no encontrado
├── MetadataError               ← metadata mal formado o inconsistente
├── BinarizationError           ← feature sin regla, rule inválida, valor no numérico
├── VigilanceError              ← ρ fuera de [0.0, 1.0]
└── DatasetError                ← dataset vacío, sin columna ID, max_clusters agotado
```

Cualquier otra `Exception` no listada arriba se trata como error inesperado y se reporta con exit code 2.

### 7.2. Tabla de errores y códigos de salida

| Excepción | Código | Causa típica | Acción sugerida |
|-----------|--------|--------------|------------------|
| `FileNotFoundCarGrossError` | 1 | El CSV de entrada o el `metadata.csv` no existen en la ruta indicada. | Verificar la ruta. Si es relativa, recordar que se evalúa respecto al *working directory* actual. |
| `MetadataError` | 1 | El `metadata.csv` falta columnas requeridas (`dataset`, `feature`, `threshold`, `rule`); un `threshold` no parsea como float; o no hay filas para el dataset actual. | Revisar `data/metadata.csv`. Confirmar que la columna `dataset` coincide con el `stem` del CSV de entrada. |
| `BinarizationError` | 1 | Una columna del CSV de entrada no tiene entrada en el metadata; o una `rule` es desconocida; o un valor no parsea como número. | Revisar que todas las features del CSV (excepto la columna de ID) tengan una fila en el metadata con la `rule` correcta. |
| `VigilanceError` | 1 | El valor pasado a `--vigilance` no está en $[0.0, 1.0]$. | Verificar el rango. Típicamente se trabaja con $\rho \in [0.4, 0.9]$. |
| `DatasetError` | 1 | El CSV está vacío, sin columna de ID (`id` o `sensor_id`); una fila no tiene la longitud de features esperada; se alcanzó `max_clusters` durante el `fit`. | Revisar el CSV. Si se alcanzó `max_clusters`, subir el valor con `--max-clusters` o bajar $\rho$. |
| Otros `Exception` | 2 | Cualquier `Exception` no listada arriba. | Reportar como issue adjuntando el comando ejecutado y el traceback (activar `--verbose`). |

### 7.3. Mensaje típico al usuario

Cuando se produce un error, el módulo imprime a `stderr`:

```
[ERROR] <mensaje específico de la excepción>
Usa --man para ver el manual completo.
```

**Ejemplo concreto** (corrida con un CSV inexistente):

```powershell
PS> python src\CarGross.py data\mi_dataset_inexistente.csv -r 0.6
[ERROR] No existe CSV de entrada: data\mi_dataset_inexistente.csv
Usa --man para ver el manual completo.
```

Exit code: 1. El módulo no continúa la ejecución ni sobrescribe los archivos de salida.

**Con `--verbose`**, en errores inesperados (exit code 2), se imprime además el traceback completo en `stderr`:

```
[ERROR] Inesperado: <tipo de excepción>: <mensaje>
<traceback completo, una línea por frame>
```

Para los errores del módulo (exit code 1), `--verbose` no agrega información adicional: el mensaje de la excepción ya es suficiente.

### 7.4. Recuperación recomendada

Ante un error del módulo (exit code 1):

1. Leer el mensaje específico (incluye fila, columna o path cuando aplica).
2. Si el error es de archivos: verificar que las rutas existen y los permisos son correctos.
3. Si el error es de metadata o binarización: abrir `data/metadata.csv` y compararlo con el header del CSV de entrada.
4. Si el error es de vigilancia: revisar que `--vigilance` esté en $[0.0, 1.0]$.
5. Si el error es de `max_clusters`: subir el valor con `--max-clusters 2000` o bajar $\rho$.
6. Si nada de lo anterior aplica: correr con `--verbose` y revisar el traceback.

---

## 8. FAQ (preguntas frecuentes)

**Q1. ¿Por qué mi corrida produce muy pocos clusters?**
Probablemente el parámetro de vigilancia $\rho$ es muy alto. ART1 con $\rho$ cercano a 1.0 exige coincidencia casi exacta entre la entrada y algún exemplar existente, lo que en datasets heterogéneos lleva a crear un cluster nuevo para casi cada fila. En el dataset de pacientes (`dataset1_pacientes.csv`, 55 filas), $\rho = 0.95$ produce aproximadamente 45–55 clusters (uno por fila). Para obtener una partición interpretable, bajá $\rho$ a un valor en el rango $[0.4, 0.7]$ y volvé a correr. Ver `docs/05_corridas_y_evaluacion.md` para los valores usados en el TFI.

**Q2. ¿Por qué algunos clusters aparecen como "exemplar vacío" (`[0 0 0 0 ...]`)?**
Cuando una fila del CSV binariza a todos-ceros (todos los features bajo sus respectivos umbrales), ART1 crea un cluster con ese vector como exemplar. Estos clusters suelen tener un solo miembro y representan el caso "ningún factor de riesgo activo" (en pacientes) o "ningún indicador operativo fuera de rango" (en sensores). Se reportan **por separado** en el TXT para no contaminar la lectura principal, porque su presencia infla artificialmente el conteo $K$. Ver `docs/06_limitaciones_y_etica.md` §1 y `docs/informe_corridas.md` §6.1 para el análisis cuantitativo de su frecuencia.

**Q3. ¿Puedo usar el módulo con features categóricos o continuos sin binarizar?**
No directamente. ART1 requiere entradas binarias (0 o 1) según el paper Lau (1992). Para features continuos, definí un umbral por feature en `data/metadata.csv` (una fila por umbral), usando la `rule` que corresponda (`gte` para "mayor o igual", `lte` para "menor o igual", etc.). Para features categóricos, primero one-hot encodeá las categorías y luego mapeá cada columna one-hot a un threshold = 1 en el metadata con `rule=gte` (categoría presente = bit activo, ausente = bit apagado). Ver `docs/03_dataset_y_preprocesamiento.md` para ejemplos completos de binarización.

**Q4. ¿Por qué dos corridas con el mismo input y mismo $\rho$ producen asignaciones distintas?**
ART1 es sensible al orden de presentación de los datos: procesa las entradas en serie, una a una, en el orden en que aparecen. Es un comportamiento esperado del algoritmo, no un bug del módulo. Para cuantificar esta sensibilidad, usá el flag `--shuffle N`: ejecuta N corridas con órdenes barajados del dataset y reporta el acuerdo *pairwise* medio contra la corrida base (run 0). En nuestros experimentos sobre 55 filas, el acuerdo promedio observado es ~0.50, con desvío estándar entre 0.057 y 0.246 según la celda (`results/resumen_corridas.md`). Si necesitás una asignación invariante al orden, deberías complementar ART1 con un mecanismo de votación o consenso, lo que está fuera del alcance de este TFI.

**Q5. ¿Por qué el dataset de sensores produce siempre 5 clusters para cualquier $\rho$?**
El dataset `dataset2_sensores.csv` tiene 55 filas pero sólo 6 vectores únicos en el espacio binario de 8 dimensiones (la octava dimensión es el desdoblamiento del voltaje en `voltaje_lt` + `voltaje_gt`). ART1 produce 5 clusters efectivos, no porque el dataset sea rígidamente de 5 clases, sino porque el sexto vector único (presente en la fila S027) **queda absorbido por el cluster 2 existente durante el fit** por un artefacto del *fast learning* combinado con el orden de presentación. Sobre los 5 vectores que efectivamente terminan como exemplares, $\rho$ deja de tener margen para discriminar: cualquiera de ellos supera fácilmente cualquier vigilancia por debajo de 1.0 porque ya son vectores discretos. Este dataset aporta información sobre **consistencia** del algoritmo, no sobre su respuesta a la vigilancia. Ver `docs/informe_corridas.md` §4.1 para el análisis completo.

**Q6. ¿Cómo comparo ART1 con K-means u otros algoritmos de clustering?**
Este TFI implementa únicamente ART1, según lo exige la consigna oficial. La comparación cuantitativa con K-means, DBSCAN, clustering jerárquico o Spectral Clustering queda fuera del alcance del trabajo. Si en el futuro se quisiera hacer esa comparación, el camino natural sería: (1) implementar K-means sobre la misma matriz binarizada usando `scikit-learn.cluster.KMeans`; (2) ejecutar ambos algoritmos con el mismo número objetivo de clusters $K$ (o variar $K$ en ambos y comparar curvas); (3) calcular el **Adjusted Rand Index (ARI)** entre las asignaciones, métrica estándar para comparar particiones con o sin etiquetas verdaderas. Ninguno de estos pasos se entrega en el TFI actual.

---

## 9. Referencias

### 9.1. Documentación interna del proyecto

- [`docs/README.md`](README.md) — índice general del proyecto y orden de lectura sugerido.
- [`docs/01_marco_teorico.md`](01_marco_teorico.md) — marco teórico de ART1, dilema estabilidad–plasticidad y posición en la taxonomía de Lippmann/Lau.
- [`docs/02_problema_y_alcance.md`](02_problema_y_alcance.md) — qué hace y qué NO hace el sistema, motivación, usuarios objetivo y mensaje honesto.
- [`docs/03_dataset_y_preprocesamiento.md`](03_dataset_y_preprocesamiento.md) — datasets canónicos y reglas de binarización con justificación clínica/operativa.
- [`docs/04_algoritmo.md`](04_algoritmo.md) — transcripción comentada del Box 3 (Lau 1992) paso a paso, con notación idéntica al paper.
- [`docs/05_corridas_y_evaluacion.md`](05_corridas_y_evaluacion.md) — diseño experimental completo, métricas no supervisadas y formato de las tablas de resultados.
- [`docs/06_limitaciones_y_etica.md`](06_limitaciones_y_etica.md) — limitaciones técnicas de ART1, consideraciones éticas del dominio clínico, disclaimer formal y vacíos para producción real.
- [`docs/07_iteraciones.md`](07_iteraciones.md) — bitácora de iteraciones del TFI.
- [`docs/informe_corridas.md`](informe_corridas.md) — informe narrativo de las 30 corridas realizadas, con interpretación clínica tentativa y análisis de estabilidad.

### 9.2. Documentación externa y artefactos

- [`_legacy/CarGross_TP/consignas_TP.md`](../_legacy/CarGross_TP/consignas_TP.md) — consigna oficial del TFI (UADER — IDTI Lab, materia *Redes Neuronales*), actividades obligatorias #1–#6.
- [`_legacy/CarGross_TP/lau_contenido.md`](../_legacy/CarGross_TP/lau_contenido.md) — transcripción completa en español del paper Lau (1992), fuente primaria del algoritmo Box 3.
- [`Lau.pp5.a.11.pdf`](../Lau.pp5.a.11.pdf), [`Lau.pp12.a.14.pdf`](../Lau.pp12.a.14.pdf) — papers originales provistos con la consigna.
- [`src/CarGross.py`](../src/CarGross.py) — implementación del módulo (660 líneas). Contiene el manual extendido (`--man`), el smoke test (`--test`), y todos los flags documentados en §5.
- [`data/metadata.csv`](../data/metadata.csv) — 14 reglas de binarización con justificación trazable a guías AHA / ATP III / ADA / OMS / ISO 10816.
- [`data/dataset1_pacientes.csv`](../data/dataset1_pacientes.csv) — 55 pacientes simulados con 7 features clínicos.
- [`data/dataset2_sensores.csv`](../data/dataset2_sensores.csv) — 55 lecturas de sensores simulados con 7 features operativos (8 después de binarizar voltaje).
- [`results/resumen_corridas.md`](../results/resumen_corridas.md) — tablas agregadas de las 30 corridas; fuente de los números cuantitativos citados en este manual.
- [`results/r_pacientes_r{0.4,0.6,0.8}_s{42-46}.txt`](../results/) — reportes TXT individuales del dataset pacientes, 15 archivos.
- [`results/r_sensores_r{0.5,0.65,0.8}_s{42-46}.txt`](../results/) — reportes TXT individuales del dataset sensores, 15 archivos.
- [`requirements.txt`](../requirements.txt) — dependencias opcionales para análisis posterior (no requeridas por el módulo).

### 9.3. Referencias bibliográficas

- [1] Lau, C. (Ed.) (1992). *Artificial Neural Networks: Concepts and Control Applications*. IEEE Press. Box 3, pp. 12–14 ("El clasificador Carpenter/Grossberg"). Referencia primaria del algoritmo implementado.
- [2] Carpenter, G.A. & Grossberg, S. (1987). *A massively parallel architecture for a self-organizing neural pattern recognition machine*. Computer Vision, Graphics, and Image Processing, 37, 54–115. Paper original de ART1.
- [3] Lippmann, R.P. (1987). *An Introduction to Computing with Neural Nets*. IEEE ASSP Magazine, April 1987, pp. 4–22. Reproducido en [1]. Taxonomía de redes neuronales útiles para clasificación.
- [4] Sánchez-Sinencio, E. & Lau, C. (Eds.) (1992). *Artificial Neural Networks*. IEEE Press. Referencia completa del libro del que se transcriben los capítulos 5–14 provistos como material de la consigna.

---

> **Descargo de responsabilidad**: este manual describe una herramienta **educativa** desarrollada como Trabajo Final Integrador de la materia *Redes Neuronales* de la UADER — IDTI Lab. **No** constituye una herramienta de diagnóstico médico, prescripción farmacológica ni derivación a especialistas. Toda decisión clínica debe ser tomada por profesionales médicos matriculados sobre la base de evidencia clínica validada, y está fuera del alcance de este trabajo. Ver el disclaimer completo en `docs/06_limitaciones_y_etica.md` §3.