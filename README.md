# Red de Carpenter-Grossberg (ART1)

Implementación de la red ART1 (Adaptive Resonance Theory 1) según el Box 3 de Lau (1992), aplicada al agrupamiento sin supervisión de pacientes a partir de sus síntomas.

Trabajo Final Integrador de Inteligencia Artificial, UADER.

Los resultados son orientativos y tienen que ser revisados por un profesional médico.

## Cómo funciona

ART1 agrupa vectores binarios: cada paciente es una lista de 0 y 1 que indica qué síntomas tiene. Los pacientes se van procesando de a uno. Para cada uno se busca la categoría más parecida y se calcula el matching ratio, que es cuántos de sus síntomas coinciden con la plantilla de esa categoría. Ese valor se compara con el parámetro de vigilancia `rho` (entre 0 y 1):

- Si el matching ratio es mayor o igual que `rho`, el paciente entra en la categoría y la plantilla se actualiza con un AND lógico.
- Si es menor, se descarta esa categoría y se prueba con la siguiente. Si ninguna alcanza, se crea una categoría nueva.

## Requisitos e instalación

Hace falta Python 3.10 o superior. Los datasets están en `data/*.csv` y se leen con la biblioteca estándar, así que para correr `CarGross.py` con CSV no hay que instalar nada. También acepta `.xlsx`.

`requirements.txt` incluye lo opcional: `openpyxl` (para `.xlsx`), `pandas` y `matplotlib` (notebook) y `nbconvert` e `ipykernel` (para ejecutar el notebook desde la terminal).

```powershell
git clone <url-del-repo>
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Notebook y visualizador

`notebooks/01_demo_art1.ipynb` recorre el código bloque por bloque, con una explicación en cada uno y gráficos de las corridas.

`visualizador/index.html` muestra ART1 paso a paso en el navegador. Se abre con doble click, sin servidor ni instalación. Con la tecla E se activa el modo exposición.

## Cómo ejecutar

Los comandos se corren desde la raíz del proyecto.

### 1. Autotest

```powershell
python src/CarGross.py --test
```

Corre un caso chico y, si sale bien, imprime `TEST PASSED`.

### 2. Dataset de entrenamiento

```powershell
python src/CarGross.py data/dataset_entrenamiento_ART1.csv -o results/resultados_entrenamiento.csv --save-txt results/resultados_entrenamiento.txt -v
```

Procesa 100 pacientes con 5 síntomas. Con `rho = 0.65` da 14 clusters y matching ratio 0.9500.

Por consola muestra la ruta y el rho, la cantidad de registros y síntomas cargados, el resumen de métricas (clusters, matching ratio, estabilidad y pureza si el dataset tiene etiquetas) y la plantilla de cada cluster con su especialidad. `-v` agrega la lista de síntomas detectados.

Genera `results/resultados_entrenamiento.csv` (una fila por paciente con su cluster) y `results/resultados_entrenamiento.txt` (el reporte completo).

### 3. Dataset de validación

```powershell
python src/CarGross.py data/dataset_validacion_ART1.csv -r 0.65 -o results/resultados_validacion.csv --save-txt results/resultados_validacion.txt
```

Es igual que el anterior, pero este dataset trae una columna "Especialidad esperada", así que además calcula la pureza. Con `rho = 0.65` da 12 clusters, matching ratio 0.9600 y pureza 69%.

### 4. Paciente individual (interactivo)

```powershell
python src/CarGross.py --input data/dataset_entrenamiento_ART1.csv --interactive --save-txt results/informe_paciente_individual.txt
```

Primero entrena con el dataset y después pide un nombre o ID (con Enter se asigna uno por defecto) y, síntoma por síntoma, 1 si lo tiene o 0 si no. Imprime una ficha con los síntomas, el cluster asignado, la especialidad sugerida y algunas recomendaciones, y la guarda en el TXT. Las recomendaciones son orientativas y no reemplazan a un profesional de la salud.

Para ver la ficha de un paciente que ya está en el dataset se usa `--patient-id` (por ejemplo `--patient-id P104` con el dataset de validación).

### 5. Manual de referencia

```powershell
python src/CarGross.py --man
```

Imprime el mismo texto que está en `docs/manual_referencia.txt`: opciones, ejemplos, instalación, salida esperada, limitaciones y preguntas frecuentes. Las corridas con distintos valores de `rho` están en [`docs/informe.md`](docs/informe.md).

### 6. Estabilidad con el orden mezclado

```powershell
python src/CarGross.py data/dataset_entrenamiento_ART1.csv -r 0.65 --shuffle 5 --seed 42 -o results/estable.csv --save-txt results/estable.txt
```

Repite la corrida 5 veces con los pacientes en orden aleatorio y muestra los clusters y el matching ratio de cada una, más el promedio. Sirve para ver cuánto depende ART1 del orden de entrada: con estos datasets la cantidad de clusters varía en 1 o 2.

## Estructura

```
src/CarGross.py            implementación y línea de comandos
notebooks/                 notebook con la demostración
visualizador/index.html    visualización paso a paso
data/                      datasets en CSV y XLSX
docs/                      manual e informe de corridas
results/                   salidas generadas (no se versionan)
requirements.txt
```

## Integrantes

- Escar, Camilo
- Gonzalez, Claudio
- Laballeja, Sofia
- Meriano, Patricia

## Referencias

- Lau, C. (1992). _Adaptive Resonance Theory_, Box 3, pp. 12-14. Transcripción en `_legacy/CarGross_TP/lau_contenido.md`; el PDF original es `Lau.pp12.a.14.pdf` (local, los PDF no se versionan).
- Carpenter, G. A. y Grossberg, S. (1987). A massively parallel architecture for a self-organizing neural pattern recognition machine. _Computer Vision, Graphics, and Image Processing_, 37, 54-115.
