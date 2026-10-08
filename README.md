# Redes de Carpenter-Grossberg (ART1)

Implementación de la red de Carpenter-Grossberg (Adaptive Resonance Theory 1) según Box 3 de Lau (1992), aplicada a clustering no supervisado de datos tabulares clínicos y operativos.

Trabajo Final Integrador de la materia **Redes Neuronales** — UADER, IDTI Lab.

-------------------------------------------------------------------------------------------

============================================================================================
 NOTA IMPORTANTE: Los informes generado mediante Inteligencia Artificial (Red ART1)
 con Aprendizaje No Supervisado. Debe ser validado y supervisado por un profesional médico.
============================================================================================

-----------------------------------------------------------------------------------------------------------------------

## Algoritmo

ART1 (Adaptive Resonance Theory 1) es un algoritmo de clustering no supervisado diseñado para agrupar patrones sintomáticos de manera incremental a partir de vectores de entrada binarios, donde cada variable representa la presencia (`1`) o ausencia (`0`) de un síntoma [1, 2].

El nivel de exigencia para la asignación de pacientes a las distintas categorías está regulado por el **parámetro de vigilancia** `ρ ∈ [0,1]` 

`Si el marching ratio es:` 
`* ** mayor que p:` el paciente es aceptado en el cluster y la plantilla del grupo se actualiza mediante un AND logico.
`* ** menor que p:` El grupo es rechazado temporalmente y la red prueba con la siguiente categoria. Si no encuentra ningun grupo crea un cluster nuevo para ese paciente.


-----------------------------------------------------------------------------------------------------------------------

## Requisitos

- **Python 3.10+**
- **Entrada en formato CSV**: los dos datasets incluidos están en `data/*.csv`. El módulo
  también acepta `.xlsx`, detectando el formato por la extensión.
- **Los `.csv` se procesan sin dependencias externas** (biblioteca estándar de Python).
- **`pip install -r requirements.txt`** instala lo opcional: `openpyxl` (para leer `.xlsx`),
  `pandas` / `matplotlib` (para el notebook) y `nbconvert` / `ipykernel` (para ejecutar el
  notebook desde la terminal).


-----------------------------------------------------------------------------------------------------------------------

## Instalación

```powershell
git clone <url-del-repo>
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt  
```

-----------------------------------------------------------------------------------------------------------------------

## Se puede ejecutar el archivo CarGross.py siguiendo los pasos que se detallan abajo 
## o se puede ejecutar el archivo notebooks/01_demo_art1.ipynb, en ese archivo se va generando una breve
## explicacion por cada bloque de ejecucion de codigo. Ademas cuenta con graficos estadisticos sobre las distintas
## ejecuciones y datos procesados.

-----------------------------------------------------------------------------------------------------------------------

## Cómo ejecutar / pruebas

> Todos los comandos se asumen ejecutados desde la raíz del proyecto. En bash son análogos, salvo la activación del venv.

### 1. Autotest de verificación

```powershell
python src/CarGross.py --test
```

Ejecuta un caso sintético pequeño y verifica que la red aprende correctamente. Imprime `AUTOTEST EXITOSO`

-----------------------------------------------------------------------------------------------------------------------

### 2. Ejecución con Dataset de Entrenamiento

```powershell
python src/CarGross.py data/dataset_entrenamiento_ART1.csv -o results/resultados_entrenamiento.csv --save-txt results/resultados_entrenamiento.txt -v
```

Aplica ART1 al dataset clínico (100 pacientes y 5 síntomas). Con `rho = 0.65` esperado: **14 clusters** y matching ratio **0.9500**. Genera:

- `CONSOLA`:
- Cabecera con la ruta de entrada y el rho, la línea "Carga completada: 100 registros, 5 síntomas.",
  el "RESUMEN DE EVALUACIÓN Y MÉTRICAS" (clusters, matching ratio, estabilidad y pureza si hay
  etiquetas) y las plantillas de cada cluster con su especialidad.

- `ARCHIVOS`:
- `results/resultados_entrenamiento.csv`: 100 filas con la asignación de cluster por paciente.
- `results/resultados_entrenamiento.txt`: reporte narrativo con los clusters formados ART1 - DIAGNOSTICO MEDICO

-----------------------------------------------------------------------------------------------------------------------

### 3. Ejecución con Dataset de Validación

```powershell
python src/CarGross.py data/dataset_validacion_ART1.csv -r 0.65 -o results/resultados_validacion.csv --save-txt results/resultados_validacion.txt
```

Aplica ART1 al dataset clínico (100 pacientes y 5 síntomas) y además calcula **pureza**, porque trae la columna de especialidad esperada. Con `rho = 0.65` esperado: **12 clusters**, matching ratio **0.9600** y **pureza 69%**. Genera:

- `CONSOLA`:
- Cabecera con la ruta de entrada y el rho, la línea "Carga completada: 100 registros, 5 síntomas.",
  el "RESUMEN DE EVALUACIÓN Y MÉTRICAS" (clusters, matching ratio, estabilidad y pureza si hay
  etiquetas) y las plantillas de cada cluster con su especialidad.

- `ARCHIVOS`:
- `results/resultados_validacion.csv`: 100 filas con la asignación de cluster por paciente.
- `results/resultados_validacion.txt`: reporte narrativo con los clusters formados ART1 - DIAGNOSTICO MEDICO

-----------------------------------------------------------------------------------------------------------------------

### 4. Consulta Interactivas (Paciente Individual)

```powershell
python src/CarGross.py --input data/dataset_entrenamiento_ART1.csv --interactive -o results/informe_paciente_individual.csv --save-txt results/informe_paciente_individual.txt
```

- `CONSOLA`:
- En esta sesion nos permite cargar un paciente nuevo interatuando con la consola, se le pregunta el nombre o id, en caso de no saber que ponerle se le apreta enter y el sistema le asignara uno por defecto, nos pregunta que sintomas tenemos y tenemos que cargarlo con 1 si lo tiene y cero si no lo tiene. En la consola va a aprecera un pequeño informe de que sintomas tiene y a que especialidad medica se le asigna y algunas pequeñas recomendacion sobre su problemas de salud, siempre advirtiendo que las recomendaciones tienen que estar supervisada por un profesional de la salud, porque este sistema tecnologico da sugerencias no 100% confiable para la tomar decisiones sobre la salud humana. 

- `ARCHIVOS`:
- `results/informe_paciente_individual.csv`: nos genera un archivo con el informe del paciente indivualmente.

-----------------------------------------------------------------------------------------------------------------------

### 5. Manual de referencia

```powershell
python src/CarGross.py --man
```

El manual completo está en el repo en `docs/manual_referencia.txt` (se regenera con el
comando de arriba). Tiene: descripción, arquitectura, sintaxis, ejemplos, proceso de
**instalación**, **datasets y salida esperada**, **alcances y limitaciones** y
**preguntas frecuentes (FAQ)**. El informe con las corridas reales y su evaluación
está en [`docs/informe.md`](docs/informe.md).

-----------------------------------------------------------------------------------------------------------------------

### 6. Test de estabilidad con barajado

```powershell
python src/CarGross.py data/dataset_entrenamiento_ART1.csv -r 0.65 --shuffle 5 --seed 42 -o results/estable.csv --save-txt results/estable.txt
```

Ejecuta la corrida **5 veces** con órdenes aleatorios de entrada y reporta por consola los clusters y el matching ratio de cada corrida más el promedio contra la base. Mide la sensibilidad de ART1 al orden de presentación (con estos datasets da ±1 o ±2 clusters) y deja el detalle en **results/estable.txt**.

-----------------------------------------------------------------------------------------------------------------------

## Estructura del proyecto

```
IA-red-neuronal/
├── src/
│   └── CarGross.py
├── notebooks/
│   └── 01_demo_art1.ipynb
├── data/
│   ├── dataset_entrenamiento_ART1.csv     # entrada (formato CSV)
│   ├── dataset_validacion_ART1.csv
│   ├── dataset_entrenamiento_ART1.xlsx    # misma data en Excel
│   └── dataset_validacion_ART1.xlsx
├── docs/
│   ├── manual_referencia.txt
│   └── informe.md
├── results/          # archivos generados (no se versionan)
└── requirements.txt
```

-----------------------------------------------------------------------------------------------------------------------

## Integrantes

- Escar, Camilo
- Gonzalez, Claudio
- Laballeja, Sofia
- Meriano, Patricia

-----------------------------------------------------------------------------------------------------------------------

## Referencias

- **Lau, C. (1992).** *Adaptive Resonance Theory*, Box 3, pp. 12-14. Transcripción disponible en `_legacy/CarGross_TP/lau_contenido.md`. PDF original en `Lau.pp12.a.14.pdf`.
- **Carpenter, G. A. & Grossberg, S. (1987).** *ART 2: Self-organization of stable category recognition codes for analog input patterns*. Applied Optics.

-----------------------------------------------------------------------------------------------------------------------