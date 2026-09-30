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
- **Sin dependencias externas** en tiempo de ejecución (pure stdlib: `csv`, `argparse`, `random`, `math`, `os`)
- **Datasets en `.csv` o `.xlsx`**: el formato se detecta automáticamente por la extensión del archivo
  (también en mayúsculas). Para `.xlsx` se usa `openpyxl` si está instalado; si no, se usa un lector
  interno basado en `zipfile` + `xml.etree`, por lo que **no es obligatorio instalar nada**.


-----------------------------------------------------------------------------------------------------------------------

## Instalación

```powershell
git clone <url-del-repo>
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt  
```
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
python src/CarGross.py data/dataset_entrenamiento_ART1.xlsx -o results/resultados_entrenamiento.csv --save-txt results/resultados_entrenamiento.txt -v
```

Aplica ART1 al dataset clínico (100 pacientes y 5 sintomas). Genera:

- `CONSOLA`:
- La consola al momento de la ejecucion nos brinda una descripcion. sobre el archivo ejecutado, cuanta cantidad de pacientes y sintomas se entreno la red. Y nos brinda una informacion sobre RESUMEN DE EVALUACION Y METRICAS, Nos arma una matriz con los cluster que fueron necesario crear por la red. 

- `ARCHIVOS`:
- `results/resultados_entrenamiento.csv`: 100 filas con la asignación de cluster por paciente.
- `results/resultados_entrenamiento.txt`: reporte narrativo con los clusters formados ART1 - DIAGNOSTICO MEDICO

-----------------------------------------------------------------------------------------------------------------------

### 3. Ejecución con Dataset de Validación

```powershell
python src/CarGross.py data/dataset_validacion_ART1.xlsx -r 0.65 -o results/resultados_validacion.csv --save-txt results/resultados_validacion.txt -
```

Aplica ART1 al dataset clínico (100 pacientes y 5 sintomas). Genera:

- `CONSOLA`:
- La consola al momento de la ejecucion nos brinda una descripcion. sobre el archivo ejecutado, cuanta cantidad de pacientes y sintomas se entreno la red. Y nos brinda una informacion sobre RESUMEN DE EVALUACION Y METRICAS, Nos arma una matriz con los cluster que fueron necesario crear por la red. 

- `ARCHIVOS`:
- `results/resultados_validacion.csv`: 100 filas con la asignación de cluster por paciente.
- `results/resultados_validacion.txt`: reporte narrativo con los clusters formados ART1 - DIAGNOSTICO MEDICO

-----------------------------------------------------------------------------------------------------------------------

### 4. Consulta Interactivas (Paciente Individual)

```powershell
python src/CarGross.py --input data/dataset_entrenamiento_ART1.xlsx --interactive -o results/informe_paciente_individual.csv --save-txt results/informe_paciente_individual.txt
```

- `CONSOLA`:
- En esta sesion nos permite cargar un paciente nuevo interatuando con la consola, se le pregunta el nombre o id, en caso de no saber que ponerle se le apreta enter y el sistema le asignara uno por defecto, nos pregunta que sintomas tenemos y tenemos que cargarlo con 1 si lo tiene y cero si no lo tiene. En la consola va a aprecera un pequeño informe de que sintomas tiene y a que especialidad medica se le asigna y algunas pequeñas recomendacion sobre su problemas de salud, siempre advirtiendo que las recomendaciones tienen que estar supervisada por un profesional de la salud, porque este sistema tecnologico da sugerencias no 100% confiable para la tomar decisiones sobre la salud humana. 

- `ARCHIVOS`:
- `results/informe_paciente_individual.csv`: nos genera un archivo con el informe del paciente indivualmente.

-----------------------------------------------------------------------------------------------------------------------

### 5. Manual extendido

```powershell
python src/CarGross.py --man > docs/manual_referencia.txt
```

Con este comando creamos un archivo.txt en la carpeta docs donde encontraremos una documentacion sobre el funcionamiento de la red neuronal y se detallan paso por paso el manual de intrucciones para que se utiliza cada comando al momento que ejecutamos la consola.

-----------------------------------------------------------------------------------------------------------------------

### 6. Test de estabilidad con barajado

```powershell
python src/CarGross.py data/dataset_entrenamiento_ART1.xlsx -r 0.60 --shuffle 5 --seed 42 -o results/estable.csv --save-txt results/estable.txt
```

Ejecuta la corrida **5 veces** con órdenes aleatorios de entrada y reporta el **acuerdo pairwise** entre corridas en el TXT. Mide la sensibilidad de ART1 al orden de presentación, da un informe por consola y crea un archivo con los detalles de cada corrida aleatoria en **results/estable.txt**

-----------------------------------------------------------------------------------------------------------------------

## Estructura del proyecto

```
IA-red-neuronal/
├── src/
│ └── CarGross.py
├── data/
│ ├── dataset_entrenamiento_ART1.xlsx
│ └── dataset_validacion_ART1.xlsx
├── docs/
│ └── manual_referencia.txt
├── results/
│ └── (archivos .csv y .txt generados)
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