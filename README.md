# Redes de Carpenter-Grossberg (ART1)

Implementación de la red de Carpenter-Grossberg (Adaptive Resonance Theory 1) según Box 3 de Lau (1992), aplicada a clustering no supervisado de datos tabulares clínicos y operativos.

Trabajo Final Integrador de la materia **Redes Neuronales** — UADER, IDTI Lab.

---

## Algoritmo

ART1 es un algoritmo de clustering no supervisado que forma categorías de manera incremental a partir de vectores de entrada binarios. El parámetro de vigilancia `ρ ∈ [0,1]` controla la rigidez del matching: a mayor `ρ`, los clusters son más finos y específicos; a menor `ρ`, se generaliza más. Las entradas se obtienen binarizando features continuas con umbrales definidos en `data/metadata.csv`.

---

## Requisitos

- **Python 3.10+**
- **Sin dependencias externas** en tiempo de ejecución (pure stdlib: `csv`, `argparse`, `random`, `math`, `os`)
- Opcional: `pandas`, `matplotlib` listados en `requirements.txt` para análisis posterior de los CSV de salida

---

## Instalación

```powershell
git clone <url-del-repo>
cd ia2026
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt   # opcional, solo para análisis
python src/CarGross.py --test     # smoke test
```

Si `--test` imprime `TEST PASSED` y termina con código de salida `0`, la instalación está OK. Los comandos son portables a bash cambiando solo la activación del entorno virtual.

---

## Cómo ejecutar / pruebas

> Todos los comandos se asumen ejecutados desde la raíz del proyecto. En bash son análogos, salvo la activación del venv.

### 1. Smoke test

```powershell
python src/CarGross.py --test
```

Ejecuta un caso sintético pequeño y verifica que la red aprende correctamente. Imprime `TEST PASSED` y termina con código `0`. Útil para validar la instalación sin tocar datos reales.

### 2. Demo con dataset de pacientes (ρ=0.6)

```powershell
python src/CarGross.py data/dataset1_pacientes.csv -r 0.6 --output results/demo.csv --save-txt results/demo.txt --verbose
```

Aplica ART1 al dataset clínico (55 pacientes × 8 features). Genera:

- `results/demo.csv`: 55 filas con la asignación de cluster por paciente.
- `results/demo.txt`: reporte narrativo con los clusters formados — típicamente **3 perfiles**: "edad", "multi-riesgo" y "cardio-metabólico".

### 3. Demo con dataset de sensores (ρ=0.65)

```powershell
python src/CarGross.py data/dataset2_sensores.csv -r 0.65 --output results/demo_sensores.csv --save-txt results/demo_sensores.txt --verbose
```

Aplica ART1 al dataset operativo (55 lecturas × 8 features). Genera los CSV y TXT equivalentes. En este dataset la cantidad de clusters es **invariante a `ρ`** porque el dataset solo tiene 6 vectores únicos; la red estabiliza en **5 clusters** por absorción de S027 en otro nodo.

### 4. Manual extendido

```powershell
python src/CarGross.py --man
```

Imprime el manual completo en español con descripción de parámetros, formato de E/S y notas de uso. Útil para referencia rápida sin abrir el código.

### 5. Test de estabilidad con barajado

```powershell
python src/CarGross.py data/dataset1_pacientes.csv -r 0.6 --shuffle 5 --seed 42 --output results/estable.csv --save-txt results/estable.txt
```

Ejecuta la corrida **5 veces** con órdenes aleatorios de entrada (semilla base 42) y reporta el **acuerdo pairwise** entre corridas en el TXT. Mide la sensibilidad de ART1 al orden de presentación.

---

## Estructura del proyecto

```
ia2026/
├── .gitignore
├── README.md
├── requirements.txt
├── CONTRIBUTING.md
├── Lau.pp12.a.14.pdf          # paper de referencia (Box 3, pp. 12-14)
├── Lau.pp5.a.11.pdf           # introducción del Lau (pp. 5-11)
├── docs/                      # 11 documentos del TFI (ver índice abajo)
├── src/
│   └── CarGross.py            # implementación ART1 (~660 líneas) — incluye DataLoader y excepciones
├── data/
│   ├── dataset1_pacientes.csv # 55 pacientes × 8 features continuas
│   ├── dataset2_sensores.csv  # 55 lecturas × 8 features continuas
│   └── metadata.csv           # 14 umbrales de binarización
├── results/
│   ├── resumen_corridas.md    # tabla consolidada de las 30 corridas
│   └── r_<dataset>_r<rho>_s<seed>.{csv,txt}  # 60 archivos (30 × 2)
├── tests/                     # smoke test (entrypoint: src/CarGross.py --test)
└── _legacy/
    ├── README.md
    └── CarGross_TP/           # intento anterior — NO entregado
```

---

## Documentación

El índice completo está en [`docs/README.md`](docs/README.md).

| Documento | Propósito |
|-----------|-----------|
| `01_marco_teorico.md` | Teoría ART1 y fundamentos |
| `02_problema_y_alcance.md` | Alcance, objetivos y limitaciones |
| `03_dataset_y_preprocesamiento.md` | Datasets, features y binarización |
| `04_algoritmo.md` | Box 3 paso a paso, en pseudocódigo |
| `05_corridas_y_evaluacion.md` | Diseño experimental y métricas |
| `06_limitaciones_y_etica.md` | Ética, sesgos y disclaimer |
| `07_iteraciones.md` | Bitácora del proceso de desarrollo |
| `informe_corridas.md` | Informe narrativo de las 30 corridas |
| `manual_referencia.md` | Manual formal de la implementación |
| `ppt_outline.md` | Outline de la presentación con audio |
| `caratula_template.md` | Template de la carátula |

---

## Resultados

- **`results/resumen_corridas.md`** consolida las **30 corridas** en una tabla: 15 pacientes (ρ ∈ {0.4, 0.6, 0.8} × 5 semillas) + 15 sensores (idem).
- **60 archivos individuales** en `results/`: 30 CSV con asignación de clusters por fila + 30 TXT con reporte narrativo (perfiles, tamaños, métricas de cohesión).

### Hallazgos clave

1. **Pacientes — `ρ` controla la granularidad**: ρ=0.4 produce pocos clusters muy generales; ρ=0.6 produce 3 perfiles interpretables ("edad", "multi-riesgo", "cardio-metabólico"); ρ=0.8 fragmenta más y aparecen singletons.
2. **Sensores — rigidez estructural**: el dataset tiene solo 6 vectores únicos. ART1 no puede formar más categorías que las que existen en los datos; absorbe S027 en otro nodo y se estabiliza en **5 clusters** independientemente de `ρ`.

---

## Integrantes

- Escar, Camilo
- Gonzalez, Claudio
- Laballeja, Sofia
- Meriano, Patricia

---

## Referencias

- **Lau, C. (1992).** *Adaptive Resonance Theory*, Box 3, pp. 12-14. Transcripción disponible en `_legacy/CarGross_TP/lau_contenido.md`. PDF original en `Lau.pp12.a.14.pdf`.
- **Carpenter, G. A. & Grossberg, S. (1987).** *ART 2: Self-organization of stable category recognition codes for analog input patterns*. Applied Optics.
