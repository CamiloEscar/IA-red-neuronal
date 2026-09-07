# CarGross — Red Neuronal ART1 (Carpenter/Grossberg)

Implementación del algoritmo **ART1 (Adaptive Resonance Theory 1)** para
clustering no supervisado, basado en el trabajo de Carpenter & Grossberg
y descripto en Lau, C. (Ed.), *Artificial Neural Networks*, IEEE Press, 1992.

---

## Estructura del proyecto

```
CarGross_TP/
├── src/
│   ├── CarGross.py       ← Red ART1: punto de entrada principal
│   ├── data_loader.py    ← Lectura de CSV y metadatos
│   ├── utils.py          ← Binarización, reportes, man page
│   └── __init__.py
├── data/
│   ├── dataset1.csv      ← Datos médicos (55 pacientes, 8 variables)
│   ├── dataset2.csv      ← Sensores industriales (55 registros, 8 variables)
│   └── metadata/
│       └── metadata.csv  ← Umbrales de binarización custom (opcional)
├── tests/
│   └── test_demo.py      ← Suite de tests unitarios e integración
├── results/
│   ├── resultado_dataset1.txt
│   ├── resultado_dataset2.txt
│   └── graficos/
├── docs/
│   ├── manual.md
│   └── informe.pdf
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Instalación

No requiere dependencias externas. Solo Python 3.6+.

```bash
# Verificar Python
python --version   # debe ser 3.6 o superior

# Clonar / descomprimir el proyecto y entrar a la carpeta
cd CarGross_TP
```

---

## Uso rápido

```bash
# Corrida básica
python src/CarGross.py data/dataset1.csv

# Con vigilancia personalizada y salida CSV
python src/CarGross.py data/dataset1.csv --vigilance 0.7 --output results/out.csv

# Guardar resumen en .txt
python src/CarGross.py data/dataset1.csv --save-txt results/resultado_dataset1.txt

# Con metadatos custom de umbrales
python src/CarGross.py data/dataset1.csv --metadata data/metadata/metadata.csv

# Ver manual completo
python src/CarGross.py --help
```

---

## Parámetros

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--vigilance` | `0.5` | Umbral rho ∈ [0.0, 1.0]. Alto = más clusters. |
| `--max-clusters` | `50` | Límite de clusters a crear. |
| `--output` | — | Ruta del CSV de salida con columna `cluster`. |
| `--metadata` | — | CSV con umbrales custom por columna. |
| `--save-txt` | — | Guarda el resumen en un archivo `.txt`. |
| `--verbose` | — | Muestra el detalle de cada paso del algoritmo. |

---

## Ejecutar los tests

```bash
python tests/test_demo.py
```

Salida esperada: todos los tests en `[PASS]`.

---

## Datasets incluidos

| Archivo | Descripción | Filas | Variables |
|---------|-------------|-------|-----------|
| `dataset1.csv` | Datos clínicos de pacientes (edad, presión, colesterol, etc.) | 55 | 8 |
| `dataset2.csv` | Sensores industriales (temperatura, vibración, RPM, etc.) | 55 | 8 |

---

## Referencias

- Carpenter, G.A. & Grossberg, S. (1987). *A massively parallel architecture for a self-organizing neural pattern recognition machine.*
- Lau, C. (Ed.) (1992). *Artificial Neural Networks.* IEEE Press, pp. 12–14.
- Lippmann, R.P. (1987). *An Introduction to Computing with Neural Nets.* IEEE ASSP Magazine.
